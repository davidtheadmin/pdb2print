"""Thin FastAPI layer for pdb2print.

Serves the static front end in ``frontend/`` and wraps the existing geometry
pipeline. All geometry lives in the ``pdb2print`` package; this module only
maps HTTP form fields to :class:`PrintParams`, calls ``build_all``, writes the
export files, and returns their URLs. Run with::

    uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import uuid

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pdb2print.config import (
    PrintParams, Representation, MinWallMode, BaseStyle, BackboneStyle,
    ConnectionParams, NoMagnetMethod, MagnetShape,
    color_for_index,
)
from pdb2print.pipeline import build_all
from pdb2print import export


HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(HERE, "frontend")

# Every generation writes its outputs into a fresh sub-directory here, which is
# served read-only at /files/<token>/... for both <model-viewer> and downloads.
OUTPUT_ROOT = tempfile.mkdtemp(prefix="pdb2print_out_")

UPLOAD_EXTS = {".pdb", ".ent", ".cif", ".mmcif", ".bcif"}

app = FastAPI(title="pdb2print")
app.mount("/files", StaticFiles(directory=OUTPUT_ROOT), name="files")


def _rgb_to_hex(rgb) -> str:
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


def _error_payload(message: str) -> dict:
    """The standard failure JSON shape (matches the streamed result event)."""
    return {"ok": False, "warning": message, "report": message,
            "glb_url": None, "threemf_url": None, "stl_url": None, "chains": []}


def _bool(x) -> bool:
    """Parse a permissive form boolean ("true"/"1"/"on"/"yes")."""
    return str(x).strip().lower() in {"1", "true", "on", "yes"}


#: Characters allowed in a generated download filename; everything else in a
#: user-supplied name collapses to "_" so the name is safe on every OS and
#: needs no quoting in a URL path.
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def _download_stem(source: str, params: PrintParams) -> str:
    """A descriptive base filename for this build, e.g. ``1zaa_pdb2print_1p5mm``.

    Downloads used to land in the user's folder as ``out.3mf`` / ``out_stl.zip``
    for every structure, which is useless the moment you have built more than
    one. The stem carries what distinguishes two builds in a downloads folder:
    which structure, and at what scale.

    ``source`` is either a bare PDB ID or the path of an uploaded file; either
    way only the basename (without extension) contributes.
    """
    stem = os.path.splitext(os.path.basename(source))[0]
    stem = _SAFE_STEM.sub("_", stem).strip("._-").lower()[:40] or "model"
    scale = f"{params.scale_mm_per_angstrom:g}".replace(".", "p")
    return f"{stem}_pdb2print_{scale}mm"


def _map_connections(fields: dict) -> ConnectionParams:
    """Build a :class:`ConnectionParams` from the (small) connection form fields."""
    return ConnectionParams(
        connect=_bool(fields.get("connect", False)),
        use_magnets=_bool(fields.get("use_magnets", False)),
        no_magnet_method=NoMagnetMethod(fields.get("no_magnet_method", "inflate")),
        connector_diameter_mm=float(fields.get("connector_diameter", 4.0)),
        magnet_thickness_mm=float(fields.get("magnet_thickness", 2.0)),
        magnet_shape=MagnetShape(fields.get("magnet_shape", "round")),
        magnet_count=int(float(fields.get("magnet_count", 1))),
        dna_magnet_count=int(float(fields.get("dna_magnet_count", 1))),
        socket=_bool(fields.get("socket", True)),
        socket_wall_mm=float(fields.get("socket_wall", 1.5)),
        magnet_fit_clearance_mm=float(fields.get("magnet_fit_clearance", 0.2)),
        basepair_connect=_bool(fields.get("basepair_connect", False)),
    )


def _map_params(fields: dict) -> PrintParams:
    """Map raw HTTP form fields to a :class:`PrintParams` (may raise ValueError)."""
    return PrintParams(
        scale_mm_per_angstrom=float(fields["scale"]),
        grid_spacing_mm=float(fields["grid_spacing"]),
        min_wall_mm=float(fields["min_wall"]),
        min_wall_mode=MinWallMode(fields["min_wall_mode"]),
        protein_representation=Representation(fields["protein_rep"]),
        nucleic_representation=Representation(fields["nucleic_rep"]),
        backbone_style=BackboneStyle(fields["backbone_style"]),
        base_style=BaseStyle(fields["base_style"]),
        nucleic_radius_mm=float(fields["nucleic_radius"]),
        protein_tube_radius_mm=float(fields.get("protein_tube_radius", 1.2)),
        cartoon_thickness_mm=float(fields.get("cartoon_thickness", 2.0)),
        slab_thickness_mm=float(fields["slab_thickness"]),
        slab_scale=float(fields["base_width"]),
        connector_radius_mm=float(fields["connector_radius"]),
        atom_radius_mm=float(fields["atom_radius"]),
        bond_radius_mm=float(fields["bond_radius"]),
        probe_radius_ang=float(fields["probe_radius"]),
        surface_atom_padding_ang=float(fields["surface_padding"]),
        connections=_map_connections(fields),
    )


def _run_and_export(source: str, params: PrintParams, progress) -> dict:
    """Blocking build + export; returns the result JSON dict (never raises).

    ``progress(frac, msg)`` is forwarded straight into ``build_all`` and reused
    for the export phase, so the SSE stream keeps ticking after meshing too.
    """
    try:
        report = build_all(source, params, progress=progress)
    except Exception as exc:  # watertight-gate RuntimeError, ValueError, etc.
        return _error_payload(str(exc))

    progress(0.96, "Writing export files…")
    token = uuid.uuid4().hex
    out_dir = os.path.join(OUTPUT_ROOT, token)
    os.makedirs(out_dir, exist_ok=True)

    # Files are named after the structure so a download is self-describing; the
    # uuid directory (not the filename) is what keeps concurrent builds apart.
    stem = _download_stem(source, params)

    export.write_glb(report.built, os.path.join(out_dir, f"{stem}.glb"),
                     markers=report.connection_markers)
    export.write_stl_zip(report.built, os.path.join(out_dir, f"{stem}_stl.zip"))

    threemf_url = None
    warning = None
    try:
        export.write_3mf(report.built, os.path.join(out_dir, f"{stem}.3mf"))
        threemf_url = f"/files/{token}/{stem}.3mf"
    except RuntimeError as exc:
        # 3MF unavailable / non-manifold — still ship GLB + STL, surface message.
        warning = str(exc)

    chains = [
        {"id": chain.chain_id, "name": chain.name,
         "color": _rgb_to_hex(color_for_index(i))}
        for i, (chain, _mesh) in enumerate(report.built)
    ]

    # Overall printed bounding box (mm) across every built chain, plus the scale
    # that produced it, so the UI can show the size live as the scale changes.
    size_mm = None
    if report.built:
        mins = [min(m.bounds[0][k] for _, m in report.built) for k in range(3)]
        maxs = [max(m.bounds[1][k] for _, m in report.built) for k in range(3)]
        size_mm = [float(maxs[k] - mins[k]) for k in range(3)]

    progress(1.0, "Done.")
    return {
        "ok": True,
        "warning": warning,
        "report": report.summary(),
        "glb_url": f"/files/{token}/{stem}.glb",
        "threemf_url": threemf_url,
        "stl_url": f"/files/{token}/{stem}_stl.zip",
        "chains": chains,
        "connections": report.connections,
        "size_mm": size_mm,
        "scale_used": params.scale_mm_per_angstrom,
    }


@app.post("/api/generate")
async def generate(
    pdb_id: str = Form(""),
    scale: float = Form(1.5),
    grid_spacing: float = Form(0.5),
    min_wall: float = Form(1.0),
    min_wall_mode: str = Form("uniform"),
    protein_rep: str = Form("surface"),
    nucleic_rep: str = Form("tube_slab"),
    backbone_style: str = Form("tube"),
    base_style: str = Form("slab"),
    nucleic_radius: float = Form(1.2),
    protein_tube_radius: float = Form(1.2),
    cartoon_thickness: float = Form(2.0),
    slab_thickness: float = Form(1.2),
    base_width: float = Form(1.0),
    connector_radius: float = Form(0.6),
    atom_radius: float = Form(1.0),
    bond_radius: float = Form(0.5),
    probe_radius: float = Form(1.4),
    surface_padding: float = Form(0.0),
    # --- connector / joinery system ---
    connect: str = Form("false"),
    use_magnets: str = Form("false"),
    no_magnet_method: str = Form("inflate"),
    connector_diameter: float = Form(4.0),
    magnet_thickness: float = Form(2.0),
    magnet_shape: str = Form("round"),
    magnet_count: int = Form(1),
    dna_magnet_count: int = Form(1),
    socket: str = Form("true"),
    socket_wall: float = Form(1.5),
    magnet_fit_clearance: float = Form(0.2),
    basepair_connect: str = Form("false"),
    file: UploadFile | None = None,
):
    """Build a model and stream progress as Server-Sent Events.

    The response is ``text/event-stream``: zero or more ``event: progress`` lines
    (``{frac, msg}``) followed by a single ``event: result`` line carrying the
    same JSON shape the endpoint returned before streaming. Validation problems
    (bad file type, missing source, unparseable params) short-circuit to a plain
    JSON error with a 4xx status, so the client can tell the two apart by
    content-type.
    """
    # 1. resolve source: uploaded file OR a PDB-ID string
    source = None
    tmp_upload_dir = None
    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in UPLOAD_EXTS:
            return JSONResponse(
                _error_payload(f"Unsupported file type '{ext}'. "
                               f"Accepted: {', '.join(sorted(UPLOAD_EXTS))}."),
                status_code=400,
            )
        tmp_upload_dir = tempfile.mkdtemp(prefix="pdb2print_up_")
        source = os.path.join(tmp_upload_dir, os.path.basename(file.filename))
        with open(source, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
    elif pdb_id.strip():
        source = pdb_id.strip()

    if not source:
        return JSONResponse(
            _error_payload("Enter a PDB ID or upload a .pdb/.cif file."),
            status_code=400,
        )

    # 2. map fields -> PrintParams
    try:
        params = _map_params({
            "scale": scale, "grid_spacing": grid_spacing, "min_wall": min_wall,
            "min_wall_mode": min_wall_mode, "protein_rep": protein_rep,
            "nucleic_rep": nucleic_rep, "backbone_style": backbone_style,
            "base_style": base_style, "nucleic_radius": nucleic_radius,
            "protein_tube_radius": protein_tube_radius,
            "cartoon_thickness": cartoon_thickness,
            "slab_thickness": slab_thickness, "base_width": base_width,
            "connector_radius": connector_radius, "atom_radius": atom_radius,
            "bond_radius": bond_radius, "probe_radius": probe_radius,
            "surface_padding": surface_padding,
            "connect": connect, "use_magnets": use_magnets,
            "no_magnet_method": no_magnet_method,
            "connector_diameter": connector_diameter,
            "magnet_thickness": magnet_thickness, "magnet_shape": magnet_shape,
            "magnet_count": magnet_count, "dna_magnet_count": dna_magnet_count,
            "socket": socket, "socket_wall": socket_wall,
            "magnet_fit_clearance": magnet_fit_clearance,
            "basepair_connect": basepair_connect,
        })
    except (ValueError, TypeError) as exc:
        if tmp_upload_dir:
            shutil.rmtree(tmp_upload_dir, ignore_errors=True)
        return JSONResponse(
            _error_payload(f"Invalid parameter: {exc}"), status_code=400,
        )

    # 3. stream progress while the (blocking) build runs in a worker thread.
    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress(frac, msg):
            loop.call_soon_threadsafe(
                queue.put_nowait, ("progress", {"frac": frac, "msg": msg}))

        def work():
            try:
                result = _run_and_export(source, params, progress)
            except Exception as exc:  # defensive: _run_and_export shouldn't raise
                result = _error_payload(str(exc))
            finally:
                if tmp_upload_dir:
                    shutil.rmtree(tmp_upload_dir, ignore_errors=True)
            loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
            loop.call_soon_threadsafe(queue.put_nowait, ("__done__", None))

        loop.run_in_executor(None, work)
        while True:
            kind, data = await queue.get()
            if kind == "__done__":
                break
            yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the rest of the front-end assets (kept last so /api and / win first).
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
