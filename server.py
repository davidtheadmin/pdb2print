"""Thin FastAPI layer for pdb2print.

Serves the static front end in ``frontend/`` and wraps the existing geometry
pipeline. All geometry lives in the ``pdb2print`` package; this module only
maps HTTP form fields to :class:`PrintParams`, calls ``build_all``, writes the
export files, and returns their URLs. Run with::

    uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pdb2print.config import (
    PrintParams, Representation, MinWallMode, BaseStyle, BackboneStyle,
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
    slab_thickness: float = Form(1.2),
    base_width: float = Form(1.0),
    connector_radius: float = Form(0.6),
    atom_radius: float = Form(1.0),
    bond_radius: float = Form(0.5),
    probe_radius: float = Form(1.4),
    surface_padding: float = Form(0.0),
    file: UploadFile | None = None,
):
    # 1. resolve source: uploaded file OR a PDB-ID string
    source = None
    tmp_upload_dir = None
    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in UPLOAD_EXTS:
            return JSONResponse(
                {"ok": False, "warning": None,
                 "report": f"Unsupported file type '{ext}'. "
                           f"Accepted: {', '.join(sorted(UPLOAD_EXTS))}.",
                 "glb_url": None, "threemf_url": None, "stl_url": None,
                 "chains": []},
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
            {"ok": False, "warning": None,
             "report": "Enter a PDB ID or upload a .pdb/.cif file.",
             "glb_url": None, "threemf_url": None, "stl_url": None, "chains": []},
            status_code=400,
        )

    # 2. map fields -> PrintParams
    try:
        params = PrintParams(
            scale_mm_per_angstrom=float(scale),
            grid_spacing_mm=float(grid_spacing),
            min_wall_mm=float(min_wall),
            min_wall_mode=MinWallMode(min_wall_mode),
            protein_representation=Representation(protein_rep),
            nucleic_representation=Representation(nucleic_rep),
            backbone_style=BackboneStyle(backbone_style),
            base_style=BaseStyle(base_style),
            nucleic_radius_mm=float(nucleic_radius),
            slab_thickness_mm=float(slab_thickness),
            slab_scale=float(base_width),
            connector_radius_mm=float(connector_radius),
            atom_radius_mm=float(atom_radius),
            bond_radius_mm=float(bond_radius),
            probe_radius_ang=float(probe_radius),
            surface_atom_padding_ang=float(surface_padding),
        )
    except (ValueError, TypeError) as exc:
        return JSONResponse(
            {"ok": False, "warning": None, "report": f"Invalid parameter: {exc}",
             "glb_url": None, "threemf_url": None, "stl_url": None, "chains": []},
            status_code=400,
        )

    # 3. run the pipeline (may raise on watertight-gate failure)
    try:
        report = build_all(source, params)
    except Exception as exc:  # RuntimeError (watertight gate), ValueError, etc.
        return JSONResponse(
            {"ok": False, "warning": str(exc), "report": str(exc),
             "glb_url": None, "threemf_url": None, "stl_url": None, "chains": []},
            status_code=200,
        )
    finally:
        if tmp_upload_dir:
            shutil.rmtree(tmp_upload_dir, ignore_errors=True)

    # 4. write outputs into a served sub-directory
    token = uuid.uuid4().hex
    out_dir = os.path.join(OUTPUT_ROOT, token)
    os.makedirs(out_dir, exist_ok=True)

    glb_path = os.path.join(out_dir, "out.glb")
    stl_path = os.path.join(out_dir, "out_stl.zip")
    export.write_glb(report.built, glb_path)
    export.write_stl_zip(report.built, stl_path)

    threemf_url = None
    warning = None
    threemf_path = os.path.join(out_dir, "out.3mf")
    try:
        export.write_3mf(report.built, threemf_path)
        threemf_url = f"/files/{token}/out.3mf"
    except RuntimeError as exc:
        # 3MF unavailable / non-manifold — still ship GLB + STL, surface message.
        warning = str(exc)

    chains = [
        {"id": chain.chain_id, "color": _rgb_to_hex(color_for_index(i))}
        for i, (chain, _mesh) in enumerate(report.built)
    ]

    # 5. respond
    return JSONResponse({
        "ok": True,
        "warning": warning,
        "report": report.summary(),
        "glb_url": f"/files/{token}/out.glb",
        "threemf_url": threemf_url,
        "stl_url": f"/files/{token}/out_stl.zip",
        "chains": chains,
    })


# Serve the rest of the front-end assets (kept last so /api and / win first).
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
