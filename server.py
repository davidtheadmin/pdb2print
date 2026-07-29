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
import threading
import time
import uuid
from typing import Optional

from collections import OrderedDict

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pdb2print.config import (
    PrintParams, Representation, MinWallMode, BaseStyle, BackboneStyle,
    ConnectionParams, NoMagnetMethod, MagnetShape, StandParams, ColumnShape,
    PlaqueRelief, PlaqueFont,
    MoleculeType, LigandStyle, color_for_index,
)
from pdb2print.pipeline import build_all, BuildCancelled
from pdb2print import export
from pdb2print import cache as cache_mod
from pdb2print.cache import Cache, DEFAULT_CACHE_DIR


HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(HERE, "frontend")

# Every generation writes its outputs into a fresh sub-directory here, which is
# served read-only at /files/<token>/... for both <model-viewer> and downloads.
OUTPUT_ROOT = tempfile.mkdtemp(prefix="pdb2print_out_")

# The build cache. Shipped entries live in the repo and survive a restart, which
# the temp directory above deliberately does not.
#
# PDB2PRINT_CACHE_RO=1 turns off writing on misses. Set it wherever the disk is
# ephemeral (a free Space resets on every cold start): the pre-generated entries
# are still served, and a miss is simply rebuilt each time rather than written to
# a filesystem that will throw the result away anyway.
CACHE_DIR = os.environ.get("PDB2PRINT_CACHE_DIR", DEFAULT_CACHE_DIR)
CACHE_READ_ONLY = os.environ.get("PDB2PRINT_CACHE_RO", "").strip().lower() in {
    "1", "true", "on", "yes"}


def _cache_max_bytes():
    """Cache ceiling from PDB2PRINT_CACHE_MAX_GB, or the module default."""
    raw = os.environ.get("PDB2PRINT_CACHE_MAX_GB", "").strip()
    if not raw:
        return None
    try:
        return int(float(raw) * 1024 ** 3)
    except ValueError:
        return None


cache = Cache(CACHE_DIR, read_only=CACHE_READ_ONLY,
              max_bytes=_cache_max_bytes())
os.makedirs(CACHE_DIR, exist_ok=True)

UPLOAD_EXTS = {".pdb", ".ent", ".cif", ".mmcif", ".bcif"}

#: Bumped whenever the front end and the server change together.
#:
#: The front end carries the same string, and the preview compares them. Three
#: separate rounds were lost to the same misreading: index.html is refetched on
#: every page load and server.py is not, so a change to both shows up as the new
#: control appearing and doing the old thing — which is indistinguishable from a
#: bug in the new control, and sends everybody looking in the wrong place.
CODE_STAMP = '2026-07-29.1'

#: When this process started, for /api/health.
_STARTED = time.time()

app = FastAPI(title="pdb2print")
app.mount("/files", StaticFiles(directory=OUTPUT_ROOT), name="files")
app.mount("/cache", StaticFiles(directory=CACHE_DIR), name="cache")


def _rgb_to_hex(rgb) -> str:
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


@app.get("/")
def index() -> FileResponse:
    # No-store on the shell page. The whole UI is one HTML file, so a browser
    # holding a cached copy is a browser running last week's front end against
    # this week's API — which presents as a fixed bug that is still there, and
    # sends everyone hunting in the wrong place. The file is small and the page
    # is loaded once per session; there is nothing to gain by caching it.
    return FileResponse(
        os.path.join(FRONTEND_DIR, "index.html"),
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/api/health")
def health() -> JSONResponse:
    """What the *running process* actually has. Open it in a browser.

    Written after losing two rounds to the same misdiagnosis twice over: a
    setting appears in the panel, does nothing, and there is no way to tell from
    the outside whether the cause is a missing dependency, a stale process still
    running last week's code, or a real bug. All three look identical, and
    guessing between them costs a round trip each time.

    So this reports the loaded code rather than the code on disk — the fields
    actually present on the dataclasses, the fonts that actually resolve, the
    dependencies that actually import — and compares the newest source file
    against the process start time, which is the one question a person cannot
    answer by looking at their editor.
    """
    import dataclasses

    from pdb2print import config as cfg, typeset
    from pdb2print.pipeline import BuildReport

    def version(name: str) -> str:
        try:
            module = __import__(name)
            return str(getattr(module, "__version__", "present"))
        except Exception as exc:
            return f"MISSING — {type(exc).__name__}: {exc}"

    newest, newest_name = 0.0, ""
    package = os.path.dirname(os.path.abspath(cfg.__file__))
    sources = [os.path.abspath(__file__)]
    for dirpath, _dirs, names in os.walk(package):
        sources.extend(os.path.join(dirpath, n) for n in names
                       if n.endswith(".py"))
    for path in sources:
        try:
            stamp = os.path.getmtime(path)
        except OSError:
            continue
        if stamp > newest:
            newest, newest_name = stamp, path
    stale = newest > _STARTED

    stand_fields = {f.name for f in dataclasses.fields(cfg.StandParams)}
    report_fields = {f.name for f in dataclasses.fields(BuildReport)}
    code = {
        "plaque_split": "plaque_split" in stand_fields,
        "plaque_relief": "plaque_relief" in stand_fields,
        "plaque_font": "plaque_font" in stand_fields,
        "column_pins": "column_pins" in stand_fields,
        "apron_rake": "apron_rake_deg" in stand_fields,
        "build_report_carries_title": "title" in report_fields,
        "stand_preview_endpoint": any(
            getattr(route, "path", "") == "/api/stand/preview"
            for route in app.routes),
    }

    fonts = {}
    for key in ("line", "sans", "serif"):
        face = typeset.face(key)
        fonts[key] = {"using": face.key, "ok": face.key == key,
                      "why": typeset.unavailable(face)}

    problems = []
    if stale:
        problems.append(
            f"This process started {time.strftime('%H:%M:%S', time.localtime(_STARTED))} "
            f"but {os.path.basename(newest_name)} was changed at "
            f"{time.strftime('%H:%M:%S', time.localtime(newest))} — it is running "
            f"older code than you have on disk. Restart the server.")
    problems.extend(f["why"] for f in fonts.values() if f["why"])
    problems.extend(f"{name} is missing" for name in
                    ("fontTools", "mapbox_earcut")
                    if version(name).startswith("MISSING"))

    return JSONResponse({
        "ok": not problems,
        "code_stamp": CODE_STAMP,
        "problems": problems or ["nothing — the running server is up to date."],
        "process_started": time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(_STARTED)),
        "newest_source": {
            "file": os.path.relpath(newest_name, os.path.dirname(
                os.path.abspath(__file__))) if newest_name else None,
            "changed": time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime(newest)) if newest else None,
            "newer_than_this_process": stale,
        },
        "code_loaded": code,
        "fonts": fonts,
        "dependencies": {name: version(name) for name in
                         ("fontTools", "mapbox_earcut", "manifold3d",
                          "trimesh", "biotite")},
    })


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
        cartoon_helix_width_mm=float(fields.get("cartoon_helix_width", 4.5)),
        cartoon_strand_width_mm=float(fields.get("cartoon_strand_width", 4.0)),
        cartoon_coil_radius_mm=float(fields.get("cartoon_coil_radius", 0.9)),
        slab_thickness_mm=float(fields["slab_thickness"]),
        slab_scale=float(fields["base_width"]),
        connector_radius_mm=float(fields["connector_radius"]),
        atom_radius_mm=float(fields["atom_radius"]),
        bond_radius_mm=float(fields["bond_radius"]),
        backbone_atom_radius_mm=float(fields.get("backbone_atom_radius", 1.0)),
        backbone_bond_radius_mm=float(fields.get("backbone_bond_radius", 0.5)),
        probe_radius_ang=float(fields["probe_radius"]),
        surface_atom_padding_ang=float(fields["surface_padding"]),
        # Defaults to False when the field is absent, matching the checkbox: a
        # caller that says nothing about ligands gets the plain structure, which is
        # both the old behaviour and the conservative one.
        include_ligands=_bool(fields.get("include_ligands", False)),
        ligand_style=LigandStyle(fields.get("ligand_style", "ball_stick")),
        ligand_atom_mm=float(fields.get("ligand_atom", 2.2)),
        ligand_bond_mm=float(fields.get("ligand_bond", 1.2)),
        ligand_vdw_scale=float(fields.get("ligand_vdw_scale", 1.0)),
        connections=_map_connections(fields),
    )


#: How long an uncached build's files stay downloadable under /files/.
#:
#: Most builds are cached and their temp copy is deleted immediately. This only
#: covers the ones that are not — uploads, and builds whose 3MF the watertight
#: gate refused — which would otherwise sit in the container until it restarted.
#: Two hours is far longer than the gap between a build finishing and someone
#: clicking download, while still bounding the leak.
OUTPUT_TTL_SECONDS = 2 * 60 * 60


def _sweep_output_root(now: Optional[float] = None) -> int:
    """Delete per-build output directories older than the TTL.

    Cheap enough to run on every build: one ``stat`` per directory, against work
    that takes tens of seconds. Doing it here rather than on a timer avoids a
    background task that has to be shut down cleanly.

    A build in progress is safe — its directory was created moments ago, so it
    cannot be older than the TTL.
    """
    now = time.time() if now is None else now
    removed = 0
    try:
        names = os.listdir(OUTPUT_ROOT)
    except OSError:
        return 0
    for name in names:
        path = os.path.join(OUTPUT_ROOT, name)
        try:
            if not os.path.isdir(path):
                continue
            if now - os.path.getmtime(path) < OUTPUT_TTL_SECONDS:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


# --------------------------------------------------------------------------
# Recently built meshes, kept in memory for the display stand
# --------------------------------------------------------------------------
#: token -> {"built", "params", "source"} for the last few completed builds.
#:
#: The stand is generated from the *finished* meshes and changes nothing about
#: how they were built, so re-running the pipeline to add one would spend a
#: minute of marching cubes reproducing geometry the server had in its hands a
#: moment ago — and would spend it again for every change of column count. This
#: is the short-lived hand-back that makes the button feel like a button.
#:
#: It is a cache of convenience and never one of record: a miss (server
#: restarted, entry evicted, or the build came from the disk cache and so never
#: existed in memory) falls back to a full rebuild, which is correct, just slow.
_RECENT_BUILDS: "OrderedDict[str, dict]" = OrderedDict()
_RECENT_LOCK = threading.Lock()

#: How many builds to hold. Deliberately small: a large complex is hundreds of
#: megabytes of triangles, and this process also has to mesh the next one.
_RECENT_MAX = 2


def _remember_build(token: str, source: str, params: PrintParams, built,
                    meta: Optional[dict] = None) -> None:
    with _RECENT_LOCK:
        _RECENT_BUILDS[token] = {
            "built": built, "params": params, "source": source,
            # What the plaque would print. Kept with the build because that is
            # the last moment it is cheap: an uploaded file is deleted when the
            # build finishes, and a fetched one is a download away.
            "meta": dict(meta or {}),
        }
        _RECENT_BUILDS.move_to_end(token)
        while len(_RECENT_BUILDS) > _RECENT_MAX:
            _RECENT_BUILDS.popitem(last=False)


def _recall_build(token: str):
    if not token:
        return None
    with _RECENT_LOCK:
        entry = _RECENT_BUILDS.get(token)
        if entry is not None:
            _RECENT_BUILDS.move_to_end(token)
        return entry


def _cached_result(meta: dict, source: str = "") -> dict:
    """Rebuild the result payload for a cache hit.

    Everything except the three URLs was stored verbatim at build time, so a hit
    is indistinguishable from a fresh build to the front end — same report text,
    same chain list and colours, same printed size. Only the URLs are rewritten,
    to point into the cache directory instead of a per-build temp directory.
    """
    files = meta.get("files") or {}
    key = meta["key"]

    def url(kind):
        name = files.get(kind)
        return f"/cache/{key}/{name}" if name else None

    result = dict(meta.get("result") or {})

    # Entries written before the plaque carried its own text have no
    # ``plaque_meta``, and a cache hit skips the build that would have read it —
    # so a structure served from cache arrived with no name on its plaque and no
    # hint as to why. Backfilled here rather than by throwing the cache away:
    # the pre-generated entries are the whole reason a popular structure is
    # instant, and re-reading one header is cheaper than re-meshing a complex by
    # several orders of magnitude. ``resolve_source`` memoises the fetch, so an
    # ID costs at most one download per process however many hits it serves.
    if source and not (result.get("plaque_meta") or {}).get("pdb_id"):
        result["plaque_meta"] = _plaque_meta(source)

    result.update({
        "ok": True,
        "glb_url": url("glb"),
        "threemf_url": url("threemf"),
        "stl_url": url("stl_zip"),
        "cached": True,
    })
    return result


def _run_and_export(source: str, params: PrintParams, progress,
                    should_cancel=None) -> dict:
    """Blocking build + export; returns the result JSON dict (never raises).

    ``progress(frac, msg)`` is forwarded straight into ``build_all`` and reused
    for the export phase, so the SSE stream keeps ticking after meshing too.
    ``should_cancel`` is polled by the pipeline so a disconnected client stops
    the build instead of leaving it to run to completion unwatched.
    """
    # Cache first: the overwhelming majority of requests are a handful of famous
    # structures at preset settings, and serving those as static files is the
    # difference between a download and a minute of marching cubes. Checked
    # before the cancellation machinery matters, because a hit finishes far
    # faster than a client can disconnect.
    try:
        hit = cache.lookup(source, params)
    except Exception:
        hit = None          # a broken cache must never take the app down
    if hit:
        progress(1.0, "Loaded from cache.")
        return _cached_result(hit, source)

    try:
        report = build_all(source, params, progress=progress,
                           should_cancel=should_cancel)
    except BuildCancelled:
        # Nobody is listening any more; unwind quietly rather than reporting.
        return _error_payload("Build cancelled.")
    except Exception as exc:  # watertight-gate RuntimeError, ValueError, etc.
        return _error_payload(str(exc))

    progress(0.96, "Writing export files…")
    _sweep_output_root()
    token = uuid.uuid4().hex
    out_dir = os.path.join(OUTPUT_ROOT, token)
    os.makedirs(out_dir, exist_ok=True)

    # Hold the meshes so "Create display stand" can work from them instead of
    # meshing the whole structure again. Done before the export rather than
    # after, so an export that fails on disk space still leaves the build
    # reachable.
    plaque_meta = {
        "pdb_id": _plaque_id(source),
        "title": report.title,
        "why": "" if report.title else
               "this structure's header has no title record",
    }
    _remember_build(token, source, params, report.built, plaque_meta)

    # Files are named after the structure so a download is self-describing; the
    # uuid directory (not the filename) is what keeps concurrent builds apart.
    stem = _download_stem(source, params)

    # A full disk surfaces here first, because the exporters write before the
    # cache does. Caught explicitly so it reports as a server problem the
    # operator can act on, rather than as an unhandled traceback that reads like
    # the structure was at fault.
    try:
        export.write_glb(report.built, os.path.join(out_dir, f"{stem}.glb"),
                         markers=report.connection_markers)
        export.write_stl_zip(report.built, os.path.join(out_dir, f"{stem}_stl.zip"))
    except OSError as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        return _error_payload(
            "The server ran out of disk space while writing the export files. "
            f"This is not a problem with your structure. ({exc.strerror or exc})"
        )

    threemf_url = None
    warning = None
    try:
        export.write_3mf(report.built, os.path.join(out_dir, f"{stem}.3mf"))
        threemf_url = f"/files/{token}/{stem}.3mf"
    except RuntimeError as exc:
        # 3MF unavailable / non-manifold — still ship GLB + STL, surface message.
        warning = str(exc)
    except OSError as exc:
        warning = f"Could not write the 3MF: {exc.strerror or exc}"

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
    result = {
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
        # What a plaque would print for this structure. Sent with every build so
        # the stand sketch can show the real ID and the real title from the
        # moment it opens, rather than the words "Structure name" standing in
        # for them until the first solve comes back — a preview whose whole
        # point is showing what you get should not be showing a placeholder.
        "plaque_meta": plaque_meta,
        # Names the in-memory meshes for a follow-up stand request. Deliberately
        # excluded from what gets cached below: a token identifies one process's
        # memory, and serving a stale one from disk would send the client
        # chasing a build that no longer exists.
        "build_token": token,
    }

    # Store on miss. Only a build that produced a 3MF is worth keeping — that is
    # the artefact people come for, and an entry without one would be served in
    # place of a retry that might succeed. Caching is strictly an optimisation,
    # so any failure here is swallowed: the user already has their files.
    if threemf_url:
        stored = None
        try:
            stored = cache.store(
                source, params,
                files={
                    "threemf": os.path.join(out_dir, f"{stem}.3mf"),
                    "glb": os.path.join(out_dir, f"{stem}.glb"),
                    "stl_zip": os.path.join(out_dir, f"{stem}_stl.zip"),
                },
                meta={
                    "stem": stem,
                    # What each object *was*, so a later display-stand request
                    # can reopen this entry as geometry instead of re-meshing
                    # the whole structure. Cheap to write and the only thing
                    # standing between a cache hit and a second full build.
                    "objects": cache_mod.describe_objects(report.built),
                    # The URLs are per-build and meaningless once this temp
                    # directory is gone; a hit rewrites them from the entry.
                    "result": {k: v for k, v in result.items()
                               if k not in ("glb_url", "threemf_url", "stl_url",
                                            "build_token")},
                },
            )
        except Exception:
            pass

        # Once the files are in the cache, the copy under OUTPUT_ROOT is dead
        # weight — the cache serves the same bytes from a stable URL. Keeping
        # both doubled disk use per build, and nothing ever deleted the temp
        # copy, so it accumulated until the container restarted.
        if stored:
            result["glb_url"] = f"/cache/{stored}/{stem}.glb"
            result["threemf_url"] = f"/cache/{stored}/{stem}.3mf"
            result["stl_url"] = f"/cache/{stored}/{stem}_stl.zip"
            shutil.rmtree(out_dir, ignore_errors=True)

    return result


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
    cartoon_helix_width: float = Form(4.5),
    cartoon_strand_width: float = Form(4.0),
    cartoon_coil_radius: float = Form(0.9),
    slab_thickness: float = Form(1.2),
    base_width: float = Form(1.0),
    connector_radius: float = Form(0.6),
    atom_radius: float = Form(1.0),
    bond_radius: float = Form(0.5),
    backbone_atom_radius: float = Form(1.0),
    backbone_bond_radius: float = Form(0.5),
    probe_radius: float = Form(1.4),
    surface_padding: float = Form(0.0),
    include_ligands: str = Form("false"),
    ligand_style: str = Form("ball_stick"),
    ligand_atom: float = Form(2.2),
    ligand_bond: float = Form(1.2),
    ligand_vdw_scale: float = Form(1.0),
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
            "cartoon_helix_width": cartoon_helix_width,
            "cartoon_strand_width": cartoon_strand_width,
            "cartoon_coil_radius": cartoon_coil_radius,
            "slab_thickness": slab_thickness, "base_width": base_width,
            "connector_radius": connector_radius, "atom_radius": atom_radius,
            "bond_radius": bond_radius,
            "backbone_atom_radius": backbone_atom_radius,
            "backbone_bond_radius": backbone_bond_radius,
            "probe_radius": probe_radius,
            "surface_padding": surface_padding,
            "include_ligands": include_ligands,
            "ligand_style": ligand_style, "ligand_atom": ligand_atom,
            "ligand_bond": ligand_bond,
            "ligand_vdw_scale": ligand_vdw_scale,
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
        # Set when the client goes away (Cancel button, closed tab, dropped
        # connection).  The worker polls it through ``should_cancel`` so an
        # abandoned build stops at the next chain instead of tying up a core
        # meshing a complex nobody is waiting for any more.
        cancelled = threading.Event()

        def progress(frac, msg):
            loop.call_soon_threadsafe(
                queue.put_nowait, ("progress", {"frac": frac, "msg": msg}))

        def work():
            try:
                result = _run_and_export(source, params, progress,
                                         should_cancel=cancelled.is_set)
            except Exception as exc:  # defensive: _run_and_export shouldn't raise
                result = _error_payload(str(exc))
            finally:
                if tmp_upload_dir:
                    shutil.rmtree(tmp_upload_dir, ignore_errors=True)
            loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
            loop.call_soon_threadsafe(queue.put_nowait, ("__done__", None))

        loop.run_in_executor(None, work)
        try:
            while True:
                kind, data = await queue.get()
                if kind == "__done__":
                    break
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        finally:
            # Reached on normal completion *and* when the client disconnects
            # (the generator is closed / cancelled).  Setting it after a normal
            # finish is harmless — the worker has already returned.
            cancelled.set()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Display stand
# --------------------------------------------------------------------------
def _map_stand(fields: dict) -> StandParams:
    """Build a :class:`StandParams` from the stand form fields."""
    return StandParams(
        enabled=True,
        columns=int(float(fields.get("stand_columns", 0) or 0)),
        orbit_theta_deg=float(fields.get("orbit_theta", 0.0) or 0.0),
        orbit_phi_deg=float(fields.get("orbit_phi", 75.0) or 75.0),
        roll_deg=float(fields.get("orbit_roll", 0.0) or 0.0),
        plate_margin_mm=float(fields.get("plate_margin", 7.0) or 7.0),
        plate_thickness_mm=float(fields.get("plate_thickness", 4.0) or 4.0),
        plate_corner_mm=float(fields.get("plate_corner", 4.0) or 4.0),
        column_diameter_mm=float(fields.get("column_diameter", 8.0) or 8.0),
        column_flared=_bool(fields.get("column_flared", True)),
        column_capital=_bool(fields.get("column_capital", False)),
        column_edge_frac=float(fields.get("column_edge_frac", 0.45) or 0.45),
        cradle_clearance_mm=float(fields.get("cradle_clearance", 0.35) or 0.35),
        cradle_depth_mm=float(fields.get("cradle_depth", 4.0) or 4.0),
        stand_off_mm=float(fields.get("stand_off", 6.0) or 6.0),
        plaque=_bool(fields.get("plaque", True)),
        plaque_pdb_id=_bool(fields.get("plaque_pdb_id", True)),
        plaque_title_text=str(fields.get("plaque_title_text", "") or "")[:120],
        plaque_note=str(fields.get("plaque_note", "") or "")[:80],
        plaque_scalebar=_bool(fields.get("plaque_scalebar", True)),
        plaque_legend=_bool(fields.get("plaque_legend", True)),
        plaque_legend_labels=str(fields.get("plaque_legend_labels", "") or "")[:2000],
        plaque_tile=_bool(fields.get("plaque_tile", True)),
        plaque_relief=PlaqueRelief(fields.get("plaque_relief", "raised")),
        plaque_text_mm=float(fields.get("plaque_text", 5.0) or 5.0),
        plaque_font=PlaqueFont(fields.get("plaque_font", "sans")),
        plaque_info_mm=max(0.0, min(200.0,
                                    float(fields.get("plaque_info_mm", 0) or 0))),
        plaque_min_stroke_mm=float(fields.get("plaque_min_stroke", 0.45) or 0.45),
        apron_rake_deg=float(fields.get("apron_rake", 0.0) or 0.0),
        column_shape=ColumnShape(fields.get("column_shape", "square")),
        column_pins=_bool(fields.get("column_pins", False)),
        pin_diameter_mm=float(fields.get("pin_diameter", 4.0) or 4.0),
        pin_depth_mm=float(fields.get("pin_depth", 3.0) or 3.0),
    )


def _plaque_id(source: str) -> str:
    """What the plaque calls this structure: its ID, or an uploaded file's name."""
    stem = os.path.splitext(os.path.basename(source))[0]
    return stem.upper() if len(stem) <= 8 else stem


def _plaque_meta(source: str) -> dict:
    """The PDB ID and structure title the plaque prints, from scratch.

    The slow path, and the fallback. Prefer :func:`_stand_meta`, which uses what
    the build already read: this one re-resolves the source, which for a PDB ID
    means going back to RCSB and for a deleted upload means failing.

    ``why`` records what went wrong rather than dropping it. A title that is
    absent because the header has none and a title that is absent because the
    file could not be opened look identical on a plaque, and only one of them is
    the user's problem.
    """
    from pdb2print import io as p2p_io
    from pdb2print.names import structure_title

    meta = {"pdb_id": _plaque_id(source), "title": None, "why": ""}
    try:
        meta["title"] = structure_title(p2p_io.resolve_source(source))
        if not meta["title"]:
            meta["why"] = "this structure's header has no title record"
    except Exception as exc:
        meta["why"] = f"the structure could not be re-read ({exc})"
    return meta


def _stand_meta(source: str, token: str) -> dict:
    """The plaque's text, from the build that produced these meshes if possible."""
    entry = _recall_build(token)
    if entry and entry.get("meta", {}).get("pdb_id"):
        return dict(entry["meta"])
    return _plaque_meta(source)


def _built_from_cache(hit: dict):
    """Reopen a cache entry as ``[(object, mesh), ...]``, or ``None``.

    The per-chain STL zip an entry already carries *is* the finished geometry —
    the same meshes the 3MF was written from, after the connections pass. Paired
    with the object metadata stored beside it, that is everything a display stand
    needs, so a structure served from cache can be stood up without meshing it a
    second time.

    This matters more than it looks. The in-memory registry only holds builds
    this process actually performed, and the whole point of the cache is that
    the popular structures are *never* built — so before this, the exact
    structures most likely to be asked for were the ones guaranteed to pay for a
    full rebuild before they could get a stand.
    """
    import io as _io
    import zipfile

    import trimesh

    name = (hit.get("files") or {}).get("stl_zip")
    if not name:
        return None
    path = os.path.join(cache.entry_dir(hit["key"]), name)
    if not os.path.isfile(path):
        return None

    by_label = {}
    order = []
    try:
        with zipfile.ZipFile(path) as zf:
            # Zip order is write order is build order, which is the order the
            # colour palette was handed out in. Keep it.
            for entry in zf.namelist():
                if not entry.lower().endswith(".stl"):
                    continue
                with zf.open(entry) as fh:
                    mesh = trimesh.load(_io.BytesIO(fh.read()), file_type="stl")
                label = os.path.splitext(os.path.basename(entry))[0]
                by_label[label] = mesh
                order.append(label)
    except Exception:
        return None

    # Entries written before the metadata existed still carry everything needed
    # in their filenames, so fall back to those rather than rebuilding.
    objects = cache_mod.objects_from_meta(hit)
    if not objects:
        objects = cache_mod.objects_from_labels(order)
    if not objects:
        return None

    built = []
    for obj in objects:
        mesh = by_label.get(obj.label())
        if mesh is None:
            return None            # a partial model is worse than a rebuild
        built.append((obj, mesh))
    return built or None


def _stand_meshes(source: str, params: PrintParams, token: str, progress=None):
    """``(built, base_params, where)`` for a stand, without ever rebuilding.

    Two routes, in cost order: the meshes this process still has in memory for
    ``token``, then the disk cache entry for this exact build — whose per-chain
    STL zip *is* the finished geometry. Returns ``(None, params, None)`` when
    neither has it, and leaves the decision about whether a rebuild is worth it
    to the caller: it is, for a stand the user asked for and is watching a
    progress bar for; it is not, for a preview that has to answer in a moment or
    not at all.
    """
    entry = _recall_build(token)
    if entry is not None:
        if progress:
            progress(0.30, "Using the model already built…")
        return entry["built"], entry["params"], "meshes in memory"
    try:
        hit = cache.lookup(source, params)
    except Exception:
        hit = None
    if hit:
        if progress:
            progress(0.20, "Reopening the cached model…")
        built = _built_from_cache(hit)
        if built is not None:
            return built, params, "the cached build"
    return None, params, None


def _run_stand(source: str, params: PrintParams, token: str, progress,
               should_cancel=None) -> dict:
    """Generate a display stand and export the model standing on it.

    Uses the in-memory meshes for ``token`` when they are still there, and falls
    back to a full rebuild when they are not — which happens after a restart, an
    eviction, or when the original build was served from the disk cache and so
    never existed as meshes in this process at all.

    **Nothing here is written to the disk cache, deliberately.**  A stand is a
    one-off: it is keyed on an orientation the user arranged by hand and will
    almost never arrange identically twice, so an entry would be written once
    and read never — while costing the same hundreds of megabytes per structure
    that the pre-generated entries are budgeted in.  The cache exists to make
    the *famous structure at preset settings* instant, and a stand is the
    opposite of that.  (The rebuild fallback below calls ``build_all`` directly
    rather than ``_run_and_export`` for the same reason: it must not store
    either.)
    """
    from pdb2print import stand as stand_mod

    built, base_params, where = _stand_meshes(source, params, token, progress)

    if built is None:
        progress(0.02, "Rebuilding the model to stand it…")
        try:
            report = build_all(source, params, progress=lambda f, m: progress(
                0.02 + 0.55 * f, m), should_cancel=should_cancel)
        except BuildCancelled:
            return _error_payload("Build cancelled.")
        except Exception as exc:
            return _error_payload(str(exc))
        built = report.built
        base_params = params
        where = "a fresh build"

    # The stand block is the only thing that may differ from the build that
    # produced these meshes: everything else has to match, or the stand would be
    # sized for geometry that is not there.
    import dataclasses
    effective = dataclasses.replace(base_params, stand=params.stand)

    progress(0.62, "Placing the stand…")
    try:
        oriented, stand_parts, notes = stand_mod.build_stand(
            built, effective, meta=_stand_meta(source, token))
    except Exception as exc:
        return _error_payload(f"Could not build the display stand: {exc}")

    combined = list(oriented) + list(stand_parts)

    progress(0.86, "Writing export files…")
    _sweep_output_root()
    out_token = uuid.uuid4().hex
    out_dir = os.path.join(OUTPUT_ROOT, out_token)
    os.makedirs(out_dir, exist_ok=True)
    stem = _download_stem(source, effective) + "_stand"

    try:
        # The preview is written Y-up so it stands the right way in the viewer
        # and the orbit's poles agree with the model's own up; the print files
        # stay Z-up, which is what a slicer means by up.
        export.write_glb(stand_mod.to_view_frame(combined),
                         os.path.join(out_dir, f"{stem}.glb"))
        export.write_stl_zip(combined, os.path.join(out_dir, f"{stem}_stl.zip"))
    except OSError as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        return _error_payload(
            "The server ran out of disk space while writing the export files. "
            f"({exc.strerror or exc})")

    threemf_url = None
    warning = None
    try:
        export.write_3mf(combined, os.path.join(out_dir, f"{stem}.3mf"))
        threemf_url = f"/files/{out_token}/{stem}.3mf"
    except (RuntimeError, OSError) as exc:
        warning = str(exc)

    # The viewer legend names the *molecule*. A base plate and a lettering tile
    # are not chains and listing them there turns a legend into an inventory —
    # the point of the pills is to say which colour is which subunit, and a
    # "stand_plaque_tile" pill answers a question nobody asked.
    chains = [
        {"id": getattr(chain, "chain_id", "-"),
         "name": chain.display_name(),
         "color": _rgb_to_hex(color)}
        for (chain, _mesh), color in zip(combined, export.object_colors(combined))
        if getattr(chain, "mtype", None) != MoleculeType.STAND
    ]

    mins = [min(m.bounds[0][k] for _, m in combined) for k in range(3)]
    maxs = [max(m.bounds[1][k] for _, m in combined) for k in range(3)]

    progress(1.0, "Done.")
    # Say what was actually produced, and from which switches. A plaque element
    # that should not be there is otherwise only visible as a shape in a
    # preview, which makes "it is still doing X" impossible to check against
    # what the server believes it did.
    sp = effective.stand
    lines = [
        f"Display stand: {len(stand_parts)} object(s) — "
        + ", ".join(part.object_name() for part, _m in stand_parts),
        f"  from {where or 'a fresh build'}"
        f" · columns {sp.columns or 'auto'} {sp.column_shape.value}"
        f"{' flared' if sp.column_flared else ' straight'}"
        f"{' + capital' if sp.column_capital else ''}"
        f" · orbit {sp.orbit_theta_deg:.0f}/{sp.orbit_phi_deg:.0f}"
        f" roll {sp.roll_deg:.0f}",
        f"  plaque={sp.plaque} id={sp.plaque_pdb_id} "
        f"name={sp.plaque_title_text!r} "
        f"scale={sp.plaque_scalebar} legend={sp.plaque_legend} "
        f"tile={sp.plaque_tile} {sp.plaque_relief.value} "
        f"font={sp.plaque_font.value} "
        f"rake={sp.apron_rake_deg:.0f}°"
        f"{' · pinned columns' if sp.column_pins else ''}"
        + (f" note={sp.plaque_note!r}" if sp.plaque_note else ""),
    ]
    lines.extend(f"  ! {n}" for n in notes)
    return {
        "ok": True,
        "warning": warning,
        "report": "\n".join(lines),
        "glb_url": f"/files/{out_token}/{stem}.glb",
        "threemf_url": threemf_url,
        "stl_url": f"/files/{out_token}/{stem}_stl.zip",
        "chains": chains,
        "connections": [],
        "size_mm": [float(maxs[k] - mins[k]) for k in range(3)],
        "scale_used": effective.scale_mm_per_angstrom,
        "stand_notes": notes,
        # The stand build is not remembered: standing it again would start from
        # the already-rotated meshes and tilt it twice.
        "build_token": token,
    }


def _stand_form(form: dict) -> dict:
    """The stand routes take the form wholesale, so they owe it the defaults.

    ``_map_params`` indexes the fields the generate endpoint declares with a
    ``Form(...)`` default, so those defaults have to be supplied here too —
    rather than have a second thirty-parameter signature drift out of step with
    the first.
    """
    fields = {
        "scale": 1.5, "grid_spacing": 0.5, "min_wall": 1.0,
        "min_wall_mode": "uniform", "protein_rep": "surface",
        "nucleic_rep": "tube_slab", "backbone_style": "tube",
        "base_style": "slab", "nucleic_radius": 1.2, "slab_thickness": 1.2,
        "base_width": 1.0, "connector_radius": 0.6, "atom_radius": 1.0,
        "bond_radius": 0.5, "probe_radius": 1.4, "surface_padding": 0.0,
    }
    fields.update({k: v for k, v in form.items() if v != ""})
    return fields


@app.post("/api/stand/preview")
async def stand_preview(request: Request):
    """Solve a stand's layout — and nothing else — for the live sketch.

    Everything expensive about a stand happens *after* the layout: carving a
    cradle out of a surface mesh, sweeping a few hundred stroke solids, meshing
    and writing the result. The arithmetic that decides where the plate goes,
    how deep the apron has to be and which points on the underside the columns
    will rise to is a fraction of a second, and it is the whole of what a
    preview needs.

    The point is that the front end does not *re-derive* any of it. A sketch
    that solves for its own column positions is a second implementation of the
    hardest judgement in this module, and the day the two disagree the picture
    will be confidently wrong — which is worse than a picture that admits it is
    generic.

    Never rebuilds. If the meshes are neither in memory nor in the cache this
    answers ``ready: false`` at once and the sketch falls back to its generic
    drawing, because a preview that takes a minute is not a preview.
    """
    form = dict(await request.form())
    token = str(form.get("build_token", "")).strip()
    source = str(form.get("pdb_id", "")).strip()
    if not source:
        remembered = _recall_build(token)
        if remembered is not None:
            source = remembered["source"]
    if not source:
        return JSONResponse({"ok": True, "ready": False, "reason": "no-model"})

    try:
        params = _map_params(_stand_form(form))
        params.stand = _map_stand(_stand_form(form))
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse(
            _error_payload(f"Invalid parameter: {exc}"), status_code=400)

    def work():
        from pdb2print import stand as stand_mod
        import dataclasses

        built, base_params, _where = _stand_meshes(source, params, token)
        if built is None:
            return {"ok": True, "ready": False, "reason": "not-in-memory",
                "stamp": CODE_STAMP}
        effective = dataclasses.replace(base_params, stand=params.stand)
        layout = stand_mod.solve_layout(built, effective,
                                        meta=_stand_meta(source, token))
        if layout is None:
            return {"ok": True, "ready": False, "reason": "empty"}
        summary = stand_mod.layout_summary(layout, effective)
        summary.update({"ok": True, "ready": True, "stamp": CODE_STAMP})
        return summary

    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(None, work)
    except Exception as exc:                 # a preview must never take the page down
        return JSONResponse({"ok": True, "ready": False, "reason": str(exc)})
    return JSONResponse(payload)


@app.post("/api/stand")
async def stand(request: Request):
    """Add a display stand to a model that has already been built.

    Takes the same form fields as ``/api/generate`` plus a ``build_token`` and
    the stand block, and streams progress the same way, so the front end reuses
    its whole result-handling path.
    """
    form = dict(await request.form())
    token = str(form.get("build_token", "")).strip()
    source = str(form.get("pdb_id", "")).strip()

    # An *uploaded* structure has no PDB ID to resend, and the temp file it was
    # read from is deleted the moment its build finishes. The remembered build
    # is therefore the only route to standing one — so take the source from
    # there when the form cannot supply it, and only refuse when neither can.
    if not source:
        remembered = _recall_build(token)
        if remembered is not None:
            source = remembered["source"]
    if not source:
        return JSONResponse(
            _error_payload("A display stand needs a model that is still in "
                           "memory on the server. Generate the model again, "
                           "then add the stand."),
            status_code=400)

    fields = _stand_form(form)

    try:
        params = _map_params(fields)
        params.stand = _map_stand(fields)
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse(
            _error_payload(f"Invalid parameter: {exc}"), status_code=400)

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()

        def progress(frac, msg):
            loop.call_soon_threadsafe(
                queue.put_nowait, ("progress", {"frac": frac, "msg": msg}))

        def work():
            try:
                result = _run_stand(source, params, token, progress,
                                    should_cancel=cancelled.is_set)
            except Exception as exc:
                result = _error_payload(str(exc))
            loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
            loop.call_soon_threadsafe(queue.put_nowait, ("__done__", None))

        loop.run_in_executor(None, work)
        try:
            while True:
                kind, data = await queue.get()
                if kind == "__done__":
                    break
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        finally:
            cancelled.set()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the rest of the front-end assets (kept last so /api and / win first).
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
