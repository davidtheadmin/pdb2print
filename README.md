# pdb2print

Convert protein / nucleic-acid structures from the PDB into **3D-printable,
per-chain-coloured files** for a Prusa Core One with multi-material (INDX).

The primary output is a **multi-object 3MF** — one named, separately-coloured
object per chain — that PrusaSlicer opens with a filament assignable per chain.
Per-chain **STL** (zipped) is offered as a fallback, and an interactive **GLB**
preview is shown in the app.

## What it does

- **Input:** a 4-character PDB ID (fetched from RCSB) or an uploaded
  `.pdb` / `.cif` / `.mmcif` / `.bcif` file.
- **Chains:** splits the structure into chains and classifies each as protein or
  nucleic acid (ligands, ions and solvent are dropped).
- **Representation** (configurable independently per molecule type):
  - Protein → **Gaussian metaball molecular surface** (default).
  - Nucleic acid → **tube-and-slab** (backbone tube + base slabs, default).
  - Either can be switched to the other; the choice is a clean parameter into
    the geometry module, designed so new representations (cartoon, ladder-only,
    backbone-tube …) are easy to add.
- **Minimum wall thickness** (default 1.0 mm) enforced as its own pass *after*
  representation, so nothing prints too thin.
- **Watertight / manifold** meshes, suitable for slicing.

## How the geometry works

Every representation rasterises its primitives into a scalar field on a regular
voxel grid, then a single **marching-cubes** extraction produces the shell.
This makes each mesh **watertight and manifold by construction**, avoiding the
self-intersections that a true analytic solvent-excluded surface or naïve CSG
union would create.

- **Grid spacing** is an explicit parameter (mesh resolution vs. speed).
- **Min-wall** runs in voxel space: `selective` thickens only regions measured
  below the target thickness; `uniform` grows every feature outward. Selective
  automatically falls back to uniform if it misbehaves.
- **Tube-and-slab** is pure Python (no ChimeraX dependency): the backbone is a
  Catmull-Rom spline swept as a tube; each base slab is an oriented box placed
  on the base plane fitted from that nucleotide's ring atoms.

The `pdb2print/` package has **no web-framework dependency** — `app.py` (Gradio)
is a thin shell — so the core can later be ported to a client-side WASM build.

```
pdb2print/
  app.py                    # Gradio UI only
  pdb2print/
    io.py                   # RCSB fetch + file loading (biotite)
    chains.py               # chain split + protein/nucleic classification
    config.py               # PrintParams, Representation enums, palette
    geometry.py             # generate_chain_mesh(chain, params) — dispatch
    representations/        # surface.py, tube_slab.py, _common.py
    meshops.py              # repair() + enforce_min_wall()
    export.py               # 3MF (lib3mf), STL, GLB
    pipeline.py             # build_all() end-to-end orchestration
  tests/                    # end-to-end tests + tiny offline fixture
```

## Run locally

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the printed local URL. Try example IDs **1UBQ** (protein), **1BNA**
(DNA), or **1ZAA** (a compact zinc-finger–DNA complex that exercises both paths).

### Use the core without the UI

```python
from pdb2print.config import PrintParams
from pdb2print.pipeline import build_all
from pdb2print import export

params = PrintParams(scale_mm_per_angstrom=0.4, grid_spacing_mm=0.6, min_wall_mm=1.0)
report = build_all("1ZAA", params)
print(report.summary())
export.write_3mf(report.built, "1zaa.3mf")
```

## Tests

```bash
pip install pytest
pytest tests/                       # offline: uses a bundled tiny structure
PDB2PRINT_TEST_RCSB=1 pytest tests/ # also fetches 1UBQ / 1BNA / 1ZAA from RCSB
```

## Deploy to a Hugging Face Space

1. Create a new **Gradio** Space.
2. Add `app.py`, the `pdb2print/` package, and `requirements.txt` to the repo
   (the Space runs `app.py` automatically).
3. `lib3mf` and `pymeshlab` install from PyPI on the Space; if `lib3mf` is ever
   unavailable the app still produces a coloured GLB and per-chain STLs, and
   reports it.

The Space needs outbound network access to fetch PDB IDs from RCSB (enabled by
default); uploaded files work without network access.

## Dependencies

biotite (parsing) · trimesh + scikit-image + scipy (meshing/voxels) · pymeshlab
(heavy repair, optional) · lib3mf (3MF, Prusa's own library) · gradio (UI).

## Roadmap

- More representations (cartoon, ladder-only, backbone tube).
- Client-side WASM port of the geometry core.
