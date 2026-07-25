# pdb2print

Convert protein / nucleic-acid structures from the PDB into **3D-printable,
per-chain-coloured files** for a Prusa Core One with multi-material (INDX).

The primary output is a **multi-object 3MF** — one named, separately-coloured
object per chain — that PrusaSlicer opens with a filament assignable per chain.
Per-chain **STL** (zipped) is offered as a fallback, and an interactive **GLB**
preview is shown in the app.

The gap this fills: no existing tool (ChimeraX, NIH 3D Print Exchange, Protein
Imager) does per-chain colour export, DNA backbone thickening, a clean SES
protein surface, *and* printable joinery between chains in one pipeline.

## What it does

- **Input:** a 4-character PDB ID (fetched from RCSB) or an uploaded
  `.pdb` / `.cif` / `.mmcif` / `.bcif` file.
- **Chains:** splits the structure into chains and classifies each as protein or
  nucleic acid (ligands, ions and solvent are dropped).
- **Representation**, configurable independently per molecule type:
  - Protein → **solvent-excluded surface** (true Connolly SES), or a backbone
    **tube**.
  - Nucleic acid → **tube-and-slab** (backbone tube + base rungs), or SES.
- **Joinery** so a multi-chain model actually holds together — magnets, a fixed
  bridge, or inflate-to-weld. See below.
- **Minimum wall thickness** enforced per representation, so nothing prints too
  thin.
- **Watertight / manifold** meshes, gated hard before export.

## Joinery

Chains whose surfaces come within a threshold can be joined three ways:

| method | what it does |
|---|---|
| **inflate** | grows both surfaces at the contact until they weld — organic, no visible strut |
| **bridge** | a clean flat-ended cylinder split across a shared face |
| **magnets** | a press-fit bore in each part so they snap together |

**Flush socket** (on by default, applies to magnets and bridges). Each part gets
a flat-ended collar driven from a shared mid-plane into its own body, so the two
halves meet on one clean disc instead of two ragged molecular surfaces touching
wherever they happen to. Anything of one part reaching past that face inside the
joint footprint is cut away, so the parts can actually close.

**Press fit, not nominal fit.** An FDM hole printed to a magnet's exact size
comes out undersize and will not accept it. The bore is cut wider
(`magnet_fit_clearance_mm`, 0.2 mm default, exposed in the UI because it is
printer-specific) and deeper, with a 45° lead-in at the mouth.

**Where the magnets go** is scored, not guessed: candidate contacts are
shortlisted from the surface point clouds, then ranked against the real solids by
how much material is actually there. The joint axis is chosen by testing three
candidate directions against the geometry — see `NOTES.md`, which explains why
the obvious construction (the line between local centres of mass) fails on DNA.

## How the geometry works

Two kernels, deliberately:

- **SES protein surface** — a signed distance field on a voxel grid
  (`distance_transform_edt` on the solvent-accessible solid), marching-cubed at
  the zero level set. Watertight by construction.
- **Everything else** — exact analytic primitives fused with `manifold3d`
  booleans (capsules, oriented boxes, frusta). No voxel grid, so no stairstep.

`manifold3d` is a **hard** dependency, not an optional fallback: shipping
known-bad geometry is worse than failing, and it is the planned kernel for the
client-side WASM milestone.

The `pdb2print/` package has **no web-framework dependency**, so the core can
later be ported to WASM.

```
pdb2print/
  server.py                 # FastAPI: serves frontend/ + /api/generate (SSE)
  app.py                    # legacy Gradio shell (kept; server.py is current)
  frontend/index.html       # the UI — single file, no build step
  pdb2print/
    io.py                   # RCSB fetch + file loading (biotite)
    names.py                # per-chain subunit names from the file header
    chains.py               # chain split + protein/nucleic classification
    config.py               # PrintParams / ConnectionParams, enums, palette
    geometry.py             # generate_chain_mesh(chain, params) — dispatch
    representations/        # surface.py, tube_slab.py, _common.py, _manifold.py
    connections.py          # joinery: magnets / bridge / inflate / base pairs
    meshops.py              # repair() + enforce_min_wall()
    export.py               # 3MF (lib3mf), STL, GLB
    pipeline.py             # build_all() end-to-end orchestration
  tests/                    # end-to-end tests + offline fixtures
```

## Run locally

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

Then open <http://localhost:7860>. Try **1UBQ** (protein), **1BNA** (DNA),
**1ZAA** (compact zinc-finger–DNA complex) or **1TUP** (p53 + DNA, five chains —
the joinery stress test).

### Use the core without the UI

```python
from pdb2print.config import PrintParams, ConnectionParams
from pdb2print.pipeline import build_all
from pdb2print import export

params = PrintParams(scale_mm_per_angstrom=1.5, grid_spacing_mm=0.5,
                     min_wall_mm=1.0)
params.connections = ConnectionParams(connect=True, use_magnets=True,
                                      connector_diameter_mm=5.0,
                                      magnet_thickness_mm=3.0)
report = build_all("1ZAA", params)
print(report.summary())
export.write_3mf(report.built, "1zaa_pdb2print.3mf")
```

## Tests

```bash
pip install pytest
pytest tests/                       # offline: uses bundled fixtures
PDB2PRINT_TEST_RCSB=1 pytest tests/ # also fetches 1UBQ / 1BNA / 1ZAA
```

## Settings that interact (worth knowing)

- **Grid spacing is in print millimetres but samples a structure in Ångström.**
  What governs mesh quality is `grid_spacing_mm / scale_mm_per_angstrom`, so
  *lowering the scale coarsens the mesh* even though the grid slider did not
  move. At the 1.5 mm/Å default with a 0.5 mm grid that is 0.33 Å.
- **pymeshlab is optional but load-bearing.** Without it the repair path is only
  hole-filling, which cannot fix a non-manifold edge, so a mesh it would have
  rescued is refused at the export gate instead. Install it if you hit
  watertight-gate failures.

## Deploy to a Hugging Face Space

Free CPU tier is enough (no GPU). The Space needs outbound network access to
fetch PDB IDs from RCSB; uploaded files work without it.

## Dependencies

biotite (parsing) · trimesh + scikit-image + scipy (meshing) · **manifold3d**
(booleans; hard dependency) · pymeshlab (heavy repair, optional but recommended)
· lib3mf (3MF, Prusa's own library) · fastapi + uvicorn (server).

## Roadmap

- PrusaSlicer acceptance test at real INDX print scale (see `NOTES.md`).
- A proper ChimeraX-style cartoon representation (the first attempt is withdrawn
  — `NOTES.md` has the design for the rework).
- Client-side WASM port of the geometry core.
