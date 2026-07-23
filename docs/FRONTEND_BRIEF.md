# pdb2print — front-end design brief

Build a **new custom front end** for pdb2print to replace the current Gradio app
(`app.py`). It must look and feel like a polished, modern desktop tool — **not**
an auto-generated form. Reference the layout and interactions below exactly.

## Golden rule
**Do not touch the geometry pipeline.** The Python package `pdb2print/`
(`chains.py`, `geometry.py`, `representations/`, `meshops.py`, `export.py`,
`config.py`, `pipeline.py`, `io.py`) is working and verified. Only replace the
UI layer and add a thin API in front of the existing `pipeline.build_all(...)`.

---

## 1. Look & feel

- **Dark mode by default.** Neutral, tool-like. No gradients, no shadows beyond
  hairlines.
- Palette:
  - page background `#0e0f11`, panel `#17191c`, raised `#1d2024`
  - hairline border `#2a2e33`
  - text `#e6e8ea`, secondary `#9aa0a6`, muted `#6b7178`
  - **accent = dark green** `#2f9e6f` (hover `#1f6b4a`) — active states, primary button
  - **dark red** `#b3453f` — errors / watertight-failure warnings only
- Type: system sans, compact. Sentence case everywhere. No ALL CAPS except tiny
  section eyebrows.
- **Everything fits one screen** on a normal laptop — no long vertical scroll of
  controls. Use compact controls and progressive disclosure.

## 2. Layout (match this)

A slim **top bar**: wordmark `pdb2print` on the left, a PDB-ID text field in the
middle, an upload icon-button, and a green **Generate** button on the right.

Below, a **two-pane split**:

- **Left (~60%, the hero): 3D preview.** A real interactive viewer (use
  `<model-viewer>` from CDN, loading the returned GLB). Overlaid on the preview:
  a **chain-colour legend** (pills: "Chain A", "Chain B", … with the chain's
  colour dot) bottom-left, and **download buttons** (3MF, STL) bottom-right.
  Below the viewer, a small monospace **report** area.
- **Right (~40%): settings**, as compact cards:
  - **Appearance** card. Segmented toggles (pill groups, active segment filled
    green), not radios:
    - `Protein`: Surface (only option for now, keep it a toggle for future)
    - A nested block (left green accent bar) titled **DNA / RNA**:
      - `Representation`: Tube-slab
      - `Backbone`: Tube · Molecule
      - `Base (rung)`: Slab · Rod · Molecule
  - **Print** card:
    - **Preset chips** (pill buttons): Balanced · Chunky · Fine detail. Clicking
      one sets the advanced sliders (values below). Optionally re-generate.
    - Two headline sliders: **Scale** and **Min wall**.
    - An **"Advanced dimensions"** disclosure (closed by default) holding the
      remaining sliders. The **ball-and-stick sizing** sliders (atom, bond)
      inside it are **hidden unless** Backbone or Base is set to `Molecule`.

## 3. Controls → parameters (exact)

Every control maps to a field on `pdb2print.config.PrintParams`. Ranges/defaults:

| Control | Param | Min | Max | Step | Default |
|---|---|---|---|---|---|
| Scale (mm/Å) | `scale_mm_per_angstrom` | 0.2 | 6.0 | 0.1 | 1.5 |
| Min wall (mm) | `min_wall_mm` | 0.0 | 5.0 | 0.1 | 1.0 |
| Tube radius (mm) | `nucleic_radius_mm` | 0.4 | 8.0 | 0.1 | 1.2 |
| Base thickness (mm) | `slab_thickness_mm` | 0.4 | 8.0 | 0.1 | 1.2 |
| Base width scale | `slab_scale` | 0.3 | 3.0 | 0.05 | 1.0 |
| Connector radius (mm) | `connector_radius_mm` | 0.2 | 8.0 | 0.1 | 0.6 |
| Atom radius (mm) | `atom_radius_mm` | 0.3 | 8.0 | 0.1 | 1.0 |
| Bond radius (mm) | `bond_radius_mm` | 0.2 | 8.0 | 0.1 | 0.5 |
| Grid spacing (mm) | `grid_spacing_mm` | 0.2 | 1.5 | 0.05 | 0.5 |
| Min-wall mode | `min_wall_mode` | — | — | — | `uniform` (or `selective`) |

Enums (string values):
- `protein_representation`, `nucleic_representation`: `surface` \| `tube_slab`
  (defaults: protein `surface`, nucleic `tube_slab`)
- `backbone_style`: `tube` \| `molecule` (default `tube`)
- `base_style`: `slab` \| `rod` \| `molecule` (default `slab`)

Preset values — tuple order is
`(grid_spacing, min_wall, nucleic_radius, slab_thickness, base_width, connector_radius, atom_radius, bond_radius)`:
- **Balanced**: `0.5, 1.0, 1.2, 1.2, 1.0, 0.6, 1.0, 0.50`
- **Chunky**: `0.5, 1.5, 2.2, 2.2, 1.4, 1.4, 1.8, 1.10`
- **Fine detail**: `0.3, 0.8, 0.9, 0.9, 0.9, 0.6, 0.8, 0.45`

Structure input: a PDB ID (default `1UBQ`) **or** an uploaded file
(`.pdb .ent .cif .mmcif .bcif`). Example buttons: `1UBQ, 1BNA, 1ZAA, 1TUP, 2HHB`.

Chain colours (match `config.CHAIN_PALETTE`, cycled) for the legend:
`#D93333, #3373D9, #33B359, #F2BF26, #994DBF, #33BFBF, #E68033, #D959A6`.

## 4. Backend contract (add this thin layer)

Add a small **FastAPI** app (e.g. `server.py`) that serves the static front end
and wraps the existing pipeline. Do not reimplement geometry — call it:

```python
from pdb2print.config import (PrintParams, Representation, MinWallMode,
                              BaseStyle, BackboneStyle)
from pdb2print.pipeline import build_all
from pdb2print import export

# POST /api/generate  (multipart: params as fields + optional 'file')
#   1. resolve source: uploaded file path OR pdb_id string
#   2. params = PrintParams(**mapped_fields)
#   3. report = build_all(source, params)
#   4. write outputs to a temp dir:
#        export.write_glb(report.built, ".../out.glb")
#        export.write_3mf(report.built, ".../out.3mf")      # may raise RuntimeError
#        export.write_stl_zip(report.built, ".../out_stl.zip")
#   5. return JSON: {glb_url, threemf_url, stl_url, report: report.summary(),
#                    chains: [{id, color}], ok: bool, warning: str|null}
# GET /  -> serves the front-end index.html
# Static file routes to serve the generated glb/3mf/zip for download + <model-viewer>.
```

`build_all` raises `RuntimeError` if a chain isn't watertight and `write_3mf`
can raise too — surface that message in the report area styled in the dark-red
accent, and still offer the GLB/STL if present.

`report.summary()` returns plain text (chain count, residues, face counts,
watertight status, warnings) — show it verbatim in the report area.

## 5. Tech constraints

- Single-page front end. `<model-viewer>` via CDN
  (`https://unpkg.com/@google/model-viewer`). Vanilla JS or a light framework —
  your call, but keep it self-contained and buildable.
- Put the front end under `frontend/` and the API in `server.py`. Update the
  README run command to `uvicorn server:app` (keep the old `python app.py` note
  as legacy).
- Must still be deployable to Hugging Face Spaces (CPU). Don't add GPU or paid
  services.

## 6. Interactions checklist

- Segmented toggles switch active state on click (one active per group).
- Preset chips set the advanced sliders; highlight the active chip.
- "Advanced dimensions" opens/closes; ball-and-stick sliders appear only for a
  molecule style.
- Generate: disable + show a progress/spinner state, POST params, then load the
  GLB into the viewer, populate downloads, legend, and report.
- Slider values show a live numeric readout.
