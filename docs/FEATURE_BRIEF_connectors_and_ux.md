# pdb2print — feature brief: connectors + UX (v2)

This brief is for a fresh build session. It has two halves:

1. **Established knowledge** — the project, its architecture, and every decision
   locked in during the session that produced the current code. Read this first.
2. **What to build** — three UX improvements and a connector/joinery system,
   with the design already agreed with the user (David).

**Do not re-architect the geometry core.** Extend it.

---

## PART 1 — ESTABLISHED KNOWLEDGE (read before building)

### What pdb2print is
An open-source tool that converts RCSB PDB/CIF structures into 3D-printable,
**per-chain-coloured multi-object 3MF** files for **PrusaSlicer**, targeting a
**Prusa Core One + INDX multi-material** setup. Each chain is a separate,
independently-filament-assignable object. The whole complex normally prints as
**one multi-material job** (this matters for the connector design).

### Architecture (do not rewrite)
- `pdb2print/` — geometry core, framework-free (so a future WASM port can reuse it):
  - `config.py` — `PrintParams` dataclass (all tunables), enums
    (`Representation`, `MoleculeType`, `MinWallMode`, `BaseStyle`,
    `BackboneStyle`), `CHAIN_PALETTE`, `color_for_index`.
  - `io.py` — `load_any(source)` (PDB id via RCSB, or file path); uses biotite.
  - `chains.py` — `split_chains` → `Chain(chain_id, atoms, mtype)`; classifies
    protein vs nucleic.
  - `geometry.py` — `generate_chain_mesh(chain, params)` dispatches by
    representation (the single extension point; `_BUILDERS` registry).
  - `representations/surface.py` — protein: true SES via signed-distance field +
    marching cubes.
  - `representations/tube_slab.py` — nucleic: backbone (tube|molecule) + base
    (slab|rod|molecule); analytic `manifold3d` primitives fused by one boolean
    union (watertight by construction, no voxel stairstep).
  - `representations/_manifold.py` — `capsule`, `sphere`, `oriented_box`,
    `union`, `to_trimesh` (manifold3d kernel — also the planned WASM kernel).
  - `meshops.py` — `repair` (watertight/manifold cleanup) and min-wall pass.
  - `export.py` — `write_glb`, `write_3mf` (lib3mf; watertight gate + GLB
    fallback if lib3mf missing), `write_stl_zip`.
  - `pipeline.py` — `build_all(source, params, progress=None) -> BuildReport`.
    `BuildReport.built` is `List[(Chain, trimesh.Trimesh)]`; `.summary()` is a
    plain-text report; hard watertight gate raises `RuntimeError` on failure.
    **`build_all` already accepts a `progress(fraction, message)` callback** and
    calls it with messages like "Loading structure…", "Meshing chain X (i/n)…".
- `server.py` — **FastAPI** front end (this is the real UI now; Gradio is legacy):
  - `GET /` serves `frontend/index.html`; static assets under `/`.
  - `POST /api/generate` (multipart): parameter fields + optional `file`. Maps
    fields → `PrintParams`, runs `build_all`, writes GLB/3MF/STL-zip into a
    per-request token dir served at `/files/<token>/…`. Returns JSON:
    `{ok, warning, report, glb_url, threemf_url, stl_url, chains:[{id,color}]}`.
    Watertight-gate failure → `ok:false` with message; 3MF-only failure → GLB+STL
    still returned, message in `warning`.
- `frontend/index.html` — self-contained single-page app (vanilla JS). Dark tool
  UI, `<model-viewer>` (CDN) preview as hero, segmented toggles, preset chips,
  advanced drawer. Palette: bg `#0e0f11`, panel `#17191c`, raised `#1d2024`,
  border `#2a2e33`, text `#e6e8ea`/`#9aa0a6`/`#6b7178`, **accent green
  `#2f9e6f`** (hover `#1f6b4a`), **error red `#b3453f`**.
- `app.py` — legacy Gradio UI. Keep as fallback; do not invest in it.
- `requirements.txt` (core) + `requirements-server.txt` (fastapi, uvicorn,
  python-multipart). `lib3mf` needed for true 3MF. Run:
  `uvicorn server:app --host 0.0.0.0 --port 7860`.

### Geometry decisions locked in this session
- **Nucleic styles**: backbone `tube`|`molecule`, base `slab`|`rod`|`molecule`
  (ball-and-stick). All build watertight, one connected body per chain.
- **Rod = clean ladder rung**: ONE tube from the backbone point to the base
  centroid, radius `= nucleic_radius_mm * 0.7` (slightly thinner than backbone),
  floored to min-wall. **No separate connector strut** — the rung is the link.
- **Molecule base** joins the backbone with a single **bond-thickness** link at
  the glycosidic atom (reads as a real bond, not a spike).
- **Slab** connector grown to `max(connector_radius, slab_thickness*0.45)` so it
  blends instead of spiking.
- **Min-wall** is a parametric offset applied to primitives before the boolean
  (tube/rod/connector radius ≥ min_wall/2; slab thickness ≥ min_wall). SES
  surface is thick by construction and declines the pass.
- **`meshops.repair` fast-path (IMPORTANT):** a mesh that is already
  `is_watertight` and single-body is returned untouched (only `fix_normals`).
  The old cleanup ran `merge_vertices` unconditionally, which welded
  near-coincident vertices at overlapping-primitive seams and **pinched valid
  analytic meshes into non-manifold**. Keep this fast-path; any new geometry that
  comes out of the manifold kernel is already watertight — don't let repair
  damage it.

### Presets (current)
Three chips; each sets the DNA/RNA **style toggles** AND chunky (~#9),
print-safe **dimensions** in one click. Default on load = **Clean ladder**.
- **Clean ladder** — backbone tube + base rod.
- **Molecular** — backbone molecule + base molecule.
- **Tube + molecule bases** — backbone tube + base molecule.
Chunky dimension set: `scale 1.5, min_wall 1.0, grid 0.5, nucleic_radius 2.4,
slab_thickness 2.4, base_width 1.5, connector_radius 1.4, atom_radius 2.2,
bond_radius 1.2` ("Tube + molecule bases" uses atom 2.0 / bond 1.1).

### Verification status
All 6 style combinations × (1BNA, 1ZAA) build watertight and single-body through
the real `repair` path. Server end-to-end verified: generate returns valid GLB
(~1.3 MB), true 3MF via lib3mf, and per-chain STL zip; all download.

### Environment / testing notes (the build sandbox has limits)
- **RCSB is blocked** from the sandbox and **large PDB files truncate** through
  the web fetch (1TUP could not be retrieved). Test with **uploaded local
  files**; complete copies of `1BNA` and `1ZAA` (stripped to valid PDB records)
  were used. PDB-id fetch and protein-SES paths work but need real network on the
  user's machine.
- **No GPU / OpenGL** in the sandbox → headless previews were done with
  **matplotlib 3D** plus `fast-simplification` decimation (preview only; never
  for print output). `<model-viewer>` renders fine in the user's browser.
- Use `fastapi.testclient.TestClient` for end-to-end server checks offline.
- Test structures: `1UBQ` (protein), `1BNA` (DNA), `1ZAA` (zinc-finger+DNA),
  `1TUP` (p53+DNA, 5 components), `2HHB` (haemoglobin, 4 chains).

### How the user works
David agrees the architecture collaboratively before building, wants honest
diagnosis, and iterates on look-and-feel visually. Show rendered results before
committing. He values clean, chunky, printable output over fussy detail.

---

## PART 2 — WHAT TO BUILD

### A. Real subunit names in the viewer legend
Today the legend shows "Chain A … F". Show the **actual molecule/subunit name**
where available, with the chain id secondary (e.g., "DNA-binding domain (A)",
"Hemoglobin alpha (A)").
- Extract per-chain names from the structure header: **PDB** `COMPND` records
  (`MOL_ID` → `MOLECULE` name, mapped to chains via `CHAIN:`); **mmCIF**
  `_entity.pdbx_description` mapped through `_entity_poly`/`_struct_asym` to
  chain ids. biotite exposes CIF categories; PDB COMPND may need light parsing.
- Fall back to "Chain X" when no name is found.
- Plumb through: `Chain` gains an optional `name`; `BuildReport`/API `chains[]`
  gains `name`; the frontend legend renders name (chain id as a small suffix).

### B. Generation progress indication
`build_all` already emits `(fraction, message)`. Surface it live.
- Recommended: a **streaming endpoint** (Server-Sent Events) — e.g.
  `POST /api/generate` streams `event: progress {frac,msg}` lines then a final
  `event: result {…json…}`. (Alternative: job id + `GET /api/progress/{id}`
  polling — simpler but chattier.)
- Frontend: replace the button's indeterminate spinner with a **progress bar +
  current message** ("Meshing chain B (2/5)…"). Keep the busy/disabled state.

### C. Relocate + highlight the download buttons
The overlaid 3MF/STL buttons are hard to see.
- Move them to the **top bar** (near Generate).
- **Disabled/greyed until a result exists**, then **highlight** (accent green,
  subtle pulse or fill) when a download is ready. 3MF and STL independently
  (3MF may be unavailable if lib3mf/watertight fails — keep STL enabled then).

### D. Connector / joinery system (the big one)
Add a post-build **`connections.py`** pass: given the per-chain meshes plus a
list of connections, modify the meshes with `manifold3d` booleans
(union/difference) so **every object stays watertight**. 3MF export is unchanged
(still individual coloured objects — now touching and/or pocketed). This aligns
with the WASM goal (same kernel).

**Core abstraction.** A `Connection` = two objects + a location/axis (their
nearest points, or a biologically meaningful pair) + a **join type** + params.
Detection is **automatic with options** (user's choice): auto-find candidate
pairs, list them as checkboxes, apply a global type/size with per-pair overrides.
Defer true manual 3D point-picking (model-viewer hotspots) to a later phase.

**The fuse-vs-assemble distinction (drives defaults).** In one INDX
multi-material print, objects whose surfaces touch bond automatically — so
"fuse" means *guarantee contact / no air gap*. Magnets and pegs are for parts
printed (or cut) **separately** and joined by hand. Support **both**.

**Join types and the user's decisions:**

1. **Fuse / no air gap** (proteins joined as one print, and general):
   - Preferred method: **grow the surfaces slightly until two parts touch**
     (localized dilation at the contact region), and/or add a short bridging
     cylinder spanning the gap, with a deliberate tiny overlap (~0.1 mm) so they
     weld. User explicitly likes "increasing the surface a bit until two proteins
     touch."
   - Result: one solid print, no separate assembly.

2. **Magnets** (so parts *can* be printed separately if wanted):
   - Subtract a cylindrical pocket = magnet Ø + clearance (press-fit ~0.05–0.1,
     glue ~0.2), depth = magnet height, **chamfered mouth**.
   - **Half-embedded protruding magnet as the connector (user's idea):** seat the
     magnet only **half** in the pocket so it **sticks out and forms the
     connecting cylinder** while staying half-embedded to set the correct gap;
     the mating part has a matching pocket. Support this "protruding/spacer"
     magnet mode in addition to flush and recessed.
   - Organic surfaces have no flat seat → optionally add a small **flat boss/pad**
     to host the pocket. Make bosses optional (user is OK with magnets sticking
     out; keep surfaces as clean as possible).
   - Common disc sizes as presets (e.g. 3×1, 4×2, 5×2, 6×3, 8×3 mm).

3. **Pegs** ("pegs might be fine"): union a cylinder on one part, subtract a
   matching hole (peg Ø + clearance) on the other; both chamfered. Bonus: two
   offset pegs (or a keyed peg) fix orientation. Good default alternative to
   magnets when no snap is needed.

**Specific connection targets the user wants:**
- **Protein ↔ protein**: magnets (separable) OR fuse (no gap). Both offered.
- **DNA interstrand — connect every base-pair step** (makes a double helix hold
  together, otherwise it falls into two strands). Method: **make the rungs meet
  in the middle** — extend each strand's rung to the helix axis so opposing rungs
  touch/fuse. **Note this is harder for the `molecule` base style** (bases are
  atom clusters, not a single extendable rung) → for molecule bases use a
  **bridging cylinder between paired bases** instead. Optionally **tiny magnets +
  a relatively big tube** at each base pair. Pair complementary residues by
  nearest base centroid across strands (not raw mesh nearest points).
- **DNA ↔ protein**: connect where they contact (e.g. zinc-finger to DNA) — same
  magnet/fuse/peg options.

**Options to expose (automatic + options):**
- Connection mode: None / Fuse / Magnet / Peg (global default), with a
  **detected-pairs checklist** and per-pair override.
- DNA base-pair connect: on/off; method (rungs-meet-middle | bridge cylinder |
  magnet+tube); size.
- Magnet: size, clearance (press-fit/glue), chamfer, seating (flush / recessed /
  **half-protruding**), optional boss.
- Peg: Ø, length, clearance, chamfer, single/keyed.
- Fuse: dilation amount / bridge radius / overlap.

**Printability constraints to honor:**
- Respect **min-wall** for every pocket/peg/bridge (don't blow through thin
  backbones).
- Horizontal pockets/holes sag on the ceiling → **teardrop** them or expose an
  "assume vertical axis" option (print orientation isn't known at model-gen).
- Recessed vs flush vs half-protruding magnets is a real toggle (clean plastic
  contact vs magnet-to-magnet strength vs spacer function).
- Keep each object watertight after every boolean; run through `repair` (whose
  fast-path preserves already-good meshes).

**Where it plugs in:** call the connections pass in `pipeline.build_all` after
per-chain meshing (behind params), extend `PrintParams`/API fields, add a
**Connections** card to the frontend with the mode + options + detected-pairs
checklist. Nearest-point/contact detection can use trimesh proximity or manifold
distance; base-pair pairing uses residue geometry.

### Suggested build order
1. Subunit names (small, high value, isolated).
2. Download relocation + highlight (frontend-only).
3. Progress streaming (SSE) — touches server + frontend.
4. Connections: start with **DNA base-pair fuse (rungs-meet-middle)** and
   **protein↔protein fuse (surface touch)**, then **magnets** (incl.
   half-protruding), then **pegs**, then **DNA↔protein** and the molecule-base
   bridging variant.

### Definition of done for each connector change
Every affected object still passes the watertight gate and is a single connected
body; a generated 3MF opens in PrusaSlicer as separately-assignable coloured
objects with the intended contact/pockets. Verify offline with uploaded 1BNA
(interstrand) and 1ZAA (protein↔DNA) via `TestClient`.
