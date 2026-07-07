# pdb2print — developer handoff notes

Context for a fresh session. Covers architecture, the geometry pipeline as it
actually works today, the bugs fixed so far, and known limitations that are
*inherent to the approach* (not bugs). See `README.md` for user-facing docs.

## Architecture

Geometry core lives in the `pdb2print/` package and has **no web-framework
import**; `app.py` (Gradio) is a thin shell that only builds `PrintParams`, calls
`pipeline.build_all`, and wires results to widgets. This separation is
deliberate — a later milestone ports the core to client-side WASM, so nothing in
`pdb2print/` may depend on Gradio.

```
app.py                         # Gradio UI ONLY
pdb2print/
  io.py            # load_any(): RCSB fetch (biotite) or file; solvent/altloc cleanup
  chains.py        # split_chains() -> [Chain]; protein/nucleic classification
  config.py        # PrintParams dataclass, Representation/MinWallMode enums,
                   #   palette, needs_min_wall()/MIN_WALL_EXEMPT
  geometry.py      # generate_chain_mesh(chain, params) — the dispatcher seam
  representations/
    _common.py     # Grid, field_to_mesh(), catmull_rom(),
                   #   rasterize_capsule(), rasterize_box()
    surface.py     # build(chain, params) — Gaussian metaball surface
    tube_slab.py   # build(chain, params) — backbone tube + base slabs + connectors
  meshops.py       # repair(), enforce_min_wall()
  export.py        # write_3mf() (lib3mf), write_stl_zip(), write_glb()
  pipeline.py      # build_all() end-to-end orchestration + BuildReport
tests/             # test_pipeline.py + tests/data/tiny.pdb (offline fixture)
```

### The dispatcher seam
`geometry.generate_chain_mesh(chain, params)` is the single extension point.
It picks the representation for the chain's molecule type
(`params.representation_for(chain.mtype)`), looks it up in the module-level
`_BUILDERS` dict (`{Representation.SURFACE: surface.build,
Representation.TUBE_SLAB: tube_slab.build}`), calls `build(chain, params)`, and
stamps `mesh.metadata["representation"]` (+ chain_id, molecule_type).

**To add a representation** (cartoon, ladder-only, backbone-tube, …): write a
`build(chain, params)` in `representations/`, add one enum value in
`config.Representation`, and one line in `geometry._BUILDERS`. Nothing else
changes. Each builder must return a single closed trimesh in print-mm space.

### Units
Structure coords are in ångström; everything downstream works in **print
millimetres**. Builders multiply coords by `params.scale_mm_per_angstrom` on
entry, so `grid_spacing_mm`, `min_wall_mm`, `nucleic_radius_mm`, etc. are all
interpreted directly in the working (mm) space.

## The voxel → marching-cubes pipeline

Every representation follows the same shape: **rasterize primitives into a scalar
field on a regular voxel grid → one marching-cubes extraction → repair.** This is
why meshes are watertight/manifold *by construction* (marching cubes on a sampled
field yields a closed surface), instead of relying on CSG + heavy repair.

Shared helpers in `representations/_common.py`:
- `Grid.covering(points, spacing, pad)` builds the voxel grid; `Grid.window()`
  returns the local sub-grid around a primitive so rasterization is cheap.
- `field_to_mesh(field, grid, level)` runs `skimage.measure.marching_cubes` and
  offsets verts by the grid origin.

**Where voxelization happens per representation:**
- **surface.py** — field is a *smooth Gaussian density*: each atom adds
  `exp(-d²/2σ²)` into a local window (σ sized from the vdW radius so `ISO_LEVEL =
  0.5` crosses near the atom radius). Marching cubes at 0.5. Because the field is
  smooth, the surface mesh itself is smooth (roughness p95 ≈ 4.4°).
- **tube_slab.py** — field is *binary occupancy* (0/1): backbone tube via
  `rasterize_capsule` along a Catmull-Rom spline, base slabs via `rasterize_box`,
  connector struts via `rasterize_capsule`. Marching cubes at 0.5. The occupancy
  (0/1) field is why tube-slab shows some voxel-grid stairstep (see limitations).

`meshops.repair()`: dedupe/degenerate cleanup → `merge_vertices` → `fix_normals`
→ pymeshlab heavy repair only if not watertight (trimesh fallback if pymeshlab
absent) → **component pruning that keeps all bodies ≥ 2% of the largest**
(`DEFAULT_MIN_COMPONENT_FRAC`).

`meshops.enforce_min_wall()`: voxelizes the finished mesh
(`trimesh.voxelized(pitch).fill()`), grows it (uniform binary dilation, or
selective local-thickness inflation with auto-fallback to uniform), and
re-meshes via marching cubes. `pitch = min(grid_spacing_mm, min_wall_mm/2)`.

## Bugs fixed this session

1. **Scale slider too small / capped too low.** `app.py` scale slider default
   0.5→**1.5** mm/Å, range 0.1–2.0 → **0.2–6.0**; `PrintParams.scale_mm_per_angstrom`
   default 0.5→1.5.

2. **Surface artifacts.** Diagnosed (1UBQ, min-wall off vs on): the raw metaball
   surface is smooth (p95 4.4°); the **min-wall pass** re-voxelized it to a binary
   occupancy grid and re-meshed, spiking roughness to p95 24°. So the artifacts
   were the min-wall re-voxelization, **not** metaball lumpiness or marching-cubes
   stairstep of the base surface. Fixed by #3 (surfaces no longer get min-wall).

3. **Protein inflated.** min-wall grew the protein 1.11× in volume (+~1 mm per
   bbox axis). Fix: **min-wall is now representation-scoped.**
   `config.needs_min_wall(rep)` / `config.MIN_WALL_EXEMPT = {SURFACE}`;
   `enforce_min_wall` reads `mesh.metadata["representation"]` and returns the mesh
   untouched for exempt reps. Protein is back to p95 4.4° and un-inflated.

4. **DNA base slabs missing.** Slabs were generated and rasterized fine (12/12,
   ~24k occupancy voxels) but came out as mesh **islands disconnected** from the
   backbone tube, and repair's old "keep only the largest component" step silently
   discarded them. Two-part fix:
   - `tube_slab.build` now rasterizes a **connector strut** (`rasterize_capsule`,
     radius `max(connector_radius_mm, grid_spacing)`) from each residue's backbone
     point (on the tube) to its slab center (inside the slab), so each nucleotide
     is a single connected body.
   - `repair()` pruning **loosened** to keep all components ≥ 2% of the largest,
     as a safety net so intentional geometry is never silently dropped even if a
     connector ever fails.
   Verified fused: `bodies=1`, 57–61% of vertices 2.2–14.3 mm off the tube axis
   (tube radius is 1.2 mm), so that off-axis geometry is the slabs.

### Representation-scoped min-wall rule (important)
- **surface** → declines min-wall (already thick everywhere by construction).
- **tube-slab** → keeps min-wall (a thin backbone tube can print too fragile).
Implemented in `config.needs_min_wall` + `enforce_min_wall`'s metadata check.

### Min-wall pass ordering (per chain, in `pipeline.build_all`)
- **tube-slab:** `generate_chain_mesh` (tube + slabs + connectors rasterized into
  one field → **single fused body**) → `enforce_min_wall` (re-voxelizes an
  already-connected mesh, so it stays connected) → `repair` (loosened pruning).
- **surface:** `generate_chain_mesh` → `enforce_min_wall` **no-ops** → `repair`.

The order matters: the connector must fuse slabs to the tube *before* min-wall
re-voxelizes, or min-wall's re-voxelization could sever a slab again and repair
would prune the orphan. The pre-min-wall `repair` from the first version was
removed for this reason.

## Known limitations (inherent to the voxel-everything approach — NOT bugs)

- **Protein surface looks blobby / less detailed than a ChimeraX SES.** It is a
  **Gaussian metaball**, not a true analytic solvent-excluded surface. Metaballs
  fuse atoms into smooth blobs by construction — sharper crevices and reentrant
  detail of a real SES are not reproduced. This was a deliberate choice: metaballs
  are watertight/manifold by construction and print-robust; true SES (e.g. MSMS)
  self-intersects and needs heavy repair. Tunables: `surface_blobbiness`,
  `surface_atom_padding_ang`, `ISO_LEVEL` in `surface.py`.
- **DNA shows voxel-grid stairstep artifacts.** tube-slab routes through a
  **binary occupancy** field, and min-wall re-voxelizes it again, so its surface
  carries grid texture (roughness p95 ≈ 28°). This is the cost of the
  voxel-everything pipeline for the tube-slab path, not a regression. If a print
  looks too coarse, the right fix is a *gentler* min-wall remesh (finer voxel or a
  stronger Gaussian blur than the current `sigma=0.6` in
  `meshops._mesh_from_occupancy`) — **not** reverting the connector/pruning work.

Both limitations trace to the same design decision (rasterize everything to a
voxel field, then marching cubes). Escaping them would mean a non-voxel path for
that representation (e.g. analytic SES for protein, or smooth swept-surface
tube/slab), which is a larger change than any tuning knob.

## Test structures
`1UBQ` (protein), `1BNA` (DNA), `1ZAA` (zinc-finger–DNA complex, exercises both
paths). Offline tests use `tests/data/tiny.pdb`; set `PDB2PRINT_TEST_RCSB=1` to
run the RCSB fetch tests.
