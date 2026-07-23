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
                   #   rasterize_capsule(), rasterize_box() (voxel toolkit)
    _manifold.py   # capsule(), oriented_box(), union(), to_trimesh()
                   #   (analytic mesh-boolean toolkit, manifold3d kernel)
    surface.py     # build(chain, params) — solvent-excluded surface (SES)
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

## Two mesh kernels (post voxel-everything migration)

The pipeline no longer voxelizes everything. Each representation now uses the
kernel that gives it a watertight mesh *without* the artifacts the voxel grid
caused. The two kernels coexist by design — this is not a half-finished
migration:

- **surface.py → grid-sampled SES.** A real solvent-excluded (Connolly) surface,
  not a Gaussian metaball. Build the solvent-accessible solid (union of atom
  balls grown by the probe radius) on a grid, then `SES = { x : inward-distance
  to the SAS boundary > probe }` via one `scipy.ndimage.distance_transform_edt`;
  marching-cubes its zero level set. The sharpness comes from the *field
  definition*, not grid density — the grid was never why the old metaball looked
  blobby (its smooth field measured p95 ≈ 4.4°; the blob was the Gaussian sum
  fusing atoms). Watertight by construction (marching cubes on a signed field).
  Tunables: `probe_radius_ang` (1.4 Å = ChimeraX water probe),
  `surface_atom_padding_ang`, `grid_spacing_mm`.
- **tube_slab.py → analytic mesh booleans.** Backbone tube (a capsule per
  Catmull-Rom segment), base slabs (oriented boxes) and connector struts
  (capsules) are exact `manifold3d` primitives fused by one `batch_boolean`
  union → guaranteed-watertight mesh, no grid, no stairstep. Helpers live in
  `representations/_manifold.py`. `manifold3d` is a **hard dependency** and is
  the same kernel the planned WASM build uses.

Voxel toolkit still in `representations/_common.py` (used by the SES field and
the retained min-wall fallback):
- `Grid.covering(points, spacing, pad)` builds the grid; `Grid.window()` returns
  the local sub-grid around a primitive so rasterization is cheap.
- `field_to_mesh(field, grid, level)` runs `skimage.measure.marching_cubes` and
  offsets verts by the grid origin. SES extracts at `level=0.0`.
- `rasterize_capsule`/`rasterize_box` remain for any future voxel-based rep;
  neither current representation uses them.

`meshops.repair()`: dedupe/degenerate cleanup → `merge_vertices` → `fix_normals`
→ pymeshlab heavy repair only if not watertight (trimesh fallback if pymeshlab
absent) → **component pruning that keeps all bodies ≥ 2% of the largest**
(`DEFAULT_MIN_COMPONENT_FRAC`).

`meshops.enforce_min_wall()`: **now a no-op for both current representations.**
Min-wall moved to build time (surface is thick by construction; tube-slab grows
its primitives parametrically — see below). The voxel implementation
(`trimesh.voxelized(pitch).fill()` → dilate → re-mesh) is retained only as a
fallback for hypothetical future thin-shell representations, gated by
`config.needs_min_wall` / `MIN_WALL_EXEMPT = {SURFACE, TUBE_SLAB}`.

`pipeline.build_all()` applies a **hard watertight gate**: any chain that meshes
but is not watertight raises `RuntimeError` and aborts the export — a
non-manifold object never reaches the 3MF writer. `export.write_3mf` re-checks
the same invariant as defence in depth.

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
- **surface** → no min-wall (SES is thick everywhere by construction).
- **tube-slab** → still honours min-wall, but as a **parametric offset applied at
  build time**: `tube_slab._min_wall_dims` grows the tube radius (`≥ min_wall/2`),
  slab thickness (`≥ min_wall`) and connector radius (`≥ min_wall/2`) *before* the
  mesh boolean. This is exact and adds no grid texture — the whole point of the
  migration. Both reps are therefore in `MIN_WALL_EXEMPT` (they decline the voxel
  `enforce_min_wall` pass), and "tube-slab keeps min-wall" is still true — it just
  lives in the builder now.

## The migration off voxel-everything (this session)

Replaced the two representation builders and added the watertight gate. All three
test structures build watertight at production defaults (scale 1.5, grid 0.5):
1UBQ, 1BNA, 1ZAA — every chain `bodies=1`, `is_watertight=True`, 3MF writes.

**Protein: Gaussian metaball → solvent-excluded surface.** The old blob was the
*field* (a sum of Gaussians fuses atoms), not the grid. Swapped to the Connolly
SES distance field (see kernel section above). Adjacent-face roughness rises from
p95 ≈ 4.4° (smooth blob) to ≈ 23° — that increase *is the detail*: real reentrant
crevices matching ChimeraX. Still watertight-by-construction (signed field →
marching cubes), so no MSMS-style self-intersection/repair risk.

**DNA: voxel occupancy → analytic manifold booleans.** The stairstep was the 0/1
occupancy grid; removing the grid removes it. Tube/slab/connector are now exact
`manifold3d` primitives unioned into one watertight solid. Roughness reads p95 ≈
91°, but that is **clean designed geometry** (the ~90° edges of the base slabs),
not noise — tube facets sit at ~18°, flat slab faces at 0°. Slabs are present and
fused (`bodies=1`).

**Watertightness is now enforced, not assumed:** hard per-chain gate in
`build_all` + a re-check in `write_3mf`.

### Prior-session bug fixes (still relevant)
- **Scale slider** default 0.5→1.5 mm/Å, range 0.2–6.0.
- **Protein inflation / surface artifacts** — were caused by the old min-wall
  re-voxelizing the surface; fixed by making min-wall representation-scoped (the
  rule above), now reinforced by min-wall moving entirely to build time.
- **DNA base slabs missing** — the old voxel path produced disconnected slab
  islands that repair pruned; the connector strut + loosened pruning fixed it.
  The manifold union makes this structural: slabs and connectors are unioned into
  one solid, so there is no orphan to prune. `repair()`'s ≥2%-of-largest
  component pruning is retained as a safety net.

## Known limitations (current, NOT bugs)

- **SES fidelity is still bounded by `grid_spacing_mm`.** The field is exact, but
  its extraction samples a grid, so features finer than ~one voxel round off. At
  the default 0.5 mm / scale 1.5 that is ~0.33 Å — finer than atomic detail — so
  this is not visible in practice; drop grid spacing if a very large scale is
  used. (A fully grid-free analytic SES, e.g. MSMS, self-intersects and would
  break the watertight guarantee — deliberately not taken.)
- **DNA slab edges are sharp by design.** If a print catches on them, chamfer via
  a small `manifold.minkowski_sum(sphere)` on the fused solid — the kernel already
  supports it; not enabled by default.

## Test structures
`1UBQ` (protein), `1BNA` (DNA), `1ZAA` (zinc-finger–DNA complex, exercises both
paths). Offline tests use `tests/data/tiny.pdb`; set `PDB2PRINT_TEST_RCSB=1` to
run the RCSB fetch tests.
