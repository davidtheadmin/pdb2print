# pdb2print — developer handoff notes

Context for a fresh session. Covers architecture, the geometry pipeline as it
actually works today, the bugs fixed so far, and known limitations that are
*inherent to the approach* (not bugs). See `README.md` for user-facing docs.

## v2 — connectors + UX (this session)

Four features from `docs/FEATURE_BRIEF_connectors_and_ux.md` Part 2, built by
*extending* the geometry core (never rewriting it) with every object kept
watertight:

- **A. Subunit names.** New `pdb2print/names.py` parses per-chain molecule names
  from PDB `COMPND` (multi-`MOL_ID` blocks; robust to a missing `;` before a
  `MOL_ID`) and mmCIF `_entity`/`_entity_poly`. `io.load_with_names()` returns
  `(atoms, {chain_id: name})`; `Chain` gains `name` + `display_name()` (falls
  back to "Chain X"); API `chains[]` and the viewer legend show it.
- **B. Progress streaming.** `POST /api/generate` now returns
  `text/event-stream` — `event: progress {frac,msg}` lines then one
  `event: result {…}` with the unchanged JSON shape. The build runs in a worker
  thread; `build_all`'s existing `progress` callback is pumped through an
  `asyncio` queue. Validation errors still return plain JSON (4xx), so the
  client tells them apart by content-type. Frontend shows a progress bar + live
  message.
- **C. Downloads** moved to the top bar; disabled until a result exists, then
  accent-green with a one-shot pulse (3MF/STL independent).
- **D. Connector / joinery system.** New `pdb2print/connections.py` runs *after*
  per-chain meshing (behind `PrintParams.connections`, a `ConnectionParams`).
  Same manifold3d kernel as the representations. **Deliberately small option
  set** (reworked from a larger first cut per user feedback): two independent
  switches —
  - **connect** joins chains whose surfaces come within `contact_threshold_mm`.
    - **magnets**: a press-fit bore in a flush socket on each part — see the v3
      section below, which replaced this entirely.
    - **inflate** (default): `_rebuild_inflated` re-meshes each contacting chain
      slightly larger at *build* time — protein via `surface_atom_padding_ang`,
      DNA via the tube/base radii — so neighbours swell until their surfaces
      overlap and weld, no strut, no re-mesh artefact.  Growth is `gap/2 + weld`
      per side, capped at 1.2 mm (wide gaps should use bridge).
    - **bridge**: a `connector_diameter` cylinder across the gap — since v3 the
      same joint as a magnet, minus the bore.
  - **basepair_connect** ties the two strands of a DNA duplex at every base pair.
    Pairing is **register-based** (`_pair_bases`): it evaluates every antiparallel
    / parallel diagonal and keeps the register with the most in-cutoff pairs —
    this locks onto the true Watson–Crick partner at the helix ends (a greedy
    nearest-neighbour swaps them for the diagonal neighbour) and the
    `basepair_max_dist_ang` cutoff leaves an unwound bubble open instead of
    mis-pairing it. Rod/slab rungs are *continued at their own radius to the
    midline* so opposing rungs meet as one smooth bar; molecule bases get thin
    bond-like spokes.

  Detection is automatic (nearest surface gap); the earlier per-pair checklist /
  `/api/detect` / overrides were removed for simplicity. **Watertight
  guarantee:** every boolean goes through `_commit`, which accepts the result
  only if it is still one connected body — so a pocket that would blow through a
  thin backbone is skipped (min-wall honoured) rather than shipping a broken
  object, and the pipeline re-gates watertightness after the pass.

Kernel additions in `representations/_manifold.py`: `frustum` (flat/tapered
cone, for chamfers & pegs), `from_trimesh` (re-enter the kernel), `difference`.
`tube_slab.base_centroids_mm()` exposes base centroids for base-pair pairing.

Magnet positions are returned as `connection_markers` and drawn as bright
magenta solids in the GLB preview only.

## v3 — magnet axis, flush socket, press fit (this session)

**RESOLVED — magnet axis and seating (v3).** The old failure was that a magnet
was placed on the *nearest-point line* of one contact pair: the smallest gap on
the interface, but not the direction either part actually has material in, so on
a curved interface the magnet sat visibly off-normal. Two earlier attempts were
reverted (averaging point-to-point directions fans around the curve; a plane fit
is ambiguous on a narrow contact strip). The current design replaces both.

**Joint placement is now two-stage and shared by magnets and bridges**
(`connections._joint_seats`):

1. **Stage 1 — shortlist** (`_candidate_seats`, cheap, point clouds). Contacts
   are scored by how much *consistent* contact surrounds them, so a surface
   whisker with the smallest gap never wins over a broad interface. Returns more
   candidates than are wanted, separated by a full socket diameter.
2. **Stage 2 — rank against the real solids** (`_score_seats`). Each candidate's
   probe ball is intersected with *both* manifolds. That single operation gives:
   - **the local centres of mass**, which feed the axis candidates below.
   - **the fill** — the fraction of the plastic the joint needs that is already
     solid, measured from each side's own surface inward (not from the mid-plane,
     or the score would just track the gap). Ranks the seats, so asking for two
     magnets puts the second on the second-best patch instead of beside the first.

### Orientation: three hypotheses, tested against the geometry

The centroid line alone is **not** a reliable axis, and the reason is worth
writing down because it cost a debugging round. **The centre of mass of a rod
slides along the rod.** A DNA backbone tube is locally a rod; the probe ball
clips an asymmetric length of it, so the DNA-side centroid sits well along the
helix rather than opposite the contact, and the axis swings toward the helix —
up to the 90°-wrong magnets seen at some tube thicknesses. Thicker tube, bigger
effect, which is exactly the observed dependence on the tube-thickness setting.

So `_choose_axis` proposes three candidates and *measures* them:

| candidate | what it is | fails when |
|---|---|---|
| `contact` | plain nearest-point line | noisy on a bumpy surface |
| `mass` | line between the local centres of mass | one side is locally a rod |
| `mass-flat` | `mass` with the contact strip's long direction projected out | patch isn't elongated (then it isn't offered) |

`mass-flat` is the rod fix. On an elongated contact patch the strip's *long*
direction is the best-determined thing about it — far better determined than its
normal, which is why the earlier plane-fit attempt failed — and it is precisely
the direction the centroid slides in. Projecting it out keeps the
across-interface component and discards the unreliable one. PCA on the patch
points, gated on λ1/λ2 ≥ `patch_elongation_min`.

Each candidate is then scored by `_path_census` on the sampled surface clouds:
**blocked** points (material that would collide on assembly) and **seated**
points (body for the collar to fuse into), as `seated − 6×blocked`. That is the
physical test — *can these parts actually come apart this way* — and it is what
kills an along-the-helix axis, which drives the socket lengthwise into the tube
and is massively blocked. Point-cloud rather than boolean, because it runs for
several axes at several seats. `_AXIS_PREFERENCE` breaks genuine ties toward the
mass-derived axes, which is what still fixes the merely *tilted* magnets.

Measured on the synthetic rod-and-slab fixture in `tests/test_features.py`: raw
centroid line 40.8° off the true normal; chosen axis 0.0° with a clean contact
line, 2.1° when the contact line is itself 20° off.

**The wrap guard** sits on top: any candidate more than `axis_agreement_min`
(cos 60°) from the plain contact line is rejected outright, for the
protein-wrapped-around-DNA case where the probe ball reaches right around the
duplex. Keep `mass_probe_scale` modest (2× socket radius) for the same reason.
The connection note reports `axis from contact line` or `axis flattened along
contact strip` when a fallback fired, so a suspicious magnet can be traced.

### Clearing the joint path

A joint is only real if the parts can close. Anything of one part that reaches
**past the shared face inside the joint footprint** is cut away — step 1 of
`_build_seat`, footprint plus `path_clearance_mm` sliding clearance. Without it a
lobe of protein hanging over the socket looks correct in the preview and then
collides on assembly. If that cut would sever the part, `_commit` rejects it and
the seat is abandoned in favour of the next-ranked one: a seat you cannot
assemble is not a seat.

**WITHDRAWN / TODO — protein cartoon representation.** The first cartoon pass
(cylinder helices + flat sheet planks, `pdb2print/representations/cartoon.py`)
does not read as a cartoon: helices come out as bare rods with no visible pitch,
sheets as detached planks with no continuity into the loops, and nothing reads
correctly at print scale. It is withdrawn rather than shipped half-working — the
option is gone from the UI, `Representation.CARTOON` is no longer registered in
`geometry._BUILDERS` (requesting it raises "no builder registered"), and the enum
member is kept only so old parameter sets still parse.

What a proper rework needs, roughly the ChimeraX approach:

1. **Real secondary-structure assignment** — DSSP-style H/E/C per residue
   (`biotite.structure.annotate_sse` is the cheap route), not geometric guessing.
2. **A guide spline with an explicit ribbon frame** — a smooth path through the
   CA trace carrying a per-sample normal/binormal, so ribbon *twist* is defined.
   The frame is what a plank-per-residue build is missing.
3. **Swept cross-sections, not primitives** — sweep a profile along the spline
   and vary it by SSE: a flattened ellipse for helices (a coil ribbon, not a
   cylinder), a wide rectangle plus arrowhead taper for strands, a small circle
   for coil. Blend the profile across SSE boundaries so segments stay continuous.
4. **One watertight solid per chain** — sweeps meet end-to-end along a shared
   spline, so this can stay analytic (`manifold3d` `batch_boolean`), like
   `tube_slab`, rather than going back through the voxel grid.
5. **Print-scale sanity** — a ribbon is a thin shell by nature; the profile
   minor axis must respect `min_wall_mm` parametrically *before* the union, the
   same rule `tube_slab` follows. Cartoon should stay out of `MIN_WALL_EXEMPT`
   only if it can self-thicken; otherwise it is not printable unsupported.

Acceptance: 1UBQ (mixed α/β) must be recognisably a cartoon next to a ChimeraX
screenshot, watertight, and printable at default scale without supports.

Tests: `tests/test_features.py` (23 tests) covers names, every join type/seat,
base-pair methods, per-pair override, the watertight gate, SSE, and detect.
Offline fixtures `tests/data/1bna.pdb`, `1zaa.pdb`, and `mini_complex.pdb` (1BNA
DNA + a poly-ALA peptide in contact — a synthetic stand-in for the 1ZAA protein,
whose chain the sandbox web-fetch truncates, as the brief notes).

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
