# pdb2print — developer handoff notes

Context for a fresh session. Covers architecture, the geometry pipeline as it
actually works today, the bugs fixed so far, and known limitations that are
*inherent to the approach* (not bugs). See `README.md` for user-facing docs.

## v7 — DNA controls simplified, publish pass (this session)

**The DNA sizes are now opt-in and say what they do.** Both style pickers stay
visible; every dimension moved behind an **Advanced sizes** disclosure, and only
the sliders the chosen styles actually read are shown. "Rung" is gone from the
vocabulary.

The slab used to need three sliders (thickness, footprint scale, strut) to
describe one plate on one little rod. It is now **two**: *Plate size*, which
scales the plate's area and thickness together at a fixed ratio
(`PLATE_THICKNESS_PER_SIZE = 1.2` in the frontend, so 1.0 reproduces the old
1.2 mm default and a plate can never end up wide and paper-thin), and
*Connecting rod radius*. Rod style shows one slider, molecule styles two each.
This composition lives in `getFormData` — the server API still takes
`slab_thickness` / `base_width` unchanged, so nothing in the core moved.

**Publish readiness.**

- Removed from the repo (moved to `_to_delete/`, which is gitignored — the
  Cowork bridge cannot unlink, so `git rm --cached` or deleting that folder
  finishes the job): `commit-v3.ps1` (a one-shot script whose own header says to
  delete it afterwards), `.thumbnail` (an unreferenced stray WebP), and
  `HANDOVER.md` (a v3-session handover whose first instruction is to clear a git
  lock from July 24 — superseded by this file, and still in git history).
- Personal names genericised in `docs/FEATURE_BRIEF_connectors_and_ux.md`.
  Nothing else in the tracked tree carried a local path, machine name or address.
  (`.gitignore`'s "David-OConnor" is upstream GitHub template text, not personal.)
- `README.md` still advertised the cartoon as withdrawn; it now lists cartoon as
  a protein representation, and the roadmap entry became per-residue colouring.
- `pdb2print/cache.py` (untracked WIP at the time) keyed on `cartoon_smoothing`
  and `cartoon_thickness_mm`, which no longer exist, and had not been told that
  the backbone ball-and-stick sizes are now separate from the bases'. Its
  `drop()` is `pop(..., None)`, so the stale names were harmless, but the
  *missing* ones would have over-keyed the cache. Both pairs are now keyed off
  their own style, and a rod base drops the plate footprint/strut it never reads.

## v6 — DNA controls, DNA↔DNA joins, cancel (previous session)

**Cartoon smoothing is no longer a setting.** It was exposed as a slider, tried,
and removed: the useful range was narrow enough that the default was the only
sensible value. The behaviour is unchanged — the old default is frozen as
`cartoon._SMOOTH` — so only the knob went away. `cartoon_smoothing` is gone from
`PrintParams`, the server form and the UI. The Cartoon panel is now three sliders
(Helix size, Sheet size, Tube thickness).

**Two DNA controls were lying about what they did.**

- **The rod rung ignored its own thickness slider.** `_base_solids` sized the rod
  as `tube_r * 0.7` — derived from the *backbone tube* radius — so "Base
  thickness" did nothing in rod mode and the rung resized when you touched the
  backbone. It is now `slab_t / 2`, i.e. the rung-thickness control is its
  diameter, which is what `slab_thickness_mm`'s own comment ("base-slab (or rod)
  thickness") always claimed. Side benefit: `_rebuild_inflated` grows
  `slab_thickness_mm` by `2 × amount`, so a rod now inflates by exactly the
  requested amount instead of `0.7 ×` it.
- **Base width and strut radius never applied to rods** (a rod is round and is
  its own link, so there is nothing to widen and no strut), but the UI showed
  them in rod mode. They are now slab-only.

**Backbone and base ball-and-stick sizes are now separate.**
`backbone_atom_radius_mm` / `backbone_bond_radius_mm` join the existing
`atom_radius_mm` / `bond_radius_mm`. One shared pair could only ever be right for
one of the two, since the sugar-phosphate backbone and the base rings are drawn
at the same time at different weights. Both new params are wired through
`_min_wall_dims`, `_backbone_solids`, the server form, the frontend, and
`_rebuild_inflated` (miss that last one and molecule-backbone is the single style
that silently refuses to inflate). `app.py` has one slider pair and drives both.

**DNA card layout.** Each style's dimensions now sit under the style that uses
them — tube radius moved out of the shared "Advanced dimensions" drawer and under
**Backbone**, where it belongs — and only the sliders a style actually reads are
shown. Names say what they size: Strand tube/atom/bond radius, Rung thickness,
Slab size scale, Slab strut radius, Base atom/bond radius.

**DNA↔DNA is never magnetised.** A pocket for even a small magnet is several
times wider than a backbone tube (1.2 mm default radius vs a 4 mm magnet), so the
socket cannot sink into the strand — it is a boss standing proud of it, and the
bore usually blows through, which `_commit` then rejects, losing the joint
anyway. Those pairs fall back to the bridge and say so in the connection note.
Protein↔DNA and protein↔protein are untouched.

**Builds are cancellable.** `build_all(..., should_cancel=...)` polls a predicate
between phases and between chains and raises `pipeline.BuildCancelled`, which the
per-chain and connector-pass handlers re-raise rather than swallowing into a
"skipped" warning. Cancellation is cooperative, so the granularity is one chain —
a single long boolean cannot be interrupted. The server sets a `threading.Event`
in the SSE generator's `finally`, which fires when the client disconnects; the
frontend's Cancel button aborts the fetch, so cancelling frees the machine and
not just the browser. The Generate button becomes Cancel while building (hover
swaps the live "Building… 42%" for "Cancel"); the ID field's go/Enter only ever
*starts* a build, so a stray Enter cannot kill a running one.

**One test was measuring sampling noise.**
`test_inflate_grows_and_closes_gap` asserted every pair's gap shrank to within
`1e-6`. `_nearest` compares *random vertex subsamples* (`_probe_points`), so
re-meshing reshuffles which vertices are compared and moves the reading by
microns; on the already-welded B↔P pair (0.11 mm) that showed up as a 6.6 µm
*increase* while the genuine pairs closed by ~1.8 mm. Pairs that start below the
weld threshold are now exempt. This is the same subsample-noise effect the
joint-placement findings doc documents — worth recognising before treating a
micron-scale movement in any of these measurements as a regression.

**Pre-existing failure, diagnosed but not fixed.**
`test_min_wall_thickens_tube_slab_parametrically` sets `nucleic_radius_mm` and
expects a protein tube-slab build to thicken, but the protein path reads
`protein_tube_radius_mm` (default 1.2) and both min-wall values it tests
(0.4, 2.0) floor below that, so the mesh is byte-identical. The test predates the
protein/nucleic tube-radius split and is stale; it is not evidence of a geometry
bug. (`test_magnets_seat_where_the_parts_were_interpenetrating` also fails on the
sandbox and predates this session.)

## v5 — cartoon representation shipped (this session)

The withdrawn cartoon (cylinder helices + detached planks) is replaced by a real
ChimeraX-style ribbon in `representations/cartoon.py`, registered in
`geometry._BUILDERS` and offered as a third **Protein** style ("Cartoon") in the
web UI and Gradio app. This closes the "WITHDRAWN / TODO — protein cartoon"
item in the v3 notes below.

How it works, following the intended rework point-for-point:

- **SSE** from `biotite.structure.annotate_sse` (P-SEA), all-coil fallback.
- **Carson–Bugg guide frame**, not Frenet. The ribbon width vector is the
  carbonyl `C→O` direction projected perpendicular to the tangent, with the
  180°/residue alternation removed by flipping and gaps filled by transport.
  This is the whole reason strands lie flat instead of corkscrewing — a
  tangent-derived frame is undefined exactly where a strand is straight.
- **Smoothed control points**, strength by structure (`cartoon_smoothing` is the
  master): strands smoothed fully (flattens the pleat without shortening a
  near-straight strand), helices only lightly and hard-capped (Laplacian
  smoothing shrinks a helix toward its axis, so strong smoothing collapses the
  spiral into a straight twisted stick — the first thing that looked wrong),
  coil left on the true trace.
- **Swept closed cross-sections** = a **rounded rectangle** (`_section`) whose
  corner radius is the smaller half-axis, sampled by arc length. For a ribbon
  (`hw > ht`) that is a *stadium*: flat top/bottom faces, semicircular thin
  edges — reads as a flat plank and never as a rounded square, whatever the
  thickness; when `hw == ht` it is a circle, so coil is a round tube and the two
  morph smoothly. Strands widen into a **arrowhead** over the last
  `cartoon_arrow_residues`. Fixed vertex count per ring ⇒ rings correspond ⇒
  smooth morphs.

  (v1 used a superellipse cross-section and uniform smoothing; both were
  reworked after the ribbon looked square when thickened and helices read as
  twisted tubes.  Default `cartoon_thickness_mm` lowered 2.0 → 1.4 for a flatter
  default plank.)
- **One watertight solid**: a single `M×K` ring grid closed with two cap fans,
  built as a `trimesh` and returned; `is_watertight` holds by construction. No
  manifold union needed because it is one continuous loft (chain breaks > 4.5 Å
  are bridged by the tube so the print stays one piece). Self-intersection where
  a helix twists tightly is topologically harmless and slicers resolve it; enough
  samples/residue keeps twist-per-segment small anyway.
- **Print sizing**: every dimension (widths, derived thickness, coil radius,
  blunted arrow tip) is grown to `min_wall_mm` *before* the sweep, so CARTOON
  stays in `MIN_WALL_EXEMPT` and needs no voxel pass.

**Second-pass fixes (same session, after review):**

- **"Thick ribbon turned into a tube."** The stadium rounds to a circle as its
  thickness approaches its width, so an independent thickness knob made the
  ribbon round off. Fixed by a **fixed flat aspect** (`_RIBBON_ASPECT = 0.30`):
  thickness is always 0.30 × width, so one *size* slider scales the whole ribbon
  and it stays a flat plank at any size. Independent `cartoon_thickness_mm` is
  gone.
- **"Helix crooked / sharp turns."** Twist between residues was linear-interp of
  the width vectors (non-uniform → kinks). Now **slerp** (`_slerp`), plus a light
  neighbour-average of the guide vectors (`_smooth_widths`) and samples/residue
  raised to 10.
- **"Arrows without ribbons."** `_clean_sse` demotes runs shorter than
  `_MIN_STRAND_LEN` (3) / `_MIN_HELIX_LEN` (4) to coil, and the arrowhead is
  clamped to leave ≥1 residue of shaft, so a strand is never all arrow.
- **UI cut to three sizes + smoothing.** The confusing width/thickness/arrow/
  samples sliders are gone. The panel is now **Helix size**, **Sheet size**,
  **Tube thickness** (a *diameter* in the UI → halved to the coil radius in
  `getFormData`), and **Smoothing**. Params: `cartoon_helix_width_mm`,
  `cartoon_strand_width_mm`, `cartoon_coil_radius_mm`, `cartoon_smoothing`;
  `cartoon_arrow_width_factor`, `cartoon_arrow_residues`,
  `cartoon_samples_per_residue` are now internal (config defaults, not surfaced).

Verified: 1UBQ (mixed α/β) builds watertight, single body, 3MF + GLB export
clean; two new tests in `tests/test_features.py`
(`test_cartoon_builds_watertight_single_body`, `..._degrades_to_tube_without_sse`).
Aimed at resin scale; the ribbon is a thin feature, so FDM wants a higher
scale/thickness and supports.

**Not done / possible next:** per-residue rainbow (N→C) colouring for the GLB
preview — would make e.g. GFP pop — is not wired; export still colours per chain.
`_catmull_pos_tan` was added to `representations/_common.py` (analytic
position+tangent) for the frame sweep.

## v4 — parts that actually fit (this session)

Three symptoms turned out to be one root cause: **nothing in the pipeline had a
signed notion of inside/outside between two chains.**

- **Chains meshed independently interpenetrate.** Each chain's surface is built
  from its own atoms as if the others were not there, so at a binding interface
  both solids claim the same volume. Correct as a picture, impossible as parts.
- **Magnets never landed on the overlap.** `_candidate_seats` ranked candidates
  on an *unsigned* nearest-vertex distance. For a vertex of A buried inside B,
  the nearest vertex *of B* is out on B's surface, so that distance equals the
  penetration depth — a millimetre or two — and the deepest, meatiest part of
  the interface scored as "far away". The smallest distance instead landed on
  the rim where the surfaces cross, the thinnest part of the joint.
- **One magnet came out tilted.** Same cause. The contact direction `B - A`
  reverses across that rim, so the seed axis could be tilted or inverted — and
  since every better-founded axis is vetoed for disagreeing with the seed
  (`axis_agreement_min`), one bad seed forced the fallback.

**`pdb2print/interference.py`** (new) resolves interpenetration with booleans
before anything else runs, and runs whether or not connectors are on — two
objects that are merely printed and handed over still have to fit. Rule
`AUTO`: nucleic keeps its true shape and the protein is carved (a socket that
is an exact negative of the duplex, like a mould); between two chains of the
same type the larger keeps its shape. `SYMMETRIC` has both retreat; `NONE` is
diagnostics only. Clearance comes from growing the keeper first — by a union of
14 translated copies, because a true `minkowski_sum` takes ~28 s on a
protein-sized mesh versus ~0.1 s for this. The carve is confined to boxes around
the interference lobes (merged, max 4) with a 3 mm margin: a full-length
subtraction against two parallel molecular surfaces is the pathological case for
a boolean kernel and cost seconds per pair.

Magnet seats are then taken from the overlap lobes: **position** from the local
contact between the carved solids (so the seat lands on the real mating face),
**direction** from the lobe's thin principal axis. An interference lobe is a
lens — broad across the interface, thin through it — so its smallest principal
direction *is* the interface normal, averaged over the whole patch instead of
read off one vertex pair. That axis is trusted and not put through the veto.

**The probe radius was the wrong lever, and is now bounded.** The SES field is
`EDT(atoms grown by p) − p`; on a convex patch those cancel exactly, leaving a
surface at `vdW + padding` regardless of `p`. An interface is convex-facing on
both sides, so **lowering the probe radius cannot reduce interpenetration** — it
only carves into concave pockets. What it does control is connectivity: two
atoms stay joined only while their grown balls overlap (`D < r1 + r2 + 2p`).
Measured on 1UBQ at 1.5 mm/Å, 0.5 mm grid — raw marching-cubes bodies, and the
volume `repair` then discards:

    probe   bodies   watertight   volume lost
     1.0       9        no           2.1 %
     1.2       6        yes          0.7 %
     1.4       2        yes          0.1 %
     1.5       1        yes          0.0 %

1.0 Å is exactly the setting that was destabilising builds. The floor is now
1.4 Å (the water probe), max 2.0, enforced in `config.resolve_surface_grid` and
in the slider. Note the grid spacing is *not* the mechanism — 1.4 Å stayed
watertight down to a 0.93-voxel erosion band while 1.0 Å failed at 3.0 — so grid
auto-refinement is deliberately mild and only catches the sub-one-voxel case.

Two latent bugs surfaced once multi-body chains became reachable (a carve may
legitimately split a loop the DNA threads through):

- `connections._commit` demanded exactly one body, which would have rejected
  *every* boolean on such a chain and silently dropped all its connectors. Now
  it checks the count did not increase.
- `meshops.repair` sent watertight multi-body meshes through `merge_vertices`,
  which pinched them into non-manifolds — watertight in, not watertight out,
  tripping the export gate. The fast path now covers any watertight mesh and
  speck-dropping is split/concatenate only.

**The socket collar was left alone, deliberately.** Three attempts were made at
the cosmetic complaint that a flat-ended collar shows its end cap as a flat spot
where it only half-sinks into a curved surface. All three were reverted:

1. **Rounded end + "skirt".** A hemisphere of the *collar's* radius is a ball
   wider than a DNA backbone tube — it replaced one eyesore with a bigger one —
   and the skirt (sweep a wall down, intersect with the offset body) became the
   slowest thing in the pass.
2. **Long taper.** Better, but the cone was too long and read as a nose cone
   stuck on the back of the joint.
3. **Short steep chamfer + measured depth.** Fixed the poking-through, but the
   steeper angle looked worse again, and the depth measurement changed which
   seats succeeded — placement and orientation regressed against a version that
   was already good.

`_seat_solid` and `_build_seat` are now byte-identical to the pre-v4 versions.
If this is revisited: the cosmetic problem is real but minor, the geometry is
entangled with seat selection (changing collar depth changes which seats pass
`_commit`, hence where magnets land), and the underlying issue on nucleic chains
is not shape at all — a socket sized for a 4 mm magnet is ~3x wider than a
default 1.2 mm-radius backbone, so *any* collar there is a boss standing proud.
Shrinking `connector_diameter_mm` or raising `nucleic_radius_mm` is the real fix.
Do not touch this to chase looks without a way to check placement did not move.

**Known limitation.** Connectors are added *after* the fit pass, so a collar
driven through a thin backbone into a neighbour can reintroduce interference. A
closing sweep removes what it can without cutting a part in two;
`interference.audit` names anything that survives in the build warnings rather
than shipping parts that quietly will not close. Trimming each collar against
its neighbours at build time was tried and reverted — it cost more than the
entire rest of the pass, because every retry redoes the clip.

**Performance on a real complex (1TUP: 3 x p53 core + DNA duplex, 5 chains).**
First run of the fit pass took 24-33 s. Profiling, rather than guessing, found
it in one place: `interference.dilate` was unioning **fourteen** translated
copies of a large mesh to achieve a 0.15 mm offset — 8.2 s of a 24 s build
across 70 calls. Cut to the six axis directions (worst-case offset 0.58x
nominal, so the request is scaled by 1.25 to put the spread either side of
nominal): at the default clearance that is 0.11-0.19 mm, which is well inside
what an FDM mating surface cares about.

Second, all the collar probes — `_end_is_buried`, `_collar_skirt` — were asking
local questions of the *whole chain*. On a 230k-triangle protein that is ~16 ms
per query against the full solid versus ~3 ms against a local clip, and the clip
itself costs 0.1 ms. `_near` clips once per seat and the search runs against
that; only the final commit touches the full solid. Together: 33 s -> 15.5 s
with byte-identical output. Whole 1TUP build, everything switched on
(magnets 2/1 + base pairs), is ~30 s end to end, all watertight, zero residual
overlap.

Two things worth remembering here. `manifold3d` is **lazy** — booleans build a
DAG and evaluate when something forces it — so cProfile attributes the work to
whatever call triggers evaluation, not to the operation that queued it. Read the
totals, not the per-function attribution. And the guesses that felt obvious were
both wrong: `_merge_boxes` (suspected O(n^3)) is microseconds, and overlap
detection on 1TUP is 0.00 s per pair — the DNA does not interpenetrate the
protein there at all.

**Progress reporting.** `connections.apply` now takes a `progress` callback and
reports per interface, forwarded by `build_all` into the 0.90-0.96 band. Before
this the bar parked at "Fitting and connecting objects..." for the entire pass,
so a build that was working and a build that was stuck were indistinguishable —
which is exactly how a slow 1TUP run was first reported. The phase names are
also the first diagnostic now: stalling at "Connecting X <-> Y" is boolean work
on that interface, whereas stalling at "Rebuilding meshes..." means
`meshops.repair` fell off its fast path into the heavy pymeshlab route (note
pymeshlab is *not* installed in the dev sandbox, so that path is invisible there
and will only ever show up on a real install).

**Cost.** The fit pass roughly doubles the connector pass on a heavily
interpenetrating structure (~2 s → ~6 s on the test fixture). Test fixture
`tests/data/overlap_complex.pdb` is synthetic: 1BNA plus two copies of a
34-residue ubiquitin lobe dropped onto the duplex, to force deep protein–DNA and
protein–protein interference.

**Not yet done.** 1TUP has not been run end-to-end — the sandbox cannot reach
RCSB. Still outstanding from v3: slice in PrusaSlicer at real INDX print scale.

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
