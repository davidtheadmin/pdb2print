# Handover — next session

Written at the end of the v3 session (magnet axis / flush socket / press fit).
Read `NOTES.md` for the *why* behind the architecture; this file is just "where
things stand and what to do next".

---

## 1. Do this first

**Delete the stale git lock.** There is a zero-byte `.git/index.lock` dated
Jul 24 23:58, left behind by a crashed git process. It blocks every commit.

```powershell
del .git\index.lock
```

Nothing was committed this session because of it. The working tree holds all the
v3 work and is the state to commit — see §2.

---

## 2. Commit the current state

The tree is at a **working, tested-by-hand state**: 1TUP builds and the joinery
works. Commit it before touching anything.

```powershell
del .git\index.lock
git add -A
git commit -m "v3: scored magnet placement, flush socket, press-fit bores, path clearing"
git push
```

`experiments-to-reapply.patch` in the repo root is **not** part of that state —
see §5. Either commit it as a loose file or move it out of the repo first.

---

## 3. What changed this session

**UI**
- Cartoon representation removed from the UI and unregistered in
  `geometry._BUILDERS` (see the WITHDRAWN/TODO block in `NOTES.md` for the
  rework design). `cartoon.py` is kept as a starting point.
- Generate button is sticky at the top of the settings column.
- Download buttons moved top-left of the viewer, enlarged, relabelled.
- Exports are named `1zaa_pdb2print_1p5mm.3mf` etc. instead of `out.3mf`.

**Joinery (the bulk of the work)**
- **Flush socket**, on by default: a flat-ended collar on each part so the two
  halves meet on one clean disc. Applies to magnets *and* bridges.
- **Press-fit bores**: oversize diameter + depth, 45° lead-in, clamped so a thin
  magnet keeps its grip. Clearance exposed in the UI (printer-specific).
- **Two-stage seat scoring**: cheap point-cloud shortlist, then ranked against
  the real solids. Asking for two magnets now uses the second-best patch.
- **Axis selection by test, not construction** — three candidate directions
  (`contact`, `mass`, `mass-flat`) scored by what would collide on assembly.
  This is the fix for the 90°-wrong magnets on DNA. `NOTES.md` explains why the
  centroid line alone fails (the centre of mass of a rod slides along the rod).
- **Path clearing**: material of one part reaching past the mating face inside
  the joint footprint is cut away, so the parts can actually close.
- **Bridge is now the same joint minus the bore** — a true cylinder split on a
  shared face, replacing the old capsule with its hemispherical bobble.

---

## 4. Open problems, in priority order

### 4a. Watertight-gate failure on 1TUP chains A/B — UNRESOLVED

The headline open bug. Symptom: `Watertight gate failed … chain_A_protein`.

**Confirmed:** the gate that fires runs in the per-chain meshing loop, *before*
`connections.apply` is ever called, and nothing in the surface path reads any
connector parameter. So the correlation David observed with magnet size (fine at
5×3 mm, fails at 3×1 mm) has no mechanism behind it that anyone has found.
Either something else moved at the same time, or the mechanism is still hidden.
**Do not assume it is the magnets, and do not assume it is not.**

Hypotheses tested and **discarded** (do not repeat these):

| hypothesis | how it died |
|---|---|
| Magnet settings cause it | Gate runs before the connections pass |
| Shared mutable state between runs | `default_factory` is correct; module globals are read-only |
| Probe radius landing on a discrete EDT distance (`spacing × √k`) | Normalised against local spacing of achievable distances: failing 0.32 vs working 0.36. Indistinguishable |
| Marching cubes on the SES field is fragile | 120 random atom clusters at the failing settings, zero non-watertight |
| `allow_degenerate=False` punching holes | A sphere whose radius is an exact multiple of the grid spacing stayed watertight either way |

**What would actually crack it:** the real 1TUP structure. The sandbox web-fetch
truncates it to 127 atoms, so it could not be reproduced. Run it locally, catch
the failing chain's mesh before `repair()`, and report:

```python
groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=None)
sizes  = [len(g) for g in groups]
print("boundary edges:", sum(1 for s in sizes if s == 1),
      "non-manifold edges:", sum(1 for s in sizes if s > 2),
      "bodies:", mesh.body_count)
```

Those two numbers separate *holes* from *seams*, which need different fixes.
Everything so far has been guesswork precisely because nobody has them.

**Also check whether `pymeshlab` is installed in your venv** (`pip show
pymeshlab`). Without it the repair path is only hole-filling, which cannot fix a
non-manifold edge at all — so a mesh it would have rescued is refused instead.
It lags new Python releases, so it is easy to be silently missing.

### 4b. PrusaSlicer acceptance test — still not done

Flagged across four sessions now. Slice 1ZAA at real INDX print scale and
confirm three separately filament-assignable objects with no manifold warnings.
Until this is done, "it exports" is not the same as "it prints".

### 4c. Press-fit clearance is unvalidated on hardware

0.2 mm on diameter is the textbook number, not a measured one. Print a socketed
joint and check the fit on the Core One. The slider exists because this is the
one value that cannot be derived.

### 4d. Cartoon representation

Withdrawn, not fixed. `NOTES.md` has a five-point design for a proper
ChimeraX-style rework (SSE assignment → guide spline with a ribbon frame →
swept SSE-varying cross-sections → analytic union → min-wall before the union).

---

## 5. `experiments-to-reapply.patch`

A later round of mesh-repair work was **reverted** because it made things worse:
a per-seat watertightness check converted the full chain mesh on every attempt
(1TUP got very slow), and a voxel-resolidify fallback closed broken meshes by
softening the whole model — the wrong trade.

Two things in that patch are worth re-applying **separately, on a green tree**,
because neither touches the connector path nor costs runtime:

1. **Pad the marching-cubes field** with an "outside" shell. Marching cubes emits
   nothing outside the array, so an iso-level reaching the array boundary
   produces an open surface. Verified in isolation: a solid running off every
   side closes, and a sphere's radius is unchanged to within 0.002 mm.
2. **Stop `repair()` deleting sliver triangles** (`nondegenerate_faces`).
   Removing a face from a closed mesh punches a hole. Tested clean, but deleting
   faces can only ever hurt.

Do **not** re-apply the voxel resolidify or the per-seat `_is_sound` check.

Delete the patch file once you have decided.

---

## 6. Things that are true and easy to forget

- **Grid spacing is a ratio.** `grid_spacing_mm / scale_mm_per_angstrom` is what
  governs mesh quality, so lowering the scale coarsens the mesh without the grid
  slider moving. 1.5 mm/Å + 0.5 mm grid = 0.33 Å.
- **Finer is not reliably better.** A coarse grid smooths a thin neck closed; a
  fine one resolves it into two surfaces touching at a point, which is
  non-manifold. The failure is not monotonic in resolution.
- `manifold3d` is a hard dependency on purpose. Do not add a silent fallback.
- Every boolean goes through `_commit`, which rejects anything that is not still
  one connected body. A bad connector must cost a magnet, never the export.
- Commit a working checkpoint before geometry rework. This session could not,
  because of the git lock, and that made the revert much more manual than it
  needed to be.

---

## 7. Test suite

`pytest tests/` — offline, uses bundled fixtures. The v3 additions cover the
flush faces, the oversize bore, tilted-interface axis recovery, the wrap
fallback, the rod-shaped-blob 90° case, the strip-projection rescue, path
clearing, and two magnets landing apart.

**These have not been run**: the session sandbox had no `manifold3d` or
`biotite`, and the project `.venv` is Windows-only. Run them first thing and
treat any failure as a real finding, not a stale test.
