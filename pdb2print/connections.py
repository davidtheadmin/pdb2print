"""Post-build connector / joinery pass (simplified).

The per-chain meshes come out of the geometry core as separate watertight
solids, one colour each.  This pass *modifies those solids* — with the same
``manifold3d`` kernel the representations use — so chosen chains are joined for
printing, while **every object stays watertight and a single connected body**.
The 3MF export is unchanged: still individual coloured objects, only now
touching, bulged together, or pocketed for magnets.

Two independent switches (see :class:`~pdb2print.config.ConnectionParams`):

* **connect** — join chains whose surfaces come within ``contact_threshold_mm``:
  - **magnets**: seat a press-fit magnet pocket in each side, so parts printed
    separately snap together;
  - **inflate**: grow both surfaces at the contact until they merge (organic,
    no visible strut) — the default;
  - **bridge**: the same joint minus the pocket — a clean cylinder split on the
    shared face.

**How a joint is placed.**  Magnets and bridges share one path, in two stages.
Stage 1 shortlists candidate contact patches from the surface point clouds,
scored by how much *consistent* contact surrounds each one, so a surface
whisker with the smallest gap never beats a broad interface.  Stage 2 then
measures each shortlisted candidate against the real solids: intersecting both
parts with a ball at the contact gives the local centre of mass on each side,
and the line joining those two centroids is the direction in which both parts
actually have material — the axis a magnet should lie along.  The same
intersection yields the *fill*, the fraction of the plastic the joint needs that
is already solid, which ranks the seats so a second magnet lands on the second
best patch rather than next to the first.

The centroid line is not trusted blindly: where a protein wraps *around* curved
DNA the probe ball reaches right around the duplex and drags the protein's local
centre of mass to the far side.  So it is accepted only when it agrees with the
plain nearest-point line to within ``axis_agreement_min``, and otherwise the
nearest-point line is kept (reported as ``axis from contact line``).

**The socket.**  On by default.  Each part gets a flat-ended collar driven from
the shared mid-plane into its own body, so the two halves meet on one clean disc
instead of two ragged molecular surfaces touching wherever they happen to.  The
magnet bore is cut *oversize* — wider and deeper than nominal, with a 45° lead-in
— because an FDM hole printed to a magnet's exact size comes out too small to
accept it.
* **basepair_connect** — tie the two strands of a DNA duplex together at every
  base pair.  Complementary bases are paired by centroid geometry with an
  antiparallel-register check and a distance cutoff, so a wrong pair or an
  unwound bubble is never bridged.  Rod/slab rungs are *continued at their own
  radius to the midline* so opposing rungs meet as one smooth bar; molecule
  bases are joined by thin bond-like spokes.

**Watertight guarantee.** Every boolean goes through :func:`_commit`, which
accepts the result only if it is still a single connected body — so a pocket
that would blow through a thin backbone is skipped (min-wall honoured) rather
than shipping a broken object.  The pipeline re-gates watertightness after the
pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import (
    PrintParams, ConnectionParams, NoMagnetMethod, MagnetShape,
    MoleculeType, BaseStyle,
)
from .chains import Chain
from .representations import _manifold, tube_slab


# Cap the vertex clouds used for nearest-point detection so big surface meshes
# stay fast; chunky print geometry does not need exact nearest surface points.
_MAX_PROBE_VERTS = 3000


@dataclass
class Connection:
    """One applied (or skipped) connector, for the report/UI."""

    a_id: str
    b_id: str
    kind: str            # protein-protein | dna-protein | dna-dna | dna-basepair
    method: str          # magnet | inflate | bridge | basepair
    gap_mm: float = 0.0
    count: int = 1
    applied: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "a": self.a_id, "b": self.b_id, "kind": self.kind,
            "method": self.method, "gap_mm": round(float(self.gap_mm), 2),
            "count": self.count, "applied": self.applied, "note": self.note,
        }


# --------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _frame(axis: np.ndarray) -> np.ndarray:
    """Orthonormal 3×3 whose *rows* are (u, v, axis) — for oriented boxes."""
    w = _unit(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(ref, w))
    v = np.cross(w, u)
    return np.array([u, v, w])


def _probe_points(mesh) -> np.ndarray:
    v = np.asarray(mesh.vertices, float)
    if len(v) > _MAX_PROBE_VERTS:
        idx = np.random.default_rng(0).choice(len(v), _MAX_PROBE_VERTS, replace=False)
        v = v[idx]
    return v


def _nearest(mesh_a, mesh_b) -> Tuple[np.ndarray, np.ndarray, float]:
    """Nearest (point_on_a, point_on_b, gap) between two meshes' vertex clouds."""
    pa = _probe_points(mesh_a)
    pb = _probe_points(mesh_b)
    dist, idx = cKDTree(pb).query(pa, k=1)
    k = int(np.argmin(dist))
    return pa[k], pb[idx[k]], float(dist[k])


def _min_wall_radius(params: PrintParams, r: float) -> float:
    if params.min_wall_mm > 0:
        return max(r, params.min_wall_mm / 2.0)
    return r


def _kind(a: Chain, b: Chain) -> str:
    if a.mtype == MoleculeType.PROTEIN and b.mtype == MoleculeType.PROTEIN:
        return "protein-protein"
    if a.mtype == MoleculeType.NUCLEIC and b.mtype == MoleculeType.NUCLEIC:
        return "dna-dna"
    return "dna-protein"


# --------------------------------------------------------------------------
# Watertight-safe boolean commit
# --------------------------------------------------------------------------
def _components(man) -> int:
    try:
        return len(man.decompose())
    except Exception:
        return 1


def _commit(man, tool, add: bool):
    """Apply ``man ± tool`` and accept it only if the result is one solid body.

    Returns the new manifold on success, or ``None`` if the boolean emptied the
    object or split it into more than one piece (e.g. a pocket that would blow
    through a thin wall).  This is what keeps every object watertight and honours
    minimum wall — a bad connector is skipped, never shipped.
    """
    try:
        out = (man + tool) if add else _manifold.difference(man, tool)
    except Exception:
        return None
    if out.is_empty() or _components(out) != 1:
        return None
    return out


# --------------------------------------------------------------------------
# Chain-to-chain join primitives
# --------------------------------------------------------------------------
#: Minimum number of supporting contacts under a magnet disc to accept the seat.
#: Below this the interface is a spike/sliver with too little meat — abandon it.
_MIN_MAGNET_FOOTPRINT = 3

#: A seat is rejected if less than this fraction of the plastic it needs (the
#: socket cylinder driven into the part) is actually inside the part *after* the
#: socket collar is allowed to make up the difference.  Guards against seating a
#: magnet on a spike that the collar would have to build out of thin air.
_MIN_SEAT_FILL = 0.35


def _seat_solid(face, into, length, radius):
    """A flat-ended cylinder from the mating face ``length`` deep into a part.

    ``into`` is the unit vector pointing from the face into that part's body, so
    the flat end always lands exactly on the shared mid-plane.  Both parts build
    one of these against the same plane, which is what makes them meet flush.
    """
    return _manifold.frustum(face, face + into * length, radius, radius)


def _pocket_tool(face, into, depth, radius, chamfer, shape: MagnetShape):
    """The magnet pocket cut into one part, mouth on the mating face.

    Sized for a *press fit*, not a nominal fit: the caller passes a radius and
    depth that already include the FDM clearances, because a hole printed to the
    magnet's exact size comes out undersize and will not take it.  A 45° lead-in
    at the mouth lets the magnet start square and swallows the elephant-foot
    bulge at the face.

    ``into`` points from the face into the body, and the tool is started a hair
    *outside* the face so the difference always breaks the surface cleanly rather
    than leaving a film.
    """
    lip = 0.05
    mouth = face - into * lip
    if shape == MagnetShape.SQUARE:
        parts = [_manifold.oriented_box(
            face + into * (depth / 2.0 - lip / 2.0), _frame(into),
            [radius, radius, (depth + lip) / 2.0])]
        if chamfer > 0:
            # A box has no frustum, so the lead-in is a shallow oversized step.
            parts.append(_manifold.oriented_box(
                face + into * (chamfer / 2.0 - lip / 2.0), _frame(into),
                [radius + chamfer, radius + chamfer, (chamfer + lip) / 2.0]))
        return _manifold.union(parts)

    parts = [_manifold.frustum(mouth, face + into * depth, radius, radius)]
    if chamfer > 0:
        parts.append(_manifold.frustum(
            mouth, face + into * chamfer, radius + chamfer, radius))
    return _manifold.union(parts)


def _farthest_seeds(points: np.ndarray, n: int) -> List[int]:
    """Indices of ``n`` well-spread points (farthest-point sampling)."""
    if len(points) <= n:
        return list(range(len(points)))
    seeds = [0]
    d2 = np.sum((points - points[0]) ** 2, axis=1)
    for _ in range(1, n):
        k = int(np.argmax(d2))
        seeds.append(k)
        d2 = np.minimum(d2, np.sum((points - points[k]) ** 2, axis=1))
    return seeds


@dataclass
class Seat:
    """One scored candidate joint location on an interface."""

    center: np.ndarray       # mid-plane point the two flat faces meet on
    axis: np.ndarray         # unit vector from part A into part B
    gap: float               # local surface-to-surface gap (mm)
    footprint: int           # supporting contacts under the disc (stage 1)
    patch: np.ndarray = None      # the local contact midpoints (stage 1)
    fill: float = 0.0        # fraction of the needed seat volume that is solid
    agreement: float = 1.0   # cos angle between chosen axis and nearest-point line
    blocked: int = 0         # surface points sitting in the assembly path
    axis_source: str = "contact"   # "mass" | "mass-flat" | "contact"
    score: float = 0.0


def _patch_long_axis(points: np.ndarray):
    """``(direction, elongation)`` of a contact patch, by PCA.

    On a narrow contact *strip* — a protein lying along a DNA backbone, say —
    the strip's long direction is the best-determined thing about it, far more
    stable than its normal.  That matters because it is exactly the direction a
    rod-shaped blob's centre of mass slides along, so knowing it lets us take it
    back out of the axis.  ``elongation`` is λ1/λ2; ~1 means a round patch with
    no meaningful long direction.
    """
    if points is None or len(points) < 4:
        return None, 1.0
    centred = points - points.mean(axis=0)
    try:
        _u, s, vh = np.linalg.svd(centred, full_matrices=False)
    except Exception:
        return None, 1.0
    if len(s) < 2 or s[1] < 1e-9:
        return (_unit(vh[0]), np.inf) if len(s) else (None, 1.0)
    return _unit(vh[0]), float(s[0] / s[1])


def _path_census(pa, pb, center, axis, radius, length):
    """``(blocked, seated)`` surface-point counts for one candidate axis.

    Cheap stand-in for "can these two parts actually come apart along this
    direction, and is there anything to bolt the collar to".

    * **blocked** — points of one part that sit inside the joint footprint but on
      the *other* part's side of the shared face.  Every one of those is material
      that has to be cut away, or the parts will not mate at all.
    * **seated** — points of each part inside the footprint on its own side,
      i.e. body for its collar to fuse into.

    Counted on the sampled surface clouds rather than by boolean, because this
    runs for several candidate axes at several candidate seats and the exact
    version would dominate the build time.
    """
    blocked = seated = 0
    for pts, sign in ((pa, -1.0), (pb, +1.0)):
        into = axis * sign                     # from the face into this part
        rel = np.asarray(pts, float) - center
        t = rel @ into
        radial = np.linalg.norm(rel - np.outer(t, into), axis=1)
        inside = radial <= radius
        blocked += int(np.count_nonzero(inside & (t < -0.02)))
        seated += int(np.count_nonzero(inside & (t > 0.0) & (t <= length)))
    return blocked, seated


def _axis_options(seat: Seat, cen_a, cen_b, cp: ConnectionParams):
    """Candidate joint axes for one seat, as ``(label, unit vector)``.

    Three hypotheses, because no single construction survives every interface:

    * ``contact`` — the plain nearest-point line.  Noisy, but never absurd.
    * ``mass`` — the line between the two local centres of mass.  Points along
      the material, which is what we want, *except* when one side's local blob is
      a rod (a DNA backbone tube): then the centroid slides along the rod as the
      probe ball clips it asymmetrically, and the axis swings toward the helix.
    * ``mass-flat`` — the mass line with the contact strip's long direction
      projected out.  This is the rod fix: it removes the one component the
      centroid is unreliable in, and keeps the component across the interface.

    They are not ranked here — the caller measures each against the actual
    geometry and picks whichever can really be assembled.
    """
    opts = [("contact", seat.axis)]
    if cen_a is None or cen_b is None:
        return opts

    delta = cen_b - cen_a
    if float(np.linalg.norm(delta)) <= 1e-6:
        return opts
    mass = _unit(delta)
    opts.append(("mass", mass))

    long_dir, elongation = _patch_long_axis(seat.patch)
    if long_dir is not None and elongation >= cp.patch_elongation_min:
        flat = mass - long_dir * float(mass @ long_dir)
        if float(np.linalg.norm(flat)) > 0.35:
            opts.append(("mass-flat", _unit(flat)))
    return opts


#: Tie-break between candidate axes that the path test cannot separate.  Tiny
#: next to the blocked/seated point counts, so it only decides genuine ties —
#: but it decides them toward the axes that know where the material is, which is
#: what fixes the merely *tilted* (as opposed to 90°-wrong) magnets.
_AXIS_PREFERENCE = {"mass-flat": 0.6, "mass": 0.3, "contact": 0.0}


def _choose_axis(seat: Seat, cen_a, cen_b, pa, pb, cp: ConnectionParams,
                 radius: float, length: float):
    """Pick the axis the joint can actually be assembled along.

    Every candidate is put through the same physical test: how much material
    would have to be cut out of the approach path, and how much body is left to
    seat the collar in.  An axis running *along* a DNA backbone rather than
    across the interface drives the socket lengthwise into the tube, so it is
    heavily blocked and loses — which is what makes this robust to the centroid
    sliding.  Candidates more than ``axis_agreement_min`` away from the plain
    contact line are rejected outright as wrap artefacts.
    """
    best = None
    for label, axis in _axis_options(seat, cen_a, cen_b, cp):
        agreement = float(np.dot(axis, seat.axis))
        if label != "contact" and agreement < cp.axis_agreement_min:
            continue
        blocked, seated = _path_census(pa, pb, seat.center, axis, radius, length)
        score = seated - cp.axis_blocked_weight * blocked + _AXIS_PREFERENCE[label]
        if best is None or score > best[0]:
            best = (score, label, axis, agreement, blocked)
    if best is None:                       # every candidate rejected
        blocked, _ = _path_census(pa, pb, seat.center, seat.axis, radius, length)
        return seat.axis, "contact", 1.0, blocked
    _score, label, axis, agreement, blocked = best
    return axis, label, agreement, blocked


def _local_mass(man, center, radius):
    """``(volume, centroid)`` of one part's material inside a ball at ``center``.

    This is the "how much meat is there" probe.  Only the component the contact
    actually sits on is used: a ball straddling an interface can also clip an
    unrelated lobe of the same chain, and averaging that in would drag the
    centroid sideways.  Returns ``(0.0, None)`` if the part has nothing here.
    """
    try:
        blob = _manifold.intersection(man, _manifold.sphere(center, radius))
        if blob.is_empty():
            return 0.0, None
        pieces = blob.decompose() or [blob]
        best, best_d = None, np.inf
        for piece in pieces:
            if piece.is_empty():
                continue
            mesh = _manifold.to_trimesh(piece)
            d = float(np.linalg.norm(np.asarray(mesh.center_mass, float) - center))
            if d < best_d:
                best, best_d = mesh, d
        if best is None:
            return 0.0, None
        return float(abs(best.volume)), np.asarray(best.center_mass, float)
    except Exception:
        return 0.0, None


def _candidate_seats(mesh_a, mesh_b, contact_thresh: float, socket_r: float,
                     want: int):
    """Stage 1 — cheap point-cloud shortlist of well-supported contact patches.

    "Just take the closest points" picks a surface spike: smallest gap, no
    material to seat anything in.  So each contact candidate is scored by how
    much *consistent* contact surrounds it (neighbours within a socket-sized
    radius whose contact direction agrees), which is high on a broad interface
    and low on an isolated whisker.  Returns the best well-separated candidates,
    more than are wanted, so stage 2 can rank them on real geometry.
    """
    pa = _probe_points(mesh_a)
    pb = _probe_points(mesh_b)
    d, idx = cKDTree(pb).query(pa, k=1)
    band = min(float(d.min()) + 1.5, contact_thresh)
    mask = d <= band
    if not mask.any():
        mask = np.zeros(len(d), bool)
        mask[int(np.argmin(d))] = True
    A, B, dd = pa[mask], pb[idx[mask]], d[mask]
    mids = 0.5 * (A + B)
    dirs = B - A
    dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9, None)

    reach = socket_r + 1.5                      # area a socket needs around it
    neigh = cKDTree(mids).query_ball_point(mids, reach)
    consistent, support = [], np.zeros(len(mids), int)
    for k in range(len(mids)):
        nb = np.asarray(neigh[k], dtype=int)
        cons = nb[dirs[nb] @ dirs[k] > 0.6]     # same-facing contact = real interface
        consistent.append(cons if len(cons) else np.array([k]))
        support[k] = len(cons)

    # Prefer meaty patches (high support); break ties toward the smaller gap.
    order = sorted(range(len(mids)), key=lambda k: (-support[k], dd[k]))
    picked, seats = [], []
    for k in order:
        if len(picked) >= max(1, want):
            break
        # Separated by a full socket diameter so two sockets never intersect.
        if any(np.linalg.norm(mids[k] - mids[c]) < 2.0 * socket_r + 1.0 for c in picked):
            continue
        picked.append(k)
        near = mids[consistent[k]]
        foot = int((np.linalg.norm(near - mids[k], axis=1) <= socket_r).sum())
        seats.append(Seat(center=mids[k], axis=_unit(dirs[k]), gap=float(dd[k]),
                          footprint=foot, patch=near))
    return seats, pa, pb


def _score_seats(seats: List[Seat], man_a, man_b, pa, pb, cp: ConnectionParams,
                 socket_r: float, need_depth: float) -> List[Seat]:
    """Stage 2 — re-orient and rank each candidate against the real solids.

    For every candidate we intersect both parts with a ball at the contact.  That
    one operation yields both things we need:

    * **the axis** — the line joining the two local centres of mass.  This is the
      direction in which each part actually *has* material, so a magnet laid
      along it sits square in the meat instead of following whatever tilt the
      single nearest-point pair happened to have.
    * **the score** — how much material is there at all, per side.

    The centroid line is not trusted blindly.  Where a protein wraps *around*
    curved DNA the probe ball reaches right around the duplex and the protein's
    local centre of mass lands on the far side, which would flip the magnet.  So
    the mass axis is accepted only when it agrees with the plain nearest-point
    line to within ``axis_agreement_min``; otherwise the nearest-point line is
    kept and the seat records ``axis_source="contact"``.

    Finally each seat is checked for *fill*: the fraction of the socket cylinder
    it needs that is already solid.  A seat on a spike scores near zero here and
    is dropped, because the collar would otherwise be built out of thin air.
    """
    probe_r = max(socket_r * cp.mass_probe_scale, socket_r + 1.0)
    scored: List[Seat] = []
    for seat in seats:
        vol_a, cen_a = _local_mass(man_a, seat.center, probe_r)
        vol_b, cen_b = _local_mass(man_b, seat.center, probe_r)
        if vol_a <= 0.0 or vol_b <= 0.0:
            continue

        axis, source, agreement, blocked = _choose_axis(
            seat, cen_a, cen_b, pa, pb, cp, socket_r + cp.path_clearance_mm,
            need_depth)
        seat.axis, seat.axis_source = axis, source
        seat.agreement, seat.blocked = agreement, blocked

        # Fill: how much of the plastic each side must supply is already there.
        # Measured from that side's own surface inward, not from the mid-plane —
        # the half-gap in between is air on every interface and would otherwise
        # make the score depend on the gap rather than on the material.
        fills = []
        for man, sign in ((man_a, -1.0), (man_b, +1.0)):
            into = seat.axis * sign
            start = seat.center + into * (seat.gap / 2.0)
            need = _seat_solid(start, into, need_depth, socket_r)
            want = _manifold.volume(need)
            try:
                have = _manifold.volume(_manifold.intersection(man, need))
            except Exception:
                have = 0.0
            fills.append(have / want if want > 1e-9 else 0.0)
        seat.fill = float(min(fills))

        # Rank on the weakest side's fill first — a joint is only as good as its
        # thinner half — then on contact footprint, then prefer a tight gap, and
        # finally shy away from seats that need a lot cut out of the path.
        seat.score = (seat.fill * 100.0
                      + min(seat.footprint, 40) * 0.5
                      - seat.gap * 2.0
                      - min(seat.blocked, 60) * 0.5)
        scored.append(seat)

    scored.sort(key=lambda s: -s.score)
    return scored


def _joint_seats(mesh_a, mesh_b, man_a, man_b, count: int,
                 cp: ConnectionParams, socket_r: float,
                 need_depth: float) -> List[Seat]:
    """The ranked seats to actually build, best first (shared by magnet+bridge)."""
    count = max(1, count)
    shortlist, pa, pb = _candidate_seats(mesh_a, mesh_b, cp.contact_threshold_mm,
                                         socket_r, count + cp.seat_shortlist_extra)
    ranked = _score_seats(shortlist, man_a, man_b, pa, pb, cp, socket_r, need_depth)
    # Prefer seats with real material behind them, but if none clears the bar
    # keep the ranked list anyway: ``_build_seat``'s watertight gate is the hard
    # limit, and refusing everything here would silently drop a joint the user
    # asked for on a genuinely thin (but printable) interface.
    good = [s for s in ranked if s.fill >= _MIN_SEAT_FILL]
    return (good or ranked)[:count]


def _build_seat(mans, i, j, seat: Seat, socket_r: float, embed: float,
                pocket: dict | None, socket_on: bool,
                clearance: float = 0.3) -> Tuple[bool, str]:
    """Build one joint at ``seat`` on both parts, or leave both untouched.

    This is the single geometry path behind both joint types, in three steps per
    part:

    1. **Clear the path.**  Anything of this part that reaches past the shared
       face, inside the joint footprint, is cut away.  Without this a lobe of the
       protein hanging over the socket simply collides with the other half and
       the parts never close — the joint looks right in the preview and does not
       assemble.  The cut is the footprint plus a sliding ``clearance``.
    2. **Raise the collar** — a flat-ended cylinder driven from the shared
       mid-plane into this part's own body, so the two halves meet on one clean
       disc instead of two ragged surfaces touching wherever they happen to.
    3. **Cut the bore**, for a magnet joint.

    The collar has to reach far enough back to fuse with the body; how far is not
    known in advance on a bumpy surface, so it is retried at increasing depths
    and ``_commit`` rejects any length that would leave it floating.  Both parts
    are committed together — a joint that only half-builds is worse than none, so
    on failure neither side is modified.
    """
    if not socket_on and pocket is None:
        return False, "socket disabled and no pocket to cut"

    for grow in (1.0, 1.6, 2.4):
        length = embed * grow
        # -1 drives into part A (the axis points A→B), +1 into part B.
        halves, ok = {}, True
        for idx, sign in ((i, -1.0), (j, +1.0)):
            man = mans[idx]
            into = seat.axis * sign
            # 1. Clear this part's material out of the other's approach path.
            #    Starts a hair past the face so the collar's own flat end (which
            #    lies exactly on it) is never shaved by this cut.
            path = _seat_solid(seat.center - into * 0.002, -into,
                               length + seat.gap + 1.0, socket_r + clearance)
            cleared = _commit(man, path, add=False)
            if cleared is None:
                # Cutting the overhang would sever the part — this seat is not
                # assemblable; the caller moves on to the next-ranked one.
                ok = False
                break
            man = cleared
            if socket_on:
                man = _commit(man, _seat_solid(seat.center, into, length, socket_r),
                              add=True)
                if man is None:
                    ok = False
                    break
            if pocket is not None:
                tool = _pocket_tool(seat.center, into, pocket["depth"],
                                    pocket["radius"], pocket["chamfer"],
                                    pocket["shape"])
                man = _commit(man, tool, add=False)
                if man is None:
                    ok = False
                    break
            halves[idx] = man
        if ok:
            mans[i], mans[j] = halves[i], halves[j]
            return True, ""
    return False, "would break watertightness or sever an overhang"


def _apply_magnet(mans, i, j, mesh_a, mesh_b, count: int, cp: ConnectionParams,
                  params: PrintParams, markers: list) -> Tuple[bool, str]:
    """Seat ``count`` press-fit magnet pockets on the best-scoring contacts.

    The pocket is cut oversize on purpose (see ``magnet_fit_clearance_mm`` /
    ``magnet_depth_clearance_mm``): an FDM hole printed to the magnet's nominal
    size comes out too small to accept it at all.  Positions, sizes and the
    resulting axis are recorded in ``markers`` so the preview can highlight them.
    """
    d, t, shape = cp.connector_diameter_mm, cp.magnet_thickness_mm, cp.magnet_shape
    pocket_r = d / 2.0 + cp.magnet_fit_clearance_mm / 2.0
    depth = t + cp.magnet_depth_clearance_mm
    socket_r = pocket_r + (cp.socket_wall_mm if cp.socket else 0.0)
    # The collar must bury the pocket and still have wall behind it.
    embed = depth + max(cp.socket_wall_mm, 1.0)

    # The lead-in eats grip: the top of the bore is oversized, so on a thin
    # magnet an unclamped chamfer would leave almost nothing holding it.
    chamfer = min(cp.magnet_chamfer_mm, 0.3 * t)

    seats = _joint_seats(mesh_a, mesh_b, mans[i], mans[j], count, cp,
                         socket_r, embed)
    pocket = {"radius": pocket_r, "depth": depth,
              "chamfer": chamfer, "shape": shape}

    placed, reasons = 0, []
    for seat in seats:
        if seat.footprint < _MIN_MAGNET_FOOTPRINT and seat.fill < _MIN_SEAT_FILL:
            reasons.append("too little contact area for a magnet")
            continue
        if not cp.socket and 2.0 * t <= seat.gap + 1e-6:
            # Without a collar to close the gap, the two magnets never meet.
            reasons.append(f"gap {seat.gap:.1f} mm ≥ 2×thickness {2*t:.1f} mm "
                           f"(turn the socket on to bridge it)")
            continue
        ok, why = _build_seat(mans, i, j, seat, socket_r, embed, pocket,
                              cp.socket, cp.path_clearance_mm)
        if not ok:
            reasons.append(why)
            continue
        placed += 1
        markers.append({
            "center": [float(x) for x in seat.center],
            "axis": [float(x) for x in seat.axis],
            "diameter": float(d), "thickness": float(t), "shape": shape.value,
            "socket_diameter": float(2.0 * socket_r) if cp.socket else None,
            "fill": round(seat.fill, 3), "axis_source": seat.axis_source,
            "blocked": int(seat.blocked),
        })

    return _joint_note(placed, len(seats), reasons, "magnet", seats)


def _apply_bridge(mans, i, j, mesh_a, mesh_b, count: int, cp: ConnectionParams,
                  params: PrintParams) -> Tuple[bool, str]:
    """Join two parts with clean flat-ended cylinders on the best contacts.

    Same seat selection and same collar as the magnet joint, minus the pocket —
    which is the point: the peg is now a true cylinder split on a shared plane,
    rather than the capsule-with-round-ends grown off the raw nearest-point pair
    that this used to be (that one landed at whatever angle the closest two
    vertices implied, and its hemispherical cap left a bobble on the surface).
    """
    r = _min_wall_radius(params, cp.connector_diameter_mm / 2.0)
    seats = _joint_seats(mesh_a, mesh_b, mans[i], mans[j], count, cp, r, r * 2.0)
    embed = max(2.0 * r, 2.0)
    placed, reasons = 0, []
    for seat in seats:
        # The peg must span the gap as well as bite into both bodies.
        ok, why = _build_seat(mans, i, j, seat, r, embed + seat.gap / 2.0,
                              None, True, cp.path_clearance_mm)
        if ok:
            placed += 1
        else:
            reasons.append(why)
    return _joint_note(placed, len(seats), reasons, "bridge", seats)


#: Axis labels worth surfacing in the UI.  "mass" is the expected case, so it is
#: left unsaid; the other two mean a fallback fired and are worth knowing about
#: if a magnet still looks wrong.
_AXIS_NOTE = {
    "contact": "axis from contact line",
    "mass-flat": "axis flattened along contact strip",
}


def _joint_note(placed: int, attempted: int, reasons, what: str, seats):
    """The (ok, human note) pair reported back to the UI for one interface."""
    used = {s.axis_source for s in seats[:max(placed, 1)]}
    extra = [_AXIS_NOTE[a] for a in sorted(used) if a in _AXIS_NOTE]
    if placed and not reasons:
        parts = ([f"{placed} {what}s"] if placed > 1 else []) + extra
        return True, "; ".join(parts)
    joined = "; ".join(sorted(set(reasons)) + extra)
    if placed:
        return True, f"placed {placed}/{attempted} — skipped: {joined}"
    return False, f"no {what} placed — " + (joined or "no contact")


def _rebuild_inflated(chain: Chain, params: PrintParams, amount_mm: float):
    """Rebuild one chain's mesh grown outward by ``amount_mm`` (the inflate join).

    "Inflate" is a small, uniform size increase applied at *build* time — exactly
    the surface-padding knob for a protein and the tube/base radii for DNA — so
    two neighbours swell until their surfaces overlap and weld, with no strut and
    no re-meshing artefact.  The growth is bounded by the contact threshold, so
    it stays small.
    """
    import dataclasses
    from . import geometry, meshops
    if chain.mtype == MoleculeType.PROTEIN:
        q = dataclasses.replace(
            params,
            surface_atom_padding_ang=params.surface_atom_padding_ang
            + amount_mm / params.scale_mm_per_angstrom,
        )
    else:
        q = dataclasses.replace(
            params,
            nucleic_radius_mm=params.nucleic_radius_mm + amount_mm,
            slab_thickness_mm=params.slab_thickness_mm + 2.0 * amount_mm,
            connector_radius_mm=params.connector_radius_mm + amount_mm,
            atom_radius_mm=params.atom_radius_mm + amount_mm,
            bond_radius_mm=params.bond_radius_mm + amount_mm,
        )
    mesh = geometry.generate_chain_mesh(chain, q)
    mesh = meshops.enforce_min_wall(mesh, q)
    return meshops.repair(mesh)


# --------------------------------------------------------------------------
# DNA interstrand base-pair connect
# --------------------------------------------------------------------------
def _pair_bases(cen_a: np.ndarray, cen_b: np.ndarray, max_dist: float):
    """Complementary base pairs between two strands (indices, distance).

    A duplex pairs bases in a fixed *register*: antiparallel means strand A
    residue ``i`` pairs strand B residue ``s - i`` for one constant ``s`` (a
    parallel register — e.g. when the file stores one strand reversed — is
    ``i + off``).  We evaluate every register, keep only pairs within
    ``max_dist``, and choose the register that pairs the most bases (shortest
    total distance breaks ties).  This is one-to-one and crossing-free by
    construction — it locks onto the true partner at the helix ends where a
    greedy nearest-neighbour swaps a base for its diagonal neighbour — and the
    cutoff naturally drops overhangs and unwound bubbles rather than mis-pairing
    them.
    """
    n_a, n_b = len(cen_a), len(cen_b)
    if n_a == 0 or n_b == 0:
        return []
    dmat = np.linalg.norm(cen_a[:, None, :] - cen_b[None, :, :], axis=-1)

    def register(index_of_b):
        pairs, total = [], 0.0
        for i in range(n_a):
            j = index_of_b(i)
            if 0 <= j < n_b and dmat[i, j] <= max_dist:
                pairs.append((i, j, float(dmat[i, j])))
                total += dmat[i, j]
        return pairs, total

    best, best_key = [], (-1, np.inf)
    # Antiparallel diagonals (j = s - i) and parallel diagonals (j = i + off).
    registers = ([(lambda i, s=s: s - i) for s in range(n_a + n_b - 1)]
                 + [(lambda i, o=o: i + o) for o in range(-(n_a - 1), n_b)])
    for idx_of_b in registers:
        pairs, total = register(idx_of_b)
        key = (len(pairs), -total)
        if key > best_key:
            best, best_key = pairs, key
    return best


def _apply_basepairs(man_a, man_b, chain_a: Chain, chain_b: Chain,
                     params: PrintParams):
    """Tie two DNA strands together at each base pair; returns (a, b, n_links)."""
    cp = params.connections
    cen_a = tube_slab.base_centroids_mm(chain_a, params)
    cen_b = tube_slab.base_centroids_mm(chain_b, params)
    max_dist = cp.basepair_max_dist_ang * params.scale_mm_per_angstrom
    pairs = _pair_bases(cen_a, cen_b, max_dist)
    if not pairs:
        return man_a, man_b, 0

    molecule = params.base_style == BaseStyle.MOLECULE
    if molecule:
        # Thin bond-like spokes between the paired bases.
        r = _min_wall_radius(params, params.bond_radius_mm)
    else:
        # Continue each rung at its own radius to the midline so opposing rungs
        # meet as one smooth bar (rod rungs are nucleic_radius*0.7 thick).
        r = _min_wall_radius(params, params.nucleic_radius_mm * 0.7)

    # The link runs *along the base-pair axis*, continuing each rung to the
    # middle (not a separate strut off the backbone).  It starts a little behind
    # the centroid — a back-step *onto* the existing rung — so it always fuses to
    # the strand body while still reading as the rung reaching the axis.
    back = max(0.6, r)
    fwd = 0.4
    n_done = 0
    for ia, ib, _d in pairs:
        ca, cb = cen_a[ia], cen_b[ib]
        mid = 0.5 * (ca + cb)
        u = _unit(cb - ca)                     # A-centroid → B-centroid (toward axis)
        sa = _commit(man_a, _manifold.capsule(ca - u * back, mid + u * fwd, r), add=True)
        sb = _commit(man_b, _manifold.capsule(cb + u * back, mid - u * fwd, r), add=True)
        if sa is not None and sb is not None:
            man_a, man_b = sa, sb
            n_done += 1
    return man_a, man_b, n_done


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _nucleic_strand_pairs(built) -> List[Tuple[int, int]]:
    """Pair up nucleic chains into duplexes by nearest-mesh proximity."""
    idx = [i for i, (c, _m) in enumerate(built) if c.mtype == MoleculeType.NUCLEIC]
    if len(idx) < 2:
        return []
    if len(idx) == 2:
        return [(idx[0], idx[1])]
    pairs, used = [], set()
    for a in range(len(idx)):
        if idx[a] in used:
            continue
        best, best_d = None, np.inf
        for b in range(len(idx)):
            if a == b or idx[b] in used:
                continue
            _, _, gap = _nearest(built[idx[a]][1], built[idx[b]][1])
            if gap < best_d:
                best, best_d = idx[b], gap
        if best is not None:
            pairs.append((idx[a], best))
            used.add(idx[a])
            used.add(best)
    return pairs


def apply(built: List[Tuple[Chain, "object"]], params: PrintParams):
    """Apply the connections pass; returns ``(new_built, [connection dicts])``.

    Input meshes must already be watertight (the pipeline gates them first).
    Output meshes are re-repaired (the fast-path preserves the already-good
    manifold result) and remain watertight single bodies.
    """
    from . import meshops
    cp = params.connections
    if not cp.enabled() or len(built) < 1:
        return built, [], []

    chains = [c for c, _m in built]
    meshes = [m for _c, m in built]
    applied: List[Connection] = []
    markers: list = []   # magnet positions for the preview highlight
    inflate = cp.connect and not cp.use_magnets \
        and cp.no_magnet_method == NoMagnetMethod.INFLATE

    # 1a) Detect chain-to-chain contacts (skipping DNA↔DNA when base-pairing).
    contacts = []  # (i, j, pa, pb, gap)
    if cp.connect:
        n = len(built)
        for i in range(n):
            for j in range(i + 1, n):
                if (cp.basepair_connect
                        and chains[i].mtype == MoleculeType.NUCLEIC
                        and chains[j].mtype == MoleculeType.NUCLEIC):
                    continue
                pa, pb, gap = _nearest(meshes[i], meshes[j])
                if gap <= cp.contact_threshold_mm:
                    contacts.append((i, j, pa, pb, gap))

    # 1b) Inflate rebuilds each contacting object slightly larger (before the
    #     manifold conversion) so neighbours swell until they overlap.
    if inflate and contacts:
        grow = [0.0] * len(built)
        for i, j, _pa, _pb, gap in contacts:
            # Half the gap on each side, plus a small weld; capped so a wide gap
            # can't balloon a thin chain (use bridge for those instead).  Kept
            # deliberately gentle — just enough to overlap.
            amt = min(max(0.0, gap) / 2.0 + 0.05, 1.0)
            grow[i] = max(grow[i], amt)
            grow[j] = max(grow[j], amt)
        for idx, amt in enumerate(grow):
            if amt > 0:
                try:
                    meshes[idx] = _rebuild_inflated(chains[idx], params, amt)
                except Exception:
                    pass
        for i, j, _pa, _pb, gap in contacts:
            applied.append(Connection(
                chains[i].chain_id, chains[j].chain_id,
                _kind(chains[i], chains[j]), "inflate", gap_mm=gap, applied=True))

    mans = [_manifold.from_trimesh(m) for m in meshes]

    # 1c) Magnet / bridge joins operate on the manifolds.
    if cp.connect and not inflate:
        for i, j, pa, pb, gap in contacts:
            kind = _kind(chains[i], chains[j])
            # Protein↔protein and DNA↔protein each get their own count; the
            # bridge reuses the same counts, since it is now the same joint
            # minus the magnet pocket.
            n_joints = cp.magnet_count if kind == "protein-protein" else cp.dna_magnet_count
            if cp.use_magnets:
                method = "magnet"
                ok, note = _apply_magnet(
                    mans, i, j, meshes[i], meshes[j], n_joints, cp, params, markers)
            else:
                method = "bridge"
                ok, note = _apply_bridge(
                    mans, i, j, meshes[i], meshes[j], n_joints, cp, params)
            applied.append(Connection(
                chains[i].chain_id, chains[j].chain_id, kind, method,
                gap_mm=gap, applied=ok, note=note))

    # 2) DNA interstrand base-pair connect.
    if cp.basepair_connect:
        for i, j in _nucleic_strand_pairs(built):
            mans[i], mans[j], n_links = _apply_basepairs(
                mans[i], mans[j], chains[i], chains[j], params)
            applied.append(Connection(
                chains[i].chain_id, chains[j].chain_id, "dna-basepair", "basepair",
                count=n_links, applied=n_links > 0,
                note=f"{n_links} base-pair link(s)" if n_links else "no base pairs found",
            ))

    # 3) Back to meshes; repair fast-path keeps the already-watertight results.
    new_built = []
    for (chain, old_mesh), man in zip(built, mans):
        mesh = _manifold.to_trimesh(man)
        mesh.metadata.update(old_mesh.metadata)
        mesh = meshops.repair(mesh)
        new_built.append((chain, mesh))
    return new_built, [c.as_dict() for c in applied], markers
