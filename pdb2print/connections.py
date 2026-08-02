"""Post-build fit + connector / joinery pass.

The per-chain meshes come out of the geometry core as separate watertight
solids, one colour each.  This pass *modifies those solids* — with the same
``manifold3d`` kernel the representations use — so that they **fit together**,
and so that chosen chains are joined for printing, while every object stays
watertight.  The 3MF export is unchanged: still individual coloured objects,
only now disjoint, touching, bulged together, or pocketed for magnets.

**Fit comes first.**  Every chain is meshed from its own atoms alone, so at a
binding interface both solids claim the same volume — right as a picture,
impossible as a set of parts.  :mod:`pdb2print.interference` carves them apart
before anything else runs (see that module for why the probe radius cannot do
this job, and why it should not be asked to).  It runs even with connectors
switched off: two objects that are simply printed and handed over still have to
fit.  It also runs first because the joint search depends on it — the volume
that had to be carved away is exactly where a magnet belongs, and until the
parts are disjoint the distances the search measures do not mean what they look
like.

Note that "watertight" no longer implies "one body".  A chain can legitimately
end up in several pieces — a protein loop that a DNA duplex genuinely threads
through has to come apart somewhere — so the booleans check that a step does not
*increase* the piece count rather than insisting on exactly one.

Two independent switches (see :class:`~pdb2print.config.ConnectionParams`):

* **connect** — join chains whose surfaces come within ``contact_threshold_mm``:
  - **magnets**: seat a press-fit magnet pocket in each side, so parts printed
    separately snap together;
  - **inflate**: grow the surfaces at the contact until they merge (organic,
    no visible strut) — the default.  The growth is placed on the protein
    wherever there is one, and a nucleic chain only ever grows its *backbone*:
    a base is a recognisable shape at a fixed scale, so padding it distorts it
    rather than enlarging it (see :func:`_inflate_growth`);
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
accepts the result only if it did not shatter the object — so a pocket that
would blow through a thin backbone is skipped (min-wall honoured) rather than
shipping a broken object.  The pipeline re-gates watertightness after the pass.

**Fit guarantee.** Exact where it can be: after the fit pass no two chains share
any volume.  Connectors are added afterwards and can reintroduce a little — a
closing sweep removes what it can without cutting a part in two, and
:func:`interference.audit` reports by name anything that survives, rather than
shipping parts that quietly will not close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import (
    PrintParams, ConnectionParams, NoMagnetMethod, MagnetShape,
    MoleculeType, BaseStyle, InterferenceRule, Representation,
)
from .chains import Chain
from .representations import _manifold, tube_slab
from . import interference


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
    if a.mtype == MoleculeType.LIGAND or b.mtype == MoleculeType.LIGAND:
        return "ligand"
    if a.mtype == MoleculeType.PROTEIN and b.mtype == MoleculeType.PROTEIN:
        return "protein-protein"
    if a.mtype == MoleculeType.NUCLEIC and b.mtype == MoleculeType.NUCLEIC:
        return "dna-dna"
    return "dna-protein"


def _is_ligand(chain: Chain) -> bool:
    return chain.mtype == MoleculeType.LIGAND


def _joinable(a: Chain, b: Chain) -> bool:
    """True if this pair should be offered a chain-to-chain joint at all.

    **Ligands never are.**  Not one of the three methods makes sense on one:

    * a *magnet* is bigger than the molecule.  A 4 mm disc against a drug that is
      12 Å across — 18 mm at the default scale, and only a few millimetres thick
      through the ring — means a pocket wider and deeper than the part it is cut
      into, so ``_commit`` rejects it and the joint is lost anyway;
    * a *bridge* or an *inflate* weld would fuse the ligand to its host, which
      destroys the only interesting thing about printing it separately: that it
      comes out and goes back in;
    * and none of it is needed.  The fit pass has already carved the host into an
      exact negative of the ligand, so the pocket grips it on every face at the
      print clearance.  Friction is the joint.
    """
    return not (_is_ligand(a) or _is_ligand(b))


# --------------------------------------------------------------------------
# Watertight-safe boolean commit
# --------------------------------------------------------------------------
def _components(man) -> int:
    try:
        return len(man.decompose())
    except Exception:
        return 1


def _commit(man, tool, add: bool):
    """Apply ``man ± tool`` and accept it only if the boolean did not shatter it.

    Returns the new manifold on success, or ``None`` if the boolean emptied the
    object or broke it into *more* pieces than it started with (e.g. a pocket
    that would blow through a thin wall).  This is what keeps every object
    watertight and honours minimum wall — a bad connector is skipped, never
    shipped.

    The test is "no more pieces than before", not "exactly one piece".  A chain
    can legitimately arrive here in several pieces — the interference pass may
    have had to cut a loop that a DNA duplex genuinely threads through — and an
    absolute one-body rule would then reject *every* subsequent boolean and
    silently drop all its connectors.
    """
    try:
        out = (man + tool) if add else _manifold.difference(man, tool)
    except Exception:
        return None
    if out.is_empty():
        return None
    # ``decompose`` is expensive on a full-size chain and this runs several times
    # per seat, so the input is only decomposed when the *result* is already in
    # more than one piece — a single-body result can never be a regression.
    after = _components(out)
    if after > 1 and after > _components(man):
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

#: Below this fraction of the collar being buried, the joint will visibly stand
#: off the surface and the user is told so.  Not a rejection: a proud joint still
#: prints and still holds, it just looks stuck on rather than built in, and on a
#: genuinely small contact patch there may be nowhere better to put it.
_MIN_SEAT_EMBEDDING = 0.5

#: Burial above which an exposed back cap means the collar came out the *far*
#: side of a thin part, rather than out of a surface that fell away under it.
#:
#: The two look identical to :func:`_cap_exposure` — the cap is in open air
#: either way — and they want opposite remedies: lengthen into the material
#: that is still there, or pull back out of the void beyond. This is the only
#: signal that separates them, and it is measured: on a 3 mm slab with a 3.7 mm
#: collar the burial reads 0.81 while the cap is fully exposed, where a
#: fallen-away surface reads well under 0.5.
#:
#: It decides *how* to close the back, never whether to. Suppressing the cone
#: outright was tried and left a flat disc showing, which is the thing the cone
#: exists to remove.
_PUNCH_THROUGH_EMBEDDING = 0.5


def _seat_solid(face, into, length, radius):
    """A flat-ended cylinder from the mating face ``length`` deep into a part.

    ``into`` is the unit vector pointing from the face into that part's body, so
    the flat end always lands exactly on the shared mid-plane.  Both parts build
    one of these against the same plane, which is what makes them meet flush.
    """
    return _manifold.frustum(face, face + into * length, radius, radius)

def _cap_exposure(blob, face, into, length, radius, skin: float = 0.2) -> float:
    """Fraction of the socket's flat back face standing in open air.

    Only the *back face* — the disc you look at edge-on when a cylinder is stuck
    to an uneven surface and the surface has fallen away under part of it.  The
    socket's side wall is a different thing and is deliberately not measured
    here: a socket emerging from a bumpy surface always shows some wall, and that
    reads as a socket, whereas a floating flat disc reads as a mistake.

    Measured on a thin slice at the very end rather than on the face itself,
    because a zero-thickness disc has no volume to compare; over a slice this
    thin the volume fraction and the area fraction are the same number.
    """
    if blob is None:
        return 1.0
    thick = min(skin, max(length * 0.25, 1e-3))
    disc = _seat_solid(face + into * max(length - thick, 0.0), into, thick, radius)
    want = _manifold.volume(disc)
    if want <= 1e-9:
        return 0.0
    try:
        have = _manifold.volume(_manifold.intersection(blob, disc))
    except Exception:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - have / want)))


def _close_the_back(blob, face, into, depth, radius, cp: ConnectionParams,
                    min_mult: float = 1.0, buried: float = 0.0):
    """``(length multiplier, needs a cone)`` for one side of a joint.

    Two remedies in the order they should be tried, and only when the back is
    actually exposed — a socket already sitting in solid material is left exactly
    as it was.

    1. **Carry the walls further down.**  Lengthening the socket keeps the mating
       face and the axis where they are and pushes only its back deeper, which on
       a surface that falls away is often enough to reach material again.  The
       shortest length that closes is taken, so a joint never grows more than it
       must, and the search is bounded by ``socket_extend_max``.
    2. **Pull the walls back in.**  An exposed cap on a collar that is otherwise
       well buried is the opposite problem: the socket came out the *far* side of
       a part thinner than itself, and lengthening drives it further out.  The
       search only ever grew, which is why a socket on a thin part punched
       through and then had a cone stacked on the spike.  Bounded by
       ``min_mult`` — the caller knows how short the collar may be and still
       hold its bore.
    3. **Cone the back.**  Where no length in either direction reaches anything,
       there is nothing to close onto.  The flat disc is replaced by a truncated
       cone instead — see :func:`_collar_solid`.

    Each probe is one boolean against the small local solid, not the chain, so
    the whole search costs a handful of cheap intersections.
    """
    limit = max(0.0, float(cp.socket_cap_exposed_max))
    if _cap_exposure(blob, face, into, depth, radius) <= limit:
        return 1.0, False

    def _longer():
        for mult in (1.25, 1.5, 1.75, 2.0):
            if mult - 1.0 > cp.socket_extend_max + 1e-9:
                return None
            if _cap_exposure(blob, face, into, depth * mult, radius) <= limit:
                return mult
        return None

    def _shorter():
        for mult in (0.9, 0.8, 0.7, 0.6, 0.5):
            if mult < min_mult - 1e-9:
                return None
            if _cap_exposure(blob, face, into, depth * mult, radius) <= limit:
                return mult
        return None

    # Order matters, and burial is what decides it. A well-buried collar with an
    # exposed cap has come out the far side of something thin, and driving it
    # further out is the opposite of a remedy — worse, a longer probe can find
    # an unrelated piece of the same chain across the void and report the cap
    # closed. Both are still tried; only the order changes.
    first, second = ((_shorter, _longer) if buried >= _PUNCH_THROUGH_EMBEDDING
                     else (_longer, _shorter))
    for attempt in (first, second):
        found = attempt()
        if found is not None:
            return found, False
    return 1.0, bool(cp.socket_back_taper)


def _collar_solid(face, into, length, radius, top_ratio: float = 0.0,
                  nose_height: float = 0.0):
    """The socket, with an optional cone built **onto** its flat back face.

    Strictly additive: the cylinder keeps its full length and full radius, and
    the cone is stacked on the far end.  Taking the taper *out* of the socket
    instead — chamfering the back edge — is wrong twice over.  It removes the
    material the joint is made of, and because the socket is only about a
    millimetre and a half longer than the magnet pocket is deep, a chamfer of any
    useful size starts biting before the bottom of the pocket: it thins the wall
    around the magnet and then undercuts it, so the magnet ends up standing proud
    of a socket that has been carved out from behind it.  Building on the back
    cannot do that, because nothing is subtracted.

    What it fixes is the flat disc you look at edge-on when a socket stands on a
    surface that has fallen away beneath it.  Capping that disc with a cone
    leaves only the cone's small flat top, whose area is ``top_ratio²`` of the
    original — 5% at the default, which is the same fraction that was allowed to
    show before any of this was triggered.
    """
    body = _seat_solid(face, into, length, radius)
    if top_ratio <= 0.0 or top_ratio >= 1.0 or nose_height <= 1e-6:
        return body
    back = face + into * length
    return _manifold.union([
        body,
        _manifold.frustum(back, back + into * nose_height,
                          radius, radius * top_ratio),
    ])


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
    embedding: float = 0.0   # fraction of the collar as built that is buried
    extend_a: float = 1.0    # length multiplier that closes A's back face
    extend_b: float = 1.0    # ...and B's
    taper_a: bool = False    # A's back could not be closed — cone it instead
    taper_b: bool = False    # ...and B's
    #: The local solids the back-face search reads, kept on the seat so that
    #: search can run late -- see :func:`_resolve_back_faces`.
    emb_a: object = None
    emb_b: object = None
    back_done: bool = False
    agreement: float = 1.0   # cos angle between chosen axis and nearest-point line
    blocked: int = 0         # surface points sitting in the assembly path
    axis_source: str = "contact"   # "overlap" | "mass" | "mass-flat" | "contact"
    edge_offset: float = 0.0 # lateral lopsidedness of the patch (mm); 0 = interior
    #: Material found around the seat, as a multiple of the collar's own volume.
    #: 1.0 means the probe ball found only as much plastic as the collar
    #: displaces — a strut, not a body.  See ``seat_depth_weight``.
    depth_ratio: float = 0.0
    #: How well this joint agrees with the line between the two chains as whole
    #: objects (1 = along it), and how far off that line it sits (mm).
    global_axis: float = 0.0
    global_offset: float = 0.0
    score: float = 0.0
    #: mm³ of interpenetration this seat was derived from (0 for point-cloud
    #: seats).  A joint that sits where the two parts *were* fighting for the
    #: same space is the natural one: there is guaranteed material on both sides
    #: and the interface normal is well determined.
    overlap_volume: float = 0.0


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


def _edge_offset(patch, center, axis) -> float:
    """Lateral lopsidedness of a contact patch around a seat centre (mm).

    Projects the patch into the plane perpendicular to ``axis`` and returns how
    far the patch's centroid sits from the seat centre *in that plane*.  A seat
    ringed by contact on every side returns ~0; a seat on the rim of an
    interface, whose support is all to one side, returns a value approaching the
    socket radius — the far side of its collar is overhanging open air.  This is
    the cheap "is the socket on the edge" probe; the score turns it into a
    preference for the interior spot that usually sits a little further in.
    """
    if patch is None or len(patch) < 3:
        return 0.0
    n = _unit(axis)
    rel = np.asarray(patch, float) - center
    perp = rel - np.outer(rel @ n, n)
    return float(np.linalg.norm(perp.mean(axis=0)))


def _recenter_into_patch(seat: "Seat", frac: float, max_shift: float) -> None:
    """Slide a seat toward the interior of its contact patch, in the mating plane.

    The physical companion to the edge penalty: where the whole shortlist sits on
    the rim (a narrow interface, so there is no better candidate to prefer), walk
    the socket inward instead.  The shift is ``frac`` of the measured rim offset,
    taken purely in the plane perpendicular to the axis — never along it, so the
    two flat faces still land on the shared mid-plane — and capped at
    ``max_shift`` so a badly lopsided patch cannot fling the seat off the contact.
    Mutates ``seat.center`` in place; a no-op when ``frac`` is zero.
    """
    if frac <= 0.0 or seat.patch is None or len(seat.patch) < 3:
        return
    n = _unit(seat.axis)
    rel = np.asarray(seat.patch, float) - seat.center
    perp = rel - np.outer(rel @ n, n)
    shift = perp.mean(axis=0) * float(frac)
    d = float(np.linalg.norm(shift))
    if d > max_shift > 0.0:
        shift = shift * (max_shift / d)
    seat.center = seat.center + shift


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


def _patch_normal(patch, center, radius: float, seed_axis):
    """Interface normal from a plane fit to the *local* contact patch, or ``None``.

    The radius bound is the whole trick, and it is why this is not the
    mating-plane refinement that was tried and reverted.  That one fitted the
    whole contact cloud: on a protein wrapped around a duplex it came out **32
    degrees** from the true local normal, because averaging around a curve is
    not averaging along a face.  Restricted to the patch the socket actually
    covers -- twice its radius, never more -- the same estimator measured
    **1.2-3.0 degrees** on that identical wrapped interface, because a wrapped
    surface *is* locally flat at that scale.  So the bound is not a tuning knob;
    widening it reintroduces the failure it exists to avoid.

    The planarity test here is only "is this a sheet at all", not
    "is this flat rather than wrapped" -- that second question was measured to
    be unanswerable (a rough flat patch and a 140-degree wrap both score ~0.67
    on any thin/mid ratio), and this does not need to answer it.

    Offered as a candidate and never snapped to: it still has to win the census
    and the embedding contest like every other axis, so a normal that cannot
    actually be assembled along loses.
    """
    if patch is None or len(patch) < 6:
        return None
    pts = np.asarray(patch, float)
    local = pts[np.linalg.norm(pts - center, axis=1) <= radius]
    if len(local) < 6:
        return None
    try:
        _u, s, vh = np.linalg.svd(local - local.mean(axis=0), full_matrices=False)
    except Exception:
        return None
    if s[1] <= 1e-9 or s[2] / s[1] > 0.55:
        return None                    # a blob, not a sheet — no normal to read
    normal = np.asarray(vh[2], float)
    if float(normal @ seed_axis) < 0.0:
        normal = -normal
    return _unit(normal)


def _axis_options(seat: Seat, cen_a, cen_b, cp: ConnectionParams,
                  socket_r: float = None, global_dir=None):
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
    * ``normal`` — the normal of a plane fitted to the contact patch within
      twice the socket radius.  The only one of the four that reads the
      *interface* rather than inferring it from where mass or vertices happen to
      be, and the most accurate where it applies (see :func:`_patch_normal`).
      Absent when the local patch is not a sheet.

    They are not ranked here — the caller measures each against the actual
    geometry and picks whichever can really be assembled.
    """
    opts = [("contact", seat.axis)]
    if global_dir is not None:
        # The line between the two chains as wholes. The coarsest hypothesis
        # available and the only one that knows what is being connected to what.
        opts.append(("global", global_dir))
    if socket_r is not None and socket_r > 0.0:
        normal = _patch_normal(seat.patch, seat.center, 2.0 * socket_r, seat.axis)
        if normal is not None:
            opts.append(("normal", normal))
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


#: Nudge toward the axes that know where the material is, on the same 0-100
#: scale as the normalised census below.
#:
#: These used to be 0.0-0.6 against a census spanning *hundreds*, which made
#: them three orders of magnitude too small to do anything but break exact ties
#: — and exact ties between point counts do not occur.  The docstring said
#: tie-break and meant it; the effect was zero.
_AXIS_PREFERENCE = {"normal": 6.0, "mass-flat": 4.0, "global": 3.5,
                    "mass": 3.0, "contact": 0.0}
#: (The census term runs 0-100 for a clean candidate but has no floor — a fully
#: obstructed one reaches -600 at the default blocked weight — so these are a
#: nudge against the top of the range, not against its whole span.)

#: How *clean* the joint axis from each source is, for seat ranking (not axis
#: selection).  An overlap lobe's thin axis and the local mass line are true
#: interface normals, so the disc sits square and the joint reads clean;
#: ``mass-flat`` is a good recovery but only fires on an awkward strip; the plain
#: ``contact`` line is the noisy fallback and a seat that can only reach its axis
#: that way is the one that ends up looking tilted.  Scaled by
#: ``axis_quality_weight`` so a nearby seat with a better-founded normal can win.
#: Deliberately *not* above ``mass``: the normal already earns its keep in the
#: axis contest through _AXIS_PREFERENCE, and stacking a seat-ranking bonus on
#: top of that would systematically promote seats on flat interfaces twice over
#: for one property.
_AXIS_QUALITY = {"overlap": 1.0, "normal": 0.9, "mass": 0.9, "global": 0.85,
                 "mass-flat": 0.7, "contact": 0.3}


def _choose_axis(seat: Seat, cen_a, cen_b, pa, pb, cp: ConnectionParams,
                 radius: float, length: float,
                 blob_a=None, blob_b=None, socket_r: float = None,
                 global_dir=None):
    """Pick the axis the joint can actually be assembled along.

    Every candidate is put through the same physical test: how much material
    would have to be cut out of the approach path, and how much body is left to
    seat the collar in.  An axis running *along* a DNA backbone rather than
    across the interface drives the socket lengthwise into the tube, so it is
    heavily blocked and loses — which is what makes this robust to the centroid
    sliding.  Candidates more than ``axis_agreement_min`` away from the plain
    contact line are rejected outright as wrap artefacts.

    The census alone cannot see the thing that actually looks wrong, though.  It
    counts *surface points* that are in the way or behind the face, which answers
    "can these come apart" but says nothing about how deeply the collar ends up
    buried — so an axis that leaves half the socket standing in open air scores
    just as well as one that sinks it into the body, as long as few points
    obstruct it.  Each candidate is therefore also measured for embedding
    (:func:`_embedding`) against the local solids, and the tilt that buries the
    joint wins.  With ``blob_a``/``blob_b`` omitted this term is simply absent and
    the behaviour is the older census-only one.
    """
    # Census first, for every candidate, so they can share one denominator.
    #
    # The division alone reorders nothing — a positive scalar shared by every
    # candidate cannot — and it is not meant to. What it does is put the census
    # in fixed units so the *other* two terms can be given weights that mean
    # something. Before, the census was a raw count whose spread between
    # candidates grew with the sample (measured on one fixture at 300 / 3,000 /
    # 10,000 probe points: 66 / 529 / 2,141) while the embedding term was
    # bounded by ``axis_embedding_weight`` however fine the mesh — so the term
    # added specifically to stop magnets tilting was worth 1-5% of the decision
    # and got less relevant the better the model. On a 0-100 census it is worth
    # a stated fraction of it, at any mesh density. The behaviour change comes
    # from that, and from _AXIS_PREFERENCE being rescaled to match.
    surveyed = []
    for label, axis in _axis_options(seat, cen_a, cen_b, cp, socket_r, global_dir):

        agreement = float(np.dot(axis, seat.axis))
        if label != "contact" and agreement < cp.axis_agreement_min:
            continue
        blocked, seated = _path_census(pa, pb, seat.center, axis, radius, length)
        surveyed.append((label, axis, agreement, blocked, seated))
    # Denominator from ``seated`` alone. Including ``blocked`` would let one
    # hopeless candidate — heavily obstructed, so a large blocked count — inflate
    # ref and compress every good candidate's census term against the fixed
    # preference and embedding terms.
    ref = max((s for _l, _a, _g, _b, s in surveyed), default=0) or 1

    best = None
    for label, axis, agreement, blocked, seated in surveyed:
        score = ((seated - cp.axis_blocked_weight * blocked) * 100.0 / ref
                 + _AXIS_PREFERENCE[label])
        if socket_r is not None and cp.axis_embedding_weight > 0.0:
            buried = min(_embedding(blob_a, seat.center, -axis, socket_r, length),
                         _embedding(blob_b, seat.center, axis, socket_r, length))
            score += cp.axis_embedding_weight * buried
        if best is None or score > best[0]:
            best = (score, label, axis, agreement, blocked)
    if best is None:                       # every candidate rejected
        blocked, _ = _path_census(pa, pb, seat.center, seat.axis, radius, length)
        return seat.axis, "contact", 1.0, blocked
    _score, label, axis, agreement, blocked = best
    return axis, label, agreement, blocked


def _local_solid(man, center, radius):
    """One part's material inside a ball at ``center``, or ``None``.

    Cut once per seat and then reused: every later question about this seat — how
    much material is here, where its centre of mass is, how deeply a collar on
    any candidate axis would bury itself — is answerable from this small solid,
    and asking them of the full chain instead would mean a full-size boolean
    apiece.  The ball must be big enough to contain whatever is measured against
    it, or material simply outside the ball reads as material that is not there.
    """
    try:
        blob = _manifold.intersection(man, _manifold.sphere(center, radius))
        return None if blob.is_empty() else blob
    except Exception:
        return None


def _embedding(blob, center, into, radius, length) -> float:
    """How much of the collar this side would build is already inside the part.

    1.0 means the socket is entirely buried in existing material and only the
    mating disc shows; 0.0 means it would be built out of thin air and stand
    proud of the surface.  This is the direct measure of the thing that looks
    wrong on a finished model — a magnet or its collar sticking out — and unlike
    ``fill`` it is taken on the collar *as built*: from the shared mid-plane,
    so the stub spanning the half-gap counts against it, because that stub is
    exactly the part with nothing behind it.

    ``blob`` must contain the collar; the caller sizes it to guarantee that, so
    intersecting against the blob gives the same answer as against the whole
    chain for a fraction of the cost.
    """
    if blob is None:
        return 0.0
    collar = _seat_solid(center, into, length, radius)
    want = _manifold.volume(collar)
    if want <= 1e-9:
        return 0.0
    try:
        have = _manifold.volume(_manifold.intersection(blob, collar))
    except Exception:
        return 0.0
    return float(max(0.0, min(1.0, have / want)))


def _local_mass(blob, center):
    """``(volume, centroid)`` of the lobe of ``blob`` the contact sits on.

    This is the "how much meat is there" probe.  Only the component the contact
    actually sits on is used: a ball straddling an interface can also clip an
    unrelated lobe of the same chain, and averaging that in would drag the
    centroid sideways.  Returns ``(0.0, None)`` if the part has nothing here.
    """
    if blob is None:
        return 0.0, None
    try:
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
    # The band that counts as "in contact", measured out from the closest
    # approach.  Proportional to the socket rather than a fixed 1.5 mm: at
    # scale 0.6 that constant is two and a half Ångström, so relief shorter than
    # a side chain was pooled into the contact patch with the flat background
    # around it.
    #
    # Widened from 0.4 to 1.2 socket radii, because the narrow band was the
    # first of three separate reasons this search wanted two *parallel* faces.
    # Where two surfaces meet at an angle, the region within a millimetre of
    # closest approach is a thin strip at the apex of the wedge — the thinnest
    # and worst place to put a joint — and everywhere with real material behind
    # it was excluded before it could be considered. A magnet does not need
    # parallel faces: the socket cuts its own flat mating disc into each side,
    # so what matters is depth along the axis, not how the surfaces happen to
    # lie.
    band = min(float(d.min()) + 1.2 * socket_r, contact_thresh)
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
        # Rank on how much contact is nearby, not on how much of it is
        # *parallel*. The direction-agreement filter is still what defines the
        # patch a seat is measured on and where its seed axis comes from — it is
        # a good answer to "which surface am I on" — but it was also the ranking
        # key, and as a ranking key it is a bias: a broad flat interface scores
        # every neighbour, while a curved or angled one scores a fraction of
        # them and loses, however much material it has. That is the second of
        # the three reasons this search preferred parallel faces.
        support[k] = len(nb)

    # Rank on the *smoothed* support, not the raw count.
    #
    # ``support`` is a neighbour count over a random 3000-point subsample, so
    # every value is a noisy estimate, and taking the best of a thousand noisy
    # estimates lands wherever the sampling happened to be kindest rather than
    # where the geometry is best.  Measured on a fixed fixture: changing only
    # the subsample seed moved the chosen seat over an 18 mm range.  Averaging
    # each value over its neighbourhood — the same lists already built above, so
    # this costs about 2 ms — halved the error against a known optimum (5.55 mm
    # to 2.75 mm mean, 12.9 mm to 5.7 mm worst).
    #
    # Note the seed is fixed, so this was never run-to-run randomness. It is
    # worse than that: the draw depends on the vertex count, so *any* change —
    # a nudged scale, one more atom — reshuffles it and can move a joint 18 mm.
    smoothed = np.array([support[np.asarray(neigh[k], dtype=int)].mean()
                         for k in range(len(mids))])
    order = sorted(range(len(mids)), key=lambda k: (-smoothed[k], dd[k]))
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
        # Seed axis averaged over the consistent neighbourhood rather than read
        # off the single nearest vertex pair.  That pair is quantised by vertex
        # spacing across the gap, and on two *flat* facing blocks it measured
        # 7-50 degrees off true depending on mesh density.  It matters out of
        # proportion to its job: every other candidate axis is vetoed for
        # disagreeing with this one, so a 50-degree seed vetoed the correct
        # answer and the recovery was to keep the bad seed.  Averaging is free —
        # the neighbourhood is already computed — and measured 27.5 to 9.5
        # degrees on the flat case, 5.9 to 0.6 on a ball resting on a plate.
        seats.append(Seat(center=mids[k],
                          axis=_unit(dirs[consistent[k]].mean(axis=0)),
                          gap=float(dd[k]), footprint=foot, patch=near))
    return seats, pa, pb


def _overlap_seats(overlap, mesh_a, mesh_b, socket_r: float) -> List[Seat]:
    """Seats derived from where the two parts *were* interpenetrating.

    These are the joint positions the old point-cloud search could never find,
    and the reason is worth stating: it ranked candidates on an **unsigned**
    nearest-vertex distance.  For a vertex of A buried deep inside B the nearest
    vertex *of B* is out on B's surface, so that distance equals the penetration
    depth — a millimetre or two — and the search read the deepest, meatiest part
    of the interface as "far away".  The smallest distance instead landed on the
    rim where the two surfaces cross, which is the thinnest part of the joint.
    Worse, the contact direction ``B - A`` reverses across that rim, so the seed
    axis could come out tilted or inverted — and since every other candidate
    axis is vetoed for disagreeing with it, one bad seed forced the fallback and
    produced exactly the tilted magnet.

    The interference pass has already carved these regions apart, so here we
    take the two things each source is actually good for:

    * **position** from the local contact between the *carved* solids, which now
      genuinely touch, so the seat lands on the real mating face;
    * **direction** from the overlap lobe's thin principal axis.  An
      interference lobe is a lens — broad across the interface, thin through it
      — so its smallest principal direction *is* the interface normal, averaged
      over the whole patch instead of read off one vertex pair.
    """
    if overlap is None or not overlap.pieces:
        return []
    pa = _probe_points(mesh_a)
    pb = _probe_points(mesh_b)
    tree_a, tree_b = cKDTree(pa), cKDTree(pb)
    reach = socket_r + 2.5

    seats: List[Seat] = []
    for piece in overlap.pieces:
        ia = np.asarray(tree_a.query_ball_point(piece.center, reach), dtype=int)
        ib = np.asarray(tree_b.query_ball_point(piece.center, reach), dtype=int)
        if len(ia) == 0 or len(ib) == 0:
            continue
        local_a, local_b = pa[ia], pb[ib]

        # Position: closest approach of the two carved surfaces near this lobe.
        dist, near = cKDTree(local_b).query(local_a, k=1)
        k = int(np.argmin(dist))
        point_a, point_b = local_a[k], local_b[near[k]]
        center = 0.5 * (point_a + point_b)
        gap = float(dist[k])

        # Direction: the lobe's thin axis, signed from A's material into B's.
        axis = piece.normal
        if axis is None:
            axis = _unit(point_b - point_a)
        else:
            lead = local_b.mean(axis=0) - local_a.mean(axis=0)
            sign = float(np.dot(axis, lead))
            if abs(sign) < 1e-9:
                sign = float(np.dot(axis, point_b - point_a))
            axis = _unit(axis * (1.0 if sign >= 0.0 else -1.0))

        patch = np.vstack([local_a, local_b])
        foot = int((np.linalg.norm(local_a - center, axis=1) <= socket_r).sum())
        seats.append(Seat(center=center, axis=axis, gap=gap, footprint=foot,
                          patch=patch, axis_source="overlap",
                          overlap_volume=float(piece.volume)))

    # Two sockets must not intersect, so thin the list to well-separated lobes,
    # biggest first — lobe volume is a direct measure of how much material the
    # joint has to work with.
    seats.sort(key=lambda s: -s.overlap_volume)
    kept: List[Seat] = []
    for seat in seats:
        if any(np.linalg.norm(seat.center - k.center) < 2.0 * socket_r + 1.0
               for k in kept):
            continue
        kept.append(seat)
    return kept


def _score_seats(seats: List[Seat], man_a, man_b, pa, pb, cp: ConnectionParams,
                 socket_r: float, need_depth: float, com_a=None, com_b=None) -> List[Seat]:
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
    # The embedding test intersects the collar with the local solid, so that
    # solid has to *contain* the collar — anything outside the ball would read as
    # missing material and score a perfectly buried socket as proud.  With the
    # stock proportions the mass probe is already wide enough and the same solid
    # serves both; a deep magnet in a thin wall is the case where it is not, and
    # then a second, wider cut is taken for the embedding only.  The mass probe
    # itself is deliberately *not* widened: it is kept modest so that a protein
    # wrapping a duplex cannot drag the local centre of mass around it.
    # Sized for the *longest* socket the back-face search may ask for, so a
    # lengthened socket is still measured against material rather than against
    # the edge of the ball it is being tested in.
    # A divisor since the score terms became scale-relative, and it arrives
    # from a form field: connector_diameter 0 with no clearance and no socket
    # wall lands here as 0.0. Clamped rather than validated because a zero-width
    # connector is not a request worth honouring either way.
    socket_r = max(float(socket_r), 1e-3)
    # The two chains as whole objects: the direction one has to travel to leave
    # the other, and the line joining them. Everything else here is measured
    # within a few millimetres of a contact point and cannot see either.
    global_dir, com_mid, com_span = None, None, 0.0
    if com_a is not None and com_b is not None:
        delta = np.asarray(com_b, float) - np.asarray(com_a, float)
        span = float(np.linalg.norm(delta))
        if span > 1e-6:
            global_dir = delta / span
            com_mid = 0.5 * (np.asarray(com_a, float) + np.asarray(com_b, float))
            com_span = span
    collar_reach = float(np.hypot(need_depth * (1.0 + cp.socket_extend_max),
                                  socket_r)) + 0.25
    scored: List[Seat] = []
    for seat in seats:
        # Optionally walk the seat off the rim and into the interior of its
        # contact patch *before* it is measured, so the axis, fill and edge
        # offset are all read at the position the joint will actually be built
        # at.  Off (frac 0) unless the user turns it on.
        _recenter_into_patch(seat, cp.seat_recenter_frac, socket_r)

        blob_a = _local_solid(man_a, seat.center, probe_r)
        blob_b = _local_solid(man_b, seat.center, probe_r)
        vol_a, cen_a = _local_mass(blob_a, seat.center)
        vol_b, cen_b = _local_mass(blob_b, seat.center)
        if vol_a <= 0.0 or vol_b <= 0.0:
            continue
        # How much plastic is actually here, on the thinner side, as a multiple
        # of what the collar displaces.  These two volumes were already being
        # measured and then used only for the "> 0" test above.
        collar_volume = float(np.pi * socket_r * socket_r * need_depth)
        seat.depth_ratio = (float(min(vol_a, vol_b) / collar_volume)
                            if collar_volume > 1e-9 else 0.0)
        if collar_reach <= probe_r:
            emb_a, emb_b = blob_a, blob_b
        else:
            emb_a = _local_solid(man_a, seat.center, collar_reach)
            emb_b = _local_solid(man_b, seat.center, collar_reach)

        if seat.axis_source == "overlap":
            # The lobe's thin axis is measured over the whole interference patch
            # and is already the interface normal, so it is not put through the
            # candidate search — and above all it is not judged against the
            # nearest-point line, which is the noisy quantity the search exists
            # to escape.  Only the path census is still wanted, for the score.
            seat.blocked, _seated = _path_census(
                pa, pb, seat.center, seat.axis,
                socket_r + cp.path_clearance_mm, need_depth)
            seat.agreement = 1.0
        else:
            axis, source, agreement, blocked = _choose_axis(
                seat, cen_a, cen_b, pa, pb, cp, socket_r + cp.path_clearance_mm,
                need_depth, emb_a, emb_b, socket_r, global_dir)
            seat.axis, seat.axis_source = axis, source
            seat.agreement, seat.blocked = agreement, blocked

        # How buried the collar ends up on its worse side, on the axis finally
        # chosen — a joint is only as hidden as its more exposed half.
        seat.embedding = min(
            _embedding(emb_a, seat.center, -seat.axis, socket_r, need_depth),
            _embedding(emb_b, seat.center, seat.axis, socket_r, need_depth))

        # Whether each side's flat back face is left hanging is decided from
        # these small local solids -- ``_build_seat`` only has the full chains,
        # where the same probes would each be a full-size boolean -- so they are
        # kept on the seat.  The measurement itself is deferred: see
        # ``_resolve_back_faces``.
        seat.emb_a, seat.emb_b = emb_a, emb_b

        # Fill: how much of the plastic each side must supply is already there.
        # Measured from that side's own surface inward, not from the mid-plane —
        # the half-gap in between is air on every interface and would otherwise
        # make the score depend on the gap rather than on the material.
        # Measured against the local blob where the blob provably contains the
        # cylinder, and against the full chain otherwise.  ``_local_solid``'s
        # whole reason for existing is that every question about a seat is
        # answerable from a small solid; this was the one question still asking
        # the whole chain, at 5.4 ms a call against 1.7.  The guard is not
        # optional: outside the ball, material simply beyond it would read as
        # material that is not there, and the seat would rank low for the wrong
        # reason.
        emb_r = probe_r if collar_reach <= probe_r else collar_reach
        far = float(np.hypot(seat.gap / 2.0 + need_depth, socket_r))
        fills = []
        for man, blob, sign in ((man_a, emb_a, -1.0), (man_b, emb_b, +1.0)):
            into = seat.axis * sign
            start = seat.center + into * (seat.gap / 2.0)
            need = _seat_solid(start, into, need_depth, socket_r)
            want = _manifold.volume(need)
            against = blob if (blob is not None and far <= emb_r) else man
            try:
                have = _manifold.volume(_manifold.intersection(against, need))
            except Exception:
                have = 0.0
            fills.append(have / want if want > 1e-9 else 0.0)
        seat.fill = float(min(fills))

        # How far off the interface's rim this seat sits, measured against the
        # axis just chosen.  Interior seats score ~0; a seat whose support is all
        # to one side (collar overhanging open air) scores up to the socket
        # radius and is penalised for it below.
        seat.edge_offset = _edge_offset(seat.patch, seat.center, seat.axis)

        # Rank on the weakest side's fill first — a joint is only as good as its
        # thinner half — then on contact footprint, then prefer a tight gap, and
        # shy away from seats that need a lot cut out of the path.  A seat
        # recovered from an interference lobe gets a bounded bonus: the two parts
        # were competing for that volume, so material on both sides is guaranteed
        # and the normal is well determined (bounded, so a huge lobe cannot
        # outrank a seat that is simply better).  Two cosmetic-but-real terms sit
        # under those: reward a well-founded joint axis (a cleaner-looking disc),
        # and penalise a seat that sits on the edge of the interface (a socket
        # that sticks out) so a tidier interior spot wins when one exists.
        # Three of these terms used to be absolute millimetres or cubic
        # millimetres inside a score whose leading term spans a hundred, which
        # meant they quietly changed weight with the model's size.  The worst
        # was the overlap bonus: measured in mm3, it decays as scale cubed, so a
        # lobe worth 30 points at scale 1.5 is worth 1.1 at 0.5 and 0.14 at
        # 0.25.  Overlap seats are the *good* ones — they carry a real interface
        # normal and skip the axis search entirely — and small scale is exactly
        # where they stopped being preferred.  All three are ratios now, tuned
        # so the stock socket reproduces roughly what it did before.
        edge_span = 0.6 * socket_r
        seat.score = (seat.fill * 100.0
                      + min(seat.footprint, 40) * 0.5
                      - (seat.gap / socket_r) * 8.0
                      - min(seat.blocked, 60) * 0.5
                      + min(1.0, seat.overlap_volume / collar_volume) * 30.0
                      + cp.axis_quality_weight * _AXIS_QUALITY.get(seat.axis_source, 0.3)
                      # Clamped at 0.6 x the radius, not the full radius: past
                      # that the far side of the collar is entirely over air and
                      # further offset is not meaningfully worse.  Divided back
                      # out so the maximum penalty is what it always was.
                      - cp.edge_center_weight * min(seat.edge_offset, edge_span) / 0.6
                      + cp.seat_embedding_weight * seat.embedding
                      + cp.seat_depth_weight * min(1.0, seat.depth_ratio / 3.0))
        if global_dir is not None:
            # Does this joint pull the two objects apart, and does it sit where
            # they actually meet? Both bounded to 0..1 so they refine a choice
            # between comparable seats rather than overriding "is there material
            # here", which is still what the leading term measures.
            seat.global_axis = abs(float(np.dot(seat.axis, global_dir)))
            off = seat.center - com_mid
            lateral = float(np.linalg.norm(off - global_dir * float(off @ global_dir)))
            seat.global_offset = lateral
            near = max(0.0, 1.0 - lateral / max(0.25 * com_span, socket_r))
            seat.score += (cp.global_axis_weight * seat.global_axis
                           + cp.global_line_weight * near)
        scored.append(seat)

    scored.sort(key=lambda s: -s.score)
    return scored


def _resolve_back_faces(seat: Seat, need_depth: float, socket_r: float,
                        cp: ConnectionParams, min_mult: float = 1.0) -> None:
    """Work out ``extend_*`` / ``taper_*`` for a seat that is about to be built.

    Deferred out of :func:`_score_seats` on purpose.  None of these four values
    appears anywhere in ``seat.score`` -- only :func:`_build_seat` reads them --
    and the shortlist scores about eight seats to build one, so seven sets of
    them were measured and thrown away.  Each set is up to ten
    ``_manifold.volume`` calls on an intersection, which is where the ~440
    volume calls per build were coming from.

    Must be called with the same ``need_depth`` and ``socket_r`` that were
    passed to :func:`_joint_seats`, not the ones :func:`_build_seat` is given:
    the result is a *multiple* of the depth it was measured at, which is how it
    carries over to the bridge, which sizes its socket differently.
    """
    if seat.back_done:
        return
    seat.back_done = True
    if seat.emb_a is None or seat.emb_b is None:
        return                        # nothing to measure; keep the defaults
    seat.extend_a, seat.taper_a = _close_the_back(
        seat.emb_a, seat.center, -seat.axis, need_depth, socket_r, cp, min_mult,
        seat.embedding)
    seat.extend_b, seat.taper_b = _close_the_back(
        seat.emb_b, seat.center, seat.axis, need_depth, socket_r, cp, min_mult,
        seat.embedding)


def _object_centre(mesh):
    """The whole object's centre — volume if it is closed, vertices if not."""
    try:
        if mesh.is_watertight and abs(mesh.volume) > 1e-9:
            return np.asarray(mesh.center_mass, float)
    except Exception:
        pass
    try:
        return np.asarray(mesh.vertices, float).mean(axis=0)
    except Exception:
        return None


def _joint_seats(mesh_a, mesh_b, man_a, man_b, count: int,
                 cp: ConnectionParams, socket_r: float,
                 need_depth: float, overlap=None) -> List[Seat]:
    """The ranked seats to actually build, best first (shared by magnet+bridge).

    Candidates come from two sources and are scored together on the same
    footing: the interference lobes this pair had before they were carved apart
    (``overlap``), and the plain contact patches of the point-cloud search.  The
    lobes are the natural joint positions and normally win, but they are not
    forced through — a pair that merely touches has no lobes at all, and a lobe
    on a spike still has to survive the fill test like anything else.
    """
    count = max(1, count)
    shortlist, pa, pb = _candidate_seats(mesh_a, mesh_b, cp.contact_threshold_mm,
                                         socket_r, count + cp.seat_shortlist_extra)
    from_overlap = _overlap_seats(overlap, mesh_a, mesh_b, socket_r)
    if from_overlap:
        # Drop point-cloud candidates that would collide with a lobe seat; the
        # lobe is the better-founded of the two in the same place.
        shortlist = [s for s in shortlist
                     if all(np.linalg.norm(s.center - o.center) >= 2.0 * socket_r + 1.0
                            for o in from_overlap)]
        shortlist = from_overlap[:count + cp.seat_shortlist_extra] + shortlist
    ranked = _score_seats(shortlist, man_a, man_b, pa, pb, cp, socket_r, need_depth,
                          _object_centre(mesh_a), _object_centre(mesh_b))
    # Prefer seats with real material behind them, but if none clears the bar
    # keep the ranked list anyway: ``_build_seat``'s watertight gate is the hard
    # limit, and refusing everything here would silently drop a joint the user
    # asked for on a genuinely thin (but printable) interface.
    good = [s for s in ranked if s.fill >= _MIN_SEAT_FILL]
    return (good or ranked)[:count]


def _build_seat(mans, i, j, seat: Seat, socket_r: float, embed: float,
                pocket: dict | None, socket_on: bool,
                clearance: float = 0.3, cap_limit: float = 0.0,
                extend_max: float = 0.0) -> Tuple[bool, str]:
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

    # Per side: how far the socket has to run to bury its own back face, and
    # whether that back has to be coned instead.  Both were measured while the
    # seat was scored; a flat top of this radius carries ``cap_limit`` of the
    # disc's area, which is the same fraction that was allowed to show anyway.
    reach = {i: seat.extend_a, j: seat.extend_b}
    cone = {i: seat.taper_a, j: seat.taper_b}
    top_ratio = float(np.sqrt(cap_limit)) if cap_limit > 0.0 else 0.0
    # A 45° cone is the natural shape and needs no arbitrary constant — its
    # height is just how far the radius has to come in.
    nose_height = (min(socket_r * (1.0 - top_ratio), embed * extend_max)
                   if top_ratio > 0.0 else 0.0)
    # How short the *cylindrical* part may be and still have a wall behind the
    # magnet, for the containment case below.
    min_body = (float(pocket["depth"]) + 0.6) if pocket else embed * 0.5

    for grow in (1.0, 1.6, 2.4):
        # -1 drives into part A (the axis points A→B), +1 into part B.
        halves, ok = {}, True
        for idx, sign in ((i, -1.0), (j, +1.0)):
            # ``grow`` escalates a collar that would not commit. A side whose
            # reach was pulled *in* is not asking to be escalated -- it was
            # shortened because there is nothing further out to reach -- and
            # multiplying the two drove it back out past where it started: a
            # side pulled to 0.8 came back at grow 1.6 as 1.28x, 28% longer than
            # if the shortening had never happened.
            length = embed * (grow * reach[idx] if reach[idx] >= 1.0
                              else reach[idx])
            man = mans[idx]
            into = seat.axis * sign
            # 1. Clear this part's material out of the other's approach path.
            #    Starts a hair past the face so the collar's own flat end (which
            #    lies exactly on it) is never shaved by this cut.  Sized from the
            #    nominal depth, not the lengthened one: how far this part's
            #    overhang has to be cut back is a fact about the *other* part's
            #    approach and has nothing to do with how deep our own socket
            #    happens to run.
            path = _seat_solid(seat.center - into * 0.002, -into,
                               embed * grow + seat.gap + 1.0, socket_r + clearance)
            cleared = _commit(man, path, add=False)
            if cleared is None:
                # Cutting the overhang would sever the part — this seat is not
                # assemblable; the caller moves on to the next-ranked one.
                ok = False
                break
            man = cleared
            if socket_on:
                # Cone the back wherever the search could not close it, and
                # never at the cost of the joint: if the coned socket will not
                # commit, the plain one is tried before the seat is abandoned, so
                # the worst case is exactly the old geometry.
                #
                # The cone is fitted *within* ``length`` rather than stacked on
                # the end of it, so the collar's overall reach is unchanged and
                # a coned socket can never stick out further than a plain one.
                # Stacking is what made a punched-through socket worse: measured
                # on a 3 mm part with a 3.7 mm collar, 28 mm3 already stood past
                # the back face and the taper took that to 76 mm3, 2.8 mm of
                # extra nose. Contained, the tip lands exactly where the flat cap
                # would have been — same envelope, no flat disc. The trade is a
                # slightly shorter full-radius body, floored at ``min_body``.
                nose = top_ratio if cone[idx] else 0.0
                grown = None
                if nose > 0.0:
                    # Two different situations produce an exposed back cap, and
                    # they want opposite treatment.
                    #
                    # The surface fell away under the socket: there is nothing
                    # behind the cap but air, so the cone is stacked on the end
                    # exactly as it always was — full 45 degrees, full wall
                    # behind the bore, and the taper simply replaces a flat disc
                    # that was hanging in space anyway.
                    #
                    # The collar came out the far side of a part thinner than
                    # itself: stacking then adds a visible spike on the back of
                    # the model. Measured on a 3 mm part with a 3.7 mm collar,
                    # 28 mm3 already stood past the back face and the taper took
                    # it to 76 mm3. There the cone is fitted *inside* the
                    # collar's own length instead, so the tip lands where the
                    # flat cap would have been and the envelope is unchanged.
                    # Blunter, because there is only so much room in front of
                    # the bore — but a blunt taper beats a flat disc, and it
                    # beats a spike.
                    body, nose_h = length, nose_height
                    if seat.embedding >= _PUNCH_THROUGH_EMBEDDING:
                        nose_h = max(0.0, min(nose_height, length - min_body))
                        body = length - nose_h
                    if nose_h > 1e-6:
                        grown = _commit(
                            man, _collar_solid(seat.center, into, body,
                                               socket_r, nose, nose_h),
                            add=True)
                if grown is None:
                    grown = _commit(man, _seat_solid(seat.center, into, length,
                                                     socket_r), add=True)
                man = grown
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

#: Stock neodymium disc diameters, largest first. Used only to suggest a size
#: that would fit; nothing here changes a setting on the user's behalf.
_STOCK_MAGNET_MM = (8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.5)

#: A socket wider than this fraction of a part's *narrowest* dimension is
#: reported. A quarter is generous — at that point the joint is a visible
#: feature of the model rather than a detail on it.
_SOCKET_SPAN_WARN = 0.25


def _socket_scale_note(built, cp: ConnectionParams) -> str:
    """One line when the magnet is simply too big for the model at this scale.

    Nothing anywhere related the connector diameter to
    ``scale_mm_per_angstrom``: the scale slider spans 0.2 to 6.0, a thirty-fold
    range, while the socket stays a fixed 7.2 mm across by 3.7 mm deep at stock
    settings.  At scale 0.4 a 60 Angstrom domain prints 24 mm wide and the
    socket is thirty percent of it; at 0.25 it is nearly half.  Every symptom of
    a badly placed joint gets worse together at small scale, and this is the
    reason they do — so it is worth saying plainly rather than leaving the user
    to infer it from a joint that looks wrong.

    Measured against each part's *narrowest* bounding-box dimension, because
    that is the one the socket has to fit inside, and against the smallest part,
    because the joint is only as good as its thinner half.

    Advisory only. The joint search may well still find somewhere good, and
    refusing to build would be worse than building something the user can look
    at and judge.
    """
    if not cp.use_magnets or not cp.connect:
        return ""
    socket_d = cp.connector_diameter_mm + cp.magnet_fit_clearance_mm + \
        (2.0 * cp.socket_wall_mm if cp.socket else 0.0)
    narrowest, where = None, ""
    for chain, mesh in built:
        try:
            span = float(np.min(mesh.bounds[1] - mesh.bounds[0]))
        except Exception:
            continue
        if span <= 0.0:
            continue
        if narrowest is None or span < narrowest:
            narrowest, where = span, chain.display_name()
    if narrowest is None or socket_d <= _SOCKET_SPAN_WARN * narrowest:
        return ""

    budget = _SOCKET_SPAN_WARN * narrowest
    fits = [d for d in _STOCK_MAGNET_MM
            if d + cp.magnet_fit_clearance_mm
            + (2.0 * cp.socket_wall_mm if cp.socket else 0.0) <= budget]
    advice = (f"a {fits[0]:g} mm magnet would fit" if fits
              else "no stock magnet is small enough — raise the scale, or use "
                   "printed pins instead")
    return (f"The magnet is large for this model: its socket is {socket_d:.1f} mm "
            f"across and the narrowest part ({where}) is only {narrowest:.1f} mm "
            f"thick, so the joint is a feature of the model rather than a detail "
            f"on it — {advice}.")


def _apply_magnet(mans, i, j, mesh_a, mesh_b, count: int, cp: ConnectionParams,
                  params: PrintParams, markers: list,
                  overlap=None) -> Tuple[bool, str]:
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
                         socket_r, embed, overlap=overlap)
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
        # How short the collar may go and still have a wall behind the bore.
        _resolve_back_faces(seat, embed, socket_r, cp,
                            min_mult=min(1.0, (depth + 0.6) / max(embed, 1e-6)))
        ok, why = _build_seat(mans, i, j, seat, socket_r, embed, pocket,
                              cp.socket, cp.path_clearance_mm,
                              cp.socket_cap_exposed_max, cp.socket_extend_max)
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
            "overlap_mm3": round(seat.overlap_volume, 2),
            "edge_offset": round(seat.edge_offset, 2),
            "embedding": round(seat.embedding, 3),
            "depth_ratio": round(seat.depth_ratio, 2),
            "global_axis": round(seat.global_axis, 3),
            "global_offset": round(seat.global_offset, 1),
        })

    return _joint_note(placed, len(seats), reasons, "magnet", seats)


def _apply_bridge(mans, i, j, mesh_a, mesh_b, count: int, cp: ConnectionParams,
                  params: PrintParams, overlap=None) -> Tuple[bool, str]:
    """Join two parts with clean flat-ended cylinders on the best contacts.

    Same seat selection and same collar as the magnet joint, minus the pocket —
    which is the point: the peg is now a true cylinder split on a shared plane,
    rather than the capsule-with-round-ends grown off the raw nearest-point pair
    that this used to be (that one landed at whatever angle the closest two
    vertices implied, and its hemispherical cap left a bobble on the surface).
    """
    r = _min_wall_radius(params, cp.connector_diameter_mm / 2.0)
    seats = _joint_seats(mesh_a, mesh_b, mans[i], mans[j], count, cp, r, r * 2.0,
                         overlap=overlap)
    embed = max(2.0 * r, 2.0)
    placed, reasons = 0, []
    for seat in seats:
        # The peg must span the gap as well as bite into both bodies.
        # Measured at the depth _build_seat will actually use, not the one
        # _joint_seats scored at: the multiplier is applied to this, so
        # verifying it against anything else under-delivers the correction.
        # A pin has no bore to protect, so it may pull back further.
        _resolve_back_faces(seat, embed + seat.gap / 2.0, r, cp, min_mult=0.5)
        ok, why = _build_seat(mans, i, j, seat, r, embed + seat.gap / 2.0,
                              None, True, cp.path_clearance_mm,
                              cp.socket_cap_exposed_max, cp.socket_extend_max)
        if ok:
            placed += 1
        else:
            reasons.append(why)
    return _joint_note(placed, len(seats), reasons, "bridge", seats)


#: Axis labels worth surfacing in the UI.  "mass" is the expected case and
#: "normal" is the best-founded one there is, so both are left unsaid; the other
#: two mean a fallback fired and are worth knowing about if a magnet still looks
#: wrong.
_AXIS_NOTE = {
    "contact": "axis from contact line",
    "mass-flat": "axis flattened along contact strip",
}


def _joint_note(placed: int, attempted: int, reasons, what: str, seats):
    """The (ok, human note) pair reported back to the UI for one interface."""
    seated = seats[:max(placed, 1)]
    used = {s.axis_source for s in seated}
    extra = [_AXIS_NOTE[a] for a in sorted(used) if a in _AXIS_NOTE]
    # A seat can clear the watertight gate and still be sunk into very little
    # plastic — a socket wider than the backbone it lands on, typically.  It is
    # printable, so it is not refused, but the user should hear about it before
    # the magnet pulls out of the part.
    if placed:
        thin = min((s.fill for s in seated), default=1.0)
        if thin < _MIN_SEAT_FILL:
            extra.append(f"thin seat ({thin * 100:.0f}% solid) — the joint is "
                         f"wider than the material it lands on; use a smaller "
                         f"connector Ø or a thicker backbone")
        # Separately from "is there material here", say so when the collar will
        # visibly stand off the surface — that is a look, not a failure, so it is
        # reported rather than refused.
        proud = min((s.embedding for s in seated), default=1.0)
        if proud < _MIN_SEAT_EMBEDDING:
            extra.append(f"joint stands proud ({proud * 100:.0f}% of the collar "
                         f"is buried) — the parts only meet over a small area "
                         f"here; a smaller connector Ø sinks it further in")
        # And separately again from both of those: there may be plenty of
        # material *around* the collar and still very little of it. Burial and
        # fill are both fractions of the collar, so both read high on a strut
        # barely wider than the socket — which is the "magnet on a thin little
        # arm" nobody was told about.
        # Magnets only: the threshold is calibrated against a magnet collar
        # (measured across arm widths at the stock 7.2 mm socket), and a pin's
        # collar is small enough that ordinary flat contacts sit near it.
        strut = min((s.depth_ratio for s in seated), default=99.0)
        if what == "magnet" and strut < 1.5:
            extra.append(f"joint sits on a thin feature — only {strut:.1f}× the "
                         f"collar's own volume of material around it; it will "
                         f"print, but a smaller connector Ø would sit better")
    if placed and not reasons:
        parts = ([f"{placed} {what}s"] if placed > 1 else []) + extra
        return True, "; ".join(parts)
    joined = "; ".join(sorted(set(reasons)) + extra)
    if placed:
        return True, f"placed {placed}/{attempted} — skipped: {joined}"
    return False, f"no {what} placed — " + (joined or "no contact")


#: Weld overlap added on top of the measured gap, so the two surfaces actually
#: intersect rather than kissing at a single tangent point.
_INFLATE_WELD_MM = 0.05

#: Most any one chain may grow.  A wide gap closed by inflation would balloon a
#: thin chain out of all proportion; use a bridge for those instead.
_INFLATE_MAX_MM = 1.0


def _inflate_growth(chain: Chain, params: PrintParams, amount_mm: float):
    """``params`` with ``chain``'s *structural* geometry grown by ``amount_mm``.

    Returns ``None`` when this chain has no knob that moves its whole surface
    outward by a known distance — the caller then leaves it alone and puts the
    growth on its neighbour.

    **Only structural dimensions grow, and for a nucleic chain that means the
    backbone alone.**  The bases are recognisable shapes drawn at a fixed scale,
    and padding them is not a size increase but a distortion:

    * the slab has a fixed footprint (``4.5 x 3.0 Å``) and only its *thickness*
      is a parameter, so growing it fattens the plate through-plane without
      moving its edges outward at all — the join does not get any closer to
      welding, and a flat rung turns into a brick;
    * in the molecule style the ring atoms sit ~1.4 Å apart, so an atom radius
      grown by even half a millimetre at print scale swallows the hole in the
      middle of the ring and the base reads as a blob;
    * the base-to-backbone connector grows past the slab it lands on and spikes.

    A fatter backbone welds just as well and leaves the bases reading as bases.
    """
    import dataclasses
    rep = params.representation_for(chain.mtype)

    # A ligand is never inflated.  Its radii are fixed constants rather than
    # parameters precisely because its proportions are the information — growing
    # the beads by half a millimetre closes the rings and it stops being a
    # recognisable molecule.  ``_joinable`` already keeps ligands out of every
    # contact, so this is the belt to that braces.
    if chain.mtype == MoleculeType.LIGAND:
        return None

    if chain.mtype == MoleculeType.PROTEIN:
        if rep == Representation.SURFACE:
            # Padding every atom's vdW radius moves the whole solvent-excluded
            # surface out by that distance — an exact uniform offset.
            return dataclasses.replace(
                params,
                surface_atom_padding_ang=params.surface_atom_padding_ang
                + amount_mm / params.scale_mm_per_angstrom,
            )
        if rep == Representation.TUBE_SLAB:
            return dataclasses.replace(
                params,
                protein_tube_radius_mm=params.protein_tube_radius_mm + amount_mm,
            )
        # Cartoon: a ribbon's thickness is locked to its width by a fixed
        # aspect, so there is no way to push its surface out uniformly without
        # visibly changing the ribbon's proportions.  Leave it be.
        return None

    return dataclasses.replace(
        params,
        nucleic_radius_mm=params.nucleic_radius_mm + amount_mm,
        # The backbone ball-and-stick has its own radii, so it has to be grown
        # here too — otherwise a molecule-backbone strand is the one style that
        # silently refuses to inflate.
        backbone_atom_radius_mm=params.backbone_atom_radius_mm + amount_mm,
        backbone_bond_radius_mm=params.backbone_bond_radius_mm + amount_mm,
    )


def _can_inflate(chain: Chain, params: PrintParams) -> bool:
    """True if inflating ``chain`` would actually move its surface."""
    return _inflate_growth(chain, params, 1.0) is not None


def _rebuild_inflated(chain: Chain, params: PrintParams, amount_mm: float):
    """Rebuild one chain's mesh grown outward by ``amount_mm`` (the inflate join).

    "Inflate" is a small size increase applied at *build* time, so two
    neighbours swell until their surfaces overlap and weld, with no strut and no
    re-meshing artefact.  The growth is bounded by the contact threshold, so it
    stays small.  Returns ``None`` if this chain cannot be grown.
    """
    from . import geometry, meshops
    q = _inflate_growth(chain, params, amount_mm)
    if q is None:
        return None
    mesh = geometry.generate_chain_mesh(chain, q)
    mesh = meshops.enforce_min_wall(mesh, q)
    return meshops.repair(mesh)


def _inflate_shares(i: int, j: int, gap: float, chains, params: PrintParams):
    """How much each side of one contact should grow: ``[(index, amount), ...]``.

    The gap is closed **protein first**.  A protein surface swells smoothly and
    the two parts read as having melted together; all a nucleic chain can move
    is its backbone tube, which thickens the strand without making the join look
    any better.  So the protein takes as much of the gap as it can carry, and
    the nucleic side is only called on for whatever is left over — which for an
    ordinary sub-millimetre contact is nothing at all, leaving the DNA exactly as
    it was drawn.

    Within one tier the growth is split evenly, so protein↔protein still swells
    symmetrically from both sides.  No chain ever grows more than
    ``_INFLATE_MAX_MM``, so a wide gap closes exactly as far as it used to —
    what changed is only *where* that growth is placed.  Returns ``[]`` when
    neither side can grow.
    """
    remaining = max(0.0, gap) + _INFLATE_WELD_MM
    can = [k for k in (i, j) if _can_inflate(chains[k], params)]
    tiers = [
        [k for k in can if chains[k].mtype == MoleculeType.PROTEIN],
        [k for k in can if chains[k].mtype != MoleculeType.PROTEIN],
    ]
    out = []
    for tier in tiers:
        if remaining <= 1e-9:
            break
        if not tier:
            continue        # no protein in this contact — fall through to DNA
        share = min(remaining / len(tier), _INFLATE_MAX_MM)
        out.extend((k, share) for k in tier)
        remaining -= share * len(tier)
    return out


def _unify_nucleic_growth(grow: List[float], chains) -> List[float]:
    """Level every nucleic chain's inflate growth to the largest one. In place.

    The nucleic backbone is a uniform *gauge* — one tube radius for the whole
    model — so two strands of a duplex coming out at different thicknesses is a
    visible defect, not a subtlety.  Per-chain amounts drift apart easily: with
    ``basepair_connect`` on there is no strand↔strand contact to keep a pair in
    step, so each strand's growth comes only from its own protein contacts, and
    a strand facing a wide gap ends up far fatter than its partner facing a
    close one.

    Protein has no equivalent and is left per-chain: padding an irregular
    solvent-excluded surface is a local offset, not a gauge, and two proteins
    padded differently is not something you can see.
    """
    # Nucleic *specifically*, not "everything that is not protein": a ligand also
    # fails that test, and levelling it to a strand's growth would inflate the one
    # object that must not be inflated (and that nothing asked to grow, since
    # ``_joinable`` excluded it from every contact).
    nucleic = [k for k, c in enumerate(chains)
               if c.mtype == MoleculeType.NUCLEIC]
    if nucleic:
        uniform = max(grow[k] for k in nucleic)
        for k in nucleic:
            grow[k] = uniform
    return grow


# --------------------------------------------------------------------------
# DNA interstrand base-pair connect
# --------------------------------------------------------------------------
#: Watson–Crick partners, by one-letter base code.  Inosine pairs cytosine.
_WATSON_CRICK = {
    ("A", "T"), ("T", "A"), ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"), ("I", "C"), ("C", "I"),
}

#: How far a partner may sit out of the base plane, as ``|d̂ · n̂|``.
#:
#: This is the test that separates a real partner from the base *stacked* on top
#: of it, and it is geometric rather than chemical on purpose.  A Watson–Crick
#: partner lies across the helix, so the line between the two ring centroids runs
#: *within* both base planes and its component along either normal is near zero.
#: The neighbour one step up the same helix sits directly on top, so that line
#: runs along the normal and the component is near one.  0.5 accepts anything
#: within 30° of the plane, which covers propeller twist and buckle with room to
#: spare while still rejecting a stacked base outright.
_BASEPAIR_COPLANAR_MAX = 0.5


def _base_letter(res_name) -> str:
    """One-letter base code from a residue name (``DA`` -> ``A``, ``DI`` -> ``I``)."""
    n = (res_name or "").strip().upper()
    if len(n) > 1 and n.startswith("D"):
        n = n[1:]
    return n[:1]


def _pair_bases(cen_a: np.ndarray, cen_b: np.ndarray, max_dist: float,
                frames_a=None, frames_b=None):
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
    delta = cen_b[None, :, :] - cen_a[:, None, :]
    dmat = np.linalg.norm(delta, axis=-1)

    ok = dmat <= max_dist

    # Coplanarity gate.  Distance alone cannot tell a partner from the base
    # stacked on top of it: a stacked neighbour is only ~3.4 Å away, well inside
    # any cutoff loose enough to tolerate a distorted duplex.  Requiring the
    # centroid-to-centroid line to lie in *both* base planes removes the whole
    # stacked column from consideration, which is what stops a one-base overhang
    # dragging the register off by one.
    if frames_a and frames_b:
        with np.errstate(invalid="ignore", divide="ignore"):
            unit_d = delta / np.where(dmat[..., None] > 1e-9, dmat[..., None], 1.0)
        na = np.array([f["normal"] for f in frames_a], float)
        nb = np.array([f["normal"] for f in frames_b], float)
        out_a = np.abs(np.einsum("ijk,ik->ij", unit_d, na))
        out_b = np.abs(np.einsum("ijk,jk->ij", unit_d, nb))
        ok &= (out_a <= _BASEPAIR_COPLANAR_MAX) & (out_b <= _BASEPAIR_COPLANAR_MAX)

    # Watson–Crick complementarity, used to score registers rather than to gate
    # pairs — a real duplex can carry a mismatch, and refusing to connect there
    # would leave a visible hole in the ladder.
    wc = np.zeros_like(ok)
    if frames_a and frames_b:
        la = [_base_letter(f.get("res_name")) for f in frames_a]
        lb = [_base_letter(f.get("res_name")) for f in frames_b]
        for i, a in enumerate(la):
            for j, b in enumerate(lb):
                if (a, b) in _WATSON_CRICK:
                    wc[i, j] = True

    def register(index_of_b):
        pairs, total, n_wc = [], 0.0, 0
        for i in range(n_a):
            j = index_of_b(i)
            if 0 <= j < n_b and ok[i, j]:
                pairs.append((i, j, float(dmat[i, j])))
                total += dmat[i, j]
                n_wc += int(wc[i, j])
        return pairs, total, n_wc

    best, best_key = [], (-1, -1, np.inf)
    # Antiparallel diagonals (j = s - i) and parallel diagonals (j = i + off).
    registers = ([(lambda i, s=s: s - i) for s in range(n_a + n_b - 1)]
                 + [(lambda i, o=o: i + o) for o in range(-(n_a - 1), n_b)])
    for idx_of_b in registers:
        pairs, total, n_wc = register(idx_of_b)
        # Complementary pairs first, then how many bases are paired at all, then
        # total distance.  Counting bases first is what produced the "clean but
        # wrong" ladder on a duplex with a one-base overhang at each end: the
        # off-by-one register pairs every base including the two overhangs, so it
        # won on count while pairing each base to its neighbour's partner.  The
        # true register pairs two fewer and every one of them complementary.
        key = (n_wc, len(pairs), -total)
        if key > best_key:
            best, best_key = pairs, key
    return best


def _apply_basepairs(man_a, man_b, chain_a: Chain, chain_b: Chain,
                     params: PrintParams):
    """Tie two DNA strands together at each base pair; returns (a, b, n_links)."""
    cp = params.connections
    cen_a, frames_a = tube_slab.base_link_frames_mm(chain_a, params)
    cen_b, frames_b = tube_slab.base_link_frames_mm(chain_b, params)
    max_dist = cp.basepair_max_dist_ang * params.scale_mm_per_angstrom
    pairs = _pair_bases(cen_a, cen_b, max_dist, frames_a, frames_b)
    if not pairs:
        return man_a, man_b, 0

    # The link continues each rung to the midline, so it has to be sized from
    # whatever that rung actually is — never from the backbone tube.  Those two
    # were coupled once, which is where the old ``nucleic_radius_mm * 0.7`` came
    # from; they are separate sliders now, so sizing off the tube produced a link
    # visibly thinner or fatter than the rung it was supposed to be continuing.
    #
    # Each branch mirrors the matching case in ``tube_slab._base_solids``:
    #   MOLECULE — ball-and-stick rungs, so the link is another bond.
    #   ROD      — ``rung_r = slab_t / 2``, so the same half-thickness here.
    #   SLAB     — a plate whose through-plane thickness is ``slab_t``; matching
    #              its thickness (not its much larger in-plane footprint) keeps
    #              the link reading as the plate reaching the axis rather than a
    #              slab-sized block bridging the duplex.
    #
    # ``_min_wall_radius`` reproduces the clamp the builder applies: it floors at
    # ``min_wall/2``, which is exactly what ``slab_t = max(slab_t, min_wall)``
    # then halved comes to, so a thin setting lands on the same number both ways.
    if params.base_style == BaseStyle.MOLECULE:
        r = _min_wall_radius(params, params.bond_radius_mm)
    else:
        r = _min_wall_radius(params, params.slab_thickness_mm / 2.0)

    # The two halves must *overlap*, not approach each other.
    #
    # ``fit_clearance_mm`` is for parts assembled by hand — magnet joints, the
    # interference carving — where an air gap is the whole point.  It is wrong
    # here.  The two strands are separate 3MF objects only so they can take
    # different filaments; they are printed at the same time, so coincident
    # material fuses in the machine.  A clearance between them is simply 0.15 mm
    # of air, which is exactly "the helix does not hold together".
    #
    # Meeting flush is not enough either.  Two round ends that merely touch share
    # a single tangent point — zero contact area, nothing for the slicer to weld.
    # So each half runs a short distance *past* the midline and the pair shares a
    # real volume.  Kept small: enough to weld, never enough to read as a bulge.
    #
    # Safe because ``interference.resolve`` has already run by this point (see
    # the ordering note in ``apply``), so nothing carves this overlap back out.
    weld = max(0.2, 0.2 * r)
    # Slab in-plane half-width, matching ``_base_solids``: the plate footprint is
    # 4.5 x 3.0 Å scaled, and the extension runs along the 4.5 direction, so the
    # width it must keep is the 3.0 one.
    slab_half_w = 3.0 * 0.5 * params.slab_scale * params.scale_mm_per_angstrom
    slab_half_t = max(params.slab_thickness_mm,
                      params.min_wall_mm if params.min_wall_mm > 0 else 0.0) * 0.5

    def _start_point(frame, centre, u):
        """Where this style's link may begin and still be inside solid material.

        ``_commit`` drops any boolean that increases the piece count, so a link
        starting in empty space is discarded and the pair silently fails to
        connect. Rod and slab are solid at their centroid, so that is both the
        visual tip and a safe anchor. A ball-and-stick base is *hollow* at its
        centroid — that point is the middle of the ring — so the molecule style
        anchors on the ring atom furthest along the pair axis instead: the ball
        nearest the partner, which is where the real hydrogen bond would leave.
        """
        if params.base_style != BaseStyle.MOLECULE:
            return centre
        ring = frame.get("ring") or []
        if not ring:
            return centre
        return max(ring, key=lambda p: float(np.dot(p - centre, u)))

    def _link(frame, centre, u, stop):
        """One half of the link: from this base out to ``stop``.

        Every style ends on a **flat face**, never a dome.  A hemispherical cap
        is what made the rod and molecule links "barely touch": the widest part
        of a dome is a single point on the axis, so two of them facing each other
        share almost no material even when they do meet.  A flat-ended cylinder
        presents its full circle instead, and with the weld overlap above the two
        halves interpenetrate over that whole disc.
        """
        start = _start_point(frame, centre, u)
        if params.base_style == BaseStyle.SLAB:
            # The plate itself reaches the axis — no rod bridging two plates.
            # Built as a box spanning start→stop, carrying the slab's own
            # thickness and in-plane width, so it reads as the same plate
            # continuing rather than a separate part bolted on.
            d = stop - start
            length = float(np.linalg.norm(d))
            if length < 1e-6:
                return None
            axis0 = d / length
            # An in-plane axis perpendicular to the run. Falls back to the
            # base's own in-plane direction if the run happens to lie along the
            # normal (degenerate, but cheap to guard).
            axis1 = np.cross(frame["normal"], axis0)
            n1 = float(np.linalg.norm(axis1))
            if n1 < 1e-6:
                axis1 = np.asarray(frame["in_plane"], float)
                n1 = float(np.linalg.norm(axis1)) or 1.0
            axis1 = axis1 / n1
            axis2 = np.cross(axis0, axis1)
            axis2 /= (float(np.linalg.norm(axis2)) or 1.0)
            return _manifold.oriented_box(
                start + d * 0.5,
                np.array([axis0, axis1, axis2]),
                np.array([length * 0.5, slab_half_w, slab_half_t]),
            )
        # Rod and molecule: a flat-ended cylinder of the rung's own radius.
        # ``frustum`` with equal radii is exactly that, and unlike ``capsule`` it
        # ends where it is told to — no hemisphere adding a radius past the end
        # point, so ``stop`` is the face rather than the centre of a dome.
        if float(np.dot(stop - start, u)) <= 1e-6:
            return None                       # nothing left to span
        return _manifold.frustum(start, stop, r, r)

    n_done = 0
    for ia, ib, _d in pairs:
        ca, cb = cen_a[ia], cen_b[ib]
        fa, fb = frames_a[ia], frames_b[ib]
        mid = 0.5 * (ca + cb)
        u = _unit(cb - ca)                     # A-centroid → B-centroid (toward axis)
        # Each half's flat face lands ``weld`` *beyond* the midline, so the two
        # interpenetrate over a 2 x weld length of full-radius material.
        stop_a = mid + u * weld
        stop_b = mid - u * weld
        tool_a = _link(fa, ca, u, stop_a)
        tool_b = _link(fb, cb, -u, stop_b)
        if tool_a is None or tool_b is None:
            continue                           # strands already meet here
        sa = _commit(man_a, tool_a, add=True)
        sb = _commit(man_b, tool_b, add=True)
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


def apply(built: List[Tuple[Chain, "object"]], params: PrintParams,
          progress=None):
    """Apply the fit + connections pass.

    Returns ``(new_built, [connection dicts], [magnet markers], [fit notes])``.

    Input meshes must already be watertight (the pipeline gates them first).
    Output meshes are re-repaired (the fast-path preserves the already-good
    manifold result) and remain watertight.

    Order matters here.  Interference is resolved **first**, before contacts are
    detected or any joint is seated, for two reasons:

    * a joint seated against still-overlapping geometry is measured against
      distances that do not mean what they appear to (see ``_overlap_seats``);
    * contact detection itself would miss the worst cases — for two chains that
      interpenetrate deeply, the unsigned nearest-vertex distance is the
      penetration depth, which can exceed ``contact_threshold_mm`` and make the
      most intimate interface in the structure look like no contact at all.

    The overlaps found on the way in are kept and handed to the joint search,
    because the volume that had to be carved away is precisely where a magnet
    belongs.
    """
    from . import meshops
    cp = params.connections
    do_fit = params.resolve_interference != InterferenceRule.NONE
    if (not cp.enabled() and not do_fit) or len(built) < 1:
        return built, [], [], []

    # This pass does seconds of boolean work per interface on a large complex,
    # and without a running commentary a slow build and a hung one look exactly
    # alike from the outside.  ``step`` is called often enough that the caller
    # can always name what is being worked on.
    def step(frac, msg):
        if progress is not None:
            progress(max(0.0, min(1.0, frac)), msg)

    chains = [c for c, _m in built]
    meshes = [m for _c, m in built]
    applied: List[Connection] = []
    markers: list = []   # magnet positions for the preview highlight
    fit_notes: List[str] = []
    # Said once for the whole build rather than per interface: it is a fact
    # about the settings, not about any one pair.
    _scale_note = _socket_scale_note(built, cp)
    if _scale_note:
        fit_notes.append(_scale_note)
    inflate = cp.connect and not cp.use_magnets \
        and cp.no_magnet_method == NoMagnetMethod.INFLATE

    # 1a) Inflate is the one mode that *wants* the parts to overlap — it grows
    #     neighbouring surfaces until they weld into one body — so the fit pass
    #     is skipped for it rather than undoing the join it just made.
    if inflate:
        contacts = []
        n = len(built)
        for i in range(n):
            for j in range(i + 1, n):
                if not _joinable(chains[i], chains[j]):
                    continue
                if (cp.basepair_connect
                        and chains[i].mtype == MoleculeType.NUCLEIC
                        and chains[j].mtype == MoleculeType.NUCLEIC):
                    continue
                _pa, _pb, gap = _nearest(meshes[i], meshes[j])
                if gap <= cp.contact_threshold_mm:
                    contacts.append((i, j, gap))

    # 1b) Inflate rebuilds each contacting object slightly larger (before the
    #     manifold conversion) so neighbours swell until they overlap.
    if inflate and contacts:
        # Work out who grows and by how much *before* rebuilding anything: a
        # chain in several contacts is rebuilt once, at the largest amount any
        # of them asked for.  Kept deliberately gentle — just enough to overlap.
        grow = [0.0] * len(built)
        shares = []
        for i, j, gap in contacts:
            sides = _inflate_shares(i, j, gap, chains, params)
            shares.append(sides)
            for idx, amt in sides:
                grow[idx] = max(grow[idx], amt)

        _unify_nucleic_growth(grow, chains)

        grown = set()
        for idx, amt in enumerate(grow):
            if amt > 0:
                try:
                    rebuilt = _rebuild_inflated(chains[idx], params, amt)
                except Exception:
                    rebuilt = None
                if rebuilt is not None:
                    meshes[idx] = rebuilt
                    grown.add(idx)
        for (i, j, gap), sides in zip(contacts, shares):
            # Only claim the join if this contact was allocated growth *and* a
            # side actually swelled — a contact between two chains that cannot
            # be offset is reported as skipped rather than silently shipped as a
            # weld that never happened.  The note is written from what really
            # grew, which is not the same as what this contact asked for: a
            # strand can also have been thickened to match its partner.
            did = [idx for idx in (i, j) if idx in grown]
            if sides and did:
                note = ""
                if len(did) == 1:
                    note = (f"grown on {chains[did[0]].display_name()} only — "
                            f"its neighbour keeps its shape")
                applied.append(Connection(
                    chains[i].chain_id, chains[j].chain_id,
                    _kind(chains[i], chains[j]), "inflate", gap_mm=gap,
                    applied=True, note=note))
            else:
                applied.append(Connection(
                    chains[i].chain_id, chains[j].chain_id,
                    _kind(chains[i], chains[j]), "inflate", gap_mm=gap,
                    applied=False,
                    note="neither part can be grown without changing its "
                         "proportions (a cartoon ribbon's thickness is tied to "
                         "its width) — use magnets or a bridge for this joint"))

    mans = [_manifold.from_trimesh(m) for m in meshes]

    # 1c) Make the solids physically disjoint, and remember where they were not.
    overlaps = {}
    if do_fit and not inflate:
        step(0.05, "Checking how the parts fit together…")
        found = interference.pair_overlaps(mans)
        if found:
            step(0.15, f"Carving {len(found)} overlapping interface(s) apart…")
        before = list(mans)
        # extend, not rebind: fit_notes may already carry the socket-vs-scale
        # note, and reassigning here dropped it on every path that could
        # produce one.
        mans, found, _resolved = interference.resolve(mans, chains, params, found)
        fit_notes.extend(_resolved)
        overlaps = {(o.i, o.j): o for o in found}
        if len(before) != len(mans) or any(a is not b
                                           for a, b in zip(before, mans)):
            # Geometry changed, so the probe clouds the joint search runs on
            # have to be re-taken from the carved solids.
            #
            # Keyed on "did anything move", not on "was there a note".  A note
            # is only emitted above _REPORT_MIN_MM3 (0.5 mm3), so a smaller
            # carve changed the solids and said nothing — and the entire joint
            # search then ran on pre-carve surfaces, which presents as an
            # occasional inexplicably placed magnet.
            meshes = [_manifold.to_trimesh(m) for m in mans]

    # 1d) Detect contacts on the *resolved* geometry, and always include any
    #     pair that was interpenetrating — those are in contact by definition,
    #     however the surviving surface gap happens to measure.
    if cp.connect and not inflate:
        n = len(built)
        todo = [(i, j) for i in range(n) for j in range(i + 1, n)
                if _joinable(chains[i], chains[j])
                and not (cp.basepair_connect
                         and chains[i].mtype == MoleculeType.NUCLEIC
                         and chains[j].mtype == MoleculeType.NUCLEIC)]
        for done, (i, j) in enumerate(todo):
                overlap = overlaps.get((i, j))
                _pa, _pb, gap = _nearest(meshes[i], meshes[j])
                if overlap is None and gap > cp.contact_threshold_mm:
                    continue
                step(0.3 + 0.6 * done / max(1, len(todo)),
                     f"Connecting {chains[i].display_name()} ↔ "
                     f"{chains[j].display_name()} ({done + 1}/{len(todo)})…")
                kind = _kind(chains[i], chains[j])
                # Protein↔protein and DNA↔protein each get their own count; the
                # bridge reuses the same counts, since it is now the same joint
                # minus the magnet pocket.
                n_joints = (cp.magnet_count if kind == "protein-protein"
                            else cp.dna_magnet_count)
                # DNA↔DNA never gets a magnet.  A pocket for even a small magnet
                # is several times wider than a backbone tube (default radius
                # 1.2 mm vs a 4 mm magnet), so the socket cannot sink into the
                # strand: it is a boss standing proud of it, and the bore usually
                # blows straight through — which ``_commit`` then rejects, so the
                # joint is lost anyway.  Those pairs are bridged instead.
                if cp.use_magnets and kind == "dna-dna":
                    method = "bridge"
                    ok, note = _apply_bridge(
                        mans, i, j, meshes[i], meshes[j], n_joints, cp, params,
                        overlap=overlap)
                    note = ("bridged, not magnetised: a magnet pocket is wider "
                            "than the backbone tube" + (f" — {note}" if note else ""))
                elif cp.use_magnets:
                    method = "magnet"
                    ok, note = _apply_magnet(
                        mans, i, j, meshes[i], meshes[j], n_joints, cp, params,
                        markers, overlap=overlap)
                else:
                    method = "bridge"
                    ok, note = _apply_bridge(
                        mans, i, j, meshes[i], meshes[j], n_joints, cp, params,
                        overlap=overlap)
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

    # 2b) Final sweep.  Everything after the fit pass *adds* material — collars,
    #     pegs, base-pair rungs — and each of those is a chance to put two parts
    #     back into the same space.  Re-checking here makes "the exported objects
    #     do not interpenetrate" a property of the pass as a whole rather than of
    #     each step remembering to behave, which is the only version of that
    #     guarantee worth having.  It is normally a no-op and reports nothing.
    if do_fit and not inflate and any(c.applied for c in applied):
        step(0.93, "Re-checking the fit after connecting…")
        mans, _again, late = interference.resolve(mans, chains, params,
                                                  allow_split=False,
                                                  want_pieces=False,
                                                  want_boxes=True)
        fit_notes.extend(f"After connecting — {note[0].lower()}{note[1:]}"
                         for note in late)
        # ...and say so plainly if anything survived that.
        #
        # ``resolve`` was called without a pre-computed overlap list, so it ran
        # its own full sweep and ``_again`` *is* that sweep's result.  Empty means
        # it found nothing, and a resolve that finds nothing returns ``mans``
        # untouched — so ``audit`` would run a third identical O(n²) sweep over
        # identical solids to reach the identical answer.  On a clean build, which
        # is the normal one, that was a third of the interference stage spent
        # confirming a known negative.
        if _again:
            fit_notes.extend(interference.audit(mans, chains))

    # 3) Back to meshes; repair fast-path keeps the already-watertight results.
    step(0.97, "Rebuilding meshes…")
    new_built = []
    for (chain, old_mesh), man in zip(built, mans):
        mesh = _manifold.to_trimesh(man)
        mesh.metadata.update(old_mesh.metadata)
        mesh = meshops.repair(mesh)
        new_built.append((chain, mesh))
    return new_built, [c.as_dict() for c in applied], markers, fit_notes
