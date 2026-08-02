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
    footprint: int           # surface points seated under the disc
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
    #: Where the search put this seat before ``_balance_burial`` moved it, and
    #: the local solids that went with it.  Restored if the moved seat turns out
    #: not to be buildable.
    home: object = None
    blocked: int = 0         # surface points sitting in the assembly path
    axis_source: str = "probe"
    #: Fraction of the probe ball each part fills, and the ball's radius (mm).
    probe_a: float = 0.0
    probe_b: float = 0.0
    probe_r: float = 0.0
    #: How far each part reaches past the mating face, inside the footprint (mm).
    overhang_a: float = 0.0
    overhang_b: float = 0.0
    #: Fraction of the collar that ends up buried, on the worse side.
    hidden: float = 0.0
    score: float = 0.0


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
_AXIS_QUALITY = {"probe": 1.0, "overlap": 1.0, "normal": 0.9, "mass": 0.9,
                 "global": 0.85, "mass-flat": 0.7, "contact": 0.3}


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

    **A side-wall version of this was built and reverted.**  It measured a thin
    shell at the collar's radius, so the number was literally the fraction of
    the wall you can see, which is the more direct description of "visible" and
    is why it was worth trying.  On real models it placed joints *worse*, which
    is the only test that counts.  The argument for it was that volume counts
    plastic in the middle of the cylinder that nobody looks at; the answer seems
    to be that the middle of the cylinder is exactly what says whether there is
    a **body** under the joint, where a wall measure is happy with a collar
    sleeved in a thin crust.  Do not rebuild it without a model to look at.

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


def _resolve_back_faces(seat: Seat, need_depth: float, socket_r: float,
                        cp: ConnectionParams, min_mult: float = 1.0) -> None:
    """Work out ``extend_*`` / ``taper_*`` for a seat that is about to be built.

    Deferred out of the search on purpose.  None of these four values appears
    anywhere in ``seat.score`` -- only :func:`_build_seat` reads them -- and the
    shortlist scores several seats to build one, so most sets of them were
    measured and thrown away.  Each set is up to ten ``_manifold.volume`` calls
    on an intersection, which is where the ~440 volume calls per build were
    coming from.

    Must be called with the same ``need_depth`` and ``socket_r`` that were
    passed to :func:`_find_seats`, not the ones :func:`_build_seat` is given:
    the result is a *multiple* of the depth it was measured at, which is how it
    carries over to the bridge, which sizes its socket differently.
    """
    if seat.back_done:
        return
    seat.back_done = True
    if seat.emb_a is None or seat.emb_b is None:
        return                        # nothing to measure; keep the defaults
    # How buried the collar ends up in bulk, on its worse side.  Measured here
    # rather than while the seat was scored, because this is the only place that
    # reads it and it costs a boolean per side: the shortlist scores several
    # seats to build one.  Measured at the depth the joint will actually use,
    # which the bridge sizes differently.
    seat.embedding = min(
        _embedding(seat.emb_a, seat.center, -seat.axis, socket_r, need_depth),
        _embedding(seat.emb_b, seat.center, seat.axis, socket_r, need_depth))
    seat.extend_a, seat.taper_a = _close_the_back(
        seat.emb_a, seat.center, -seat.axis, need_depth, socket_r, cp, min_mult,
        seat.embedding)
    seat.extend_b, seat.taper_b = _close_the_back(
        seat.emb_b, seat.center, seat.axis, need_depth, socket_r, cp, min_mult,
        seat.embedding)


#: How much material each side needs around a candidate before the joint is
#: worth building, as a multiple of the collar's own volume.
#:
#: Measured against the *collar*, not against the probe ball, and that
#: distinction is not cosmetic: the ball's radius is floored on the socket
#: radius, so turning the socket on grows it by nearly half and its volume by
#: seven times. A fraction-of-ball threshold therefore rejected everything the
#: moment a socket was switched on, which is exactly what happened -- a joint
#: that built happily without a collar failed the *acceptance test* with one,
#: for no reason connected to the geometry.
#:
#: A multiple of the collar means the same thing at every socket size, every
#: ball size and every scale.
#: 0.5, not 1.5. The collar's volume goes as socket_r squared, so switching the
#: socket on multiplied this threshold by 3.75 while the material available
#: stayed exactly the same -- and refused joints that build perfectly well. This
#: is only meant to throw out a candidate that is mostly air; ``_build_seat``'s
#: watertight commit is the real gate and always has been.
_MIN_PROBE_COLLARS = 0.5

#: Blocked-to-seated ratio past which the parts plainly cannot come apart along
#: this axis. Deliberately generous -- a rough carved interface has plenty of
#: points on the "wrong" side of any plane through it, and this is meant to
#: catch interlocking, not roughness.
_MAX_BLOCKED_RATIO = 3.0


def _object_thickness(mesh) -> float:
    """How thick this object is, in the "you could drill into it" sense.

    Volume over surface area, scaled. Exact for the shapes that matter: a
    cylinder of radius r gives 2r, a sphere of radius R gives 4R/3. Cheap --
    both quantities are already cached on the mesh -- and it does not care about
    the bounding box, which for a DNA duplex describes the helix rather than the
    1.2 mm tube the joint would actually land on.
    """
    try:
        area = float(mesh.area)
        if area > 1e-9:
            return max(0.0, 4.0 * abs(float(mesh.volume)) / area)
    except Exception:
        pass
    try:
        return float(np.min(mesh.bounds[1] - mesh.bounds[0]))
    except Exception:
        return 0.0


def _probe_radius(mesh_a, mesh_b, socket_r: float) -> float:
    """How big a ball to judge a joint in.

    Big enough to see well past the socket, small enough not to swallow the
    part, and it **moves with the model** instead of being a fixed number of
    millimetres. That last property is the point: every threshold in the old
    search was absolute, so the same number meant something quite different at
    scale 0.4 than at scale 3, and small models came off worst across the board.

    The floor matters as much as the ceiling. Too small a ball reads the local
    bumpiness of one atom instead of the shape of the interface; too large and
    it reaches around a thin feature and the centre of mass ends up on the far
    side of it — measured at 34 degrees of axis error once the ball is ~2.9x a
    feature's own size, and 84 degrees at 4x.
    """
    lo = np.minimum(mesh_a.bounds[0], mesh_b.bounds[0])
    hi = np.maximum(mesh_a.bounds[1], mesh_b.bounds[1])
    # The ceiling comes from how thick the *thinner object* actually is, not
    # from its bounding box. A ball much wider than the material it is measuring
    # is mostly air, and then both the score and the acceptance test read low for
    # a reason that has nothing to do with the joint -- which is exactly what
    # happens on a DNA backbone, where the box says "duplex" and the material
    # says "1.2 mm tube".
    thinnest = min(_object_thickness(mesh_a), _object_thickness(mesh_b))
    floor = 1.5 * socket_r
    ceiling = max(floor, 1.5 * thinnest) if thinnest > 0.0 else 1e9
    return float(min(max(floor, 0.08 * float(np.linalg.norm(hi - lo))), ceiling))


def _blob_centre(blob):
    """Centre of mass of a small cut solid, or ``None``."""
    if blob is None:
        return None
    try:
        return np.asarray(_manifold.to_trimesh(blob).center_mass, float)
    except Exception:
        return None


def _surface_along(points, origin, axis, radius: float):
    """Where a cloud crosses a line: its extents along ``axis``, or ``None``."""
    rel = np.asarray(points, float) - origin
    t = rel @ axis
    radial = np.linalg.norm(rel - np.outer(t, axis), axis=1)
    near = radial <= radius
    return t[near] if np.any(near) else None


#: How far the mating plane may slide along the axis to even the two sides out,
#: as a fraction of the collar's own depth.
#:
#: Bounded because sliding the plane into a part means cutting a plug out of it.
#: On a thin DNA backbone meeting a fat protein domain the search would happily
#: drive deep into the protein while the backbone still cannot hide anything --
#: a big hole cut for no gain. Half the collar is as far as it is worth going.
_BALANCE_MAX_SHIFT = 0.5

#: Burial difference below which the two sides are called even, and the search
#: does not run at all. Finer than the surface is smooth.
_BALANCE_TOL = 0.05

#: How much the worse side must actually gain before the seat is moved at all.
#:
#: Without this the search takes whatever the bisection ended on, and on a badly
#: lopsided interface -- where no shift can even the sides out, because one of
#: them has nothing to bury into anywhere -- that is the far end of the travel
#: for nothing. Measured on the mini_complex protein-DNA joint: 1.5 mm of shift
#: bought 0.006 of burial and moved the mating plane so far into the chain that
#: clearing the approach path severed it. A refinement that cannot pay for
#: itself should not be taken.
_BALANCE_MIN_GAIN = 0.05


def _balance_burial(seat: Seat, man_a, man_b, socket_r: float,
                    need_depth: float, probe_r: float) -> None:
    """Slide the mating plane along the axis until both sides are equally buried.

    The plane starts midway across the gap, which is a rule about the *gap* and
    says nothing about how much of either collar you can see.  On two flat
    facing surfaces the two happen to coincide; where one surface falls away
    under the joint and the other is solid, they do not, and the result is one
    collar standing proud while the other is invisible.

    Moving the plane trades one against the other -- toward A buries A's collar
    and exposes B's -- so evening them out **maximises the worse of the two**,
    which is exactly the figure ``seat.score`` already ranks on and had no way
    to improve.  It is not a new criterion; it is the missing optimiser for the
    one that is there.  It also settles the standing complaint that
    ``seat.hidden`` is the worse side and says nothing about the other: after
    this the two sides are equal by construction.

    Run only on the seats that will actually be built.  Each step is four small
    booleans, and volume calls are what made builds slow before.

    Note it runs *after* ranking, so a lopsided seat that would balance well is
    still beaten by an already-even one.  That is the cheap direction of the
    trade: balancing the whole shortlist costs an order of magnitude more.
    """
    room = _BALANCE_MAX_SHIFT * need_depth
    # Cut a fresh pair of blobs for this. The probe ball is barely wider than
    # the collar it contains -- its floor is 1.5x the socket radius, and at
    # stock settings the collar's far corner sits at 96% of it -- so a shifted
    # collar pokes straight out of it and reads as *unburied* because the blob
    # simply stops. This ball reaches past the far end of the travel.
    #
    # Enlarging is safe here where it would not be elsewhere: the warning about
    # a big ball is that its centre of mass reaches around a thin feature and
    # flips the axis. Nothing here derives an axis. Burial is a local question
    # and a wider ball only answers it over more of the part.
    reach = float(np.hypot(need_depth + room, socket_r)) + 0.5
    blob_a = _local_solid(man_a, seat.center, reach)
    blob_b = _local_solid(man_b, seat.center, reach)
    if blob_a is None or blob_b is None:
        return

    def buried(shift):
        c = seat.center + seat.axis * shift
        return (_embedding(blob_a, c, -seat.axis, socket_r, need_depth),
                _embedding(blob_b, c, seat.axis, socket_r, need_depth))

    wa, wb = buried(0.0)
    best, best0 = (0.0, wa, wb), (wa, wb)
    if abs(wa - wb) > _BALANCE_TOL:
        # A is buried more the further the plane moves toward A and less as it
        # moves toward B; B does the opposite. The difference is therefore
        # monotone in the shift, so bisection over the half-range the sign
        # points at is enough. Four steps land within a sixteenth of it.
        lo, hi = (0.0, room) if wa > wb else (-room, 0.0)
        for _ in range(4):
            mid = 0.5 * (lo + hi)
            ma, mb = buried(mid)
            if min(ma, mb) > min(best[1], best[2]):
                best = (mid, ma, mb)
            if abs(ma - mb) <= _BALANCE_TOL:
                break
            if ma > mb:
                lo = mid          # A still the buried one: keep moving toward B
            else:
                hi = mid
    shift, wa, wb = best
    if min(wa, wb) < min(best0) + _BALANCE_MIN_GAIN:
        shift, wa, wb = 0.0, best0[0], best0[1]
    if abs(shift) > 1e-6:
        seat.home = (np.array(seat.center, float), seat.emb_a, seat.emb_b)
        seat.center = seat.center + seat.axis * shift
        # The back-face search reads blobs cut at the probe radius around the
        # seat, so recut them where the seat now is. Same radius as before, on
        # purpose: that search is calibrated against this ball and this is not
        # the place to change what it sees.
        seat.emb_a = _local_solid(man_a, seat.center, probe_r) or seat.emb_a
        seat.emb_b = _local_solid(man_b, seat.center, probe_r) or seat.emb_b
    # The score keeps the figure it was ranked on; this is what the joint
    # actually ends up being, and it is what the report should show.
    seat.hidden = float(min(wa, wb))


def _unbalance(seat: Seat) -> bool:
    """Put a balanced seat back where the search found it; ``True`` if it moved.

    Evening the two sides out is a refinement.  Being able to build the joint at
    all is not, and moving the mating plane into a part means cutting a plug out
    of it, which on a thin one severs it.  The callers try the tidier position
    first and fall back to this.
    """
    if seat.home is None:
        return False
    seat.center, seat.emb_a, seat.emb_b = seat.home
    seat.home = None
    seat.back_done = False        # the back-face search has to be redone there
    return True


def _clear_depths(seat: Seat, va, vb, radius: float) -> None:
    """How far each part reaches past the mating face, remeasured properly.

    This number is the length of the approach cut, so anything it cannot see is
    material left standing in the other part's way -- and the version taken
    while the seat was scored could not see much:

    * it ran over the **probe cloud**, which is capped at ``_MAX_PROBE_VERTS``
      and on a large chain samples roughly one vertex in ten, so the tip of a
      lobe is simply not in it;
    * it used the **socket radius**, while the cut is made at the socket radius
      plus the sliding clearance, leaving an unmeasured ring of material inside
      the cut's own footprint;
    * it was taken from the probe's centre of mass rather than from the seat.

    None of that matters to the *score* -- which is why it was fine there -- and
    all of it matters here, where being 6 mm short means a model that cannot be
    assembled.  Two dot products over the full vertex set, for the handful of
    seats that will actually be built.
    """
    for pts, sign, attr in ((va, 1.0, "overhang_a"), (vb, -1.0, "overhang_b")):
        t = _surface_along(pts, seat.center, seat.axis * sign, radius)
        setattr(seat, attr, 0.0 if t is None else float(max(0.0, t.max())))


#: Why the last interface's candidates were refused, when none survived.
_LAST_REFUSALS: dict = {}


def _probe_seat(man_a, man_b, pa, pb, centre, probe_r: float, socket_r: float,
                cp: ConnectionParams, need_depth: float, refused=None,
                gap_falloff: float = None, hidden_weight: float = None):
    """One candidate spot, measured. Returns a :class:`Seat` or ``None``.

    The whole placement rule, and it is deliberately one idea rather than
    twelve:

    1. Cut a ball of each part's material around the spot.
    2. **Score it on how much of that ball each part fills**, on the weaker of
       the two. A broad flat interface fills both halves; a spike grazing a
       surface fills almost none, and a thin arm not much more. That single
       measure replaces fill, embedding, depth ratio and footprint, which were
       four different answers to "is there material here".
    3. **Take the axis from the two centres of mass** of what was found. Not
       from a nearest-vertex pair (measured 7-50 degrees off on two *flat*
       facing blocks), not from a plane fit, not from the shape of an
       interference lobe.
    4. **Put the joint on that line**, midway across the gap it crosses —
       rather than at the closest-approach point with an axis fitted to it.

    Steps 2 and 3 protect each other, which is what makes the big ball safe
    here when it is not safe elsewhere: a centre of mass reaches around a thin
    feature and flips only when that part barely fills the ball, and step 2 has
    already thrown those candidates out.
    """
    def no(why):
        if refused is not None:
            refused.append(why)
        return None

    blob_a = _local_solid(man_a, centre, probe_r)
    blob_b = _local_solid(man_b, centre, probe_r)
    if blob_a is None or blob_b is None:
        return no("no material in the probe")
    try:
        ball = (4.0 / 3.0) * np.pi * probe_r ** 3
        vol_a = float(_manifold.volume(blob_a))
        vol_b = float(_manifold.volume(blob_b))
    except Exception:
        return None
    frac_a, frac_b = vol_a / ball, vol_b / ball
    collar = float(np.pi * socket_r * socket_r * need_depth)
    if min(vol_a, vol_b) < _MIN_PROBE_COLLARS * collar:
        return no("too little material for the collar")

    com_a, com_b = _blob_centre(blob_a), _blob_centre(blob_b)
    if com_a is None or com_b is None:
        return no("no centre of mass")
    delta = com_b - com_a
    if float(np.linalg.norm(delta)) < 1e-6:
        return no("the two centres coincide")
    axis = _unit(delta)

    # Where that line actually crosses from one part to the other. The seat goes
    # midway between the two surfaces, which is where both collars meet flush.
    ta = _surface_along(pa, com_a, axis, socket_r)
    tb = _surface_along(pb, com_a, axis, socket_r)
    if ta is None or tb is None:
        return no("the axis misses one surface")
    a_face, b_face = float(ta.max()), float(tb.min())
    if b_face < a_face - socket_r:
        return no("the axis leaves A after entering B")
    seat_centre = com_a + axis * (0.5 * (a_face + b_face))
    gap = max(0.0, b_face - a_face)

    # The one thing the volume score cannot see: whether the parts can come
    # apart along this axis at all. Interlocked fingers and a cup gripping past
    # an equator both give a beautiful flat interface. Kept as a refusal, not as
    # another weighted term.
    blocked, seated = _path_census(pa, pb, seat_centre, axis,
                                   socket_r + cp.path_clearance_mm, need_depth)
    if blocked > _MAX_BLOCKED_RATIO * seated:
        return no("the parts cannot come apart along this axis")

    # How far each part reaches past the mating face inside the joint's own
    # footprint. The approach cut is sized from this rather than from the collar
    # length: a part that leans 9 mm over the face keeps 9 mm of material on the
    # other part's side of it, and a 3.9 mm cut leaves the joint unable to close.
    # Only a placeholder here -- see ``_clear_depths``, which remeasures it over
    # the full vertex set for the seats that are kept.
    over_a = float(max(0.0, ta.max() - (0.5 * (a_face + b_face))))
    over_b = float(max(0.0, (0.5 * (a_face + b_face)) - tb.min()))
    seat = Seat(center=seat_centre, axis=axis, gap=gap,
                footprint=int(seated), axis_source="probe")
    seat.overhang_a, seat.overhang_b = over_a, over_b
    # The back-face search reads these. The probe ball is at least 2.5x the
    # socket radius, so it comfortably contains the collar it will be asked
    # about, which is the one thing that has to be true of it.
    seat.emb_a, seat.emb_b = blob_a, blob_b
    seat.blocked = int(blocked)
    # Volume alone will happily take a wider gap for slightly more material —
    # which is wrong at the end of a DNA duplex, where the strands splay and
    # the fullest probe is the one furthest from a joint that would actually
    # close. Multiplied rather than subtracted, so a gap the collar cannot span
    # cannot be bought back with volume however much of it there is.
    # How much of the collar ends up inside existing material, on the worse of
    # the two sides. 1.0 means only the mating disc shows; 0.0 means it stands
    # in open air. This is the difference between a magnet you have to look for
    # and one you can see from across the room, and it costs one small boolean
    # per side against blobs that are already cut.
    seat_hidden = min(_embedding(blob_a, seat_centre, -axis, socket_r, need_depth),
                      _embedding(blob_b, seat_centre, axis, socket_r, need_depth))

    reach = max(float(cp.contact_threshold_mm), 1e-6)
    falloff = cp.seat_gap_falloff if gap_falloff is None else float(gap_falloff)
    # Squared, so the penalty is gentle among genuinely close candidates and
    # steep once a gap is real. Linear treated "touching" and "half a millimetre
    # apart" as nearly the same, which is not how the joint behaves.
    near = 1.0 - falloff * min(1.0, gap / reach) ** 2
    hide = cp.seat_hidden_weight if hidden_weight is None else float(hidden_weight)
    seat.hidden = float(seat_hidden)
    seat.fill = float(min(frac_a, frac_b))
    # Multiplied, like the distance term, so a socket standing in open air
    # cannot be bought back with volume. At the default weight a fully exposed
    # collar keeps half its score and a fully buried one keeps all of it.
    seat.score = float(min(frac_a, frac_b) * max(near, 0.0)
                       * (1.0 - hide + hide * seat_hidden))
    seat.probe_a, seat.probe_b, seat.probe_r = float(frac_a), float(frac_b), probe_r
    return seat


def _find_seats(mesh_a, mesh_b, man_a, man_b, count: int,
                cp: ConnectionParams, socket_r: float,
                need_depth: float, gap_falloff: float = None,
                hidden_weight: float = None) -> List[Seat]:
    """The ranked places to build a joint, best first.

    Candidates are simply *everywhere the two surfaces come close*, spread
    evenly over the contact by farthest-point sampling. No preference for
    parallel faces, no direction-agreement filter, no separate source for
    interference lobes — a place the parts used to overlap is now a broad close
    contact and scores well on its own merits.
    """
    count = max(1, count)
    pa = _probe_points(mesh_a)
    pb = _probe_points(mesh_b)
    d, idx = cKDTree(pb).query(pa, k=1)
    close = d <= cp.contact_threshold_mm
    if not np.any(close):
        close = np.zeros(len(d), bool)
        close[int(np.argmin(d))] = True
    mids = 0.5 * (pa[close] + pb[idx[close]])

    probe_r = _probe_radius(mesh_a, mesh_b, socket_r)
    wanted = min(len(mids), max(10, 4 * count + 6))
    picks = _farthest_seeds(mids, wanted)

    seats, refused = [], []
    for k in picks:
        seat = _probe_seat(man_a, man_b, pa, pb, mids[k], probe_r,
                           socket_r, cp, need_depth, refused, gap_falloff,
                           hidden_weight)
        if seat is not None:
            seats.append(seat)
    if not seats and refused:
        # Never leave "no joint here" unexplained again: the tally says which
        # test threw the candidates away, which is the one thing that cannot be
        # worked out afterwards from the model.
        tally = {}
        for why in refused:
            tally[why] = tally.get(why, 0) + 1
        _LAST_REFUSALS.clear()
        _LAST_REFUSALS.update(tally)
    seats.sort(key=lambda s: -s.score)

    kept: List[Seat] = []
    for seat in seats:
        if any(float(np.linalg.norm(seat.center - k.center)) < 2.0 * socket_r + 1.0
               for k in kept):
            continue
        kept.append(seat)
        if len(kept) >= count:
            break
    if kept:
        va = np.asarray(mesh_a.vertices, float)
        vb = np.asarray(mesh_b.vertices, float)
        for seat in kept:
            # Order matters: balancing moves the mating plane, and the overhang
            # is measured from it.
            _balance_burial(seat, man_a, man_b, socket_r, need_depth, probe_r)
            _clear_depths(seat, va, vb, socket_r + cp.path_clearance_mm)
    return kept


def _build_seat(mans, i, j, seat: Seat, socket_r: float, embed: float,
                pocket: dict | None, socket_on: bool,
                clearance: float = 0.3, cap_limit: float = 0.0,
                extend_max: float = 0.0,
                nose_scale: float = 1.0) -> Tuple[bool, str]:
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
    # A 45° cone is the natural shape — its height is just how far the radius
    # has to come in — but at full height it is a visible point on the back of
    # the model, so ``nose_scale`` flattens it. Below about 0.7 the taper stops
    # being self-supporting on an FDM printer; see ``socket_nose_scale``.
    nose_height = (min(socket_r * (1.0 - top_ratio) * max(0.0, float(nose_scale)),
                       embed * extend_max)
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
            # Long enough to reach however far *this* part actually leans over
            # the face, not merely as long as our own collar. This part's own
            # overhang, because this cut removes this part's material: the two
            # were the wrong way round, so a chain reaching 9.9 mm across the
            # face was cut back by whatever its neighbour happened to reach.
            over = seat.overhang_a if idx == i else seat.overhang_b
            path = _seat_solid(seat.center - into * 0.002, -into,
                               max(embed * grow + seat.gap + 1.0, over + 1.0),
                               socket_r + clearance)
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
                  overlap=None, gap_falloff: float = None,
                  hidden_weight: float = None) -> Tuple[bool, str]:
    """Seat ``count`` press-fit magnet pockets on the best-scoring contacts.

    The pocket is cut oversize on purpose (see ``magnet_fit_clearance_mm`` /
    ``magnet_depth_clearance_mm``): an FDM hole printed to the magnet's nominal
    size comes out too small to accept it at all.  Positions, sizes and the
    resulting axis are recorded in ``markers`` so the preview can highlight them.
    """
    d, t, shape = cp.connector_diameter_mm, cp.magnet_thickness_mm, cp.magnet_shape
    pocket_r = d / 2.0 + cp.magnet_fit_clearance_mm / 2.0
    depth = t + cp.magnet_depth_clearance_mm
    # Always the socketed radius, even with the socket off. This number is not
    # only the collar's: it sets the probe ball's floor, the acceptance gate
    # (which goes as its square), the spacing between candidates and the width
    # of the approach cut. Letting it collapse to the bare pocket meant turning
    # the socket off *moved the magnets* -- 3.6 mm down to 2.1 mm at stock
    # settings, a probe ball from 5.4 to 3.2 mm and an acceptance gate 2.9x
    # easier to pass. That was never a decision, just a side effect, and it is
    # the same accident as the two thresholds that scaled with this radius
    # before it. Score one geometry; let only the build ask whether to raise a
    # collar.
    socket_r = pocket_r + cp.socket_wall_mm
    # The collar must bury the pocket and still have wall behind it.
    embed = depth + max(cp.socket_wall_mm, 1.0)

    # The lead-in eats grip: the top of the bore is oversized, so on a thin
    # magnet an unclamped chamfer would leave almost nothing holding it.
    chamfer = min(cp.magnet_chamfer_mm, 0.3 * t)

    seats = _find_seats(mesh_a, mesh_b, mans[i], mans[j], count, cp,
                        socket_r, embed, gap_falloff, hidden_weight)
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
                              cp.socket_cap_exposed_max, cp.socket_extend_max,
                              cp.socket_nose_scale)
        if not ok and _unbalance(seat):
            _resolve_back_faces(seat, embed, socket_r, cp,
                                min_mult=min(1.0, (depth + 0.6) / max(embed, 1e-6)))
            ok, why = _build_seat(mans, i, j, seat, socket_r, embed, pocket,
                                  cp.socket, cp.path_clearance_mm,
                                  cp.socket_cap_exposed_max, cp.socket_extend_max,
                                  cp.socket_nose_scale)
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
            "embedding": round(seat.embedding, 3),
            "probe_a": round(seat.probe_a, 3),
            "probe_b": round(seat.probe_b, 3),
            "probe_r": round(seat.probe_r, 2),
            "hidden": round(seat.hidden, 3),
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
    # A bridge is the same joint without the magnet, so it is placed the same
    # way -- and it is thinner on purpose: no pocket to bury means no socket
    # wall, so the peg only has to be strong, not roomy.
    r = _min_wall_radius(params, float(cp.bridge_diameter_mm) / 2.0)
    seats = _find_seats(mesh_a, mesh_b, mans[i], mans[j], count, cp, r, r * 2.0)
    embed = max(2.0 * r, 2.0)
    placed, reasons = 0, []
    for seat in seats:
        # The peg must span the gap as well as bite into both bodies.
        # Measured at the depth _build_seat will actually use, not the one
        # _find_seats scored at: the multiplier is applied to this, so
        # verifying it against anything else under-delivers the correction.
        # A pin has no bore to protect, so it may pull back further.
        _resolve_back_faces(seat, embed + seat.gap / 2.0, r, cp, min_mult=0.5)
        ok, why = _build_seat(mans, i, j, seat, r, embed + seat.gap / 2.0,
                              None, True, cp.path_clearance_mm,
                              cp.socket_cap_exposed_max, cp.socket_extend_max,
                              cp.socket_nose_scale)
        if not ok and _unbalance(seat):
            _resolve_back_faces(seat, embed + seat.gap / 2.0, r, cp, min_mult=0.5)
            ok, why = _build_seat(mans, i, j, seat, r, embed + seat.gap / 2.0,
                                  None, True, cp.path_clearance_mm,
                                  cp.socket_cap_exposed_max, cp.socket_extend_max,
                                  cp.socket_nose_scale)
        if ok:
            placed += 1
        else:
            reasons.append(why)
    return _joint_note(placed, len(seats), reasons, "bridge", seats)


def _joint_note(placed: int, attempted: int, reasons, what: str, seats):
    """The (ok, human note) pair reported back to the UI for one interface."""
    seated = seats[:max(placed, 1)]
    extra: list = []
    # One measure now, because the search takes one: how much of the probe ball
    # each part filled, on the weaker side. A broad flat interface fills both
    # halves; a spike or a thin arm fills very little. It is printable either
    # way, so this is reported rather than refused — but it is the number that
    # says whether a joint has anything to hold on to.
    if placed:
        weak = min((min(s.probe_a, s.probe_b) for s in seated), default=1.0)
        if weak < 0.15:
            extra.append(f"little material around this joint — the thinner side "
                         f"fills {weak * 100:.0f}% of the space around it; it "
                         f"will print, but a smaller connector Ø, a thicker "
                         f"backbone or a larger scale would seat it better")
    if placed and not reasons:
        parts = ([f"{placed} {what}s"] if placed > 1 else []) + extra
        return True, "; ".join(parts)
    joined = "; ".join(sorted(set(reasons)) + extra)
    if placed:
        return True, f"placed {placed}/{attempted} — skipped: {joined}"
    if not attempted and _LAST_REFUSALS:
        why = ", ".join(f"{n}x {reason}" for reason, n
                        in sorted(_LAST_REFUSALS.items(), key=lambda kv: -kv[1]))
        return False, f"no {what} placed — every candidate was refused: {why}"
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
      distances that do not mean what they appear to;
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
                # Two DNA strands get nothing unless the base-pair control
                # asked for it. They used to be silently bridged here, which is
                # a joint nobody requested appearing between two strands -- and
                # the base-pair pass is the feature that *is* meant to link
                # them, so it should be the only thing that does.
                if kind == "dna-dna":
                    continue
                if cp.use_magnets:
                    method = "magnet"
                    first_mark = len(markers)
                    ok, note = _apply_magnet(
                        mans, i, j, meshes[i], meshes[j], n_joints, cp, params,
                        markers, overlap=overlap,
                        # Two proteins meet across a broad face, and the middle
                        # of it is the right place — so distance counts for
                        # everything there. On a mixed interface the geometry is
                        # lumpier and material still has to have a say.
                        gap_falloff=(cp.seat_gap_falloff_flat
                                     if kind == "protein-protein" else None),
                        # Two proteins are where a hidden magnet is achievable
                        # and worth the most: broad faces with real depth behind
                        # them. On a backbone there is nowhere to hide one, and
                        # insisting would only cost a joint.
                        hidden_weight=(cp.seat_hidden_weight_flat
                                       if kind == "protein-protein" else None))
                    # Which two parts each magnet belongs to. Nothing carried
                    # that before, so a marker could not be told apart from a
                    # third chain that happens to cross the same line -- which
                    # is a different problem with a different answer, and one
                    # this joint is not the place to solve.
                    for mark in markers[first_mark:]:
                        mark["a"] = chains[i].chain_id
                        mark["b"] = chains[j].chain_id
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
