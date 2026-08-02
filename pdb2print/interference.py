"""Resolve interpenetration between separately-built chain solids.

Every chain is meshed **independently from its own atoms**, so at a binding
interface the two solids necessarily overlap: each one's surface is built as if
the other were not there, and both bulge into the shared space.  On screen that
looks right — it is what the complex actually looks like — but the two objects
cannot be printed and assembled, because each occupies volume the other needs.

This pass makes the set of solids *physically disjoint* before anything else
touches them.  It runs whether or not the parts are being connected: two objects
that are simply printed and handed to the user still have to fit together.

**Why the probe radius cannot do this job.**  The SES field is
``EDT(atoms grown by p) − p``; on a convex patch those cancel exactly, leaving a
surface at ``vdW + padding`` regardless of ``p``.  An interface is convex-facing
on both sides, so lowering the probe radius does not move it — it only carves
into concave pockets, opening crevices and risking pinched, non-manifold necks.
Interference is a boolean problem and is solved here with booleans.

**Who gives up the material.**  Under :attr:`InterferenceRule.AUTO`:

* **ligand ↔ anything** — the ligand keeps its true shape and its host is carved.
  This is the whole point of printing a bound ligand: the pocket comes out as an
  exact negative of the drug, so it lifts out and drops back in and you can see
  why it fits.  Carving the ligand instead would take a bite out of the one
  object small enough that a bite destroys it.
* **nucleic ↔ protein** — the nucleic acid keeps its true shape and the protein
  is carved.  This matches the biology (DNA sits in a groove) and it reads as
  intentional: the protein ends up with a socket that is an exact negative of
  the duplex, like a mould, rather than both parts looking bitten.
* **same type** — the larger part keeps its shape and the smaller one is carved,
  which is deterministic and keeps the dominant subunit intact.

:attr:`InterferenceRule.SYMMETRIC` instead has *both* parts retreat out of the
shared volume.  Neither is deformed by the other's shape, at the cost of a gap
where they used to interpenetrate (the connector socket bridges it).  Ligands are
excluded from it and keep their shape either way — see :func:`_carve_target`.

**Clearance.**  Carving with a plain subtraction gives a zero-clearance mate:
geometrically perfect, but FDM parts print a little oversize and would bind.  So
the tool is the *other* part grown by ``fit_clearance_mm`` first.  A true
Minkowski dilation is far too slow here (tens of seconds on a protein-sized
mesh), so the growth is a union of six translated copies — exact along the axes
and a little under between them, which at a 0.1–0.3 mm clearance is well inside
print tolerance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

from .config import PrintParams, MoleculeType, InterferenceRule
from .chains import Chain
from .representations import _manifold


#: Unit directions used to approximate a spherical dilation by a union of
#: translated copies.  Six axis directions, not the fourteen that adding the cube
#: corners would give: this union is the single most expensive thing in the whole
#: pass (it was 8 s of a 24 s build on a five-chain complex, because each call
#: unions that many copies of a large mesh), and the corners more than double it
#: to buy accuracy that a print clearance does not need.
_DILATE_DIRS: np.ndarray = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    dtype=float,
)

#: Overlap fragments below this fraction of the largest one are numerical
#: slivers along the interface rim, not real interpenetration lobes.  They are
#: still carved away (the carve uses the whole overlap solid); they are just not
#: offered as connector seats.
_PIECE_MIN_FRACTION = 0.05

#: Never offer more than this many seats from one interface, however finely the
#: overlap decomposes.
_MAX_PIECES = 12

#: Overlaps smaller than this (mm³) are carved away but not reported.  Booleans
#: on organic meshes leave sub-cubic-millimetre slivers wherever two surfaces
#: were built to meet exactly — a collar face against its opposite number, say —
#: and listing those as findings would bury the real ones.
_REPORT_MIN_MM3 = 0.5


# --------------------------------------------------------------------------
# Small manifold helpers
# --------------------------------------------------------------------------
def _volume(man) -> float:
    """Enclosed volume of a manifold, tolerant of ``volume`` being a property.

    ``manifold3d`` has shipped ``Manifold.volume`` both ways across releases and
    this is called often enough that going via a trimesh conversion (what
    :func:`_manifold.volume` does) would be wasteful.
    """
    if man is None or man.is_empty():
        return 0.0
    try:
        v = man.volume
        return float(abs(v() if callable(v) else v))
    except Exception:
        return _manifold.volume(man)


def _components(man) -> int:
    try:
        return len(man.decompose())
    except Exception:
        return 1


#: Six axis directions grow an axis-facing surface by the full amount and a
#: corner-facing one by ``cos(54.7°) ≈ 0.58``, so the requested amount is scaled
#: up to put the spread either side of nominal rather than all below it.  At the
#: default 0.15 mm clearance that is 0.11 mm at the tightest and 0.19 mm at the
#: loosest — both comfortably inside what an FDM mating surface cares about, and
#: the anisotropy is invisible on an organic shape.
_DILATE_COMPENSATION = 1.25


def dilate(man, amount: float):
    """Grow ``man`` outward by roughly ``amount`` mm in every direction.

    Union of translated copies (see the module docstring) — orders of magnitude
    faster than a true Minkowski sum and accurate enough for a print clearance.
    Returns ``man`` unchanged for a non-positive amount.
    """
    if man is None or amount <= 0.0 or man.is_empty():
        return man
    offsets = _DILATE_DIRS * (float(amount) * _DILATE_COMPENSATION)
    return _manifold.union([man.translate(tuple(v)) for v in offsets])


def _solid_moments(mesh):
    """``(volume, centroid, covariance)`` of the solid a closed mesh encloses.

    Exact closed-form integrals over the tetrahedra spanned from the origin, so
    the result depends on the *solid* and not at all on how it was triangulated.
    That is the whole point. The previous version ran an SVD over the mesh's
    vertices with every vertex weighted equally, and a boolean's output is not
    uniformly tessellated — slivers where two surfaces nearly coincide, dense
    clusters where they cross — so the principal axes followed the mesher.

    Measured on 126 lobes built through both production meshing routes: for the
    **same solid**, remeshed, the old vertex axis moved a median of 9.5 degrees,
    p90 37.6, max 87.0. This moves 0.000, and not as a matter of luck — nothing
    in these integrals can see the triangulation.

    Also cheaper than what it replaces: ``trimesh.center_mass`` alone was ~47 ms
    on a 47k-triangle lobe, this is ~21 ms and returns the covariance too.
    """
    verts = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, np.int64)
    if len(verts) < 4 or len(faces) < 4:
        return 0.0, None, None
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    # Signed volume of each tetrahedron (origin, a, b, c).
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    volume = float(vol6.sum()) / 6.0
    if abs(volume) < 1e-12:
        return 0.0, None, None
    centroid = (np.einsum("i,ij->j", vol6, (a + b + c)) / 4.0) / (6.0 * volume)
    # Second moment of each tet about the origin:
    #   integral of x x^T = (V/20) * (aa^T + bb^T + cc^T + s s^T),  s = a+b+c
    s = a + b + c
    outer = (np.einsum("i,ij,ik->jk", vol6, a, a)
             + np.einsum("i,ij,ik->jk", vol6, b, b)
             + np.einsum("i,ij,ik->jk", vol6, c, c)
             + np.einsum("i,ij,ik->jk", vol6, s, s)) / (6.0 * 20.0)
    cov = outer / volume - np.outer(centroid, centroid)
    return abs(volume), centroid, 0.5 * (cov + cov.T)


def _centroid_and_extent(man) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``(centre of mass, thin-axis unit vector)`` of one overlap fragment.

    An interference lobe is usually a *lens*: wide across the interface, thin
    through it.  Its smallest principal direction is then the interface normal,
    and far better conditioned than any single nearest-point pair — which is why
    a seat derived from the overlap sits square.  The sign is arbitrary here;
    the caller orients it.

    What it no longer does is read the axis off the mesh's *vertices* — see
    :func:`_solid_moments` for why that was worth 9.5 degrees of noise.

    It does **not** try to detect a lobe that is not a lens, and that omission
    is deliberate.  A gate on ``sqrt(thin/mid)`` was written and removed: the
    threshold separated the synthetic fixtures cleanly but was never measured
    against real lobes, whose ratios span 0.13 to 0.96 — so it refused a large
    and unknown share of perfectly good interfaces.  Worse, refusing here does
    not send the seat to the axis search as intended; ``_overlap_seats`` falls
    back to a single nearest-vertex pair, which is the noisiest estimator in the
    codebase.  A gate is still the right idea, but it needs a measured threshold
    and a fallback that goes somewhere better than that.
    """
    try:
        mesh = _manifold.to_trimesh(man)
    except Exception:
        return None, None
    volume, centre, cov = _solid_moments(mesh)
    if centre is None:
        return None, None
    try:
        evals, evecs = np.linalg.eigh(cov)          # ascending
    except Exception:
        return centre, None
    if float(evals[1]) <= 1e-15:
        return centre, None                # degenerate; nothing to read
    axis = np.asarray(evecs[:, 0], float)
    n = float(np.linalg.norm(axis))
    return centre, (axis / n if n > 1e-12 else None)


# --------------------------------------------------------------------------
# Overlap detection
# --------------------------------------------------------------------------
@dataclass
class OverlapPiece:
    """One connected lobe of interpenetration between two chains."""

    center: np.ndarray          # centre of mass of the lobe
    normal: np.ndarray          # interface normal (thin axis), sign unset
    volume: float               # mm³ of shared material


@dataclass
class Overlap:
    """All the interpenetration between one pair of chains."""

    i: int
    j: int
    solid: object = None        # the intersection manifold
    volume: float = 0.0
    pieces: List[OverlapPiece] = field(default_factory=list)
    #: Bounding box of every connected lobe, *unfiltered* — the carve has to
    #: cover all of them, including the slivers that are too small to be worth
    #: offering as a connector seat.
    boxes: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"a": self.i, "b": self.j, "volume_mm3": round(self.volume, 2),
                "lobes": len(self.pieces)}


def _bounds(man):
    """``(lo, hi)`` of a manifold's bounding box, or ``None`` if unavailable."""
    try:
        bb = man.bounding_box()
        return np.array(bb[:3], float), np.array(bb[3:], float)
    except Exception:
        return None


def _boxes_disjoint(a, b) -> bool:
    """True if two ``(lo, hi)`` boxes provably cannot intersect."""
    if a is None or b is None:
        return False            # unknown: fall through to the real boolean
    return bool(np.any(a[1] < b[0]) or np.any(b[1] < a[0]))


def pair_overlaps(mans, want_pieces: bool = True,
                  want_boxes: Optional[bool] = None) -> List[Overlap]:
    """Every interpenetrating pair among ``mans``, with its shared solid.

    ``want_boxes`` asks for each lobe's bounding box without the rest of the
    measurement — that is what the carve localises on, and it is cheap.
    Defaults to following ``want_pieces``, which is what every existing caller
    expects.

    ``want_pieces`` decomposes each overlap into connected lobes and measures
    them, which is what the connector pass uses to seat magnets exactly where
    the parts currently collide — the natural joint positions that the old
    nearest-point search could never find (an unsigned distance reads a deeply
    buried vertex as *far away*, so the deepest contact scored worst).

    Pairs whose bounding boxes do not touch are skipped without a boolean.  That
    is free correctness — two solids inside disjoint boxes cannot share a point —
    and it stops the quadratic cost from mattering.  It matters now because
    ligands multiplied the object count: a haemoglobin with its four haems is 8
    objects, so 28 pairs instead of 6, and all but a handful are a protein and a
    drug at opposite ends of the model.
    """
    # Boxes default to following want_pieces, so a caller that never asked
    # sees exactly what it always did.
    boxes_wanted = want_pieces if want_boxes is None else bool(want_boxes)
    out: List[Overlap] = []
    n = len(mans)
    boxes = [_bounds(m) for m in mans]
    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_disjoint(boxes[i], boxes[j]):
                continue
            try:
                solid = _manifold.intersection(mans[i], mans[j])
            except Exception:
                continue
            if solid is None or solid.is_empty():
                continue
            ov = Overlap(i=i, j=j, solid=solid, volume=_volume(solid))
            if ov.volume <= 0.0:
                continue
            if want_pieces:
                ov.pieces, ov.boxes = _measure_pieces(solid)
            elif boxes_wanted:
                ov.boxes = _lobe_boxes(solid)
            out.append(ov)
    return out


def _lobe_boxes(solid) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Just the bounding box of every lobe — no centroids, no principal axes.

    This is the half of :func:`_measure_pieces` the *carve* needs.  The other
    half — a centroid and a thin axis per lobe, each costing a mesh conversion
    and an SVD — is for the connector seat search, and the two were tied to one
    flag.  Switching the expensive measurement off therefore also emptied
    ``ov.boxes``, and :func:`_carve_tool` silently lost its entire localisation:
    with no boxes it falls back to a single box around the whole overlap, which
    its own docstring explains saves nothing on a duplex.  The local solid then
    grows from a few thousand triangles toward the whole chain, and the dilation
    with it — measured 250ms against 2,750ms.
    """
    try:
        chunks = solid.decompose() or [solid]
    except Exception:
        chunks = [solid]
    boxes: List[Tuple[np.ndarray, np.ndarray]] = []
    for chunk in chunks:
        if chunk is None or chunk.is_empty():
            continue
        if _volume(chunk) <= 0.0:
            continue
        try:
            bb = chunk.bounding_box()
            boxes.append((np.array(bb[:3], float), np.array(bb[3:], float)))
        except Exception:
            pass
    return boxes


def _measure_pieces(solid):
    """``(seat candidates, every lobe's bounding box)`` for one overlap solid."""
    try:
        chunks = solid.decompose() or [solid]
    except Exception:
        chunks = [solid]
    sized: List[Tuple[float, object]] = []
    boxes: List[Tuple[np.ndarray, np.ndarray]] = []
    for chunk in chunks:
        if chunk is None or chunk.is_empty():
            continue
        vol = _volume(chunk)
        if vol <= 0.0:
            continue
        try:
            bb = chunk.bounding_box()
            boxes.append((np.array(bb[:3], float), np.array(bb[3:], float)))
        except Exception:
            pass
        sized.append((vol, chunk))
    if not sized:
        return [], boxes
    # Rank on volume, then measure only what survives.  Volume comes straight
    # from the kernel; a centroid and a principal axis cost a mesh conversion and
    # an SVD each.  The two lines below have always thrown most of that away — a
    # 50-base-pair duplex interferes at every rung, so ~50 lobes were measured in
    # full and 12 were kept.
    sized.sort(key=lambda t: -t[0])
    cutoff = sized[0][0] * _PIECE_MIN_FRACTION
    measured: List[OverlapPiece] = []
    for vol, chunk in sized[:_MAX_PIECES]:
        if vol < cutoff:
            break
        center, normal = _centroid_and_extent(chunk)
        if center is None:
            continue
        measured.append(OverlapPiece(center=center, normal=normal, volume=vol))
    return measured, boxes


# --------------------------------------------------------------------------
# Carving
# --------------------------------------------------------------------------
def _carve_target(chain_a: Chain, chain_b: Chain, vol_a: float, vol_b: float,
                  rule: InterferenceRule) -> str:
    """Which side gives up the shared volume: ``"a"``, ``"b"`` or ``"both"``."""
    a_lig = chain_a.mtype == MoleculeType.LIGAND
    b_lig = chain_b.mtype == MoleculeType.LIGAND
    # A ligand is exempt from SYMMETRIC too, and not as a convenience: symmetric
    # retreat means "both parts give up half the shared volume", and a ligand
    # sitting *inside* its host shares essentially its whole volume, so its half
    # is a bite out of a 20-atom molecule.  The socket-shaped-like-the-drug result
    # is the only useful one, so the rule that produces it is not optional.  Two
    # ligands against each other is not a real configuration (they would be one
    # residue), and if it happens the larger still wins below.
    if a_lig != b_lig:
        return "b" if a_lig else "a"
    if rule == InterferenceRule.SYMMETRIC:
        return "both"
    a_nuc = chain_a.mtype == MoleculeType.NUCLEIC
    b_nuc = chain_b.mtype == MoleculeType.NUCLEIC
    if a_nuc and not b_nuc:
        return "b"          # nucleic keeps its shape; carve the protein
    if b_nuc and not a_nuc:
        return "a"
    return "a" if vol_a < vol_b else "b"     # same type: the larger one wins


def _box(lo, hi):
    """An axis-aligned box manifold spanning ``lo``..``hi``."""
    size = np.asarray(hi, float) - np.asarray(lo, float)
    center = 0.5 * (np.asarray(hi, float) + np.asarray(lo, float))
    return _manifold.Manifold.cube(tuple(size), center=True).translate(tuple(center))


#: Ceiling on how many separate regions one carve is split into.  Each region
#: costs three booleans, so past a handful the per-region saving is swamped;
#: adjacent lobes are merged until the count is under this.
_MAX_CARVE_REGIONS = 4

#: How far past the interference the carve region reaches (mm).  This is not
#: slack — it is the difference between "remove the shared volume" and "apply the
#: clearance".  Two parts come within the clearance of each other over a patch
#: much wider than the volume they actually share, and the thin features on that
#: flank are exactly what fouls a joint later: trim them and the connector seats;
#: leave them and the socket's approach cut has to sever them instead, which the
#: watertight gate then refuses.  Sized to a typical magnet radius.
_REGION_MARGIN_MM = 3.0


def _merge_boxes(boxes, pad: float, limit: int):
    """Fuse boxes that are within ``pad`` of each other, down to ``limit`` of them.

    A DNA duplex interferes at every base pair, so the raw lobe list is long and
    strung out along the helix — and a box per rung is the wrong trade, because
    each region carries its own clip/grow/clip and the overhead beats the saving.
    Merging first collapses that stripe into one region while still keeping two
    genuinely separate binding sites apart, which is where the locality actually
    pays.
    """
    merged = [(lo.copy(), hi.copy()) for lo, hi in boxes]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        out = []
        for lo, hi in merged:
            for k, (olo, ohi) in enumerate(out):
                if np.all(lo - pad <= ohi) and np.all(hi + pad >= olo):
                    out[k] = (np.minimum(olo, lo), np.maximum(ohi, hi))
                    changed = True
                    break
            else:
                out.append((lo, hi))
        merged = out
    while len(merged) > limit:
        # Still too many: fold the two closest together and try again.
        best, pair = np.inf, (0, 1)
        for a in range(len(merged)):
            for b in range(a + 1, len(merged)):
                ca = 0.5 * (merged[a][0] + merged[a][1])
                cb = 0.5 * (merged[b][0] + merged[b][1])
                d = float(np.linalg.norm(ca - cb))
                if d < best:
                    best, pair = d, (a, b)
        a, b = pair
        merged[a] = (np.minimum(merged[a][0], merged[b][0]),
                     np.maximum(merged[a][1], merged[b][1]))
        merged.pop(b)
    return merged


#: How many times :func:`_carve_tool` fell back to dilating a whole chain.
#:
#: Each of those is a six-copy union of a full-size solid -- measured at 2.75s on
#: a 70k-triangle chain, against ~250ms for the localised tool it stands in for
#: -- and it happens inside a swallowed exception, so today it costs three
#: seconds with no note, no warning and nothing in the report.  If this is ever
#: non-zero there is a bigger problem in this module than anything else in it.
CARVE_FALLBACKS = 0


def _carve_fallback(why: str):
    """Count and announce a whole-chain dilation. Returns nothing."""
    global CARVE_FALLBACKS
    CARVE_FALLBACKS += 1
    _LOG.warning("interference: carve fell back to dilating a whole chain "
                 "(%s); this is slow and should not happen", why)


def _carve_tool(keeper, ov: "Overlap", amount: float):
    """The cutting tool for one carve: ``keeper`` grown by ``amount``, localised.

    Growing a whole chain and subtracting it is correct but slow, and slow for a
    specific reason: two molecular surfaces that bind run *parallel and close*
    over a wide patch, which is the worst case for a boolean kernel — it is the
    configuration with the most near-coincident faces to resolve.  Over the full
    length of a DNA duplex that costs seconds per pair.

    Nothing outside the interference needs cutting, so the work is confined to
    the lobes.  Per *lobe*, not per interface: a duplex interferes at every base
    pair, so a single box around the whole overlap would still span the entire
    helix and save nothing, while a box per rung is genuinely small.

    Each region is handled the same way — clip the keeper to a generous box (so
    the grown result is still correct at the boundary), grow it there, then clip
    back to a tight one.  Inside the tight box that is exactly the full
    dilation; outside it the tool is empty.  Every lobe of shared volume lies
    inside its own tight box by construction, so the fit guarantee is untouched;
    all that is skipped is clearance in places the parts never contested.
    """
    tight = max(amount, 0.0) + _REGION_MARGIN_MM
    loose = tight + max(amount, 0.0) + 1.0

    regions = ov.boxes
    if regions:
        regions = _merge_boxes(regions, 2.0 * loose, _MAX_CARVE_REGIONS)
    if not regions:
        try:
            bb = ov.solid.bounding_box()
            regions = [(np.array(bb[:3], float), np.array(bb[3:], float))]
        except Exception:
            _carve_fallback("no bounding box for the overlap")
            return dilate(keeper, amount)
    tools = []
    for lo, hi in regions:
        try:
            local = _manifold.intersection(keeper, _box(lo - loose, hi + loose))
            if local.is_empty():
                continue
            tools.append(_manifold.intersection(dilate(local, amount),
                                                _box(lo - tight, hi + tight)))
        except Exception:
            _carve_fallback("localised tool failed")
            return dilate(keeper, amount)
    if not tools:
        return None
    try:
        return _manifold.union(tools)
    except Exception:
        _carve_fallback("union of localised tools failed")
        return dilate(keeper, amount)


def _subtract(man, tool, allow_split: bool = True):
    """``man - tool``, or ``None`` if the cut is not one we are willing to make.

    On the main pass a carve that *splits* a part is allowed through: a thin
    protein loop that a DNA duplex genuinely threads has to come apart
    somewhere, and dropping the offcut silently would be worse than reporting a
    two-piece object.  Only annihilation is rejected.

    On the closing sweep it is not.  By then the connectors are in, the
    remaining overlaps are sub-millimetre slivers, and cutting a part in half to
    chase one is strictly worse than leaving it: it would sever the very joint
    that was just seated.
    """
    if tool is None or tool.is_empty():
        return man
    try:
        out = _manifold.difference(man, tool)
    except Exception:
        return None
    if out is None or out.is_empty():
        return None
    if not allow_split and _components(out) > _components(man):
        return None
    return out


def resolve(mans, chains: List[Chain], params: PrintParams,
            overlaps: Optional[List[Overlap]] = None,
            allow_split: bool = True, want_pieces: bool = True,
            want_boxes: Optional[bool] = None):
    """Make every solid in ``mans`` disjoint from the others.

    Returns ``(new_mans, overlaps, notes)`` where ``overlaps`` are the overlaps
    that were found *before* carving — the connector pass wants those, because
    the volume that had to be removed is exactly where a magnet belongs — and
    ``notes`` are human-readable lines for the build report.
    """
    rule = params.resolve_interference
    if rule == InterferenceRule.NONE or len(mans) < 2:
        return mans, [], []

    if overlaps is None:
        # ``want_pieces`` is measurement for the *connector* search — a centroid
        # and a principal axis per lobe, each needing a mesh conversion and an
        # SVD.  The closing sweep only carves; it has no seats left to place, so
        # asking for that is pure waste and on a five-chain complex it was
        # seconds of it.
        overlaps = pair_overlaps(mans, want_pieces=want_pieces,
                                 want_boxes=want_boxes)
    if not overlaps:
        return mans, [], []

    clearance = max(0.0, float(params.fit_clearance_mm))
    mans = list(mans)
    # The tool for every carve is built from the solids as they were *before*
    # this pass, so a chain touching two neighbours is not carved against an
    # already-carved partner (which would leave the second interface short).
    original = list(mans)
    notes: List[str] = []

    # Volume is a pure function of the *original* solid and a chain is often
    # weighed against several neighbours, so it is computed once per chain.
    _volumes: dict = {}

    def vol(idx: int) -> float:
        if idx not in _volumes:
            _volumes[idx] = _volume(original[idx])
        return _volumes[idx]

    # Same argument for the piece count, which was not being cached at all: four
    # full-chain decompose() calls per reported overlap, and the ones on
    # ``original`` were recomputed from scratch for every overlap that chain took
    # part in.  A decompose on a 70k-triangle chain is ~96ms, so a handful of
    # overlaps was seconds of work to decorate a sentence.  The current solids
    # are keyed on identity and the object is held alongside the count, so a
    # carve (which replaces the object) invalidates its own entry and an id
    # cannot be reused while the entry is live.
    _orig_comp: dict = {}
    _cur_comp: dict = {}

    def orig_components(idx: int) -> int:
        if idx not in _orig_comp:
            _orig_comp[idx] = _components(original[idx])
        return _orig_comp[idx]

    def cur_components(idx: int) -> int:
        cached = _cur_comp.get(idx)
        if cached is not None and cached[0] is mans[idx]:
            return cached[1]
        count = _components(mans[idx])
        _cur_comp[idx] = (mans[idx], count)
        return count

    for ov in overlaps:
        i, j = ov.i, ov.j
        target = _carve_target(chains[i], chains[j], vol(i), vol(j), rule)
        label = f"{chains[i].label()} ↔ {chains[j].label()}"

        if target == "both":
            half = clearance / 2.0
            new_i = _subtract(mans[i], _carve_tool(original[j], ov, half),
                              allow_split)
            new_j = _subtract(mans[j], _carve_tool(original[i], ov, half),
                              allow_split)
            if new_i is None or new_j is None:
                notes.append(f"Interference at {label} left as-is: carving it out "
                             f"would have destroyed an object.")
                continue
            mans[i], mans[j] = new_i, new_j
            who = "both parts trimmed"
        else:
            cut, keep = (i, j) if target == "a" else (j, i)
            new = _subtract(mans[cut],
                            _carve_tool(original[keep], ov, clearance),
                            allow_split)
            if new is None:
                notes.append(f"Interference at {label} left as-is: carving it out "
                             f"would have destroyed or split {chains[cut].label()}.")
                continue
            mans[cut] = new
            who = f"{chains[cut].label()} carved to fit {chains[keep].label()}"

        if ov.volume < _REPORT_MIN_MM3:
            continue                      # carved, but too small to be news
        extra = ""
        if cur_components(i) > orig_components(i) or \
                cur_components(j) > orig_components(j):
            extra = " — this split a part into separate pieces"
        notes.append(f"Interference at {label}: {ov.volume:.1f} mm³ shared, "
                     f"{who}{extra}.")

    return mans, overlaps, notes


def audit(mans, chains: List[Chain]) -> List[str]:
    """Report any pair still sharing space — the honest end-of-pipeline check.

    Interference resolution is exact when it runs on the raw chain solids: the
    exported parts are disjoint, full stop.  Connectors are added afterwards
    though, and a collar driven through a thin backbone into a neighbour can
    leave interference that the closing sweep will not remove, because removing
    it would cut a part in two.  That is a real "these will not fit" condition
    and it belongs in the user's face rather than in a comment.
    """
    notes: List[str] = []
    for ov in pair_overlaps(mans, want_pieces=False):
        if ov.volume < _REPORT_MIN_MM3:
            continue
        notes.append(
            f"{chains[ov.i].label()} and {chains[ov.j].label()} still share "
            f"{ov.volume:.1f} mm³ after connecting — a connector was driven "
            f"through one of them into the other, and removing it would have "
            f"split a part. These two will not close fully; try a smaller "
            f"connector Ø, fewer joints on this interface, or a thicker backbone."
        )
    return notes


def residual_overlap(mans) -> float:
    """Total remaining shared volume across all pairs — 0.0 means printable.

    Used by the tests and by the build report as the honest end-to-end check
    that the parts really will fit together.
    """
    return float(sum(ov.volume for ov in pair_overlaps(mans, want_pieces=False)))
