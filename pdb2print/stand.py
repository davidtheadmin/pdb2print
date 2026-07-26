"""Display stand: a base plate, columns into a cradle, and an engraved apron.

This pass runs **after** a build, not during one.  It takes the finished chain
meshes and the orientation the user locked in the viewer, and returns that model
rotated upright plus a handful of new objects to print beside it.

    base plate  ── rounded slab, sized to the model's footprint, extended at
                   the front into a clear apron
    columns     ── flared uprights whose tops are carved into a cradle by the
                   model itself, so it seats in one position and no other
    plaque      ── raised lettering lying flat on that apron: structure and
                   scale at the left, a colour-matched chain legend at the right

**Why the model gets rotated.**  The stand is defined by which way is down, and
which way is down is whatever the user was looking at.  Rather than build a
tilted stand around a fixed model, the whole assembly is rotated so the plate
lands flat on ``z = 0``.  The file then arrives in the slicer already oriented
to print, and the preview shows it standing the way it will stand on a desk.

**Why the cradle is carved rather than modelled.**  The top of a column has to
match a molecular surface, which has no describable shape.  Subtracting the
model — grown by a clearance first, exactly as the magnet pockets and the
interference pass already do — gives a seat that fits that surface and nothing
else.  It is the same trick, pointed downward.

**Why the plaque is an apron rather than a wall.**  Lettering on an upright
panel has to be tall enough to read from the front, which on a short model means
a label taller than the thing it labels.  Flat on the plate it can be as wide as
the plate is, reads from the natural viewing angle looking down, and prints with
no overhang at all.  Putting it *in front of* the model's footprint rather than
under it is also what keeps it clear of the columns by construction.

Everything is built from the analytic primitives in
:mod:`pdb2print.representations._manifold` and fused with manifold booleans, so
every part comes out watertight by construction, like the rest of the pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import trimesh

from . import strokefont
from .config import (
    ColumnShape, MoleculeType, PrintParams, StandParams, color_for_index,
)
from .representations import _manifold


# --------------------------------------------------------------------------
# Coordinate conventions
# --------------------------------------------------------------------------
#: The up axis of the space the GLB is viewed in.
#:
#: ``<model-viewer>`` orbits about glTF's +Y, and trimesh writes the scene's
#: coordinates into the GLB without an axis conversion, so the camera angles
#: coming back from the browser are expressed in *these* coordinates — the same
#: ones the meshes are already in.  If a future trimesh release starts rotating
#: on export, this constant and nothing else is what has to change.
VIEW_UP = np.array([0.0, 1.0, 0.0])

#: Colour of the stand itself.  A neutral graphite: it has to read as furniture
#: rather than as part of the molecule, and it is what the eye discounts.
STAND_COLOR = (0.33, 0.35, 0.38)
#: Colour of the white backing tiles the lettering sits on.
TILE_COLOR = (0.95, 0.95, 0.94)
#: Lettering on a white tile: the plate's own graphite, for contrast.
TEXT_ON_TILE = STAND_COLOR
#: Lettering straight onto the plate, with no tile behind it.
TEXT_ON_PLATE = (0.93, 0.94, 0.95)

#: Nice round numbers a scale bar is allowed to be, in ångström.  A bar reading
#: "37 Å" is a measurement; one reading "50 Å" is a scale.
_SCALEBAR_STEPS_ANG = (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000)

#: The apron is never shallower than this (mm), so there is always somewhere for
#: the lettering to go even on a small model.
_APRON_MIN_MM = 22.0

#: The most the lettering may be shrunk to fit.  Below this a stroke drops under
#: one extrusion width and prints as a smudge, so the apron is deepened instead.
_TEXT_MIN_SHRINK = 0.62

#: Smallest cap height (mm) worth setting a one-line title at.  Below this the
#: title stops being small and starts being unprintable, and two balanced lines
#: at a readable size beat one line nobody can read.
_TITLE_MIN_CAP_MM = 1.9


# --------------------------------------------------------------------------
# Stand parts travel through export as if they were chains
# --------------------------------------------------------------------------
@dataclass
class StandPart:
    """One printable piece of the stand, duck-typed to pass for a ``Chain``.

    The exporters ask an object for :meth:`label` and :meth:`object_name` and
    read ``mtype``; nothing downstream of the build needs atoms.  Matching that
    small interface is what lets a stand piece be appended to ``report.built``
    and picked up by the 3MF, STL and GLB writers with no special case in any of
    them — the same move that lets a ligand ride through as a ``Chain``.

    ``color`` is carried rather than derived, because a legend row has to come
    out in *its chain's* colour and the palette is indexed by position.
    """

    part: str                       # "base" | "text" | "legend"
    name: str
    color: Tuple[float, float, float]
    chain_id: str = "-"
    mtype: MoleculeType = MoleculeType.STAND
    res_name: Optional[str] = None
    res_id: Optional[int] = None

    is_ligand: bool = False
    n_atoms: int = 0
    n_residues: int = 0

    def label(self) -> str:
        """Unique id, and the STL filename inside the zip — must not collide."""
        suffix = "" if self.chain_id in ("-", "") else f"_{self.chain_id}"
        return f"stand_{self.part}{suffix}"

    def display_name(self) -> str:
        return self.name

    def object_name(self) -> str:
        """What PrusaSlicer shows in its object list.

        Prefixed ``stand_`` so every piece of the furniture sorts together and
        away from the molecule, the same reason ligands are prefixed.
        """
        return self.name


# --------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------
def view_basis(theta_deg: float, phi_deg: float):
    """``(right, forward, up)`` unit vectors for a model-viewer camera orbit.

    ``theta`` is the azimuth about +Y and ``phi`` the polar angle from +Y, which
    is model-viewer's own ``camera-orbit`` convention.  ``forward`` points from
    the camera toward the model, so ``up`` is *screen* up — and screen up is the
    whole input to this feature: it is what the user set when they turned the
    model to the pose they wanted to display it in.
    """
    theta = math.radians(float(theta_deg))
    phi = math.radians(float(phi_deg))
    position = np.array([
        math.sin(phi) * math.sin(theta),
        math.cos(phi),
        math.sin(phi) * math.cos(theta),
    ])
    norm = float(np.linalg.norm(position))
    if norm < 1e-9:
        position = np.array([0.0, 0.0, 1.0])
        norm = 1.0
    forward = -position / norm

    right = np.cross(forward, VIEW_UP)
    if float(np.linalg.norm(right)) < 1e-6:
        # Looking straight down the up axis: the azimuth alone fixes the roll,
        # which would otherwise be undefined and snap unpredictably.
        right = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    right = right / float(np.linalg.norm(right))
    up = np.cross(right, forward)
    up = up / float(np.linalg.norm(up))
    return right, forward, up


def stand_rotation(stand: StandParams) -> np.ndarray:
    """3x3 rotation from model space into stand space.

    Stand space is ``+X`` right, ``+Y`` away from the viewer, ``+Z`` up, so the
    base plate lies in the XY plane and the apron sits at low ``Y`` — the near
    edge, straight out of the screen the user aimed the model with.

    ``roll_deg`` then spins that frame about the view axis, which is the degree
    of freedom an orbit camera does not have.  Positive is clockwise as seen on
    screen, matching the CSS rotation the viewer applies to the picture, so the
    two stay in step: what the user sees rolled is what gets built rolled.
    """
    right, forward, up = view_basis(stand.orbit_theta_deg, stand.orbit_phi_deg)
    roll = math.radians(float(getattr(stand, "roll_deg", 0.0) or 0.0))
    if abs(roll) > 1e-9:
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        # Rotating the picture clockwise by θ means the world direction now at
        # screen-up is the one that used to sit θ anticlockwise of it.
        right, up = (cos_r * right + sin_r * up,
                     cos_r * up - sin_r * right)
    return np.array([right, forward, up])


#: Stand space (Z up, for the slicer) expressed in glTF's frame (Y up).
#:
#: The stand is built Z-up because that is what a slicer and a print bed mean by
#: up.  glTF — and therefore ``<model-viewer>``, which orbits about +Y — means
#: something else by it.  Exporting the Z-up geometry straight to GLB puts the
#: stand on its side in the preview, and worse, silently changes what the camera
#: controls do: the orbit's poles no longer line up with the model's own up, so
#: the polar sweep runs out after a quarter turn of the *stand* instead of a
#: half turn.  The print files stay Z-up; only the preview is converted.
_Z_UP_TO_Y_UP = np.array([
    [1.0, 0.0, 0.0],     # x stays x
    [0.0, 0.0, 1.0],     # glTF y  <- stand z   (up)
    [0.0, -1.0, 0.0],    # glTF z  <- -stand y  (toward the viewer)
])


def to_view_frame(built):
    """``built`` rotated from stand space (Z up) into glTF's frame (Y up).

    Preview only.  Returns new ``(object, mesh)`` pairs; the originals — the ones
    the 3MF and STLs are written from — are left in the frame the print needs.
    """
    return [(obj, _apply_rotation(mesh, _Z_UP_TO_Y_UP, np.zeros(3)))
            for obj, mesh in built]


def _apply_rotation(mesh: trimesh.Trimesh, rotation: np.ndarray,
                    offset: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.vertices = out.vertices @ rotation.T + offset
    return out


# --------------------------------------------------------------------------
# Column siting
# --------------------------------------------------------------------------
def recommend_columns(meshes: Sequence[trimesh.Trimesh]) -> int:
    """A sensible default column count for a model of this size.

    Driven by the footprint's longest span, because that is what decides whether
    the model can rock: a compact blob is stable on one column, and a long
    complex levers itself off anything less than a pair.  Three is the ceiling —
    beyond it the columns start competing for the same underside and the extra
    one hides behind the others anyway.
    """
    if not meshes:
        return 1
    lows = np.array([m.bounds[0] for m in meshes])
    highs = np.array([m.bounds[1] for m in meshes])
    span = float(max((highs.max(axis=0) - lows.min(axis=0))[:2]))
    if span < 45.0:
        return 1
    if span < 110.0:
        return 2
    return 3


@dataclass
class _Candidate:
    """One possible column top: a point on a chain's underside."""

    point: np.ndarray
    index: int              # which built object it belongs to
    downness: float         # -normal_z, 1.0 = facing straight down
    mtype: Optional[MoleculeType] = None


def _underside_candidates(meshes: Sequence[trimesh.Trimesh],
                          stand: StandParams,
                          cell_mm: float,
                          mtypes: Optional[Sequence] = None) -> List[_Candidate]:
    """Points on the models' undersides where a column could land.

    Vertices rather than ray casts, deliberately.  A ray cast needs an
    acceleration structure that is not a hard dependency here (and blows up the
    memory on a surface mesh), while the vertices *are* the surface and already
    carry normals.  Two filters do the work here:

    * the surface must face downward — ``stand.column_normal_min`` — or the
      cradle is cut into a near-vertical wall, where it becomes a knife edge
      that neither prints nor holds the model;
    * only the lowest vertex in each grid cell is kept, so the candidate set is
      the silhouette of the underside rather than every vertex on it.

    A third and more important filter, that nothing at all lies below the
    candidate, is applied afterwards by :func:`_drop_obstructed`.
    """
    out: List[_Candidate] = []
    for idx, mesh in enumerate(meshes):
        try:
            normals = mesh.vertex_normals
        except Exception:
            continue
        verts = np.asarray(mesh.vertices, float)
        if len(verts) == 0:
            continue
        downness = -np.asarray(normals, float)[:, 2]
        keep = downness >= float(stand.column_normal_min)
        if not np.any(keep):
            continue
        pts = verts[keep]
        down = downness[keep]

        cells = np.floor(pts[:, :2] / max(cell_mm, 1e-6)).astype(np.int64)
        best: dict = {}
        for i in range(len(pts)):
            key = (int(cells[i, 0]), int(cells[i, 1]))
            prior = best.get(key)
            if prior is None or pts[i, 2] < pts[prior, 2]:
                best[key] = i
        mtype = mtypes[idx] if mtypes is not None and idx < len(mtypes) else None
        for i in best.values():
            out.append(_Candidate(point=pts[i], index=idx,
                                  downness=float(down[i]), mtype=mtype))
    return out


def _prefer_protein(candidates: List[_Candidate],
                    minimum: int = 1) -> List[_Candidate]:
    """Keep only protein candidates when there are usably many of them.

    A nucleic backbone is a thin swept tube. A column meeting one touches a
    cylinder along a tangent: the cradle either grips a sliver or, cut deep
    enough to grip, swallows the strand and hides the thing being displayed.
    Protein presents a broad surface a seat can sit in, so in a mixed structure
    the protein carries the model.

    Falls back to the full set when the protein does not offer enough — a stand
    that exists on the DNA beats no stand at all.
    """
    protein = [c for c in candidates if c.mtype == MoleculeType.PROTEIN]
    return protein if len(protein) >= max(1, minimum) else candidates


def footprint_hull(meshes: Sequence[trimesh.Trimesh]):
    """The model's plan-view silhouette as an ordered ``(N, 2)`` hull, or ``None``.

    This is the shape you see looking straight down at the model, and it is the
    thing a column should sit *inside* rather than on the rim of.
    """
    from scipy.spatial import ConvexHull

    pts = [np.asarray(m.vertices, float)[:, :2] for m in meshes
           if len(m.vertices)]
    if not pts:
        return None
    xy = np.vstack(pts)
    if len(xy) < 3:
        return None
    try:
        return xy[ConvexHull(xy).vertices]
    except Exception:
        return None


def _distance_inside(point_xy: np.ndarray, hull: np.ndarray) -> float:
    """How far an interior point sits from the nearest edge of a convex hull."""
    best = None
    count = len(hull)
    for i in range(count):
        a = hull[i]
        b = hull[(i + 1) % count]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        # On a convex hull an interior point is on the inner side of every edge,
        # so the perpendicular distance to the edge *line* is the real clearance.
        gap = abs(_cross2(edge / length, point_xy - a))
        best = gap if best is None else min(best, gap)
    return float(best or 0.0)


def _drop_edge_candidates(candidates: List[_Candidate], hull, cap_mm: float,
                          edge_frac: float, minimum: int) -> List[_Candidate]:
    """Keep candidates that sit well inside the model's plan-view silhouette.

    Measured against the **footprint**, not against distance from the centre.
    Those are the same thing only for a round model: on anything elongated — a
    duplex, a long complex — a radial rule keeps points that are near the middle
    lengthways while still hanging off the narrow sides, which is exactly the
    perched look it was supposed to prevent. Distance to the silhouette edge is
    the quantity that was actually meant all along.

    The threshold is **relative to what this model actually offers**, capped by
    an absolute one.  A fixed millimetre inset cannot serve both ends: on a large
    complex it is a rounding error, and on a small one it exceeds the whole
    underside, so every candidate fails and the columns end up wherever the
    fallback puts them — with the spread destroyed and nothing gained.  Taking a
    fraction of the deepest available inset asks the same question of every
    model: *is this point near the middle of what is on offer, or out on the
    rim?*

    When nothing clears the threshold the most inboard candidates are taken
    anyway: a column somewhat too close to the edge still beats no column.
    """
    if hull is None or not candidates:
        return candidates
    gaps = np.array([_distance_inside(c.point[:2], hull) for c in candidates])
    inset_mm = min(float(edge_frac) * float(gaps.max()), float(cap_mm))
    kept = [c for c, g in zip(candidates, gaps) if g >= inset_mm]
    if len(kept) >= minimum:
        return kept
    # Nothing clears the bar — a small model, or a pose with a narrow footprint.
    # Keep the most inboard third rather than exactly ``minimum``: cutting to the
    # bare count leaves the spread search with no choices to make, and three
    # columns picked from three candidates are wherever those three happen to be.
    take = max(minimum, int(np.ceil(len(candidates) * 0.35)))
    order = np.argsort(gaps)[::-1][:take]
    return [candidates[int(i)] for i in order]


def _drop_obstructed(candidates: List[_Candidate],
                     meshes: Sequence[trimesh.Trimesh],
                     footprint_radius: float,
                     tolerance: float = 1.0) -> List[_Candidate]:
    """Keep only candidates a column can actually *reach* from the plate.

    This is the filter that stops a column spearing the model.  A candidate is
    chosen for facing downward, but facing downward says nothing about what is
    underneath it: the underside of a chain sitting high in the structure is a
    perfectly good downward-facing surface with, very often, another whole chain
    between it and the plate.  A column raised to meet it goes straight through
    that chain — entering somewhere nobody looks, and welding two subunits
    together in a place the structure does not justify.

    So the candidate must be the lowest material anywhere within the column's
    own footprint, across **every** object.  That is a 2D neighbourhood query,
    not a ray cast: everything within ``footprint_radius`` horizontally has to
    be at or above the candidate, give or take ``tolerance``.
    """
    from scipy.spatial import cKDTree

    all_xy = []
    all_z = []
    for mesh in meshes:
        verts = np.asarray(mesh.vertices, float)
        if len(verts):
            all_xy.append(verts[:, :2])
            all_z.append(verts[:, 2])
    if not all_xy:
        return []
    all_xy = np.vstack(all_xy)
    all_z = np.concatenate(all_z)
    tree = cKDTree(all_xy)

    kept = []
    for cand in candidates:
        near = tree.query_ball_point(cand.point[:2], footprint_radius)
        if not near:
            kept.append(cand)
            continue
        if float(all_z[near].min()) >= float(cand.point[2]) - tolerance:
            kept.append(cand)
    return kept


def _drop_shared_candidates(candidates: List[_Candidate],
                            meshes: Sequence[trimesh.Trimesh],
                            clear_radius: float) -> List[_Candidate]:
    """Keep only candidates where a column would *touch* exactly one object.

    Separate from :func:`_drop_obstructed`, which is about what a column passes
    through on the way up; this is about what it lands on.  A column bridging the
    seam between two chains fuses them at a point the structure does not justify.
    """
    from scipy.spatial import cKDTree

    trees = []
    for mesh in meshes:
        verts = np.asarray(mesh.vertices, float)
        trees.append(cKDTree(verts) if len(verts) else None)

    kept = []
    for cand in candidates:
        clash = False
        for idx, tree in enumerate(trees):
            if idx == cand.index or tree is None:
                continue
            if tree.query_ball_point(cand.point, clear_radius, return_length=True):
                clash = True
                break
        if not clash:
            kept.append(cand)
    return kept


def _centre_of_mass_xy(meshes: Sequence[trimesh.Trimesh]) -> np.ndarray:
    """Volume-weighted centre of the whole model, projected to the plate."""
    total = 0.0
    accum = np.zeros(3)
    for mesh in meshes:
        try:
            volume = float(abs(mesh.volume))
            centre = np.asarray(mesh.center_mass, float)
        except Exception:
            continue
        if volume <= 0 or not np.all(np.isfinite(centre)):
            continue
        accum += centre * volume
        total += volume
    if total <= 0:
        allv = np.vstack([m.vertices for m in meshes])
        return allv.mean(axis=0)[:2]
    return (accum / total)[:2]


def _choose_columns(candidates: List[_Candidate], count: int,
                    centre_xy: np.ndarray) -> List[_Candidate]:
    """Pick ``count`` column sites: spread wide, but under the centre of mass.

    Stability is the whole criterion, and it has two halves that pull against
    each other.  Columns want to be **far apart** — the wider the base of
    support, the more torque it takes to tip the model — and they want to
    **straddle the centre of mass**, because a wide pair that both sit on the
    same side of it is a diving board.
    """
    if not candidates or count <= 0:
        return []

    pts = np.array([c.point for c in candidates])
    xy = pts[:, :2]
    # How squarely each candidate faces the plate. A point on the extreme edge
    # of a silhouette is a glancing tangent — the surface is curving away from
    # the column almost as fast as it meets it — so the cradle gets very little
    # to hold and the model looks perched on its own rim. Weighting toward the
    # flatter contact settles the columns inboard without discarding the wide
    # options outright, which a hard radial cut does.
    down = np.array([max(0.0, float(c.downness)) for c in candidates])

    def flatness(idx) -> float:
        return 0.62 + 0.38 * float(np.mean(down[list(idx)]))

    if count == 1:
        # Directly under the centre of mass, and low: a single column is a
        # cantilever and every millimetre of height is leverage against it.
        d = np.linalg.norm(xy - centre_xy, axis=1)
        score = d + 0.35 * (pts[:, 2] - pts[:, 2].min()) + 12.0 * (1.0 - down)
        return [candidates[int(np.argmin(score))]]

    if count == 2:
        order = np.argsort(np.linalg.norm(xy - centre_xy, axis=1))[::-1]
        pool = order[:120]
        best, best_score = None, -1e18
        for ai in range(len(pool)):
            a = xy[pool[ai]]
            for bi in range(ai + 1, len(pool)):
                b = xy[pool[bi]]
                seg = b - a
                length = float(np.linalg.norm(seg))
                if length < 1e-6:
                    continue
                t = float(np.dot(centre_xy - a, seg) / (length * length))
                balance = 1.0 - abs(t - 0.5) * 2.0
                perp = float(np.linalg.norm((centre_xy - a) - seg * t))
                score = ((length + 55.0 * balance - 2.2 * perp)
                         * flatness((pool[ai], pool[bi])))
                if score > best_score:
                    best_score, best = score, (pool[ai], pool[bi])
        if best is None:
            return [candidates[0]]
        return [candidates[int(best[0])], candidates[int(best[1])]]

    if count == 3 and len(candidates) >= 3:
        # Maximise the **area** of the triangle rather than the spacing between
        # its corners.  Farthest-point sampling happily returns three points in
        # a row — each one is a long way from the others, which is exactly what
        # it was asked for — and three columns in a line is a hinge: the model
        # is free to rotate about it and the third column contributes nothing
        # the outer two did not already. Area goes to zero as the points become
        # collinear, so maximising it is the same instruction with the failure
        # mode removed.
        pool = _spread_shortlist(xy, centre_xy, limit=44)
        best, best_score = None, -1e18
        for ai in range(len(pool)):
            a = xy[pool[ai]]
            for bi in range(ai + 1, len(pool)):
                b = xy[pool[bi]]
                for ci in range(bi + 1, len(pool)):
                    c = xy[pool[ci]]
                    area = _triangle_area(a, b, c)
                    if area <= 0.0:
                        continue
                    # A triangle that does not contain the centre of mass leans;
                    # one that does cannot tip without lifting a column.
                    inside = _point_in_triangle(centre_xy, a, b, c)
                    score = (area * (1.0 if inside else 0.45)
                             * flatness((pool[ai], pool[bi], pool[ci])))
                    if score > best_score:
                        best_score, best = score, (pool[ai], pool[bi], pool[ci])
        if best is not None:
            return [candidates[int(i)] for i in best]

    chosen = [int(np.argmin(np.linalg.norm(xy - centre_xy, axis=1)))]
    while len(chosen) < min(count, len(candidates)):
        taken = xy[chosen]
        gaps = np.min(np.linalg.norm(xy[:, None, :] - taken[None, :, :], axis=2),
                      axis=1)
        nxt = int(np.argmax(gaps))
        if nxt in chosen:
            break
        chosen.append(nxt)
    return [candidates[i] for i in chosen]


def _spread_shortlist(xy: np.ndarray, centre_xy: np.ndarray,
                      limit: int) -> List[int]:
    """Indices of up to ``limit`` well-spread candidates, for a brute-force pass.

    Every triple of a few hundred candidates is tens of millions of
    combinations; of a well-spread forty it is under ten thousand. Thinning by
    farthest-point sampling rather than at random keeps the extremes that any
    good triangle would want to use.
    """
    if len(xy) <= limit:
        return list(range(len(xy)))
    picked = [int(np.argmax(np.linalg.norm(xy - centre_xy, axis=1)))]
    while len(picked) < limit:
        taken = xy[picked]
        gaps = np.min(np.linalg.norm(xy[:, None, :] - taken[None, :, :], axis=2),
                      axis=1)
        nxt = int(np.argmax(gaps))
        if nxt in picked:
            break
        picked.append(nxt)
    return picked


def _cross2(u, v) -> float:
    """2D cross product. Spelled out rather than using ``np.cross``, which
    deprecated the 2-vector form in NumPy 2.0 and warns on every call."""
    return float(u[0]) * float(v[1]) - float(u[1]) * float(v[0])


def _triangle_area(a, b, c) -> float:
    return abs(_cross2(b - a, c - a)) * 0.5


def _point_in_triangle(p, a, b, c) -> bool:
    """True if ``p`` lies inside triangle ``abc`` (2D, either winding)."""
    d1 = _cross2(b - a, p - a)
    d2 = _cross2(c - b, p - b)
    d3 = _cross2(a - c, p - c)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


# --------------------------------------------------------------------------
# Primitive helpers
# --------------------------------------------------------------------------
def _rounded_slab(x0: float, x1: float, y0: float, y1: float,
                  z0: float, z1: float, radius: float):
    """A rounded rectangular slab over the given extents.

    Built as the Minkowski sum of a rectangle and a disc — a smaller box grown
    by the corner radius in each direction, plus a cylinder at each corner —
    which is exact and needs no polygon offsetting library.
    """
    half_x = 0.5 * (x1 - x0)
    half_y = 0.5 * (y1 - y0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    radius = max(0.0, min(radius, half_x * 0.9, half_y * 0.9))
    height = max(1e-4, z1 - z0)
    cz = 0.5 * (z0 + z1)
    axes = np.eye(3)
    parts = [
        _manifold.oriented_box([cx, cy, cz], axes,
                               [half_x - radius, half_y, height * 0.5]),
        _manifold.oriented_box([cx, cy, cz], axes,
                               [half_x, half_y - radius, height * 0.5]),
    ]
    if radius > 1e-6:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                parts.append(_manifold.frustum(
                    [cx + sx * (half_x - radius), cy + sy * (half_y - radius), z0],
                    [cx + sx * (half_x - radius), cy + sy * (half_y - radius), z1],
                    radius, radius))
    return _manifold.union(parts)


def _column_solid(x: float, y: float, z0: float, z1: float,
                  top_half: float, foot_half: float, shape: ColumnShape):
    """One column, square or round, flaring from ``foot_half`` to ``top_half``.

    A square column cannot be a single tapered primitive the way a frustum can,
    so the flare is a short plinth at the base rather than a continuous taper.
    That is arguably the better look anyway — it reads as a deliberate foot
    rather than as a cone someone truncated.
    """
    if shape == ColumnShape.ROUND:
        return _manifold.frustum([x, y, z0], [x, y, z1], foot_half, top_half)

    axes = np.eye(3)
    shaft = _manifold.oriented_box(
        [x, y, 0.5 * (z0 + z1)], axes,
        [top_half, top_half, max(1e-4, 0.5 * (z1 - z0))])
    plinth_h = min(max(2.0, (z1 - z0) * 0.16), max(1e-3, (z1 - z0) * 0.5))
    plinth = _manifold.oriented_box(
        [x, y, z0 + plinth_h * 0.5], axes,
        [foot_half, foot_half, plinth_h * 0.5])
    return _manifold.union([shaft, plinth])


def _stroke_solids(polylines, origin: np.ndarray, axis_u: np.ndarray,
                   axis_v: np.ndarray, normal: np.ndarray,
                   width: float, depth: float, sink: float):
    """Sweep 2D centre lines into raised ridges on a plane.

    ``origin``/``axis_u``/``axis_v`` place the font's 2D frame in space and
    ``normal`` is the direction the lettering stands proud.  Each segment
    becomes a box and each vertex a cylinder, so corners are rounded and the
    strokes join without a notch.  ``sink`` is how far the ridge continues
    *behind* the face, which is what lets the plate be cut to receive it.
    """
    half_w = max(width, 1e-3) * 0.5
    solids = []
    for line in polylines:
        pts = [origin + axis_u * float(x) + axis_v * float(y) for (x, y) in line]
        for a, b in zip(pts, pts[1:]):
            delta = b - a
            length = float(np.linalg.norm(delta))
            if length < 1e-7:
                continue
            along = delta / length
            across = np.cross(normal, along)
            across_norm = float(np.linalg.norm(across))
            if across_norm < 1e-9:
                continue
            across = across / across_norm
            centre = 0.5 * (a + b) + normal * (0.5 * (depth - sink))
            solids.append(_manifold.oriented_box(
                centre, np.array([along, across, normal]),
                [length * 0.5, half_w, 0.5 * (depth + sink)]))
        for p in pts:
            solids.append(_manifold.frustum(
                p - normal * sink, p + normal * depth, half_w, half_w))
    return solids


# --------------------------------------------------------------------------
# Plaque content
# --------------------------------------------------------------------------
def _scalebar_length(scale_mm_per_ang: float, max_mm: float):
    """``(angstrom, millimetres)`` for the longest tidy bar that fits ``max_mm``."""
    best = None
    for ang in _SCALEBAR_STEPS_ANG:
        mm = ang * float(scale_mm_per_ang)
        if mm <= max_mm:
            best = (ang, mm)
    if best is None:
        ang = max(1, int(max_mm / max(scale_mm_per_ang, 1e-6)))
        best = (ang, ang * float(scale_mm_per_ang))
    return best


#: Wrappers a header puts round the only informative part of a molecule name.
_NAME_WRAPPER = re.compile(
    r"^\s*(?:protein|polyprotein|peptide|enzyme|molecule)\s*[:(\-]\s*(.+?)\s*\)?\s*$",
    re.I)
#: Bare class words that describe nothing on their own.
_NAME_EMPTY = {"protein", "peptide", "polypeptide", "enzyme", "molecule",
               "chain", "polymer", "unknown", "uncharacterized protein"}


def legend_label(chain, index: int) -> str:
    """A legend row that says something, given a header name that often does not.

    PDB ``COMPND`` records routinely wrap the useful word in a class name —
    ``PROTEIN (ZIF268)`` — or supply only the class, ``PROTEIN``.  Printed
    verbatim on a plaque the first is noise round a name and the second is no
    name at all, and both look like the tool did not know what it was labelling.

    So: unwrap ``Protein (Zif268)`` to ``Zif268``; fall back to the chain id when
    what is left is only a class word; and name a ligand by its CCD code, which
    is the thing someone can actually look up.  The chain id is always appended,
    because on a homodimer the name alone cannot tell the two rows apart.
    """
    cid = str(getattr(chain, "chain_id", "?"))

    if getattr(chain, "mtype", None) == MoleculeType.LIGAND:
        code = getattr(chain, "res_name", None)
        return f"{code} ({cid})" if code else f"Ligand ({cid})"

    name = (getattr(chain, "name", None) or "").strip()
    match = _NAME_WRAPPER.match(name)
    if match:
        name = match.group(1).strip().strip("()")
    if not name or name.strip().lower().rstrip(".") in _NAME_EMPTY:
        mtype = getattr(chain, "mtype", None)
        name = "DNA / RNA" if mtype == MoleculeType.NUCLEIC else f"Chain {cid}"
        return name if name.startswith("Chain") else f"{name} ({cid})"
    return f"{name} ({cid})"


@dataclass
class _Row:
    """One laid-out row of the plaque, in plate-local millimetres."""

    kind: str                      # "text" | "scalebar" | "legend"
    text: str = ""
    cap_mm: float = 4.0
    height_mm: float = 6.0
    bar_mm: float = 0.0
    built_index: int = -1


def _info_rows(stand: StandParams, params: PrintParams, meta: dict,
               width_mm: float) -> List[_Row]:
    """The left-hand block: what this is, and how big it is."""
    rows: List[_Row] = []
    cap = float(stand.plaque_text_mm)

    if stand.plaque_pdb_id and meta.get("pdb_id"):
        text = str(meta["pdb_id"]).upper()
        size = strokefont.fit_cap_height([text], width_mm, cap)
        rows.append(_Row("text", text, size, size * 1.6))

    if stand.plaque_title and meta.get("title"):
        title = str(meta["title"])
        preferred = cap * 0.46
        # One line if it can be had. A title set across three lines competes with
        # the ID above it for the eye; the same words on one line read as a
        # caption to it, which is what they are. Shrinking the type to buy that
        # is worth it down to the point where the strokes stop printing — below
        # that a single line is legible only in the sense that it exists.
        floor = max(_TITLE_MIN_CAP_MM, cap * 0.26)
        single = strokefont.fit_cap_height([title], width_mm, preferred,
                                           min_cap_mm=0.1)
        if single >= floor:
            rows.append(_Row("text", title, single, single * 1.55))
        else:
            # Wrapping is second choice; wrapping *and* truncating is third. Give
            # up a little type size first, because losing the end of a structure
            # name to an ellipsis is losing information, and a tenth of a
            # millimetre of cap height is losing nothing.
            size, lines = preferred, []
            for factor in (1.0, 0.86, 0.74, 0.64):
                size = max(floor, preferred * factor)
                lines = strokefont.wrap_balanced(title, size, width_mm,
                                                 max_lines=3)
                if not any(line.endswith("...") for line in lines):
                    break
            for line in lines:
                rows.append(_Row("text", line, size, size * 1.55))

    if stand.plaque_scalebar:
        size = cap * 0.42
        ang, bar_mm = _scalebar_length(params.scale_mm_per_angstrom, width_mm * 0.5)
        rows.append(_Row("scalebar", f"{ang} Å", size, size * 2.3, bar_mm=bar_mm))
        # The bar shows the size; this line lets someone reproduce it. Both are
        # wanted: the bar is what you hold a ruler against, and the ratio is what
        # you quote when someone asks what scale the model is.
        rows.append(_Row("text", f"1 Å = {params.scale_mm_per_angstrom:g} mm",
                         size, size * 1.6))
    return rows


def _legend_rows(stand: StandParams, chain_rows: List[dict],
                 width_mm: float) -> List[_Row]:
    """The right-hand block: one colour-matched row per printed object."""
    rows: List[_Row] = []
    if not (stand.plaque_legend and chain_rows):
        return rows
    size = float(stand.plaque_text_mm) * 0.42
    # A to Z down the page. The build order is whatever the file listed — polymers
    # then ligands, chains in header order — which is fine for colour assignment
    # and useless for finding a row: someone reading the legend is looking up the
    # letter moulded on the model, so the letters have to be in order.
    for row in sorted(chain_rows, key=lambda r: str(r["chain_id"])):
        label = row["label"]
        available = width_mm - size * 2.2
        if strokefont.text_width(label, size) > available:
            shortened = strokefont.wrap(label, size, available, max_lines=1)
            label = shortened[0] if shortened else label
        rows.append(_Row("legend", label, size, size * 1.85,
                         built_index=row["index"]))
    return rows


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def build_stand(built, params: PrintParams, meta: Optional[dict] = None):
    """Rotate ``built`` upright and generate the stand around it.

    Returns ``(oriented_built, stand_parts, notes)``:

    * ``oriented_built`` is the same list of ``(chain, mesh)`` pairs with every
      mesh rotated into stand space and set down above the plate;
    * ``stand_parts`` is a list of ``(StandPart, mesh)`` ready to be appended to
      it for export;
    * ``notes`` are warnings for the user — a column that could not be sited, a
      plaque that had to shrink.

    Never raises for a geometric reason: a model that defeats the column search
    still gets its plate and its plaque, and the note says what is missing.
    """
    meta = dict(meta or {})
    stand = params.stand
    notes: List[str] = []

    chains = [c for c, _m in built]
    rotation = stand_rotation(stand)

    rotated = [_apply_rotation(m, rotation, np.zeros(3)) for _c, m in built]
    if not rotated:
        return list(built), [], ["Nothing to stand: the build produced no objects."]

    lows = np.array([m.bounds[0] for m in rotated])
    highs = np.array([m.bounds[1] for m in rotated])
    model_min = lows.min(axis=0)
    model_max = highs.max(axis=0)

    # Set the model down: centred on the plate in X, lifted so its lowest point
    # clears the plate top by ``stand_off_mm``.  The plate's *underside* lands on
    # z = 0 so the file arrives sitting on the print bed.
    centre_xy = 0.5 * (model_min[:2] + model_max[:2])
    offset = np.array([-centre_xy[0], -centre_xy[1],
                       -model_min[2] + float(stand.stand_off_mm)
                       + float(stand.plate_thickness_mm)])
    meshes = [_apply_rotation(m, np.eye(3), offset) for m in rotated]
    oriented_built = [(chains[i], meshes[i]) for i in range(len(meshes))]

    lows = np.array([m.bounds[0] for m in meshes])
    highs = np.array([m.bounds[1] for m in meshes])
    model_min = lows.min(axis=0)
    model_max = highs.max(axis=0)

    plate_bottom = 0.0
    plate_top = float(stand.plate_thickness_mm)
    margin = float(stand.plate_margin_mm)

    # ---- lay the plaque out first: it decides how deep the apron is --------
    chain_rows = []
    for i, (chain, _m) in enumerate(built):
        if getattr(chain, "mtype", None) == MoleculeType.STAND:
            continue
        chain_rows.append({
            "index": i,
            "chain_id": str(getattr(chain, "chain_id", "?")),
            "label": legend_label(chain, i),
            "color": color_for_index(i),
        })

    plate_x0 = float(model_min[0]) - margin
    plate_x1 = float(model_max[0]) + margin
    plate_width = plate_x1 - plate_x0
    pad = max(3.0, float(stand.plaque_text_mm) * 0.6)
    column_width = plate_width * 0.5 - pad * 1.5

    info_rows: List[_Row] = []
    legend_rows: List[_Row] = []
    if stand.plaque:
        info_rows = _info_rows(stand, params, meta, column_width)
        legend_rows = _legend_rows(stand, chain_rows, column_width)

    content = max(sum(r.height_mm for r in info_rows),
                  sum(r.height_mm for r in legend_rows))
    apron = 0.0
    if content > 0:
        apron = max(_APRON_MIN_MM, content + 2.0 * pad)

    plate_y0 = float(model_min[1]) - margin - apron
    plate_y1 = float(model_max[1]) + margin

    parts = [_rounded_slab(plate_x0, plate_x1, plate_y0, plate_y1,
                           plate_bottom, plate_top,
                           float(stand.plate_corner_mm))]

    # ---- columns -------------------------------------------------------
    wanted = int(stand.columns) if int(stand.columns) > 0 else recommend_columns(meshes)
    radius = max(1.0, float(stand.column_diameter_mm) * 0.5)
    foot = radius * float(stand.column_flare)
    model_centre = _centre_of_mass_xy(meshes)
    candidates = _underside_candidates(
        meshes, stand, cell_mm=max(2.0, radius),
        mtypes=[getattr(c, "mtype", None) for c in chains])
    if candidates:
        # Order matters. Reachability first: it is the cheap filter and usually
        # removes the most. Then which molecule may carry the load, then the
        # cosmetic trim of the outer fringe — narrowing by taste before
        # narrowing by physics would throw away points the physics still needed.
        candidates = _drop_obstructed(candidates, meshes, foot + 0.6)
        candidates = _drop_shared_candidates(candidates, meshes, radius + 1.0)
        if stand.column_prefer_protein:
            candidates = _prefer_protein(candidates, minimum=max(1, wanted))
        candidates = _drop_edge_candidates(
            candidates, footprint_hull(meshes),
            cap_mm=foot + float(stand.column_edge_margin_mm),
            edge_frac=float(stand.column_edge_frac),
            minimum=max(4, wanted + 2))
    if not candidates:
        notes.append(
            "No column could be placed: in this orientation there is no "
            "downward-facing surface with a clear path down to the plate. Turn "
            "the model so a flatter, more exposed face points down.")
        columns = []
    else:
        columns = _choose_columns(candidates, wanted, model_centre)
        if len(columns) < wanted:
            notes.append(
                f"Placed {len(columns)} column(s) rather than {wanted}: the "
                f"underside did not offer enough separated, unobstructed spots "
                f"in this orientation.")

    model_manifold = None
    if columns:
        try:
            model_manifold = _manifold.union(
                [_manifold.from_trimesh(m) for m in meshes])
        except Exception:
            model_manifold = None

    from . import interference

    for column in columns:
        px, py, pz = (float(v) for v in column.point)
        if py < plate_y0 + apron:
            # Cannot happen while the apron sits in front of the model's own
            # footprint, but the plaque is laid out before the columns are sited
            # and a future change to either could quietly break that. Say so
            # rather than printing lettering with a post through it.
            notes.append("A column landed over the plaque area; the lettering "
                         "there may be interrupted.")
        # Stop at the low point, do not chase the surface upward.
        #
        # Following the underside across the column's full width gave complete
        # contact and a part that cannot be assembled: the underside of a
        # molecule curves back over itself, so a cradle cut to match it is full
        # of undercuts and the model has no straight-down path into its own seat.
        # A shallow dimple near the lowest point is locally a simple saucer — the
        # model drops in — and the contact it gives up was never load-bearing
        # anyway. The column grows from the plate until it reaches the surface,
        # and stops there.
        top = pz + float(stand.cradle_depth_mm)
        solid = _column_solid(px, py, plate_top - 0.5, top, radius, foot,
                              stand.column_shape)
        if model_manifold is not None:
            try:
                # Carve only against the model *near this column*. The whole
                # model would be a boolean against a surface mesh with hundreds
                # of thousands of faces, per column, to change a few millimetres
                # at the top of a stick — the same localisation the interference
                # pass makes for the same reason. The window only has to cover
                # the column plus a margin: material the column cannot reach
                # cannot shape it.
                low, high = pz - 4.0, top + 2.0
                window = _manifold.oriented_box(
                    [px, py, 0.5 * (low + high)], np.eye(3),
                    [radius * 2.2, radius * 2.2, 0.5 * (high - low)])
                local = _manifold.intersection(model_manifold, window)
                if not local.is_empty():
                    tool = interference.dilate(
                        local, float(stand.cradle_clearance_mm))
                    carved = _manifold.difference(solid, tool)
                    if not carved.is_empty():
                        solid = carved
            except Exception:
                notes.append("One column could not be carved to fit the model; "
                             "it will meet it flat.")
        parts.append(solid)

    # ---- plaque, lying flat on the apron --------------------------------
    # Each entry is ``(part, solid, sits_on_a_tile)``.
    text_parts: List[Tuple[StandPart, "object", bool]] = []
    if info_rows or legend_rows:
        shrink = 1.0
        if content + 2.0 * pad > apron:
            shrink = max(_TEXT_MIN_SHRINK, (apron - 2.0 * pad) / max(content, 1e-6))
            for row in info_rows + legend_rows:
                row.cap_mm *= shrink
                row.height_mm *= shrink
                row.bar_mm *= shrink

        axis_u = np.array([1.0, 0.0, 0.0])
        axis_v = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        emboss = float(stand.plaque_emboss_mm)
        stroke = float(stand.plaque_stroke_mm) * (0.7 + 0.3 * shrink)
        tile_h = float(stand.plaque_tile_mm)
        sink = 0.5
        # Lettering on a tile can only sink as far as the tile is thick, or the
        # recess cut to receive it goes clean through and opens a letter-shaped
        # hole into the plate underneath.
        tile_sink = min(sink, tile_h * 0.6)

        info_text: List = []
        legend_solids: dict = {}
        tiles: List = []
        apron_top = plate_y0 + apron

        def _tile(x0, x1, y0, y1):
            """A white field the lettering sits on, standing just off the plate."""
            return _rounded_slab(x0, x1, y0, y1,
                                 plate_top - sink, plate_top + tile_h,
                                 min(2.2, max(1e-3, (y1 - y0)) * 0.2))

        def _emit(rows, x_left, x_right, target, align_right, tiled):
            """Lay one block out from the back of the apron toward the front.

            ``align_right`` sets each row flush to ``x_right`` instead of
            ``x_left``. The legend uses it so the colour dots form one straight
            edge down the right-hand side rather than a ragged one — with rows of
            different name lengths, a left-aligned legend has its dots in a line
            but its text ending anywhere, and reads as an accident.

            ``target`` may be a list or a callable taking ``(row, solids)``, which
            is how the legend routes each row's geometry into its own per-chain
            object instead of one shared one.
            """
            cursor = apron_top - pad
            block_top = cursor
            use_sink = tile_sink if tiled else sink
            base_z = plate_top + (tile_h if tiled else 0.0)
            # Actual ink extents, so a tile can be cut to the text rather than to
            # the column it was allotted. A block of three short chain names in a
            # half-plate-wide slot left most of its tile blank, which read as a
            # misprint rather than as margin.
            used = {"x0": None, "x1": None}

            def put(row, solids):
                if callable(target):
                    target(row, solids)
                else:
                    target.extend(solids)

            def mark(x0, x1):
                used["x0"] = x0 if used["x0"] is None else min(used["x0"], x0)
                used["x1"] = x1 if used["x1"] is None else max(used["x1"], x1)

            for row in rows:
                cursor -= row.height_mm
                baseline = cursor + row.height_mm * 0.26

                # Width of everything this row will draw, so a right-aligned row
                # knows where to start.
                lead = 0.0
                if row.kind == "legend":
                    lead = row.cap_mm * 1.6
                elif row.kind == "scalebar":
                    lead = row.bar_mm + row.cap_mm * 0.7
                text_w = strokefont.text_width(row.text, row.cap_mm) if row.text else 0.0
                x = (x_right - (lead + text_w)) if align_right else x_left
                mark(x, x + lead + text_w)

                if row.kind == "legend":
                    dot_r = row.cap_mm * 0.52
                    put(row, [_manifold.frustum(
                        [x + dot_r, baseline + dot_r, base_z - use_sink],
                        [x + dot_r, baseline + dot_r, base_z + emboss],
                        dot_r, dot_r)])
                    x += row.cap_mm * 1.6

                elif row.kind == "scalebar":
                    bar_h = max(0.8, row.cap_mm * 0.3)
                    bars = [_manifold.oriented_box(
                        [x + row.bar_mm * 0.5, baseline + bar_h * 0.5,
                         base_z + 0.5 * (emboss - use_sink)],
                        np.array([axis_u, axis_v, normal]),
                        [row.bar_mm * 0.5, bar_h * 0.5, 0.5 * (emboss + use_sink)])]
                    for end in (0.0, row.bar_mm):
                        bars.append(_manifold.oriented_box(
                            [x + end, baseline + row.cap_mm * 0.4,
                             base_z + 0.5 * (emboss - use_sink)],
                            np.array([axis_u, axis_v, normal]),
                            [bar_h * 0.5, row.cap_mm * 0.4,
                             0.5 * (emboss + use_sink)]))
                    put(row, bars)
                    x += row.bar_mm + row.cap_mm * 0.7

                if row.text:
                    put(row, _stroke_solids(
                        strokefont.layout(row.text, row.cap_mm),
                        np.array([x, baseline, base_z]),
                        axis_u, axis_v, normal, stroke, emboss, use_sink))

            if tiled and rows and used["x0"] is not None:
                tiles.append(_tile(used["x0"] - pad * 0.5, used["x1"] + pad * 0.5,
                                   cursor - pad * 0.45, block_top + pad * 0.45))

        info_left = plate_x0 + pad
        info_right = plate_x0 + column_width + pad
        legend_right = plate_x1 - pad
        legend_left = legend_right - column_width

        def _legend_target(row, solids):
            """Route a legend row's dot *and* its lettering into one object.

            Both carry the chain's colour, so the row is a single filament change
            in the slicer and the name is as identifiable as the dot beside it.
            """
            legend_solids.setdefault(row.built_index, []).extend(solids)

        # One switch, both blocks. Anything else means a control labelled
        # "white tile" that leaves a white tile behind when it is switched off,
        # which reads as a defect however well the exception is justified.
        tiled = bool(stand.plaque_tile)
        _emit(info_rows, info_left, info_right, info_text,
              align_right=False, tiled=tiled)
        _emit(legend_rows, legend_left, legend_right, _legend_target,
              align_right=True, tiled=tiled and bool(legend_rows))

        # ``on_tile`` decides what each solid is cut *out of*.  The plaque is
        # three layers deep now — plate, white tile, lettering — and a raised
        # solid has to be recessed into whatever it actually stands on. Cutting
        # everything out of the plate, as when the lettering sat directly on it,
        # would leave the letters overlapping the tile in shared volume, which is
        # exactly what a multi-material slicer objects to.
        if tiles:
            text_parts.append((StandPart("tile", "stand_plaque_tile", TILE_COLOR),
                               _manifold.union(tiles), False))
        if info_text:
            text_parts.append((
                StandPart("text", "stand_plaque_text",
                          TEXT_ON_TILE if tiled else TEXT_ON_PLATE),
                _manifold.union(info_text), tiled))
        for built_index, solids in legend_solids.items():
            if not solids:
                continue
            row = next(r for r in chain_rows if r["index"] == built_index)
            text_parts.append((
                StandPart("legend", f"stand_legend_{row['chain_id']}",
                          row["color"], chain_id=row["chain_id"]),
                _manifold.union(solids), True))

    # ---- fuse and hand back --------------------------------------------
    from . import meshops

    stand_parts: List = []
    base_solid = _manifold.union(parts)
    tile_index = next((i for i, (p, _s, _t) in enumerate(text_parts)
                       if p.part == "tile"), None)
    tile_solid = text_parts[tile_index][1] if tile_index is not None else None

    for part, solid, on_tile in text_parts:
        # Cut each raised solid out of whatever carries it, so the two meet on a
        # shared boundary and never share a volume.
        try:
            if on_tile and tile_solid is not None:
                trimmed = _manifold.difference(tile_solid, solid)
                if not trimmed.is_empty():
                    tile_solid = trimmed
            else:
                trimmed = _manifold.difference(base_solid, solid)
                if not trimmed.is_empty():
                    base_solid = trimmed
        except Exception:
            notes.append(f"{part.object_name()} could not be recessed into the "
                         f"surface below it; it sits on top instead.")

    stand_parts.append((
        StandPart("base", "stand_base", STAND_COLOR),
        meshops.repair(_manifold.to_trimesh(base_solid))))
    for i, (part, solid, _on_tile) in enumerate(text_parts):
        if i == tile_index:
            solid = tile_solid
        if solid is None:
            continue
        try:
            stand_parts.append((part, meshops.repair(_manifold.to_trimesh(solid))))
        except Exception:
            notes.append(f"Left out {part.object_name()}: it did not mesh.")

    return oriented_built, stand_parts, notes
