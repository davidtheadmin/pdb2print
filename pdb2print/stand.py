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
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

from . import typeset
from .config import (
    ColumnShape, MoleculeType, PlaqueRelief, PrintParams, StandParams,
    color_for_index,
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

#: Clear space (mm) between the two white tiles, at their closest.
#:
#: The blocks were separated by exactly one ``pad`` and each tile is cut half a
#: pad wider than its own lettering, so when both blocks filled their width the
#: two tiles met edge to edge — one white shape with a seam down it, rather than
#: two panels. This is the gap that survives that, and it is added to the plate
#: rather than taken out of either block: neither of them has width to spare at
#: the point where this starts to matter.
_TILE_GAP_MM = 2.5

#: The apron is never shallower than this (mm), so there is always somewhere for
#: the lettering to go even on a small model.
_APRON_MIN_MM = 22.0

#: The most the lettering may be shrunk to fit.  Below this a stroke drops under
#: one extrusion width and prints as a smudge, so the apron is deepened instead.
_TEXT_MIN_SHRINK = 0.62

#: The most the plaque face may be tipped toward the viewer.  Past about
#: twenty-five degrees the apron is a ramp rather than a lectern, the wedge
#: needed to make it starts to crowd the model, and the lettering — which is
#: laid out in the plan view and then projected — begins to look foreshortened.
_RAKE_MAX_DEG = 25.0
#: How thick a raked apron still is at its front edge (mm).  Tapering it to
#: nothing gives a knife edge that chips, prints as a stringy first layer, and
#: reads as a mistake next to a plate with a 4 mm corner radius.
_RAKE_LIP_MM = 0.8
#: And no wedge is ever taller than this at the back (mm), however deep the
#: apron or however generous the stand-off.
_RAKE_MAX_RISE_MM = 10.0

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


def _centre_of_mass_xy_placed(built, rotation: np.ndarray, offset: np.ndarray,
                              placed: Sequence[trimesh.Trimesh]) -> np.ndarray:
    """Same answer as ``_centre_of_mass_xy(placed)``, without re-measuring.

    Volume is invariant under a rotation and a translation and the centre of
    mass is equivariant under both, so there is nothing here that has to be
    measured on the placed meshes -- and measuring is what cost: those are
    fresh copies, so trimesh's mass-properties cache is cold every time and each
    preview paid for a full volume integration over every chain (measured at
    249ms on a 245k-face model, 1.8s on a 983k-face one, against 0.1ms on a warm
    mesh).  The originals are the same objects from one preview to the next, so
    they are integrated once and the answer is moved arithmetically thereafter.

    ``placed`` is only touched for the degenerate fallback, which needs vertices
    in the final frame.
    """
    total = 0.0
    accum = np.zeros(3)
    for _c, mesh in built:
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
        allv = np.vstack([m.vertices for m in placed])
        return allv.mean(axis=0)[:2]
    return ((accum / total) @ rotation.T + offset)[:2]


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
        # Plain arithmetic rather than np.mean, which is called once per
        # candidate pair or triple -- C(44,3) = 13,244 of them for three
        # columns -- and spent more time in numpy's dispatch than on the two or
        # three floats it was averaging (57ms against 5.8ms, measured).  The
        # summation order is the same, so the value is too.
        total = 0.0
        for i in idx:
            total += down[i]
        return 0.62 + 0.38 * float(total / len(idx))

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


#: A flared foot is a fixed height, not a fraction of the column (mm).
#:
#: Proportional was the obvious first choice and the wrong one: the columns on a
#: stand are all different heights, because each one rises to whatever part of
#: the underside is above it. A foot at 16% of the column gives every column a
#: different foot, and a set of feet that disagree with each other reads as a
#: fault rather than as a taper. One height for all of them is what makes them
#: look like a set. Clamped on a very short column, where a 2.6 mm foot would be
#: most of it.
_FOOT_HEIGHT_MM = 2.6
#: Likewise the base disc under a fluted shaft, and the capital at the top.
_FLUTE_BASE_MM = 2.2
_CAPITAL_MM = 2.0

#: Flutes round a classical column, and how deep each one is cut as a fraction
#: of the shaft radius.  Twenty-four would be Doric; at 8 mm across, twenty-four
#: flutes is a texture rather than a profile, and a 0.4 mm nozzle rounds them
#: off into one.  Eight reads as fluting at the size these actually print.
_FLUTE_COUNT = 8
_FLUTE_DEPTH = 0.16
#: Points per flute.  The scallop is a smooth curve, so it needs enough segments
#: not to read as a polygon — but every one of them is a face on every column.
_FLUTE_STEPS = 9


def _profile_polygon(shape: ColumnShape, segments: int = 20) -> np.ndarray:
    """A column's cross-section as an ordered CCW ``(N, 2)`` polygon of unit size.

    "Unit" means *half the across-flats width*, so a square column and a round
    one asked for the same diameter measure the same across the flats — which is
    what someone comparing them on a slider means by the same thickness.
    """
    if shape in (ColumnShape.SQUARE, ColumnShape.TAPER):
        return np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    if shape == ColumnShape.FLUTED:
        theta = np.linspace(0.0, 2.0 * np.pi, _FLUTE_COUNT * _FLUTE_STEPS,
                            endpoint=False)
        # Scalloped, never re-entrant: the radius is a smooth function of the
        # angle, so the polygon is star-shaped about its centre and cannot
        # self-intersect however deep the flutes are cut.
        radius = 1.0 - _FLUTE_DEPTH * (0.5 + 0.5 * np.cos(_FLUTE_COUNT * theta)) ** 2
        return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    theta = np.linspace(0.0, 2.0 * np.pi, max(8, int(segments)), endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _tapered_prism(profile: np.ndarray, x: float, y: float,
                   z0: float, z1: float, half0: float, half1: float):
    """``profile`` swept from ``half0`` at ``z0`` to ``half1`` at ``z1``.

    Written out as an explicit closed mesh rather than assembled from booleans.
    A square frustum has no primitive here and the kernel's convex hull would
    only cover the convex cases — this covers a scalloped section too, exactly,
    for the cost of two rings of vertices.  Watertight by construction: every
    edge is shared by exactly two triangles.
    """
    profile = np.asarray(profile, float)
    count = len(profile)
    z0, z1 = float(z0), float(z1)
    if z1 - z0 < 1e-6:
        z1 = z0 + 1e-6

    lower = np.column_stack([x + profile[:, 0] * half0,
                             y + profile[:, 1] * half0,
                             np.full(count, z0)])
    upper = np.column_stack([x + profile[:, 0] * half1,
                             y + profile[:, 1] * half1,
                             np.full(count, z1)])
    verts = np.vstack([lower, upper,
                       [[x, y, z0]], [[x, y, z1]]])
    hub_low, hub_high = 2 * count, 2 * count + 1

    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.append([i, j, count + j])              # side, lower triangle
        faces.append([i, count + j, count + i])      # side, upper triangle
        faces.append([j, i, hub_low])                # bottom fan (facing -z)
        faces.append([count + i, count + j, hub_high])   # top fan (facing +z)
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, np.int64),
                           process=False)
    return _manifold.from_trimesh(mesh)


def _column_solid(x: float, y: float, z0: float, z1: float,
                  top_half: float, foot_half: float, stand: StandParams):
    """One column in the requested style, from the plate at ``z0`` to ``z1``.

    Four profiles, one rule: whatever the style does, the shaft measures
    ``top_half`` across the flats where it meets the model, because that is the
    number the cradle, the reachability filter and the slider all agree on.
    Everything else — plinth, flare, flutes, capital — happens below or around
    it.
    """
    shape = stand.column_shape
    flared = bool(getattr(stand, "column_flared", True))
    foot = float(foot_half) if flared else float(top_half)
    height = max(1e-4, z1 - z0)
    square = _profile_polygon(ColumnShape.SQUARE)
    round_ = _profile_polygon(ColumnShape.ROUND)
    parts = []

    if shape == ColumnShape.ROUND:
        parts.append(_manifold.frustum([x, y, z0], [x, y, z1], foot, top_half))

    elif shape == ColumnShape.TAPER:
        # An obelisk: one continuous square taper, no plinth to interrupt it.
        parts.append(_tapered_prism(square, x, y, z0, z1, foot, top_half))

    elif shape == ColumnShape.FLUTED:
        base_h = min(_FLUTE_BASE_MM, height * 0.40) if flared else 0.0
        # A whisper of entasis. Straight-sided, a fluted shaft reads as a pipe;
        # the classical remedy is to have it swell slightly and narrow toward
        # the top, and a few per cent is enough for the eye to notice without
        # anyone being able to name what they noticed.
        parts.append(_tapered_prism(
            _profile_polygon(shape), x, y, z0 + base_h, z1,
            top_half * (1.05 if flared else 1.0), top_half * 0.94))
        if base_h > 0.0:
            parts.append(_tapered_prism(round_, x, y, z0, z0 + base_h,
                                        foot, top_half * 1.08))

    else:                                            # SQUARE
        parts.append(_tapered_prism(square, x, y, z0, z1, top_half, top_half))
        if flared:
            # A square column cannot be one tapered primitive the way a frustum
            # can, so its flare is a short plinth rather than a continuous
            # taper — arguably the better look anyway, reading as a deliberate
            # foot rather than as a cone someone truncated.
            plinth_h = min(_FOOT_HEIGHT_MM, max(1e-3, height * 0.45))
            parts.append(_tapered_prism(square, x, y, z0, z0 + plinth_h,
                                        foot, foot))

    if bool(getattr(stand, "column_capital", False)):
        # Never wider than the foot: the reachability filter cleared a corridor
        # the width of the foot and nothing about the capital was checked.
        cap_h = min(_CAPITAL_MM, max(1e-3, height * 0.30))
        cap_half = min(top_half * 1.32, max(top_half * 1.04, foot))
        profile = round_ if shape in (ColumnShape.ROUND, ColumnShape.FLUTED) else square
        parts.append(_tapered_prism(profile, x, y, z1 - cap_h, z1,
                                    top_half * 1.02, cap_half))

    return _manifold.union(parts) if len(parts) > 1 else parts[0]


def _place(solid, transform: np.ndarray):
    """A manifold moved by a 3x4 affine ``transform`` (identity fast-pathed)."""
    if transform is None:
        return solid
    return solid.transform(np.asarray(transform, float).tolist())


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


def _filled_solid(outer, holes, origin: np.ndarray, axis_u: np.ndarray,
                  axis_v: np.ndarray, normal: np.ndarray,
                  depth: float, sink: float):
    """A closed ring with holes, extruded into a solid on a plane.

    This is what a real typeface needs and a stroke font does not: a glyph is an
    *area*, not a path, so it has to be triangulated before it can be given a
    thickness.  ``mapbox_earcut`` does the triangulation — the same ear-clipping
    every 2D renderer uses, and the one dependency the real-font work adds
    besides ``fontTools``.

    Watertight by construction, like everything else here: the top is the
    triangulation, the bottom is the same triangulation reversed, and the walls
    are one quad per edge of every ring, so each edge belongs to exactly two
    faces.
    """
    import mapbox_earcut

    rings = [np.asarray(outer, float)] + [np.asarray(h, float) for h in holes]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return None
    verts2d = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        tri = np.asarray(
            mapbox_earcut.triangulate_float64(verts2d, ends), np.int64
        ).reshape(-1, 3)
    except Exception:
        return None
    if not len(tri):
        return None

    # Every top triangle must face the same way as the plane's normal. Ear
    # clipping follows the winding it was given, and a ring that arrived wound
    # the other way would otherwise put a hole in the letter.
    a, b, c = verts2d[tri[:, 0]], verts2d[tri[:, 1]], verts2d[tri[:, 2]]
    flip = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
    tri[flip] = tri[flip][:, ::-1]

    count = len(verts2d)
    plane = (origin + np.outer(verts2d[:, 0], axis_u)
             + np.outer(verts2d[:, 1], axis_v))
    vertices = np.vstack([plane + normal * float(depth),
                          plane - normal * float(sink)])

    faces = [tri, tri[:, ::-1] + count]
    start = 0
    for ring in rings:
        n = len(ring)
        idx = np.arange(n)
        i = start + idx
        j = start + (idx + 1) % n
        faces.append(np.column_stack([i + count, j + count, j]))
        faces.append(np.column_stack([i + count, j, i]))
        start += n

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.vstack(faces),
                           process=False)
    try:
        return _manifold.from_trimesh(mesh)
    except Exception:
        return None


def _text_solids(face, text: str, cap_mm: float, origin: np.ndarray,
                 axis_u: np.ndarray, axis_v: np.ndarray, normal: np.ndarray,
                 stroke_mm: float, depth: float, sink: float, grow: float):
    """One line of type as solids, whichever kind of font it is set in.

    A stroke font is swept; an outline font is filled.  ``grow`` fattens a
    filled outline by sweeping its own contours as well — a Minkowski sum with a
    disc, done the cheap way — which is how a face whose stems are thinner than
    the nozzle is made printable without leaving the typeface behind.
    """
    if not getattr(face, "outline", False):
        return _stroke_solids(face.strokes(text, cap_mm), origin, axis_u,
                              axis_v, normal, stroke_mm, depth, sink)

    solids = []
    rings: List = []
    for outer, holes in face.contours(text, cap_mm):
        filled = _filled_solid(outer, holes, origin, axis_u, axis_v, normal,
                               depth, sink)
        if filled is not None:
            solids.append(filled)
        rings.append(outer)
        rings.extend(holes)
    if grow > 1e-4 and rings:
        closed = [[(float(p[0]), float(p[1])) for p in ring] + [
            (float(ring[0][0]), float(ring[0][1]))] for ring in rings]
        solids.extend(_stroke_solids(closed, origin, axis_u, axis_v, normal,
                                     2.0 * grow, depth, sink))
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


def legend_overrides(raw: str) -> Dict[int, str]:
    """Parse ``index<TAB>label`` lines into ``{index: label}``.

    Best-effort by design, like everything else that reads a name: a malformed
    line is skipped rather than raised on, because a typo in one row must not
    cost the user the other rows or the stand. Blank labels are dropped so that
    clearing a box means "go back to the generated name" rather than "print an
    empty row".
    """
    out: Dict[int, str] = {}
    for line in (raw or "").splitlines():
        head, sep, label = line.partition("\t")
        if not sep:
            continue
        try:
            index = int(head.strip())
        except ValueError:
            continue
        label = label.strip()[:48]
        if label:
            out[index] = label
    return out


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
    face = typeset.face(getattr(stand, "plaque_font", "sans"))

    if stand.plaque_pdb_id and meta.get("pdb_id"):
        text = str(meta["pdb_id"]).upper()
        size = typeset.fit_cap_height(face, [text], width_mm, cap)
        rows.append(_Row("text", text, size, size * 1.6))

    title = str(getattr(stand, "plaque_title_text", "") or "").strip()
    if title:
        preferred = cap * 0.46
        # One line if it can be had. A title set across three lines competes with
        # the ID above it for the eye; the same words on one line read as a
        # caption to it, which is what they are. Shrinking the type to buy that
        # is worth it down to the point where the strokes stop printing — below
        # that a single line is legible only in the sense that it exists.
        floor = max(_TITLE_MIN_CAP_MM, cap * 0.26)
        single = typeset.fit_cap_height(face, [title], width_mm, preferred,
                                        min_cap_mm=0.1)
        if single >= floor:
            rows.append(_Row("text", title, single, single * 1.55))
        else:
            # Wrapping is second choice; truncating is not a choice at all.
            # Give up a little type size first, because a tenth of a millimetre
            # of cap height is losing nothing and the end of a structure's name
            # is losing the name. If four lines at the smallest printable size
            # still will not hold it, it takes as many lines as it takes and the
            # apron gets deeper — the apron is sized from this content, so that
            # costs a few millimetres of plate and nothing else.
            size, lines = preferred, []
            for factor in (1.0, 0.86, 0.74, 0.64):
                size = max(floor, preferred * factor)
                lines = typeset.wrap_balanced(face, title, size, width_mm,
                                              max_lines=4, truncate=False)
                if len(lines) <= 3:
                    break
            for line in lines:
                rows.append(_Row("text", line, size, size * 1.55))

    note = str(getattr(stand, "plaque_note", "") or "").strip()
    if note:
        # Set at the scale line's size, below the title: it is the user's own
        # line, and a line of your own competing with the structure's name for
        # the eye reads as a correction to it rather than as an addition.
        size = cap * 0.42
        for line in typeset.wrap(face, note, size, width_mm, max_lines=2):
            rows.append(_Row("text", line, size, size * 1.6))

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


def _info_natural_width(stand: StandParams, params: PrintParams,
                        meta: dict) -> float:
    """The width past which the left block stops getting anything out of it.

    Every line the block holds, measured at the size it is set at, unwrapped.
    Give the block this much and nothing wraps; give it more and nothing
    changes — the lettering cannot get any wider, so the white tile behind it
    stops growing and the extra millimetres go into pushing the plate out for
    no return.

    Which makes it the natural top of the width control, and that is what it is
    used for: the slider ends here, and a value past it is clamped back to it.
    """
    if not stand.plaque:
        return 0.0
    face = typeset.face(getattr(stand, "plaque_font", "sans"))
    cap = float(stand.plaque_text_mm)
    widest = 0.0
    if stand.plaque_pdb_id and meta.get("pdb_id"):
        widest = max(widest, typeset.text_width(
            face, str(meta["pdb_id"]).upper(), cap))
    title = str(getattr(stand, "plaque_title_text", "") or "").strip()
    if title:
        # The size ``_info_rows`` sets a title at when it fits on one line.
        widest = max(widest, typeset.text_width(face, title, cap * 0.46))
    note = str(getattr(stand, "plaque_note", "") or "").strip()
    if note:
        widest = max(widest, typeset.text_width(face, note, cap * 0.42))
    if stand.plaque_scalebar:
        widest = max(widest, typeset.text_width(
            face, f"1 Å = {params.scale_mm_per_angstrom:g} mm", cap * 0.42))
    return min(widest * 1.04 + 1.0, 200.0) if widest else 0.0


def _info_floor_width(stand: StandParams, meta: dict) -> float:
    """The narrowest the left block may be before it starts shrinking its own type.

    Every row on the plaque shrinks to fit the column it is given — right for a
    title that is a sentence long, wrong for a four-character PDB ID that is
    meant to be the headline. Without this floor the width control quietly sets
    the type size instead of the width, and a 5 mm ID comes out at 2.9 mm.
    """
    if not stand.plaque:
        return 0.0
    face = typeset.face(getattr(stand, "plaque_font", "sans"))
    cap = float(stand.plaque_text_mm)
    wanted = 0.0
    if stand.plaque_pdb_id and meta.get("pdb_id"):
        wanted = typeset.text_width(face, str(meta["pdb_id"]).upper(), cap)
    return min(max(14.0, wanted + 1.0), 44.0)


def _legend_natural_width(stand: StandParams, chain_rows: List[dict]) -> float:
    """How wide the legend would like to be: its longest row, unshortened.

    Asked before the plate is sized, so the plate can be made wide enough for
    both blocks instead of one of them being squeezed to fit what is left. The
    legend is the block with a natural width — a chain's name is as long as it
    is — where the information block is happy at any width and merely wraps
    differently.
    """
    if not (stand.plaque and stand.plaque_legend and chain_rows):
        return 0.0
    face = typeset.face(getattr(stand, "plaque_font", "sans"))
    size = float(stand.plaque_text_mm) * 0.42
    widest = 0.0
    for row in chain_rows:
        widest = max(widest,
                     size * 2.2 + typeset.text_width(face, row["label"], size))
    # A hair over, because ``_legend_rows`` shortens anything that is not
    # *strictly* narrower than the space it has, and a name that measures its
    # own width exactly would be truncated by a rounding error.
    return widest + 0.6 if widest else 0.0


def _legend_rows(stand: StandParams, chain_rows: List[dict],
                 width_mm: float) -> List[_Row]:
    """The right-hand block: one colour-matched row per printed object."""
    rows: List[_Row] = []
    if not (stand.plaque_legend and chain_rows):
        return rows
    size = float(stand.plaque_text_mm) * 0.42
    face = typeset.face(getattr(stand, "plaque_font", "sans"))
    # A to Z down the page. The build order is whatever the file listed — polymers
    # then ligands, chains in header order — which is fine for colour assignment
    # and useless for finding a row: someone reading the legend is looking up the
    # letter moulded on the model, so the letters have to be in order.
    for row in sorted(chain_rows, key=lambda r: str(r["chain_id"])):
        label = row["label"]
        available = width_mm - size * 2.2
        if typeset.text_width(face, label, size) > available:
            shortened = typeset.wrap(face, label, size, available, max_lines=1)
            label = shortened[0] if shortened else label
        rows.append(_Row("legend", label, size, size * 1.85,
                         built_index=row["index"]))
    return rows


# --------------------------------------------------------------------------
# Layout: everything decided before any geometry is built
# --------------------------------------------------------------------------
@dataclass
class StandLayout:
    """The stand, decided but not yet made.

    Everything here is arithmetic on the finished meshes: where the plate goes,
    how deep the apron has to be to hold the lettering, which points on the
    underside the columns will rise to.  None of it builds a solid, and the
    expensive half of :func:`build_stand` — carving four cradles out of a surface
    mesh, sweeping a few hundred stroke segments, meshing the result — happens
    entirely after it.

    Split out so the front end can draw the stand it is about to ask for.  A
    preview that re-derives any of this in JavaScript is a second implementation
    that will disagree with the first one eventually, and the disagreement will
    surface as a column that is not where the picture promised.
    """

    oriented_built: list
    meshes: list
    chains: list
    chain_rows: List[dict]
    model_min: np.ndarray
    model_max: np.ndarray

    plate_x0: float
    plate_x1: float
    plate_y0: float
    plate_y1: float
    plate_top: float
    corner_mm: float

    apron: float
    apron_top: float
    pad: float
    info_width: float
    legend_width: float
    rake_deg: float
    rake_rise: float
    rake_lip: float

    info_rows: List[_Row]
    legend_rows: List[_Row]
    shrink: float

    meta: dict
    columns: List[_Candidate]
    wanted: int
    radius: float
    foot: float
    hull: Optional[np.ndarray]

    notes: List[str]


#: Solved column sets, keyed on everything that can move a column.
#:
#: Measured control by control on a 3-chain model: of the twelve stand
#: controls, **eight produce bit-identical columns** -- plaque text, corner
#: style, plate margin, apron rake, the note, the title, the tile toggle and
#: the relief -- while the column search itself costs 0.71s on a 245k-face
#: model and 4.80s on a 983k-face one, and the plaque layout it is holding up
#: costs 0.14ms.  So two thirds of the panel was paying seconds for an answer
#: that could not change.
#:
#: Small and LRU because each entry pins the meshes it was solved against.
#: That reference is load-bearing: the key identifies the meshes by ``id()``,
#: which is only sound while they are alive -- a collected object's id can be
#: handed to a different one.
_COLUMN_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_COLUMN_CACHE_MAX = 4
_COLUMN_LOCK = threading.Lock()


def _column_cache_key(built, rotation, offset, stand, wanted: int,
                      radius: float, foot: float) -> tuple:
    """Everything the column search reads, and nothing else.

    ``plate_thickness`` and ``stand_off`` are in here through ``offset`` even
    though they are provably a pure z translation of every candidate (measured:
    xy delta exactly 0.0).  Deriving the shift instead would work and would win
    two more sliders -- but a key missing a field it needed does not miss the
    cache, it silently freezes a column somewhere wrong, which is a far worse
    failure than re-solving.  The eight controls that matter most are the free
    ones either way.
    """
    return (
        tuple(id(m) for _c, m in built),
        tuple(np.asarray(rotation, float).ravel().tolist()),
        tuple(np.asarray(offset, float).tolist()),
        int(wanted), float(radius), float(foot),
        float(stand.column_normal_min), float(stand.column_edge_frac),
        float(stand.column_edge_margin_mm), bool(stand.column_prefer_protein),
    )


def _column_cache_get(key: tuple):
    with _COLUMN_LOCK:
        hit = _COLUMN_CACHE.get(key)
        if hit is not None:
            _COLUMN_CACHE.move_to_end(key)
            return hit[1]
        return None


def _column_cache_put(key: tuple, built, value) -> None:
    with _COLUMN_LOCK:
        # built is held only to keep the id()s in the key honest.
        _COLUMN_CACHE[key] = (list(built), value)
        _COLUMN_CACHE.move_to_end(key)
        while len(_COLUMN_CACHE) > _COLUMN_CACHE_MAX:
            _COLUMN_CACHE.popitem(last=False)


def solve_layout(built, params: PrintParams,
                 meta: Optional[dict] = None) -> Optional[StandLayout]:
    """Work out the whole stand without building any of it.

    Returns ``None`` when there is nothing to stand.  Never raises for a
    geometric reason: a model that defeats the column search comes back with no
    columns and a note saying so, because a plate and a plaque are still worth
    having.
    """
    meta = dict(meta or {})
    stand = params.stand
    notes: List[str] = []

    chains = [c for c, _m in built]
    rotation = stand_rotation(stand)
    # One copy per chain, not two.  This was two passes of _apply_rotation, and
    # a trimesh copy() walks a hash over the vertex and face arrays -- on a
    # 983k-face model that is 1.18s of the solve, against 0.7ms for the
    # arithmetic it exists to carry.  The second pass only added a constant, so
    # it is applied in place below once the bounds it depends on are known.
    meshes = []
    for _c, m in built:
        out = m.copy()
        out.vertices = np.asarray(out.vertices, float) @ rotation.T
        meshes.append(out)
    if not meshes:
        return None

    lows = np.array([m.bounds[0] for m in meshes])
    highs = np.array([m.bounds[1] for m in meshes])
    model_min = lows.min(axis=0)
    model_max = highs.max(axis=0)

    # Set the model down: centred on the plate in X, lifted so its lowest point
    # clears the plate top by ``stand_off_mm``.  The plate's *underside* lands on
    # z = 0 so the file arrives sitting on the print bed.
    centre_xy = 0.5 * (model_min[:2] + model_max[:2])
    offset = np.array([-centre_xy[0], -centre_xy[1],
                       -model_min[2] + float(stand.stand_off_mm)
                       + float(stand.plate_thickness_mm)])
    for out in meshes:
        out.vertices = np.asarray(out.vertices, float) + offset
    oriented_built = [(chains[i], meshes[i]) for i in range(len(meshes))]

    lows = np.array([m.bounds[0] for m in meshes])
    highs = np.array([m.bounds[1] for m in meshes])
    model_min = lows.min(axis=0)
    model_max = highs.max(axis=0)

    plate_top = float(stand.plate_thickness_mm)
    margin = float(stand.plate_margin_mm)

    # ---- lay the plaque out first: it decides how deep the apron is --------
    chain_rows = []
    overrides = legend_overrides(getattr(stand, 'plaque_legend_labels', ''))
    for i, (chain, _m) in enumerate(built):
        if getattr(chain, "mtype", None) == MoleculeType.STAND:
            continue
        chain_rows.append({
            "index": i,
            "chain_id": str(getattr(chain, "chain_id", "?")),
            "label": overrides.get(i) or legend_label(chain, i),
            "color": color_for_index(i),
        })

    # The plate is at least wide enough for the model, and wider if the plaque
    # needs it to be. One rule, in one place:
    #
    #   the left block is as wide as it was asked to be (or half, on auto);
    #   the legend keeps the width its own names need;
    #   the plate grows only if those two do not fit side by side.
    #
    # Two earlier versions of this tried to be clever — a fraction of the plate,
    # then a fraction interpolated between measured anchors — and the cleverness
    # was the problem both times: it could not be checked by looking at it, and
    # when it disagreed with what somebody expected there was no way to tell
    # which of the three moving parts had done it.
    pad = max(3.0, float(stand.plaque_text_mm) * 0.6)
    # A margin at each edge, and between the blocks a gap wide enough that the
    # two tiles still have ``_TILE_GAP_MM`` of plate between them once each has
    # taken its own half-pad of surround.
    spare = pad * 2.0 + (pad + _TILE_GAP_MM)
    natural_half = max(abs(float(model_min[0])), abs(float(model_max[0]))) + margin
    natural_usable = max(12.0, 2.0 * natural_half - spare)

    # What the legend needs is what its longest name is. Only an absolute
    # backstop, against a pathological header.
    legend_need = 0.0
    if stand.plaque:
        legend_need = min(_legend_natural_width(stand, chain_rows), 90.0)

    # Bounded at both ends by something real. Below the floor the control would
    # be shrinking the headline rather than narrowing the block; above the
    # natural width it would be pushing the plate out to hold lettering that
    # cannot get any wider, which is where the legend used to start paying for
    # width nobody was using.
    info_max = _info_natural_width(stand, params, meta)
    asked = float(getattr(stand, "plaque_info_mm", 0.0) or 0.0)
    info_width = asked if asked > 0.0 else natural_usable * 0.5
    if info_max > 0.0:
        info_width = min(info_width, info_max)
    info_width = max(info_width, _info_floor_width(stand, meta))

    half = max(natural_half, 0.5 * (info_width + legend_need + spare))
    plate_x0, plate_x1 = -half, half
    plate_width = plate_x1 - plate_x0
    usable = plate_width - spare
    legend_width = max(6.0, usable - info_width)

    info_rows: List[_Row] = []
    legend_rows: List[_Row] = []
    if stand.plaque:
        info_rows = _info_rows(stand, params, meta, info_width)
        legend_rows = _legend_rows(stand, chain_rows, legend_width)

    if stand.plaque:
        # Said once, here, so it reaches the build report *and* the live sketch:
        # both go through this function.
        missing = typeset.unavailable(typeset.face(
            getattr(stand, "plaque_font", "sans")))
        if missing:
            notes.append(missing)

    content = max(sum(r.height_mm for r in info_rows),
                  sum(r.height_mm for r in legend_rows))
    apron = 0.0
    if content > 0:
        apron = max(_APRON_MIN_MM, content + 2.0 * pad)

    shrink = 1.0
    if content > 0 and content + 2.0 * pad > apron:
        shrink = max(_TEXT_MIN_SHRINK, (apron - 2.0 * pad) / max(content, 1e-6))
        for row in info_rows + legend_rows:
            row.cap_mm *= shrink
            row.height_mm *= shrink
            row.bar_mm *= shrink

    plate_y0 = float(model_min[1]) - margin - apron
    plate_y1 = float(model_max[1]) + margin
    apron_top = plate_y0 + apron

    # ---- how far the plaque face is tipped toward the viewer -------------
    rake_deg = max(0.0, min(float(getattr(stand, "apron_rake_deg", 0.0) or 0.0),
                            _RAKE_MAX_DEG))
    rake_rise = 0.0
    rake_lip = 0.0
    if apron > 0.0 and rake_deg > 1e-6:
        rake_lip = _RAKE_LIP_MM
        rake_rise = apron * math.tan(math.radians(rake_deg))
        # The wedge is highest at the back, right in front of the model, so past
        # a point it stops being a lectern and starts being a wall you look at
        # the model over. The stand-off is the model's own ground clearance and
        # so the natural ceiling: keep the wedge below the lowest thing it could
        # hide.
        ceiling = min(_RAKE_MAX_RISE_MM,
                      max(4.0, float(stand.stand_off_mm)) - rake_lip)
        if rake_rise > ceiling:
            rake_rise = max(0.0, ceiling)
            capped = math.degrees(math.atan2(rake_rise, apron))
            if rake_deg - capped > 1.5:
                notes.append(
                    f"The plaque rake was limited to about {capped:.0f}° so the "
                    f"wedge stays below the model; raise the stand-off, or "
                    f"shorten the plaque, for more.")
            rake_deg = capped
        if rake_rise <= 1e-6:
            rake_deg, rake_lip = 0.0, 0.0

    # ---- where the columns land -----------------------------------------
    wanted = int(stand.columns) if int(stand.columns) > 0 else recommend_columns(meshes)
    radius = max(1.0, float(stand.column_diameter_mm) * 0.5)
    foot = radius * (float(stand.column_flare)
                     if bool(getattr(stand, "column_flared", True)) else 1.0)
    # Everything from here to the end of this block depends only on the meshes,
    # the orientation and the column settings — never on the plaque — so it is
    # cached under exactly those. See _column_cache_key for what that buys.
    _ckey = _column_cache_key(built, rotation, offset, stand, wanted, radius, foot)
    _cached = _column_cache_get(_ckey)
    if _cached is not None:
        hull, columns, column_notes = _cached
    else:
        column_notes: List[str] = []
        model_centre = _centre_of_mass_xy_placed(built, rotation, offset, meshes)
        hull = footprint_hull(meshes)
        candidates = _underside_candidates(
            meshes, stand, cell_mm=max(2.0, radius),
            mtypes=[getattr(c, "mtype", None) for c in chains])
        if candidates:
            # Order matters. Reachability first: it is the cheap filter and
            # usually removes the most. Then which molecule may carry the load,
            # then the cosmetic trim of the outer fringe — narrowing by taste
            # before narrowing by physics would throw away points the physics
            # still needed.
            candidates = _drop_obstructed(candidates, meshes, foot + 0.6)
            candidates = _drop_shared_candidates(candidates, meshes, radius + 1.0)
            if stand.column_prefer_protein:
                candidates = _prefer_protein(candidates, minimum=max(1, wanted))
            candidates = _drop_edge_candidates(
                candidates, hull,
                cap_mm=foot + float(stand.column_edge_margin_mm),
                edge_frac=float(stand.column_edge_frac),
                minimum=max(4, wanted + 2))
        if not candidates:
            column_notes.append(
                "No column could be placed: in this orientation there is no "
                "downward-facing surface with a clear path down to the plate. "
                "Turn the model so a flatter, more exposed face points down.")
            columns: List[_Candidate] = []
        else:
            columns = _choose_columns(candidates, wanted, model_centre)
            if len(columns) < wanted:
                column_notes.append(
                    f"Placed {len(columns)} column(s) rather than {wanted}: the "
                    f"underside did not offer enough separated, unobstructed "
                    f"spots in this orientation.")
        _column_cache_put(_ckey, built, (hull, columns, column_notes))
    # Callers treat these as read-only; a cache hit hands out the same objects.
    notes.extend(column_notes)

    return StandLayout(
        oriented_built=oriented_built, meshes=meshes, chains=chains,
        chain_rows=chain_rows, model_min=model_min, model_max=model_max,
        plate_x0=plate_x0, plate_x1=plate_x1, plate_y0=plate_y0,
        plate_y1=plate_y1, plate_top=plate_top,
        corner_mm=float(stand.plate_corner_mm),
        apron=apron, apron_top=apron_top, pad=pad,
        info_width=info_width, legend_width=legend_width,
        rake_deg=rake_deg, rake_rise=rake_rise, rake_lip=rake_lip,
        info_rows=info_rows, legend_rows=legend_rows, shrink=shrink, meta=meta,
        columns=columns, wanted=wanted, radius=radius, foot=foot, hull=hull,
        notes=notes,
    )


def layout_summary(layout: StandLayout, params: PrintParams) -> dict:
    """A JSON-ready description of a solved layout, for the live sketch.

    Millimetres throughout, in stand space: ``+x`` right, ``+y`` away from the
    viewer, ``+z`` up, plate underside on ``z = 0``.  Row heights and cap sizes
    are the *final* ones, after any shrink-to-fit, so a preview drawing them at
    face value draws what will be printed.
    """
    stand = params.stand

    def _rows(rows: List[_Row], colours: Optional[dict] = None) -> List[dict]:
        out = []
        for row in rows:
            item = {"kind": row.kind, "text": row.text,
                    "cap_mm": round(float(row.cap_mm), 3),
                    "height_mm": round(float(row.height_mm), 3),
                    "bar_mm": round(float(row.bar_mm), 3)}
            if colours is not None and row.built_index >= 0:
                meta = colours.get(row.built_index)
                if meta:
                    item["color"] = _rgb_hex(meta["color"])
                    item["chain_id"] = meta["chain_id"]
                    # The built index, so the front end can address this row
                    # when it sends a replacement label back. chain_id will not
                    # do: a homodimer has two rows sharing one id.
                    item["index"] = int(row.built_index)
                    # ``text`` is what fits the plate — _legend_rows shortens a
                    # long name to one line. The editor has to prefill with the
                    # name itself, or accepting the box unchanged would bake the
                    # truncation in as a permanent override.
                    item["full"] = meta["label"]
            out.append(item)
        return out

    by_index = {r["index"]: r for r in layout.chain_rows}
    model_min = [round(float(v), 3) for v in layout.model_min]
    model_max = [round(float(v), 3) for v in layout.model_max]

    return {
        "plate": {
            "x0": round(layout.plate_x0, 3), "x1": round(layout.plate_x1, 3),
            "y0": round(layout.plate_y0, 3), "y1": round(layout.plate_y1, 3),
            "top": round(layout.plate_top, 3),
            "corner_mm": round(layout.corner_mm, 3),
        },
        "apron": {
            "depth": round(layout.apron, 3),
            "top": round(layout.apron_top, 3),
            "pad": round(layout.pad, 3),
            "info_width": round(layout.info_width, 3),
            "legend_width": round(layout.legend_width, 3),
            "info_mm": round(float(getattr(stand, "plaque_info_mm", 0.0) or 0.0), 1),
            # Where the width control should stop: past this the lettering
            # cannot get wider, so neither can the tile behind it.
            "info_max_mm": round(_info_natural_width(stand, params, layout.meta), 1),
            "info_floor_mm": round(_info_floor_width(stand, layout.meta), 1),
            "rake_deg": round(layout.rake_deg, 3),
            "rise": round(layout.rake_rise, 3),
            "lip": round(layout.rake_lip, 3),
        },
        "model": {"min": model_min, "max": model_max,
                  "stand_off": round(float(stand.stand_off_mm), 3)},
        "hull": ([[round(float(x), 2), round(float(y), 2)]
                  for x, y in layout.hull] if layout.hull is not None else []),
        "columns": [{"x": round(float(c.point[0]), 3),
                     "y": round(float(c.point[1]), 3),
                     "top": round(float(c.point[2])
                                  + float(stand.cradle_depth_mm), 3),
                     "seat": round(float(c.point[2]), 3)}
                    for c in layout.columns],
        "column": {
            "shape": stand.column_shape.value,
            "radius": round(layout.radius, 3),
            "foot": round(layout.foot, 3),
            "flared": bool(getattr(stand, "column_flared", True)),
            "capital": bool(getattr(stand, "column_capital", False)),
            "pins": bool(getattr(stand, "column_pins", False)),
            "wanted": int(layout.wanted),
            "placed": len(layout.columns),
        },
        # The plaque's text as text, not only as geometry. A preview at panel
        # size renders a 2 mm line about five pixels tall, which is honest about
        # the print and useless for answering "is my structure's name in there".
        "meta": {"pdb_id": layout.meta.get("pdb_id") or "",
                 "title": layout.meta.get("title") or "",
                 "why": layout.meta.get("why") or ""},
        "plaque": {
            "on": bool(stand.plaque),
            "relief": getattr(stand, "plaque_relief", PlaqueRelief.RAISED).value,
            "font": typeset.face(getattr(stand, "plaque_font", "sans")).key,
            "font_note": typeset.unavailable(
                typeset.face(getattr(stand, "plaque_font", "sans"))),
            "tile": bool(stand.plaque_tile),
            "info": _rows(layout.info_rows),
            "legend": _rows(layout.legend_rows, by_index),
        },
        "notes": list(layout.notes),
    }


def _rgb_hex(color) -> str:
    r, g, b = (max(0, min(255, int(round(float(c) * 255)))) for c in color)
    return f"#{r:02x}{g:02x}{b:02x}"


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
    stand = params.stand
    layout = solve_layout(built, params, meta)
    if layout is None:
        return list(built), [], ["Nothing to stand: the build produced no objects."]

    notes = layout.notes
    meshes = layout.meshes
    plate_top = layout.plate_top
    apron = layout.apron
    apron_top = layout.apron_top
    pad = layout.pad

    parts = [_rounded_slab(layout.plate_x0, layout.plate_x1,
                           layout.plate_y0, layout.plate_y1,
                           0.0, plate_top, layout.corner_mm)]

    # ---- the raked apron, if any ----------------------------------------
    # A wedge *added* to the plate rather than cut into it. The cut version can
    # never be deeper than the plate is thick, which runs out at about five
    # degrees on a 4 mm plate — and five degrees is not a rake, it is a defect.
    # Adding material has no such ceiling, and every face of the wedge slopes
    # upward from the bed, so none of it needs support.
    cos_a, sin_a = 1.0, 0.0
    surface_z = plate_top
    if layout.rake_rise > 1e-6:
        angle = math.atan2(layout.rake_rise, apron)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        surface_z = plate_top + layout.rake_lip + layout.rake_rise
        slope = apron / max(cos_a, 1e-6)
        thickness = surface_z + 20.0
        centre_top = np.array([
            0.5 * (layout.plate_x0 + layout.plate_x1),
            0.5 * (layout.plate_y0 + apron_top),
            plate_top + layout.rake_lip + 0.5 * layout.rake_rise])
        axis_v = np.array([0.0, cos_a, sin_a])
        normal = np.array([0.0, -sin_a, cos_a])
        wedge = _manifold.oriented_box(
            centre_top - normal * (thickness * 0.5),
            np.array([[1.0, 0.0, 0.0], axis_v, normal]),
            [0.5 * (layout.plate_x1 - layout.plate_x0), 0.5 * slope,
             0.5 * thickness])
        # Clipped to the plate's own rounded plan, so the wedge inherits the
        # corner radius instead of poking square corners out of a round plate,
        # and squared off at the back where it meets the rest of the plate.
        plan = _rounded_slab(layout.plate_x0, layout.plate_x1,
                             layout.plate_y0, layout.plate_y1,
                             0.0, surface_z + 1.0, layout.corner_mm)
        back = _manifold.oriented_box(
            [0.5 * (layout.plate_x0 + layout.plate_x1),
             0.5 * (layout.plate_y0 - 2.0 + apron_top),
             0.5 * (surface_z + 2.0)],
            np.eye(3),
            [0.5 * (layout.plate_x1 - layout.plate_x0) + 1.0,
             0.5 * (apron_top - layout.plate_y0 + 2.0),
             0.5 * (surface_z + 4.0)])
        try:
            wedge = _manifold.intersection(_manifold.intersection(wedge, plan),
                                           back)
            if not wedge.is_empty():
                parts.append(wedge)
            else:
                cos_a, sin_a, surface_z = 1.0, 0.0, plate_top
        except Exception:
            notes.append("The plaque rake could not be built; the apron is flat.")
            cos_a, sin_a, surface_z = 1.0, 0.0, plate_top

    #: Maps plaque-layout coordinates ``(x, y, lift)`` onto the apron face,
    #: where ``lift`` is height above that face.  The identity-plus-translation
    #: when the apron is flat, which is why every plaque solid below can be
    #: built in one frame and placed in another.
    face_xf = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a, apron_top * (1.0 - cos_a)],
        [0.0, sin_a, cos_a, surface_z - apron_top * sin_a],
    ])

    # ---- columns -------------------------------------------------------
    radius, foot = layout.radius, layout.foot
    model_manifold = None
    if layout.columns:
        try:
            model_manifold = _manifold.union(
                [_manifold.from_trimesh(m) for m in meshes])
        except Exception:
            model_manifold = None

    from . import interference

    # Pinned, the columns leave the plate and become parts in their own right —
    # which is the whole point: a plate nobody has welded columns to can be
    # turned over and printed with its lettering face down against the build
    # sheet, and a plaque printed against a smooth sheet is as good as the
    # sheet. Everything about that is decided here.
    pins = bool(getattr(stand, "column_pins", False))
    pin_radius = max(0.8, min(float(getattr(stand, "pin_diameter_mm", 4.0)) * 0.5,
                              radius * 0.8))
    pin_depth = float(getattr(stand, "pin_depth_mm", 3.0))
    if pins and pin_depth > plate_top - 1.0:
        pin_depth = max(1.0, plate_top - 1.0)
        notes.append(
            f"The assembly pins were shortened to {pin_depth:.1f} mm so the "
            f"sockets do not break through the underside of the plate — which "
            f"is the face you are turning over to print.")
    pin_clear = float(getattr(stand, "pin_clearance_mm", 0.15))
    column_parts: List = []
    plate_cuts: List = []

    for index, column in enumerate(layout.columns):
        px, py, pz = (float(v) for v in column.point)
        if py < apron_top:
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
        # Unpinned the column is fused to the plate, so it starts just inside it
        # to guarantee the union has volume to work with. Pinned it starts *on*
        # the plate, because the two are about to be separate objects and a
        # column buried half a millimetre into its own plate would not seat.
        solid = _column_solid(px, py, plate_top - (0.0 if pins else 0.5),
                              top, radius, foot, stand)
        if pins:
            tip = min(0.6, pin_depth * 0.3)
            solid = _manifold.union([
                solid,
                _manifold.frustum([px, py, plate_top - pin_depth],
                                  [px, py, plate_top - pin_depth + tip],
                                  max(0.3, pin_radius - tip), pin_radius),
                _manifold.frustum([px, py, plate_top - pin_depth + tip],
                                  [px, py, plate_top + 0.01],
                                  pin_radius, pin_radius),
            ])
            plate_cuts.append(_manifold.union([
                _manifold.frustum([px, py, plate_top - pin_depth - 0.15],
                                  [px, py, plate_top + 0.02],
                                  pin_radius + pin_clear, pin_radius + pin_clear),
                # A lead-in at the mouth, so the pin finds the hole instead of
                # catching the first-layer elephant's foot around its rim.
                _manifold.frustum([px, py, plate_top - 0.45],
                                  [px, py, plate_top + 0.02],
                                  pin_radius + pin_clear,
                                  pin_radius + pin_clear + 0.45),
            ]))
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
                    [radius * 2.4, radius * 2.4, 0.5 * (high - low)])
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
        if pins:
            column_parts.append((
                StandPart("column", f"stand_column_{index + 1}", STAND_COLOR,
                          chain_id=str(index + 1)), solid))
        else:
            parts.append(solid)

    # ---- plaque, lying on the apron -------------------------------------
    # Built in the apron's own frame — ``z`` is height above the face, not above
    # the plate — and placed onto the face at the end.  One transform at the
    # bottom is the whole cost of a raked plaque; without it every dot, bar and
    # stroke below would need its own trigonometry.
    info_rows, legend_rows = layout.info_rows, layout.legend_rows
    relief = getattr(stand, "plaque_relief", PlaqueRelief.FLUSH)
    engrave = relief == PlaqueRelief.ENGRAVED
    flush = relief == PlaqueRelief.FLUSH

    # Each entry is ``(part, solid, sits_on_a_tile)``.
    text_parts: List[Tuple[StandPart, "object", bool]] = []
    cutters: List = []                      # engraved lettering, to subtract
    tile_solid = None

    if info_rows or legend_rows:
        shrink = layout.shrink
        axis_u = np.array([1.0, 0.0, 0.0])
        axis_v = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        emboss = float(stand.plaque_emboss_mm)
        stroke = float(stand.plaque_stroke_mm) * (0.7 + 0.3 * shrink)
        face = typeset.face(getattr(stand, "plaque_font", "sans"))
        min_stroke = float(getattr(stand, "plaque_min_stroke_mm", 0.45))
        tile_h = float(stand.plaque_tile_mm)
        sink = 0.5
        tiled = bool(stand.plaque_tile)
        # Lettering on a tile can only sink as far as the tile is thick, or the
        # recess cut to receive it goes clean through and opens a letter-shaped
        # hole into the plate underneath.
        tile_sink = min(sink, tile_h * 0.6)

        # Engraved, the cut wants to go *through* a tile and stop on the plate
        # beneath it: the letters then read in the plate's colour on a white
        # field, with no second object involved and nothing that can come loose.
        # Straight onto the plate it is an ordinary groove.
        cut_depth = (tile_h + sink) if tiled else emboss
        cut_proud = 0.4                     # start the cutter above the face
        # Flush, everything is sunk instead: the tile drops into the apron until
        # its face is level with the plate, and the lettering drops into the
        # tile until it is level with that. Nothing stands proud, nothing is
        # hollow, and the whole apron is one plane whose colour changes partway
        # through a layer — which is the thing a multi-material printer is
        # actually good at, and the thing a flat top surface prints best as.
        flush_sink = min(emboss, tile_h * 0.75) if tiled else emboss
        # Where the tile sits relative to the face, and therefore what height
        # the lettering on it is measured from.
        tile_lo, tile_hi = (-tile_h, 0.0) if flush else (-sink, tile_h)

        info_text: List = []
        legend_solids: dict = {}
        tiles: List = []

        def _tile(x0, x1, y0, y1):
            """A white field for the lettering — proud of the face, or level with it."""
            return _rounded_slab(x0, x1, y0, y1, tile_lo, tile_hi,
                                 min(2.2, max(1e-3, (y1 - y0)) * 0.2))

        def _emit(rows, x_left, target, tiled):
            """Lay one block out from the back of the apron toward the front.

            Both blocks set from their own left edge. The legend used to be set
            flush right, on the reasoning that ragged endings read as an
            accident — but flush right puts the *colour dots* down a ragged
            edge, and the dots are the thing being scanned. A reader looking up
            which colour is which subunit wants them in a column; where the
            names happen to end is not something anyone reads.

            ``target`` may be a list or a callable taking ``(row, solids)``, which
            is how the legend routes each row's geometry into its own per-chain
            object instead of one shared one.

            Solids are classed as *ink* or not.  Ink is lettering: engraved, it
            becomes a cutter and no object at all.  A legend's colour dot is not
            ink — a dot's whole job is to be a colour, so it stays a raised
            object in its chain's filament however the text beside it is made.
            """
            cursor = apron_top - pad
            block_top = cursor
            use_sink = (flush_sink if flush
                        else (tile_sink if tiled else sink))
            base_z = tile_hi if tiled else 0.0
            # Flush lettering has no height above the face at all; that is what
            # flush means. Everything below reads these two and needs no other
            # branch on the mode.
            rise_mm = 0.0 if flush else emboss
            # Actual ink extents, so a tile can be cut to the text rather than to
            # the column it was allotted. A block of three short chain names in a
            # half-plate-wide slot left most of its tile blank, which read as a
            # misprint rather than as margin.
            used = {"x0": None, "x1": None}

            def put(row, solids, ink=True):
                if ink and engrave:
                    cutters.extend(solids)
                elif callable(target):
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
                text_w = typeset.text_width(face, row.text, row.cap_mm) if row.text else 0.0
                x = x_left
                mark(x, x + lead + text_w)

                if row.kind == "legend":
                    # Engraved, a colour dot is a lie: there is one filament, and
                    # a raised disc of it says nothing the letters beside it have
                    # not already said. So it is cut like everything else, and
                    # whoever wants it coloured can put paint in it.
                    dot_r = row.cap_mm * 0.52
                    dot_lo = base_z - (cut_depth if engrave else use_sink)
                    dot_hi = base_z + (cut_proud if engrave else rise_mm)
                    put(row, [_manifold.frustum(
                        [x + dot_r, baseline + dot_r, dot_lo],
                        [x + dot_r, baseline + dot_r, dot_hi],
                        dot_r, dot_r)], ink=engrave)
                    x += row.cap_mm * 1.6

                elif row.kind == "scalebar":
                    bar_h = max(0.8, row.cap_mm * 0.3)
                    depth = cut_depth if engrave else use_sink
                    rise = cut_proud if engrave else rise_mm
                    bars = [_manifold.oriented_box(
                        [x + row.bar_mm * 0.5, baseline + bar_h * 0.5,
                         base_z + 0.5 * (rise - depth)],
                        np.array([axis_u, axis_v, normal]),
                        [row.bar_mm * 0.5, bar_h * 0.5, 0.5 * (rise + depth)])]
                    for end in (0.0, row.bar_mm):
                        bars.append(_manifold.oriented_box(
                            [x + end, baseline + row.cap_mm * 0.4,
                             base_z + 0.5 * (rise - depth)],
                            np.array([axis_u, axis_v, normal]),
                            [bar_h * 0.5, row.cap_mm * 0.4,
                             0.5 * (rise + depth)]))
                    put(row, bars)
                    x += row.bar_mm + row.cap_mm * 0.7

                if row.text:
                    put(row, _text_solids(
                        face, row.text, row.cap_mm,
                        np.array([x, baseline, base_z]),
                        axis_u, axis_v, normal, stroke,
                        cut_proud if engrave else rise_mm,
                        cut_depth if engrave else use_sink,
                        typeset.grow_for(face, row.cap_mm, min_stroke)))

            if tiled and rows and used["x0"] is not None:
                tiles.append(_tile(used["x0"] - pad * 0.5, used["x1"] + pad * 0.5,
                                   cursor - pad * 0.45, block_top + pad * 0.45))

        def _ink_width(rows) -> float:
            """How wide a block's widest row actually draws."""
            widest = 0.0
            for row in rows:
                lead = 0.0
                if row.kind == "legend":
                    lead = row.cap_mm * 1.6
                elif row.kind == "scalebar":
                    lead = row.bar_mm + row.cap_mm * 0.7
                text_w = (typeset.text_width(face, row.text, row.cap_mm)
                          if row.text else 0.0)
                widest = max(widest, lead + text_w)
            return widest

        info_left = layout.plate_x0 + pad
        legend_right = layout.plate_x1 - pad
        # The block is anchored to the right margin, the rows are set from its
        # left. Setting the rows themselves flush right lined the *names* up and
        # left the colour dots down a ragged edge, which is the wrong thing to
        # line up; but anchoring nothing at all let the tile — which is cut to
        # the lettering, not to the space allotted — drift away from the edge of
        # the plate whenever the legend had width to spare.
        legend_left = legend_right - min(layout.legend_width,
                                         _ink_width(legend_rows) or layout.legend_width)

        def _legend_target(row, solids):
            """Route a legend row's dot *and* its lettering into one object.

            Both carry the chain's colour, so the row is a single filament change
            in the slicer and the name is as identifiable as the dot beside it.
            """
            legend_solids.setdefault(row.built_index, []).extend(solids)

        # One switch, both blocks. Anything else means a control labelled
        # "white tile" that leaves a white tile behind when it is switched off,
        # which reads as a defect however well the exception is justified.
        _emit(info_rows, info_left, info_text, tiled=tiled)
        _emit(legend_rows, legend_left, _legend_target,
              tiled=tiled and bool(legend_rows))

        cutters = [_place(s, face_xf) for s in cutters]
        if tiles:
            tile_solid = _place(_manifold.union(tiles), face_xf)
            text_parts.append((StandPart("tile", "stand_plaque_tile", TILE_COLOR),
                               tile_solid, False))
        if info_text:
            text_parts.append((
                StandPart("text", "stand_plaque_text",
                          TEXT_ON_TILE if tiled else TEXT_ON_PLATE),
                _place(_manifold.union(info_text), face_xf), tiled))
        for built_index, solids in legend_solids.items():
            if not solids:
                continue
            row = next(r for r in layout.chain_rows if r["index"] == built_index)
            text_parts.append((
                StandPart("legend", f"stand_legend_{row['chain_id']}",
                          row["color"], chain_id=row["chain_id"]),
                _place(_manifold.union(solids), face_xf), True))

    # ---- fuse and hand back --------------------------------------------
    from . import meshops

    #: Keep every connected component of a stand part, however small.
    #:
    #: ``meshops.repair`` drops components under 2% of the largest, which is
    #: right for a marching-cubes mesh where the small pieces are stray specks
    #: and wrong for lettering, where the small pieces are the dot on an 'i' and
    #: the full stop. Against a 5 mm headline letter an 'i' dot at 2.3 mm is
    #: about 1.7% by volume — just under the bar — so it silently vanished, and
    #: only from words that happened to contain one.
    #:
    #: Nothing in a stand comes from a voxel grid. It is all analytic solids out
    #: of the manifold kernel, watertight by construction, and every disconnected
    #: piece is there because something put it there.
    keep_all = dict(min_component_frac=0.0)

    stand_parts: List = []
    base_solid = _manifold.union(parts)

    for cut in plate_cuts:
        try:
            trimmed = _manifold.difference(base_solid, cut)
            if not trimmed.is_empty():
                base_solid = trimmed
        except Exception:
            notes.append("A pin socket could not be cut into the plate; that "
                         "column will have to be glued.")

    # ``on_tile`` decides what each solid is cut *out of*.  The plaque can be
    # three layers deep — plate, white tile, lettering — and a raised solid has
    # to be recessed into whatever it actually stands on. Cutting everything out
    # of the plate, as when the lettering sat directly on it, would leave the
    # letters overlapping the tile in shared volume, which is exactly what a
    # multi-material slicer objects to.
    #
    # The tile's own recess is cut from the *unengraved* tile, so that engraving
    # through it leaves a void rather than a plate-coloured pillar rising to sit
    # flush in it. Flush would be a handsome inlay on a two-material printer and
    # nothing at all on a one-material one — and not coming loose on a
    # one-material printer is the entire reason engraving is offered.
    for part, solid, on_tile in text_parts:
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

    if cutters:
        # One union then one subtraction. Cutting a few hundred stroke solids out
        # of the plate one at a time re-meshes the whole plate a few hundred
        # times, and took four times as long as building everything else put
        # together.
        try:
            tool = _manifold.union(cutters)
            if tile_solid is not None:
                trimmed = _manifold.difference(tile_solid, tool)
                if not trimmed.is_empty():
                    tile_solid = trimmed
            else:
                trimmed = _manifold.difference(base_solid, tool)
                if not trimmed.is_empty():
                    base_solid = trimmed
        except Exception:
            notes.append("The engraved lettering could not be cut into the "
                         "apron; it was left out.")

    tile_index = next((i for i, (p, _s, _t) in enumerate(text_parts)
                       if p.part == "tile"), None)

    stand_parts.append((
        StandPart("base", "stand_base", STAND_COLOR),
        meshops.repair(_manifold.to_trimesh(base_solid), **keep_all)))
    for part, solid in column_parts:
        try:
            stand_parts.append((part, meshops.repair(
                _manifold.to_trimesh(solid), **keep_all)))
        except Exception:
            notes.append(f"Left out {part.object_name()}: it did not mesh.")
    for i, (part, solid, _on_tile) in enumerate(text_parts):
        if i == tile_index:
            solid = tile_solid
        if solid is None:
            continue
        try:
            stand_parts.append((part, meshops.repair(
                _manifold.to_trimesh(solid), **keep_all)))
        except Exception:
            notes.append(f"Left out {part.object_name()}: it did not mesh.")

    return layout.oriented_built, stand_parts, notes
