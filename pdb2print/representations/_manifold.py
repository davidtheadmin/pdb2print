"""Analytic mesh primitives and booleans via the manifold kernel.

Representations that are made of exact primitives (a swept tube, oriented base
slabs, connector struts) build them here as :class:`manifold3d.Manifold` solids
and fuse them with a single guaranteed-watertight boolean union — no voxel grid,
so no stairstep.  ``manifold3d`` is the same kernel the planned client-side WASM
build uses, so this is not throwaway work.

Everything is in print-millimetre space, matching the rest of the pipeline.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import trimesh

import manifold3d as m3d
from manifold3d import CrossSection, Manifold


# Circular tessellation of tubes/spheres.  Modest on purpose: the union is
# watertight regardless, and coarser primitives keep face counts printable.
_SEGMENTS = 20


def _rot_z_to(direction: np.ndarray) -> np.ndarray:
    """Rotation whose columns map local (x, y, z) so that +z aligns to ``direction``."""
    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.eye(3)
    d = d / n
    # Any reference axis not near-parallel to d gives a stable in-plane basis.
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return np.column_stack([x, y, d])


def _cylinder_between(a, b, radius: float, segments: int) -> Manifold:
    """A flat-ended cylinder of ``radius`` spanning ``a``..``b``."""
    cyl = Manifold.cylinder(float(np.linalg.norm(b - a)), radius, radius,
                            segments, center=True)
    transform = np.column_stack([_rot_z_to(b - a), 0.5 * (a + b)])
    return cyl.transform(transform.tolist())


def capsule_parts(a, b, radius: float, segments: int = _SEGMENTS):
    """The primitives a capsule is made of, *unfused*.

    A caller that is going to union its capsules together anyway wants these
    rather than :func:`capsule`.  Union is associative, so fusing three
    primitives here and fusing the results again reaches the same solid through
    hundreds of nested booleans where one flat batch would do.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if float(np.linalg.norm(b - a)) < 1e-9:
        return [Manifold.sphere(radius, segments).translate(tuple(a))]
    return [
        _cylinder_between(a, b, radius, segments),
        Manifold.sphere(radius, segments).translate(tuple(a)),
        Manifold.sphere(radius, segments).translate(tuple(b)),
    ]


def swept_tube_parts(points, radius: float, segments: int = _SEGMENTS):
    """Unfused primitives for a tube of ``radius`` swept along ``points``.

    One capsule per segment builds the sphere at every interior sample **twice**
    — as the end cap of one segment and again as the start cap of the next — so
    an ``n``-sample spline costs ``3n - 3`` primitives where ``2n - 1`` is the
    same solid.  Emitting one sphere per sample and one cylinder per segment
    drops the duplicates and leaves the caller a single flat union.
    """
    pts = np.asarray(points, float)
    if len(pts) == 0:
        return []
    parts = [Manifold.sphere(radius, segments).translate(tuple(p)) for p in pts]
    for a, b in zip(pts[:-1], pts[1:]):
        if float(np.linalg.norm(b - a)) < 1e-9:
            continue
        parts.append(_cylinder_between(a, b, radius, segments))
    return parts


def capsule(a, b, radius: float, segments: int = _SEGMENTS) -> Manifold:
    """A capsule (cylinder with hemispherical caps) between points ``a`` and ``b``."""
    parts = capsule_parts(a, b, radius, segments)
    if len(parts) == 1:
        return parts[0]
    return Manifold.batch_boolean(parts, m3d.OpType.Add)


def sphere(center, radius: float, segments: int = _SEGMENTS) -> Manifold:
    """A sphere of ``radius`` centred at ``center`` (used for ball-and-stick)."""
    center = np.asarray(center, float)
    return Manifold.sphere(radius, segments).translate(tuple(center))


def frustum(a, b, r_a: float, r_b: float, segments: int = _SEGMENTS) -> Manifold:
    """A flat-ended (truncated) cone from ``a`` (radius ``r_a``) to ``b`` (radius ``r_b``).

    With ``r_a == r_b`` this is a plain flat-ended cylinder; a differing pair
    gives the chamfered mouths and tapered pegs the connector system needs.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    if length < 1e-9:
        length = 1e-6
    cyl = Manifold.cylinder(length, r_a, r_b, segments, center=True)
    transform = np.column_stack([_rot_z_to(b - a), 0.5 * (a + b)])
    return cyl.transform(transform.tolist())


#: Slab pitch and ceiling for :func:`sweep_up`.  0.25 mm is well under the
#: thinnest thing a cradle clearance can leave, and 64 slabs cover 16 mm of
#: sweep at that pitch -- more than a cradle is ever deep.
_SWEEP_PITCH_MM = 0.25
_SWEEP_MAX_SLABS = 64


def sweep_up(manifold: Manifold, z_top: float,
             pitch: float = _SWEEP_PITCH_MM) -> Optional[Manifold]:
    """Everything from ``manifold`` upward, as a solid, capped at ``z_top``.

    The vertical Minkowski sweep of a solid: a point is in the result when the
    solid occupies anything at or below it in the same vertical line.  Subtract
    that from a part and no material can survive *above* a place the solid
    itself would have cut -- which is the difference between a seat and a
    tunnel with a lid on it.

    Built from horizontal slices rather than as a true sweep, because the
    kernel has no Minkowski and a union of translated copies costs a boolean
    per step.  Each slab takes the cross-section at its *floor* and extrudes it
    to its ceiling, and the cross-sections accumulate going up, so the result
    only ever grows with height.  Taking the floor's section is what makes this
    safe to combine with an exact difference: a slab never reaches below the
    height at which the solid actually starts, so the imprint the plain
    difference cuts is left alone and only the material above it is removed.

    Returns ``None`` when there is nothing to sweep.
    """
    if manifold is None or manifold.is_empty():
        return None
    box = manifold.bounding_box()
    z_lo, z_hi = float(box[2]), min(float(box[5]), float(z_top))
    if z_top <= z_lo:
        return None
    pitch = max(0.05, float(pitch))
    span = max(0.0, z_hi - z_lo)
    steps = int(math.ceil(span / pitch)) + 1
    if steps > _SWEEP_MAX_SLABS:
        steps = _SWEEP_MAX_SLABS
        pitch = span / (steps - 1) if steps > 1 else pitch

    acc = None
    slabs = []
    for k in range(steps):
        z = z_lo + k * pitch
        # The last slab carries the accumulated section the rest of the way to
        # the cap, so material above the solid's own top still goes.
        ceiling = float(z_top) if k == steps - 1 else min(z + pitch, float(z_top))
        try:
            section = manifold.slice(z)
        except Exception:
            section = None
        if section is not None and not section.is_empty():
            acc = section if acc is None else acc + section
            # Contours would otherwise compound with every union; a hundredth of
            # a millimetre is far below anything a printer resolves.
            try:
                acc = acc.simplify(0.01)
            except Exception:
                pass
        if acc is None or acc.is_empty():
            continue
        height = ceiling - z
        if height <= 1e-6:
            continue
        slabs.append(acc.extrude(height).translate((0.0, 0.0, z)))

    if not slabs:
        return None
    return union(slabs)


def cross_section(contour) -> CrossSection:
    """A 2D region from one closed polygon, given as ``(N, 2)`` points."""
    return CrossSection([np.asarray(contour, float)])


def footprint_below(manifold: Manifold, z_top: float) -> CrossSection:
    """The XY shadow of everything in ``manifold`` at or below ``z_top``.

    Seen from above, this is exactly what a downward-only cut takes out of a
    column that reaches ``z_top``: subtract it from the column's own outline and
    what is left is the flat top face, pieces and all.
    """
    if manifold is None or manifold.is_empty():
        return CrossSection()
    box = manifold.bounding_box()
    if box[2] >= z_top:
        return CrossSection()
    mid_z = 0.5 * (box[2] - 1.0 + z_top)
    size = (max(1e-3, box[3] - box[0]) + 2.0,
            max(1e-3, box[4] - box[1]) + 2.0,
            max(1e-3, z_top - (box[2] - 1.0)))
    clip = (Manifold.cube(size, center=True)
            .translate((0.5 * (box[0] + box[3]), 0.5 * (box[1] + box[4]), mid_z)))
    part = Manifold.batch_boolean([manifold, clip], m3d.OpType.Intersect)
    if part.is_empty():
        return CrossSection()
    return part.project()


def section_pieces(outline: CrossSection, shadow: CrossSection):
    """Areas of the connected pieces of ``outline`` minus ``shadow``, largest first."""
    if outline is None or outline.is_empty():
        return []
    keep = (outline if shadow is None or shadow.is_empty()
            else CrossSection.batch_boolean([outline, shadow], m3d.OpType.Subtract))
    if keep.is_empty():
        return []
    return sorted((float(c.area()) for c in keep.decompose()), reverse=True)


def section_overlap(a: CrossSection, b: CrossSection) -> float:
    """Area the two 2D regions have in common."""
    if a is None or b is None or a.is_empty() or b.is_empty():
        return 0.0
    return float(CrossSection.batch_boolean(
        [a, b], m3d.OpType.Intersect).area())


def oriented_box(center, axes, half_extents) -> Manifold:
    """An oriented box.

    ``axes`` is a 3x3 matrix whose *rows* are the box's orthonormal local axes;
    ``half_extents`` are the half-sizes along each of those axes.
    """
    center = np.asarray(center, float)
    axes = np.asarray(axes, float)
    size = tuple(2.0 * np.asarray(half_extents, float))
    box = Manifold.cube(size, center=True)
    # Rows of ``axes`` are the local basis in world coords, so its transpose is
    # the rotation that maps local -> world.
    transform = np.column_stack([axes.T, center])
    return box.transform(transform.tolist())


def union(manifolds) -> Manifold:
    """Fuse many manifolds into one watertight solid (drops empties)."""
    parts = [m for m in manifolds if m is not None and not m.is_empty()]
    if not parts:
        raise ValueError("No primitives to union.")
    if len(parts) == 1:
        return parts[0]
    return Manifold.batch_boolean(parts, m3d.OpType.Add)


def difference(a: Manifold, b: Manifold) -> Manifold:
    """Boolean subtraction ``a - b`` (watertight by construction)."""
    return a - b


def intersection(a: Manifold, b: Manifold) -> Manifold:
    """Boolean intersection ``a ∩ b``.

    Goes through ``batch_boolean`` rather than the ``^`` operator because the
    operator overloads have moved around between ``manifold3d`` releases while
    the explicit ``OpType`` call has been stable.
    """
    return Manifold.batch_boolean([a, b], m3d.OpType.Intersect)


def volume(manifold: Manifold) -> float:
    """Enclosed volume of a manifold, or 0.0 if it is empty.

    Prefers the kernel's own accessor, which has been a property in some
    ``manifold3d`` releases and a method in others — hence the probe rather than
    a straight call.  The mesh conversion is kept only as a fallback: it is
    perfectly accurate but it materialises the whole mesh to measure it, which is
    wasted work on anything bigger than a probe primitive.
    """
    if manifold is None or manifold.is_empty():
        return 0.0
    try:
        v = manifold.volume
        return float(abs(v() if callable(v) else v))
    except Exception:
        pass
    try:
        return float(abs(to_trimesh(manifold).volume))
    except Exception:
        return 0.0


def to_trimesh(manifold: Manifold) -> trimesh.Trimesh:
    """Convert a :class:`Manifold` to a :class:`trimesh.Trimesh` (print-mm space)."""
    if manifold.is_empty():
        raise ValueError("Cannot convert an empty manifold to a mesh.")
    mesh = manifold.to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    faces = np.asarray(mesh.tri_verts).astype(np.int64)
    # process=False: the manifold is already indexed/de-duplicated and closed;
    # downstream repair() will merge_vertices/fix_normals before the watertight
    # gate, so no cleanup is needed here.
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def from_trimesh(mesh: trimesh.Trimesh) -> Manifold:
    """Convert a (watertight) :class:`trimesh.Trimesh` back into a :class:`Manifold`.

    Used by the connections pass to re-enter the manifold kernel and apply
    booleans (adding bridges/pegs, subtracting pockets) to an already-built
    per-chain mesh, so the result stays watertight by construction.
    """
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
    return Manifold(m3d.Mesh(vert_properties=verts, tri_verts=faces))
