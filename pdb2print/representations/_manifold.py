"""Analytic mesh primitives and booleans via the manifold kernel.

Representations that are made of exact primitives (a swept tube, oriented base
slabs, connector struts) build them here as :class:`manifold3d.Manifold` solids
and fuse them with a single guaranteed-watertight boolean union — no voxel grid,
so no stairstep.  ``manifold3d`` is the same kernel the planned client-side WASM
build uses, so this is not throwaway work.

Everything is in print-millimetre space, matching the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np
import trimesh

import manifold3d as m3d
from manifold3d import Manifold


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


def capsule(a, b, radius: float, segments: int = _SEGMENTS) -> Manifold:
    """A capsule (cylinder with hemispherical caps) between points ``a`` and ``b``."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    if length < 1e-9:
        return Manifold.sphere(radius, segments).translate(tuple(a))
    cyl = Manifold.cylinder(length, radius, radius, segments, center=True)
    transform = np.column_stack([_rot_z_to(b - a), 0.5 * (a + b)])
    parts = [
        cyl.transform(transform.tolist()),
        Manifold.sphere(radius, segments).translate(tuple(a)),
        Manifold.sphere(radius, segments).translate(tuple(b)),
    ]
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
