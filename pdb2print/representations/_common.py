"""Shared geometry helpers for all representations.

The unifying idea: every representation rasterises its primitives into a scalar
field on a regular grid, then a single marching-cubes extraction produces a
closed, manifold shell.  This is what makes the output watertight *by
construction* rather than by repairing CSG results after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from skimage import measure


@dataclass
class Grid:
    """A regular voxel grid covering a bounding box (all in print-mm)."""

    origin: np.ndarray      # world coord of voxel (0,0,0) centre, shape (3,)
    spacing: float
    shape: tuple            # (nx, ny, nz)

    @classmethod
    def covering(cls, points: np.ndarray, spacing: float, pad: float) -> "Grid":
        """Grid that covers ``points`` with ``pad`` mm of margin on every side."""
        lo = points.min(axis=0) - pad
        hi = points.max(axis=0) + pad
        shape = tuple(int(np.ceil((hi[i] - lo[i]) / spacing)) + 1 for i in range(3))
        return cls(origin=lo, spacing=float(spacing), shape=shape)

    def axis_coords(self):
        return [
            self.origin[i] + np.arange(self.shape[i]) * self.spacing
            for i in range(3)
        ]

    def window(self, center: np.ndarray, radius: float):
        """Index slices of the sub-grid within ``radius`` of ``center``.

        Returns ``(slices, coord_grids)`` where coord_grids are the world
        coordinates of the window voxels (meshgrid, 'ij' indexing).  Used to
        rasterise a primitive cheaply into just its local neighbourhood.
        """
        lo_idx = np.floor((center - radius - self.origin) / self.spacing).astype(int)
        hi_idx = np.ceil((center + radius - self.origin) / self.spacing).astype(int)
        lo_idx = np.maximum(lo_idx, 0)
        hi_idx = np.minimum(hi_idx, np.array(self.shape) - 1)
        if np.any(hi_idx < lo_idx):
            return None, None
        slices = tuple(slice(lo_idx[i], hi_idx[i] + 1) for i in range(3))
        axes = [
            self.origin[i] + np.arange(lo_idx[i], hi_idx[i] + 1) * self.spacing
            for i in range(3)
        ]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        return slices, np.stack([gx, gy, gz], axis=-1)


def field_to_mesh(field: np.ndarray, grid: Grid, level: float) -> trimesh.Trimesh:
    """Marching-cubes a scalar field into a trimesh in world (print-mm) space."""
    if not (field.min() < level < field.max()):
        raise ValueError(
            "Iso-level is outside the field range; nothing to extract "
            f"(min={field.min():.3f}, max={field.max():.3f}, level={level})."
        )
    verts, faces, normals, _ = measure.marching_cubes(
        field, level=level, spacing=(grid.spacing,) * 3, allow_degenerate=False
    )
    verts = verts + grid.origin
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                           process=False)
    return mesh


def catmull_rom(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Interpolate a smooth Catmull-Rom spline through ``points`` (N>=2, 3D)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    # Duplicate endpoints so the spline reaches the first/last control points.
    p = np.vstack([points[0], points, points[-1]])
    out = []
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    t2, t3 = t * t, t * t * t
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        seg = (
            0.5 * (
                (2 * p1)
                + (-p0 + p2) * t[:, None]
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2[:, None]
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3[:, None]
            )
        )
        out.append(seg)
    out.append(points[-1][None, :])
    return np.vstack(out)


def rasterize_capsule(field, grid, a, b, radius):
    """Union a capsule (segment ``a``-``b`` of ``radius``) into ``field`` as occupancy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    seg_center = 0.5 * (a + b)
    seg_half = 0.5 * np.linalg.norm(b - a)
    slices, coords = grid.window(seg_center, radius + seg_half)
    if slices is None:
        return
    ab = b - a
    ab_len2 = float(ab @ ab) or 1e-9
    rel = coords - a
    t = np.clip((rel @ ab) / ab_len2, 0.0, 1.0)
    closest = a + t[..., None] * ab
    dist = np.linalg.norm(coords - closest, axis=-1)
    inside = dist <= radius
    field[slices] = np.maximum(field[slices], inside.astype(field.dtype))


def rasterize_box(field, grid, center, axes, half_extents):
    """Union an oriented box into ``field`` as occupancy.

    ``axes`` is a 3x3 matrix whose rows are the box's orthonormal local axes;
    ``half_extents`` are the half-sizes along each axis.
    """
    center = np.asarray(center, float)
    axes = np.asarray(axes, float)
    half_extents = np.asarray(half_extents, float)
    reach = float(np.linalg.norm(half_extents)) + grid.spacing
    slices, coords = grid.window(center, reach)
    if slices is None:
        return
    rel = coords - center
    # Project onto each local axis; inside iff |proj| <= half extent for all.
    inside = np.ones(coords.shape[:-1], dtype=bool)
    for k in range(3):
        proj = rel @ axes[k]
        inside &= np.abs(proj) <= half_extents[k]
    field[slices] = np.maximum(field[slices], inside.astype(field.dtype))
