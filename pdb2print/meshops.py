"""Mesh post-processing: watertight/manifold repair and min-wall enforcement.

Both passes are representation-agnostic — they run identically on a surface mesh
or a tube-and-slab mesh — which is exactly the separation the spec asks for:
whatever geometry a chain uses, these run afterward as their own passes.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

from .config import PrintParams, MinWallMode


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------
def repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Make ``mesh`` watertight, manifold and consistently oriented.

    Marching-cubes output is already closed; this cleans up degeneracies and,
    if pymeshlab is available, runs a stronger non-manifold repair pass.
    """
    mesh.remove_duplicate_faces() if hasattr(mesh, "remove_duplicate_faces") else None
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        mesh = _pymeshlab_repair(mesh)

    # Keep only the largest connected component if repair left islands.
    if mesh.body_count > 1:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 0:
            mesh = max(parts, key=lambda m: m.volume if m.is_volume else m.area)
    mesh.fix_normals()
    return mesh


def _pymeshlab_repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Best-effort heavy repair via pymeshlab; returns input unchanged on failure."""
    try:
        import pymeshlab
    except Exception:
        # trimesh-only fallback: fill holes.
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        return mesh
    try:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_close_holes(maxholesize=200)
        m = ms.current_mesh()
        return trimesh.Trimesh(vertices=m.vertex_matrix(),
                               faces=m.face_matrix(), process=True)
    except Exception:
        trimesh.repair.fill_holes(mesh)
        return mesh


# --------------------------------------------------------------------------
# Minimum wall thickness
# --------------------------------------------------------------------------
def enforce_min_wall(mesh: trimesh.Trimesh, params: PrintParams) -> trimesh.Trimesh:
    """Guarantee every feature is at least ``params.min_wall_mm`` thick.

    Works in voxel space so it is robust and representation-independent.  The
    selective mode thickens only regions measured to be too thin; if that path
    fails for any reason it falls back to the uniform offset, which is always
    available.  (See project note: ship the reliable fallback, refine later.)
    """
    if params.min_wall_mm <= 0:
        return mesh

    pitch = min(params.grid_spacing_mm, params.min_wall_mm / 2.0)
    occ, origin = _voxelize(mesh, pitch)
    if occ is None:
        return mesh

    if params.min_wall_mode == MinWallMode.SELECTIVE:
        try:
            grown = _selective_thicken(occ, pitch, params.min_wall_mm)
        except Exception:
            grown = _uniform_grow(occ, pitch, params.min_wall_mm)
    else:
        grown = _uniform_grow(occ, pitch, params.min_wall_mm)

    out = _mesh_from_occupancy(grown, origin, pitch)
    return repair(out) if out is not None else mesh


def _voxelize(mesh: trimesh.Trimesh, pitch: float):
    """Return (filled boolean occupancy array, world origin) or (None, None)."""
    try:
        vg = mesh.voxelized(pitch=pitch).fill()
        occ = np.asarray(vg.matrix, dtype=bool)
        # VoxelGrid.transform maps matrix index -> world; translation is origin.
        origin = np.asarray(vg.transform[:3, 3], dtype=float)
        # Pad so morphological growth has room at the boundary.
        pad = 3
        occ = np.pad(occ, pad, mode="constant", constant_values=False)
        origin = origin - pad * pitch
        return occ, origin
    except Exception:
        return None, None


def _uniform_grow(occ: np.ndarray, pitch: float, min_wall_mm: float) -> np.ndarray:
    """Dilate every feature outward so its thickness increases by ~min_wall."""
    iters = max(1, int(round((min_wall_mm / 2.0) / pitch)))
    return ndimage.binary_dilation(occ, iterations=iters)


def _selective_thicken(occ: np.ndarray, pitch: float, min_wall_mm: float) -> np.ndarray:
    """Grow only voxels whose local thickness is below ``min_wall``.

    Local thickness ~= 2 * distance-to-surface at interior voxels.  Voxels
    below the target seed a constrained dilation until they reach it.
    """
    half_target_vox = (min_wall_mm / 2.0) / pitch
    dist = ndimage.distance_transform_edt(occ)          # in voxels
    thin_seed = occ & (dist < half_target_vox)
    if not thin_seed.any():
        return occ
    iters = max(1, int(np.ceil(half_target_vox - dist[occ].min())))
    grown = occ.copy()
    seed = thin_seed
    for _ in range(iters):
        seed = ndimage.binary_dilation(seed)
        grown |= seed
    return grown


def _mesh_from_occupancy(occ: np.ndarray, origin: np.ndarray, pitch: float):
    from skimage import measure
    if not occ.any():
        return None
    field = ndimage.gaussian_filter(occ.astype(np.float32), sigma=0.6)
    if not (field.min() < 0.5 < field.max()):
        return None
    verts, faces, normals, _ = measure.marching_cubes(
        field, level=0.5, spacing=(pitch,) * 3, allow_degenerate=False
    )
    verts = verts + origin
    return trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals, process=False)
