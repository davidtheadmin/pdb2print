"""Mesh post-processing: watertight/manifold repair and min-wall enforcement.

Both passes are representation-agnostic — they run identically on a surface mesh
or a tube-and-slab mesh — which is exactly the separation the spec asks for:
whatever geometry a chain uses, these run afterward as their own passes.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

from .config import PrintParams, MinWallMode, needs_min_wall


# Connected components smaller than this fraction of the largest are treated as
# noise and dropped; anything above it is kept.  Intentional geometry (e.g. base
# slabs, ~5% of a tube) survives, but true specks do not.
DEFAULT_MIN_COMPONENT_FRAC = 0.02


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------
def repair(mesh: trimesh.Trimesh,
           min_component_frac: float = DEFAULT_MIN_COMPONENT_FRAC) -> trimesh.Trimesh:
    """Make ``mesh`` watertight, manifold and consistently oriented.

    Marching-cubes output is already closed; this cleans up degeneracies and,
    if pymeshlab is available, runs a stronger non-manifold repair pass.

    Multiple connected components are kept as long as each is at least
    ``min_component_frac`` of the largest — so intentional geometry (base slabs,
    multi-domain chains) is never silently discarded, while tiny stray specks
    still are.  (Previously this kept only the single largest component, which
    was silently deleting disconnected base slabs.)

    Fast path: analytic representations (tube-slab, SES surface) come out of the
    manifold/marching-cubes kernel already watertight.  The cleanup below
    (notably ``merge_vertices``) can weld near-coincident vertices at
    overlapping-primitive seams and *pinch* such a mesh into a non-manifold, so
    an already-watertight mesh skips it entirely.

    That fast path covers the multi-body case too, and must.  A chain can arrive
    here watertight in several pieces — the interference pass has to cut a loop
    that a DNA duplex genuinely threads through — and sending that through the
    cleanup destroyed it: the mesh went in watertight and came out not, which
    then tripped the export gate.  Dropping specks needs only split and
    concatenate, both of which preserve watertightness.
    """
    mesh.fix_normals()
    if mesh.is_watertight:
        return mesh if mesh.body_count == 1 else _drop_specks(mesh, min_component_frac)

    mesh.remove_duplicate_faces() if hasattr(mesh, "remove_duplicate_faces") else None
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        mesh = _pymeshlab_repair(mesh)

    if mesh.body_count > 1:
        mesh = _drop_specks(mesh, min_component_frac)
    mesh.fix_normals()
    return mesh


def _drop_specks(mesh: trimesh.Trimesh, min_component_frac: float) -> trimesh.Trimesh:
    """Keep every component at least ``min_component_frac`` of the largest.

    Split and concatenate only — no vertex welding — so a watertight input stays
    watertight.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        return mesh
    sizes = np.array([abs(p.volume) if p.is_volume else p.area for p in parts])
    keep = [p for p, sz in zip(parts, sizes)
            if sz >= sizes.max() * min_component_frac]
    if not keep:
        return mesh
    out = keep[0] if len(keep) == 1 else trimesh.util.concatenate(keep)
    out.fix_normals()
    return out


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

    The pass is representation-scoped: meshes whose representation declines
    min-wall (e.g. Gaussian surfaces, already thick everywhere) are returned
    untouched, so they keep their crisp, un-inflated shape.
    """
    rep = mesh.metadata.get("representation")
    if rep is not None and not needs_min_wall(rep):
        return mesh
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
