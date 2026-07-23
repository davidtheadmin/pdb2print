"""Tube-and-slab representation for nucleic acids.

Backbone = a smooth tube swept along a spline through the phosphate/backbone
trace.  Bases = oriented slabs placed on each nucleotide's base plane, which is
fitted from that base's ring atoms using a standard reference-atom set.

Tube, slabs and connector struts are built as *exact analytic primitives*
(capsules and oriented boxes) and fused with a single guaranteed-watertight
mesh boolean via the manifold kernel — no voxel grid, so no stairstep.  The same
kernel backs the planned client-side WASM build, so this is not throwaway work.

Minimum wall thickness is a **parametric offset**: the tube radius, slab
thickness and connector radius are each grown to satisfy ``min_wall_mm`` *before*
the union, rather than by re-voxelising the finished mesh.  Because we control
the primitives, this is exact and adds no grid texture.
"""

from __future__ import annotations

import numpy as np

from ..config import PrintParams
from ._common import catmull_rom
from . import _manifold


# Ring atoms that define each base's plane and in-plane orientation.
# Purines (A, G) use the fused two-ring system; pyrimidines (C, T, U) the
# single ring.  These are the canonical PDB atom names.
_PURINE_RING = ["N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"]
_PYRIMIDINE_RING = ["N1", "C2", "N3", "C4", "C5", "C6"]

_PURINE_RESNAMES = {"DA", "DG", "A", "G", "I", "DI"}
# The glycosidic attachment atom, used to orient the slab away from the backbone.
_GLYCO_ATOM = {"purine": "N9", "pyrimidine": "N1"}

# Backbone trace atom preference order.
_BACKBONE_ATOMS = ["P", "C5'", "O5'", "C4'"]


def _residue_iter(atoms):
    """Yield ``(res_name, residue_atom_array)`` in chain order."""
    import biotite.structure as struc
    starts = struc.get_residue_starts(atoms)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else atoms.array_length()
        res = atoms[start:end]
        yield str(res.res_name[0]), res


def _atom_coord(res, name):
    idx = np.where(res.atom_name == name)[0]
    if len(idx) == 0:
        return None
    return res.coord[idx[0]].astype(float)


def _backbone_point(res):
    for name in _BACKBONE_ATOMS:
        c = _atom_coord(res, name)
        if c is not None:
            return c
    return res.coord.mean(axis=0)


def _base_frame(res_name, res):
    """Fit a base plane: return (center, normal, in_plane_dir) or ``None``.

    ``normal`` is the base-plane normal; ``in_plane_dir`` points from the
    glycosidic atom toward the ring centroid (the long axis of the slab).
    """
    is_purine = res_name in _PURINE_RESNAMES
    ring = _PURINE_RING if is_purine else _PYRIMIDINE_RING
    glyco = _GLYCO_ATOM["purine" if is_purine else "pyrimidine"]

    pts = [c for c in (_atom_coord(res, n) for n in ring) if c is not None]
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    center = pts.mean(axis=0)

    # Plane normal = smallest-singular-value direction of the centred ring.
    _, _, vh = np.linalg.svd(pts - center)
    normal = vh[2]

    glyco_c = _atom_coord(res, glyco)
    if glyco_c is not None:
        in_plane = center - glyco_c
    else:
        in_plane = pts[0] - center
    # Orthogonalise against the normal and normalise.
    in_plane = in_plane - normal * (in_plane @ normal)
    n_in = np.linalg.norm(in_plane)
    if n_in < 1e-6:
        return None
    in_plane /= n_in
    return center, normal / np.linalg.norm(normal), in_plane


def _min_wall_dims(params: PrintParams):
    """Effective (tube_radius, slab_thickness, connector_radius) after min-wall.

    A solid tube/strut of radius ``r`` is ``2r`` thick and a slab is ``slab_t``
    thick, so the minimum-wall guarantee becomes ``r ≥ min_wall/2`` and
    ``slab_t ≥ min_wall`` — a parametric growth of the primitives we control.
    """
    tube_r = params.nucleic_radius_mm
    slab_t = params.slab_thickness_mm
    conn_r = params.connector_radius_mm
    if params.min_wall_mm > 0:
        tube_r = max(tube_r, params.min_wall_mm / 2.0)
        slab_t = max(slab_t, params.min_wall_mm)
        conn_r = max(conn_r, params.min_wall_mm / 2.0)
    return tube_r, slab_t, conn_r


def build(chain, params: PrintParams):
    """Return a watertight trimesh of the tube-and-slab model for ``chain``.

    Tube, base slabs and connector struts are exact analytic primitives fused by
    one manifold boolean union, so the nucleotide comes out as a single
    watertight body with no grid texture.  Minimum wall thickness is applied as a
    parametric offset on those primitives (see :func:`_min_wall_dims`).
    """
    s = params.scale_mm_per_angstrom
    tube_r, slab_t, conn_r = _min_wall_dims(params)

    residues = list(_residue_iter(chain.atoms))
    backbone = np.array([_backbone_point(res) for _, res in residues]) * s

    solids = []

    # Backbone tube: a capsule per spline segment (consecutive capsules overlap,
    # so the union is a single smooth watertight tube).
    if len(backbone) >= 2:
        spline = catmull_rom(backbone, params.spline_samples_per_residue)
        for i in range(len(spline) - 1):
            solids.append(_manifold.capsule(spline[i], spline[i + 1], tube_r))
    elif len(backbone) == 1:
        solids.append(_manifold.capsule(backbone[0], backbone[0], tube_r))

    # Base slabs, each fused to the tube by a connector strut running from the
    # backbone point (on the tube) to the slab centre (inside the slab).
    for (res_name, res), bp in zip(residues, backbone):
        frame = _base_frame(res_name, res)
        if frame is None:
            continue
        center, normal, long_axis = frame
        center_mm = center * s
        third = np.cross(normal, long_axis)
        third /= (np.linalg.norm(third) or 1.0)
        axes = np.array([long_axis, third, normal])
        # Slab footprint: a base is roughly 4.5 x 3.0 angstrom, scaled to mm;
        # through-plane thickness is exactly slab_t.
        half = np.array([4.5, 3.0, 1.0]) * 0.5 * params.slab_scale * s
        half[2] = slab_t * 0.5
        solids.append(_manifold.oriented_box(center_mm, axes, half))
        solids.append(_manifold.capsule(bp, center_mm, conn_r))

    fused = _manifold.union(solids)
    return _manifold.to_trimesh(fused)
