"""Tube-and-slab representation for nucleic acids.

Backbone = a smooth tube swept along a spline through the phosphate/backbone
trace.  Bases = oriented slabs placed on each nucleotide's base plane, which is
fitted from that base's ring atoms using a standard reference-atom set.  Both
are rasterised into one occupancy field and meshed together, so the union is
watertight without any CSG.

This is the pure-Python equivalent of ChimeraX's ``nucleotides tube/slab`` —
kept dependency-free so it ports to the planned WASM build.
"""

from __future__ import annotations

import numpy as np

from ..config import PrintParams
from ._common import (
    Grid, field_to_mesh, catmull_rom, rasterize_capsule, rasterize_box,
)


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


def build(chain, params: PrintParams):
    """Return a watertight trimesh of the tube-and-slab model for ``chain``."""
    s = params.scale_mm_per_angstrom
    tube_r = params.nucleic_radius_mm
    slab_t = params.slab_thickness_mm

    residues = list(_residue_iter(chain.atoms))
    backbone = np.array([_backbone_point(res) for _, res in residues]) * s

    # Base slabs: (center_mm, axes 3x3, half_extents_mm)
    slabs = []
    for res_name, res in residues:
        frame = _base_frame(res_name, res)
        if frame is None:
            continue
        center, normal, long_axis = frame
        center_mm = center * s
        third = np.cross(normal, long_axis)
        third /= (np.linalg.norm(third) or 1.0)
        axes = np.array([long_axis, third, normal])
        # Slab footprint: a base is roughly 4.5 x 3.0 angstrom; scale to mm.
        half = np.array([4.5, 3.0, slab_t / s * 0.5]) * 0.5 * params.slab_scale * s
        # Keep the through-plane half-extent exactly slab_t/2 in mm.
        half[2] = slab_t * 0.5
        slabs.append((center_mm, axes, half))

    # Bounding box over everything, then rasterise.
    all_pts = [backbone]
    for center_mm, _, half in slabs:
        reach = float(np.linalg.norm(half))
        all_pts.append(center_mm + reach)
        all_pts.append(center_mm - reach)
    pts = np.vstack(all_pts)

    spacing = params.grid_spacing_mm
    pad = max(tube_r, slab_t) + 2.0 * spacing
    grid = Grid.covering(pts, spacing=spacing, pad=pad)
    field = np.zeros(grid.shape, dtype=np.float32)

    # Backbone tube.
    if len(backbone) >= 2:
        spline = catmull_rom(backbone, params.spline_samples_per_residue)
        for i in range(len(spline) - 1):
            rasterize_capsule(field, grid, spline[i], spline[i + 1], tube_r)

    # Base slabs (and a short connector so a base can't float off the tube).
    for center_mm, axes, half in slabs:
        rasterize_box(field, grid, center_mm, axes, half)

    return field_to_mesh(field, grid, level=0.5)
