"""Tube-and-slab representation for nucleic acids.

Backbone = a smooth tube swept along a spline through the phosphate/backbone
trace, *or* a ball-and-stick model of the sugar-phosphate atoms.  Bases = one of
three interchangeable "rung" styles placed on each nucleotide:

* ``slab``     — an oriented box on the base plane (the original look);
* ``rod``      — a rounded cylinder rung, chunky and very printable;
* ``molecule`` — a ball-and-stick model of the base's ring atoms.

Every primitive (capsule, oriented box, sphere, connector strut) is an *exact
analytic solid* fused with a single guaranteed-watertight manifold boolean — no
voxel grid, so no stairstep.  The same kernel backs the planned client-side WASM
build, so this is not throwaway work.

Minimum wall thickness is a **parametric offset**: tube radius, slab/rod
thickness, connector radius and the molecule atom/bond radii are each grown to
satisfy ``min_wall_mm`` *before* the union, rather than by re-voxelising the
finished mesh.  Because we control the primitives, this is exact and adds no grid
texture.

Every base is tied to the backbone by a connector strut and consecutive
backbone atoms are linked, so each chain fuses into a single connected,
watertight, printable body regardless of the style combination chosen.
"""

from __future__ import annotations

import numpy as np

from ..config import PrintParams, BaseStyle, BackboneStyle
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

# Sugar-phosphate atoms drawn as balls in the "molecule" backbone style.
_BACKBONE_SUGAR_ATOMS = [
    "P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'",
    "C3'", "O3'", "C2'", "O2'", "C1'",
]

# Covalent-bond distance cutoff (ångström, pre-scale) used to draw sticks
# between atoms in the molecule styles.
_BOND_CUTOFF_ANG = 2.0


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
    """Fit a base plane: return (center, normal, in_plane_dir, glyco) or ``None``.

    ``normal`` is the base-plane normal; ``in_plane_dir`` points from the
    glycosidic atom toward the ring centroid (the long axis of the slab);
    ``glyco`` is the glycosidic atom coord (backbone attachment) or the ring
    centroid as a fallback.
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
        glyco_c = center
        in_plane = pts[0] - center
    # Orthogonalise against the normal and normalise.
    in_plane = in_plane - normal * (in_plane @ normal)
    n_in = np.linalg.norm(in_plane)
    if n_in < 1e-6:
        return None
    in_plane /= n_in
    return center, normal / np.linalg.norm(normal), in_plane, glyco_c


def _ring_atoms(res_name, res):
    """Return the base ring-atom coords (ångström) present in ``res``."""
    is_purine = res_name in _PURINE_RESNAMES
    ring = _PURINE_RING if is_purine else _PYRIMIDINE_RING
    return [c for c in (_atom_coord(res, n) for n in ring) if c is not None]


def _min_wall_dims(params: PrintParams):
    """Effective primitive dimensions after the min-wall parametric growth.

    A solid tube/strut/rod of radius ``r`` is ``2r`` thick and a slab is
    ``slab_t`` thick, so the minimum-wall guarantee becomes ``r ≥ min_wall/2``
    and ``slab_t ≥ min_wall`` — a parametric growth of the primitives we
    control.  Returns ``(tube_r, slab_t, conn_r, atom_r, bond_r)``.
    """
    tube_r = params.nucleic_radius_mm
    slab_t = params.slab_thickness_mm
    conn_r = params.connector_radius_mm
    atom_r = params.atom_radius_mm
    bond_r = params.bond_radius_mm
    if params.min_wall_mm > 0:
        half = params.min_wall_mm / 2.0
        tube_r = max(tube_r, half)
        slab_t = max(slab_t, params.min_wall_mm)
        conn_r = max(conn_r, half)
        atom_r = max(atom_r, half)
        bond_r = max(bond_r, half)
    return tube_r, slab_t, conn_r, atom_r, bond_r


def _ball_and_stick(coords_ang, s, atom_r, bond_r):
    """Spheres at each atom plus sticks between atoms within bonding distance.

    ``coords_ang`` is a list/array of atom coordinates in ångström; ``s`` is the
    mm-per-ångström scale.  Returns a list of manifold solids.
    """
    coords_ang = np.asarray(coords_ang, float)
    solids = []
    for c in coords_ang:
        solids.append(_manifold.sphere(c * s, atom_r))
    n = len(coords_ang)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords_ang[i] - coords_ang[j]) <= _BOND_CUTOFF_ANG:
                solids.append(
                    _manifold.capsule(coords_ang[i] * s, coords_ang[j] * s, bond_r)
                )
    return solids


def _backbone_solids(residues, backbone_mm, params, s, tube_r, atom_r, bond_r):
    """Build the backbone solids for the chosen style.

    ``backbone_mm`` are the per-residue trace points already scaled to mm.
    """
    solids = []
    if params.backbone_style == BackboneStyle.MOLECULE:
        # Ball-and-stick of the sugar-phosphate atoms, per residue, with the
        # trace points linked so the whole strand is one connected body.
        for (_, res) in residues:
            atoms = [c for c in (_atom_coord(res, n) for n in _BACKBONE_SUGAR_ATOMS)
                     if c is not None]
            if atoms:
                solids.extend(_ball_and_stick(atoms, s, atom_r, bond_r))
        for i in range(len(backbone_mm) - 1):
            solids.append(
                _manifold.capsule(backbone_mm[i], backbone_mm[i + 1], bond_r)
            )
        return solids

    # Default: smooth swept tube (a capsule per spline segment; consecutive
    # capsules overlap, so the union is one smooth watertight tube).
    if len(backbone_mm) >= 2:
        spline = catmull_rom(backbone_mm, params.spline_samples_per_residue)
        for i in range(len(spline) - 1):
            solids.append(_manifold.capsule(spline[i], spline[i + 1], tube_r))
    elif len(backbone_mm) == 1:
        solids.append(_manifold.capsule(backbone_mm[0], backbone_mm[0], tube_r))
    return solids


def _base_solids(res_name, res, bp, params, s, slab_t, conn_r, atom_r, bond_r):
    """Build the solids for one base in the chosen style, tied to ``bp``.

    ``bp`` is the residue's backbone trace point (mm).  Returns ``[]`` if the
    base plane cannot be fitted.
    """
    frame = _base_frame(res_name, res)
    if frame is None:
        return []
    center, normal, long_axis, glyco = frame
    center_mm = center * s
    glyco_mm = glyco * s

    if params.base_style == BaseStyle.MOLECULE:
        # Ball-and-stick of the ring atoms, tied to the backbone at the
        # glycosidic atom.
        ring = _ring_atoms(res_name, res)
        if not ring:
            return []
        solids = _ball_and_stick(ring, s, atom_r, bond_r)
        solids.append(_manifold.capsule(bp, glyco_mm, conn_r))
        return solids

    if params.base_style == BaseStyle.ROD:
        # A cylinder rung along the base long axis, radius from slab thickness.
        rod_r = max(slab_t * 0.5, conn_r)
        half_len = 4.5 * 0.5 * params.slab_scale * s
        tip = center_mm + long_axis * half_len
        tail = center_mm - long_axis * half_len
        return [
            _manifold.capsule(tail, tip, rod_r),
            _manifold.capsule(bp, center_mm, conn_r),
        ]

    # Default: flat oriented slab on the base plane.
    third = np.cross(normal, long_axis)
    third /= (np.linalg.norm(third) or 1.0)
    axes = np.array([long_axis, third, normal])
    # Slab footprint: a base is roughly 4.5 x 3.0 ångström, scaled to mm;
    # through-plane thickness is exactly slab_t.
    half = np.array([4.5, 3.0, 1.0]) * 0.5 * params.slab_scale * s
    half[2] = slab_t * 0.5
    return [
        _manifold.oriented_box(center_mm, axes, half),
        _manifold.capsule(bp, center_mm, conn_r),
    ]


def build(chain, params: PrintParams):
    """Return a watertight trimesh of the nucleic-acid model for ``chain``.

    The backbone (tube or molecule) and each base (slab, rod or molecule) are
    built from exact analytic primitives and fused by one manifold boolean
    union, so the chain comes out as a single watertight body with no grid
    texture.  Minimum wall thickness is applied as a parametric offset on those
    primitives (see :func:`_min_wall_dims`).
    """
    s = params.scale_mm_per_angstrom
    tube_r, slab_t, conn_r, atom_r, bond_r = _min_wall_dims(params)

    residues = list(_residue_iter(chain.atoms))
    backbone_mm = np.array([_backbone_point(res) for _, res in residues]) * s

    solids = _backbone_solids(
        residues, backbone_mm, params, s, tube_r, atom_r, bond_r
    )

    for (res_name, res), bp in zip(residues, backbone_mm):
        solids.extend(
            _base_solids(res_name, res, bp, params, s, slab_t, conn_r, atom_r, bond_r)
        )

    fused = _manifold.union(solids)
    return _manifold.to_trimesh(fused)
