"""Solvent-excluded molecular surface (SES).

This is the real rolling-probe surface ChimeraX shows — sharp crevices and
reentrant detail — not a Gaussian metaball blob.  It is built as a signed
distance field and extracted with marching cubes, so it is watertight and
manifold by construction (unlike an analytic MSMS/reduced-surface mesh, which
self-intersects and needs heavy repair).

The detail comes from the *field definition*, not the grid: the SES is the
Connolly surface

    SES_solid = { x : dist(x, outside(SAS)) > probe_radius }

where ``SAS`` (the solvent-accessible solid) is the union of atom balls grown by
the probe radius.  Eroding that solid back by the probe radius reproduces the
van-der-Waals surface on convex patches and the concave reentrant surface in
every crevice — exactly what a rolling water probe traces.  A single Euclidean
distance transform yields the whole field.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import PrintParams, VDW_RADII_ANG, DEFAULT_VDW_ANG
from ._common import Grid, field_to_mesh


def _sas_radii_mm(atoms, params: PrintParams) -> np.ndarray:
    """Per-atom solvent-accessible radius (vdW + padding + probe), in print-mm."""
    elements = [str(e).upper() for e in atoms.element]
    radii_ang = (
        np.array([VDW_RADII_ANG.get(e, DEFAULT_VDW_ANG) for e in elements])
        + params.surface_atom_padding_ang
        + params.probe_radius_ang
    )
    return radii_ang * params.scale_mm_per_angstrom


def build(chain, params: PrintParams):
    """Return a watertight trimesh of the solvent-excluded surface for ``chain``."""
    coords = chain.atoms.coord.astype(float) * params.scale_mm_per_angstrom
    sas_radii = _sas_radii_mm(chain.atoms, params)
    probe_mm = params.probe_radius_ang * params.scale_mm_per_angstrom
    spacing = params.grid_spacing_mm

    pad = float(sas_radii.max()) + 2.0 * spacing
    grid = Grid.covering(coords, spacing=spacing, pad=pad)

    # 1) Rasterise the solvent-accessible solid: the union of atom balls grown
    #    by the probe radius.
    sas = np.zeros(grid.shape, dtype=bool)
    for c, r in zip(coords, sas_radii):
        slices, pts = grid.window(c, r)
        if slices is None:
            continue
        inside = np.sum((pts - c) ** 2, axis=-1) <= r * r
        sas[slices] |= inside

    if not sas.any():
        raise ValueError("Surface field is empty; the structure is smaller than "
                         "one voxel — reduce grid spacing or increase scale.")

    # 2) SES = SAS solid eroded inward by the probe radius.  The signed field is
    #    (inward distance to the SAS boundary) − probe; its zero level set is the
    #    solvent-excluded surface.  Distances are in voxels, so scale to mm.
    inward_mm = ndimage.distance_transform_edt(sas) * spacing
    field = inward_mm - probe_mm

    return field_to_mesh(field, grid, level=0.0)
