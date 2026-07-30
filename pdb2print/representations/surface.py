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

from ..config import (
    PrintParams, VDW_RADII_ANG, DEFAULT_VDW_ANG, resolve_surface_grid,
)
from ._common import Grid, field_to_mesh


def _sas_radii_mm(atoms, params: PrintParams, probe_ang: float) -> np.ndarray:
    """Per-atom solvent-accessible radius (vdW + padding + probe), in print-mm."""
    elements = [str(e).upper() for e in atoms.element]
    radii_ang = (
        np.array([VDW_RADII_ANG.get(e, DEFAULT_VDW_ANG) for e in elements])
        + params.surface_atom_padding_ang
        + probe_ang
    )
    return radii_ang * params.scale_mm_per_angstrom


def build(chain, params: PrintParams):
    """Return a watertight trimesh of the solvent-excluded surface for ``chain``.

    The probe radius and grid spacing are passed through
    :func:`~pdb2print.config.resolve_surface_grid` first, which clamps them to a
    combination that can actually be meshed: too small a probe severs the
    surface between atoms and too coarse a grid cannot place the eroded level
    set.  Whatever it changed is recorded in ``mesh.metadata["notes"]`` and
    surfaced by the pipeline as a build warning.
    """
    coords = chain.atoms.coord.astype(float) * params.scale_mm_per_angstrom
    extent = coords.max(axis=0) - coords.min(axis=0)
    safe = resolve_surface_grid(params, extent_mm=extent)

    sas_radii = _sas_radii_mm(chain.atoms, params, safe.probe_ang)
    probe_mm = safe.probe_ang * params.scale_mm_per_angstrom
    spacing = safe.spacing_mm

    pad = float(sas_radii.max()) + 2.0 * spacing
    grid = Grid.covering(coords, spacing=spacing, pad=pad)

    # 1) Rasterise the solvent-accessible solid: the union of atom balls grown
    #    by the probe radius.
    sas = np.zeros(grid.shape, dtype=bool)
    for c, r in zip(coords, sas_radii):
        slices, axes = grid.window_axes(c, r)
        if slices is None:
            continue
        # Separable squared distance.  These three broadcast to the same sum, in
        # the same order, as summing the last axis of a materialised
        # (W, W, W, 3) coordinate array — without materialising it.  That array
        # was being rebuilt once per atom in the structure.
        dx2 = (axes[0] - c[0]) ** 2
        dy2 = (axes[1] - c[1]) ** 2
        dz2 = (axes[2] - c[2]) ** 2
        inside = (dx2[:, None, None] + dy2[None, :, None]
                  + dz2[None, None, :]) <= r * r
        sas[slices] |= inside

    if not sas.any():
        raise ValueError("Surface field is empty; the structure is smaller than "
                         "one voxel — reduce grid spacing or increase scale.")

    # 2) SES = SAS solid eroded inward by the probe radius.  The signed field is
    #    (inward distance to the SAS boundary) − probe; its zero level set is the
    #    solvent-excluded surface.  Distances are in voxels, so scale to mm.
    #    Done in place: ``edt(sas) * spacing - probe`` allocates three full
    #    float64 grids where one will do, and on a large chain each of those is
    #    hundreds of megabytes that the machine has to find, fill and throw away.
    field = ndimage.distance_transform_edt(sas)
    del sas                     # the EDT is the only thing that needed it
    field *= spacing
    field -= probe_mm

    mesh = field_to_mesh(field, grid, level=0.0)
    if safe.notes:
        mesh.metadata["notes"] = list(safe.notes)
    return mesh
