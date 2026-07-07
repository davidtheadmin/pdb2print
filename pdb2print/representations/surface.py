"""Gaussian metaball molecular surface.

Each atom contributes a Gaussian density kernel sized by its van der Waals
radius; the isosurface of the summed field is a smooth, fused shell that is
watertight and manifold by construction — the print-friendly stand-in for a
true analytic solvent-excluded surface (which self-intersects and needs heavy
repair).
"""

from __future__ import annotations

import numpy as np

from ..config import PrintParams, VDW_RADII_ANG, DEFAULT_VDW_ANG
from ._common import Grid, field_to_mesh


# A single Gaussian exp(-d^2 / (2 sigma^2)) crosses ISO_LEVEL at
# d = sigma * sqrt(-2 ln level).  We size sigma so that this crossing distance
# equals the atom's radius, i.e. a lone atom meshes at (about) its vdW radius.
ISO_LEVEL = 0.5
_CROSS = np.sqrt(-2.0 * np.log(ISO_LEVEL))   # ~1.177


def _atom_radii_mm(atoms, params: PrintParams) -> np.ndarray:
    elements = [str(e).upper() for e in atoms.element]
    radii_ang = np.array(
        [VDW_RADII_ANG.get(e, DEFAULT_VDW_ANG) for e in elements]
    ) + params.surface_atom_padding_ang
    return radii_ang * params.scale_mm_per_angstrom


def build(chain, params: PrintParams):
    """Return a watertight trimesh of the Gaussian surface for ``chain``."""
    coords = chain.atoms.coord.astype(float) * params.scale_mm_per_angstrom
    radii = _atom_radii_mm(chain.atoms, params)

    spacing = params.grid_spacing_mm
    max_radius = float(radii.max())
    # Pad by a few sigma so kernels are not clipped at the box edge.
    pad = max_radius * 2.0 + spacing
    grid = Grid.covering(coords, spacing=spacing, pad=pad)

    field = np.zeros(grid.shape, dtype=np.float32)
    for c, r in zip(coords, radii):
        sigma = (r / _CROSS) * params.surface_blobbiness
        # Evaluate out to 3 sigma; beyond that the contribution is negligible.
        reach = 3.0 * sigma
        slices, pts = grid.window(c, reach)
        if slices is None:
            continue
        d2 = np.sum((pts - c) ** 2, axis=-1)
        field[slices] += np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)

    return field_to_mesh(field, grid, level=ISO_LEVEL)
