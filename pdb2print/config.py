"""Configuration types shared across the geometry pipeline.

Everything the UI can tune flows through :class:`PrintParams`.  Keeping it a plain
dataclass (no Gradio imports) is what lets the same parameters drive a future
WASM build.

Units note
----------
Structure coordinates from the PDB are in angstroms.  The pipeline works
internally in *print millimetres*: atom coordinates are multiplied by
``scale_mm_per_angstrom`` on the way in, so ``grid_spacing_mm``, ``min_wall_mm``
and ``nucleic_radius_mm`` are all interpreted directly in the working space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class Representation(str, Enum):
    """How a chain's atoms are turned into a printable solid.

    New representations (cartoon, ladder-only, backbone-tube, ...) are added
    here and wired into :func:`pdb2print.geometry.generate_chain_mesh`; nothing
    else in the pipeline needs to change.
    """

    SURFACE = "surface"        # Gaussian metaball molecular surface
    TUBE_SLAB = "tube_slab"    # backbone tube + base slabs (nucleic acids)


class MoleculeType(str, Enum):
    PROTEIN = "protein"
    NUCLEIC = "nucleic"


class MinWallMode(str, Enum):
    #: Grow every feature outward by ``min_wall`` (robust, always available).
    UNIFORM = "uniform"
    #: Thicken only regions measured to be below ``min_wall`` (falls back to
    #: :attr:`UNIFORM` if it misbehaves).
    SELECTIVE = "selective"


@dataclass
class PrintParams:
    """All user-tunable parameters for one export."""

    # --- global ---------------------------------------------------------
    scale_mm_per_angstrom: float = 0.5   # overall size control
    grid_spacing_mm: float = 0.5         # marching-cubes voxel size (explicit!)
    min_wall_mm: float = 1.0             # enforced minimum feature thickness
    min_wall_mode: MinWallMode = MinWallMode.UNIFORM

    # --- per molecule-type representation (defaults per the spec) -------
    protein_representation: Representation = Representation.SURFACE
    nucleic_representation: Representation = Representation.TUBE_SLAB

    # --- surface (metaball) tuning -------------------------------------
    #: Gaussian spread as a multiple of each atom's van der Waals radius.
    surface_blobbiness: float = 1.0
    #: Extra radius added to atoms before meshing (angstrom, pre-scale) — a
    #: small value fuses neighbouring atoms into a smooth printable shell.
    surface_atom_padding_ang: float = 0.4

    # --- tube-and-slab tuning ------------------------------------------
    nucleic_radius_mm: float = 1.2       # backbone tube radius at print scale
    slab_thickness_mm: float = 1.2       # base-slab thickness
    slab_scale: float = 1.0              # scale factor on the in-plane slab size
    spline_samples_per_residue: int = 6  # backbone smoothness

    def representation_for(self, mtype: MoleculeType) -> Representation:
        if mtype == MoleculeType.PROTEIN:
            return self.protein_representation
        return self.nucleic_representation

    # Convenience: convert a print-mm length back to angstrom (for callers
    # that still work in structure space).
    def mm_to_ang(self, mm: float) -> float:
        return mm / self.scale_mm_per_angstrom


# Van der Waals radii in angstrom, used to size the metaball kernels.
VDW_RADII_ANG: Dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80,
    "S": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "MG": 1.73, "ZN": 1.39, "NA": 2.27, "K": 2.75, "CA": 2.31,
    "FE": 2.00, "MN": 2.00,
}
DEFAULT_VDW_ANG: float = 1.70


# A distinct, print-friendly colour per chain (RGB floats 0..1).  Cycled if
# there are more chains than entries.
CHAIN_PALETTE = [
    (0.85, 0.20, 0.20),   # red
    (0.20, 0.45, 0.85),   # blue
    (0.20, 0.70, 0.35),   # green
    (0.95, 0.75, 0.15),   # amber
    (0.60, 0.30, 0.75),   # purple
    (0.20, 0.75, 0.75),   # teal
    (0.90, 0.50, 0.20),   # orange
    (0.85, 0.35, 0.65),   # pink
]


def color_for_index(i: int):
    return CHAIN_PALETTE[i % len(CHAIN_PALETTE)]
