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
    # WITHDRAWN — kept so old saved parameter sets still parse, but it is not
    # registered in ``geometry._BUILDERS`` and is not offered in the UI.  The
    # first pass (cylinder helices + flat sheet planks) did not read as a
    # cartoon; see the TODO in NOTES.md for the intended rework.
    CARTOON = "cartoon"        # secondary-structure cartoon (helices + sheets)


class BaseStyle(str, Enum):
    """How each nucleotide base (the ladder "rung") is rendered.

    All styles are built from exact analytic primitives, fused to the backbone
    by a connector strut, and grown to satisfy ``min_wall_mm`` before the union
    so every one is watertight and printable.
    """

    SLAB = "slab"          # flat oriented box on the base plane (original)
    ROD = "rod"            # rounded cylinder rung — chunky, very printable
    MOLECULE = "molecule"  # ball-and-stick of the base's ring atoms


class BackboneStyle(str, Enum):
    """How the sugar-phosphate backbone is rendered."""

    TUBE = "tube"          # smooth swept capsule tube (original)
    MOLECULE = "molecule"  # ball-and-stick of the backbone/sugar atoms


class MoleculeType(str, Enum):
    PROTEIN = "protein"
    NUCLEIC = "nucleic"


class MinWallMode(str, Enum):
    #: Grow every feature outward by ``min_wall`` (robust, always available).
    UNIFORM = "uniform"
    #: Thicken only regions measured to be below ``min_wall`` (falls back to
    #: :attr:`UNIFORM` if it misbehaves).
    SELECTIVE = "selective"


class NoMagnetMethod(str, Enum):
    """How chains are joined when magnets are *off*."""

    INFLATE = "inflate"   # grow both surfaces at the contact until they merge
    BRIDGE = "bridge"     # a short cylinder spanning the gap (peg/strut)


class MagnetShape(str, Enum):
    """Magnet cross-section for the subtracted pocket."""

    ROUND = "round"       # cylindrical disc magnet
    SQUARE = "square"     # square/block magnet


@dataclass
class ConnectionParams:
    """The (deliberately small) set of options for the connections pass.

    Two independent switches:

    * ``connect`` joins chains that touch (protein↔protein, protein↔DNA).  With
      ``use_magnets`` it subtracts a magnet pocket from each side; without, it
      either ``INFLATE``s the surfaces together or drops a ``BRIDGE`` cylinder.
      A single ``connector_diameter_mm`` sizes the magnet or the bridge.
    * ``basepair_connect`` ties the two strands of a DNA duplex together at each
      base pair (geometry-driven; complementary bases paired by centroid, with a
      distance cutoff so an unwound bubble is left open).
    """

    # --- chain-to-chain joins ------------------------------------------
    connect: bool = False
    use_magnets: bool = False
    no_magnet_method: NoMagnetMethod = NoMagnetMethod.INFLATE
    #: Magnet Ø (magnets on) or bridge Ø (bridge); inflate sizes from the gap.
    connector_diameter_mm: float = 4.0
    #: Per-magnet thickness = the pocket depth on each side.
    magnet_thickness_mm: float = 2.0
    magnet_shape: MagnetShape = MagnetShape.ROUND
    #: How many magnets per protein↔protein interface (spread across the contact
    #: patch).
    magnet_count: int = 1
    #: How many magnets per DNA↔protein interface.  These contact patches are
    #: usually smaller than protein↔protein ones, so this is exposed separately
    #: and defaults to a single magnet.
    dna_magnet_count: int = 1

    # --- flush socket (magnets *and* bridge) ---------------------------
    #: Raise a flat-faced cylindrical collar around the joint on both parts so
    #: they meet on a clean machined-looking disc instead of two ragged organic
    #: surfaces.  On by default: cutting a pocket straight into a bumpy molecular
    #: surface is what made the old joints look wrong and sit proud.
    socket: bool = True
    #: Plastic left around the magnet pocket, i.e. socket Ø = magnet Ø + 2×this.
    socket_wall_mm: float = 1.5
    #: Extra diameter on the magnet pocket over the nominal magnet Ø.  FDM holes
    #: print undersize, so a nominal-sized pocket will not accept the magnet at
    #: all; ~0.2 mm gives a press fit you still have to push.
    magnet_fit_clearance_mm: float = 0.2
    #: Extra pocket depth per side over the nominal magnet thickness.  This is
    #: *bottom relief*, not slop: the magnet is pressed in until it is flush with
    #: the mating face, and the extra depth is there so a stray blob or a bit of
    #: stringing at the bottom of the bore cannot hold it proud — which would
    #: stop the two faces meeting at all.
    magnet_depth_clearance_mm: float = 0.2
    #: 45° lead-in at the pocket mouth.  Makes the magnet start square instead of
    #: catching on the rim, and hides the elephant-foot bulge at the face.
    magnet_chamfer_mm: float = 0.4
    #: Extra radius on the bore cut through each part's *approach path*.  A joint
    #: is only assemblable if neither part has material sitting in the other's
    #: way, so anything of one part that reaches past the shared face inside the
    #: collar footprint is cut away.  This is the sliding clearance on that cut.
    path_clearance_mm: float = 0.3

    # --- DNA base-pair connect -----------------------------------------
    basepair_connect: bool = False

    # --- internal (not exposed in the UI) ------------------------------
    #: Radius of the ball used to weigh how much material ("meat") each side has
    #: around a candidate seat, as a multiple of the socket radius.  Kept modest
    #: on purpose: a large ball on a protein that *wraps* around DNA reaches
    #: right around it and drags the local centre of mass to the far side.
    mass_probe_scale: float = 2.0
    #: Cosine of the largest angle any candidate axis may differ from the plain
    #: nearest-point line before it is rejected outright (0.5 = 60°).  Catches
    #: the protein-wrapped-around-DNA case, where the probe ball reaches right
    #: around the duplex and the centroid line flips.
    axis_agreement_min: float = 0.5
    #: How strongly a candidate axis is penalised for driving the joint through
    #: material that would have to be cut away.  Large on purpose: a magnet that
    #: cannot be pulled apart is a failed print, whereas a slightly shallower
    #: seat is only cosmetic.
    axis_blocked_weight: float = 6.0
    #: Aspect ratio at which a contact patch counts as a *strip* rather than a
    #: disc.  Above this, the strip's long direction is projected out of the
    #: centroid line — on an elongated patch that direction is the one thing that
    #: is well determined, and it is exactly where a rod-shaped blob's centre of
    #: mass slides to.
    patch_elongation_min: float = 2.0
    #: How many candidate seats get the expensive exact (boolean) scoring pass,
    #: over and above the number of magnets actually wanted.
    seat_shortlist_extra: int = 3
    #: Max surface gap (mm) for two chains to count as "in contact".
    contact_threshold_mm: float = 3.0
    #: Max base-centroid distance (ångström, pre-scale) still treated as a real
    #: Watson–Crick pair — beyond this the strands have genuinely separated (an
    #: unwound bubble / melted end) and are left unconnected.  A paired B-DNA
    #: step sits ~6–7 Å; this is set generously so only a real separation stops
    #: the connection.
    basepair_max_dist_ang: float = 13.0

    def enabled(self) -> bool:
        """True if any connection work is requested."""
        return self.connect or self.basepair_connect


@dataclass
class PrintParams:
    """All user-tunable parameters for one export."""

    # --- global ---------------------------------------------------------
    scale_mm_per_angstrom: float = 1.5   # overall size control
    grid_spacing_mm: float = 0.5         # marching-cubes voxel size (explicit!)
    min_wall_mm: float = 1.0             # enforced minimum feature thickness
    min_wall_mode: MinWallMode = MinWallMode.UNIFORM

    # --- per molecule-type representation (defaults per the spec) -------
    protein_representation: Representation = Representation.SURFACE
    nucleic_representation: Representation = Representation.TUBE_SLAB

    # --- surface (solvent-excluded surface) tuning ---------------------
    #: Rolling-probe radius (ångström, pre-scale).  1.4 Å is the standard water
    #: probe and is what ChimeraX uses for its molecular surface; it sets how
    #: deep a crevice the surface reproduces.
    probe_radius_ang: float = 1.4
    #: Extra radius added to each atom before the surface is computed (ångström,
    #: pre-scale).  0 gives a true van-der-Waals-based SES; a small positive
    #: value smooths hairline gaps for a more print-robust shell.
    surface_atom_padding_ang: float = 0.0

    # --- tube-and-slab tuning ------------------------------------------
    nucleic_radius_mm: float = 1.2       # backbone tube radius at print scale
    #: Backbone tube radius for the *protein* "tubes" representation, kept
    #: separate from the nucleic tube so the two can be sized independently.
    protein_tube_radius_mm: float = 1.2
    #: Overall chunkiness of the protein "cartoon" representation: the helix
    #: cylinder radius and (scaled) the sheet-plank thickness.
    cartoon_thickness_mm: float = 2.0
    slab_thickness_mm: float = 1.2       # base-slab (or rod) thickness
    slab_scale: float = 1.0              # scale factor on the in-plane base size
    connector_radius_mm: float = 0.6     # strut fusing each base to the backbone
    spline_samples_per_residue: int = 6  # backbone smoothness

    # --- nucleic base / backbone style ---------------------------------
    base_style: BaseStyle = BaseStyle.SLAB
    backbone_style: BackboneStyle = BackboneStyle.TUBE
    #: Sphere radius (mm) for atoms in the "molecule" base/backbone styles.
    atom_radius_mm: float = 1.0
    #: Cylinder radius (mm) for the bonds ("sticks") in the molecule styles.
    bond_radius_mm: float = 0.5

    # --- connector / joinery system ------------------------------------
    connections: ConnectionParams = field(default_factory=ConnectionParams)

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


# Representations that must NOT receive the voxel ``enforce_min_wall`` pass.
# Both current representations already own their wall thickness *at build time*:
#   • SURFACE   — a solvent-excluded surface is thick everywhere by construction.
#   • TUBE_SLAB — min-wall is now a parametric offset applied to the analytic
#     primitives before the mesh boolean (see ``tube_slab.build``), so its tube,
#     slabs and connectors are already ≥ min-wall.  Re-voxelising either here
#     would only reintroduce the grid stairstep we moved off of.
# The pass therefore stays in ``meshops`` only as a fallback for hypothetical
# future representations that build a thin shell and cannot self-thicken.
MIN_WALL_EXEMPT = frozenset({
    Representation.SURFACE, Representation.TUBE_SLAB, Representation.CARTOON,
})


def needs_min_wall(representation) -> bool:
    """True if a representation needs the voxel ``enforce_min_wall`` pass.

    Accepts a :class:`Representation` or its string value (mesh metadata stores
    the value).  Both current representations apply their own wall thickness at
    build time and so decline the pass; unknown values default to keeping it
    (safe side).
    """
    if isinstance(representation, str):
        try:
            representation = Representation(representation)
        except ValueError:
            return True
    return representation not in MIN_WALL_EXEMPT
