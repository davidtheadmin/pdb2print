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
from typing import Dict, List

import numpy as np


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


class InterferenceRule(str, Enum):
    """How interpenetration between two chains is resolved before export.

    Chains are meshed independently, so at a binding interface both solids
    occupy the same volume.  That is correct as a picture of the complex and
    impossible as a set of printed parts, so one or both have to give it up.
    """

    #: Nucleic acids keep their true shape and the protein is carved to fit
    #: (a socket that is an exact negative of the duplex); between two chains
    #: of the same type the larger keeps its shape.  The default.
    AUTO = "auto"
    #: Both parts retreat out of the shared volume.  Neither is deformed by the
    #: other's shape, at the cost of a gap where they interpenetrated.
    SYMMETRIC = "symmetric"
    #: Leave the overlap in place.  The preview still looks right, but the
    #: printed parts will not fit together — diagnostics only.
    NONE = "none"


# --- solvent-excluded surface safety envelope -----------------------------
#
# The SES field is ``EDT(atoms grown by p) − p``.  On a convex patch the two
# cancel exactly — a ball of radius ``vdW + pad + p`` eroded inward by ``p`` is a
# ball of radius ``vdW + pad`` — so **the probe radius does not set the size of
# the part**.  It only sets how deep a crevice gets filled in.  Anyone reaching
# for it to stop two chains touching is pulling a lever that is not connected to
# that: use ``surface_atom_padding_ang``, or let the interference pass carve
# them apart properly.
#
# What the probe radius *does* control is connectivity.  Two atoms stay joined
# in the surface only while their grown balls overlap (``D < r1 + r2 + 2p``), so
# lowering it pulls that radius in by twice the amount and starts severing thin
# necks — first into extra loose bodies, then into pinched, non-manifold
# geometry that the watertight gate rejects.
#
# Measured on 1UBQ at 1.5 mm/Å, 0.5 mm grid — bodies in the raw marching-cubes
# result, and the volume ``meshops.repair`` then throws away as debris:
#
#     probe   bodies   watertight   volume lost
#      1.0       9        no           2.1 %
#      1.1      12        yes          1.5 %
#      1.2       6        yes          0.7 %
#      1.3       4        yes          0.4 %
#      1.4       2        yes          0.1 %
#      1.5       1        yes          0.0 %
#
# 1.0 Å is exactly the setting that was destabilising real builds, and the
# failure is monotonic rather than a cliff.  The floor is therefore set at the
# water probe: it is both the standard definition of the SES and the point where
# the surface stops shedding pieces, and below it there is nothing to gain.
PROBE_RADIUS_MIN_ANG: float = 1.4
#: Above this the surface is so inflated that side-chain detail is gone and
#: separate lobes fuse into a blob; there is nothing to gain further out either.
PROBE_RADIUS_MAX_ANG: float = 2.0
#: The erosion band (the probe radius in mm) must span at least this many
#: voxels.  Deliberately mild: the same measurement shows the band is *not* what
#: drives the failure — 1.4 Å stayed watertight down to a 0.93-voxel band, while
#: 1.0 Å failed at a 3.0-voxel one — so this only catches the degenerate case
#: where the erosion is thinner than a single sample and the level set has
#: nothing to land on.  Grid refinement is expensive and is not the fix here.
PROBE_VOXELS_MIN: float = 1.0
#: Ceiling on the auto-refinement, so a large structure cannot silently ask for
#: tens of gigabytes.  Beyond this the grid is left coarser with a warning.
SURFACE_VOXEL_BUDGET: int = 40_000_000


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
    #: How much a seat is rewarded for the socket being *buried* rather than
    #: standing proud of the surface (see ``connections._embedding``): 1.0 means
    #: the collar is entirely inside existing material and only the mating disc
    #: shows, 0.0 that it would be built out of thin air.
    #:
    #: This overlaps ``fill`` deliberately and is not the same measurement.
    #: ``fill`` starts at each part's own surface, so the stub of collar that
    #: spans the half-gap — the piece with nothing at all behind it, and the piece
    #: you actually see sticking out — is excluded from it by construction.  This
    #: is taken on the collar as built, from the shared mid-plane, so that stub
    #: counts.  Kept well below ``fill``'s weight of 100 so it refines the choice
    #: between comparable seats rather than overriding "is there material here".
    seat_embedding_weight: float = 40.0
    #: The same measure applied to *orientation*, in units of the surface-point
    #: counts the axis search scores in.
    #:
    #: The axis search had no notion of this at all: it scored candidates purely
    #: on how many sampled surface points obstruct the approach or sit behind the
    #: face, which answers "can the parts come apart" and is silent on how deeply
    #: the collar buries itself.  So a tilt that leaves half the socket in open
    #: air scored level with one that sinks it into the body.  25 puts a fully
    #: buried joint roughly on par with 25 surface points of the census — enough
    #: to decide between two otherwise similar tilts, far too little to override
    #: the blocked penalty (6 per point), so an axis that cannot be assembled
    #: still loses.  Set to 0 to score orientation the old way.
    axis_embedding_weight: float = 25.0
    #: How strongly a seat is penalised for sitting on the *rim* of an interface
    #: rather than in its interior.  Measured as the lateral lopsidedness of the
    #: contact patch around the seat (mm, clamped to the socket radius): a seat
    #: ringed by contact scores ~0, one whose support is all to one side scores up
    #: to the socket radius.  This is what stops a socket landing on the edge of a
    #: contact where its collar overhangs open air — there is usually a cleaner
    #: spot a little further in, and this makes the ranker prefer it.  Kept modest
    #: so it only decides between seats of comparable fill, never over a genuinely
    #: better-seated one.
    edge_center_weight: float = 8.0
    #: How strongly a seat is rewarded for having a *well-determined* joint axis,
    #: by the source the axis came from (overlap lobe / mass line = a real
    #: interface normal, so the disc sits square and clean; the plain contact-line
    #: fallback is noisy and can look tilted).  Lets a nearby seat with a cleaner
    #: orientation win over one that only reaches its axis by fallback.
    axis_quality_weight: float = 12.0
    #: Physically slide each seat toward the interior of its contact patch before
    #: it is built, as a fraction of the measured rim offset (0 = off, the seat
    #: stays where it was found; 0.5 pulls it halfway in).  The motion is purely
    #: in the mating plane — never along the axis — so the two flat faces still
    #: meet on the shared mid-plane; it is capped at one socket radius.  This is
    #: the physical companion to ``edge_center_weight``: the weight *prefers* an
    #: interior seat when the shortlist offers one, this *makes* one when the whole
    #: interface is narrow and every candidate sits on the rim.  Off by default —
    #: turn it up only if re-ranking alone still leaves sockets proud of an edge.
    seat_recenter_frac: float = 0.0
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

    # --- fit / interference --------------------------------------------
    #: How interpenetration between chains is resolved.  This runs whether or
    #: not the parts are being connected — two objects that are simply printed
    #: and handed over still have to fit together.
    resolve_interference: InterferenceRule = InterferenceRule.AUTO
    #: Sliding clearance left between two mating parts (mm).  A plain boolean
    #: subtraction is a geometrically perfect zero-clearance mate, but FDM parts
    #: come out slightly oversize and would bind, so the part being carved is cut
    #: against its neighbour grown by this much.
    fit_clearance_mm: float = 0.15

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


@dataclass
class SurfaceGrid:
    """The probe radius and grid spacing a surface build will actually use."""

    probe_ang: float
    spacing_mm: float
    notes: List[str] = field(default_factory=list)

    @property
    def probe_mm_at(self):
        return self.probe_ang


def resolve_surface_grid(params: "PrintParams", extent_mm=None) -> SurfaceGrid:
    """Clamp the probe radius and refine the grid until the pair is safe.

    The dominant failure mode is the probe radius, not the grid: too small a
    probe severs the surface between atoms (see the table above
    :data:`PROBE_RADIUS_MIN_ANG`), and no amount of grid refinement repairs a
    surface that has genuinely come apart.  So the probe is clamped hard and the
    grid is only nudged off the degenerate sub-one-voxel case.

    Refinement is capped by :data:`SURFACE_VOXEL_BUDGET` when ``extent_mm`` (the
    structure's bounding-box size in mm) is supplied, so a large complex cannot
    silently ask for an impossible allocation; in that case the build goes ahead
    at the coarser spacing with a note rather than failing.
    """
    notes: List[str] = []

    probe = float(params.probe_radius_ang)
    if probe < PROBE_RADIUS_MIN_ANG:
        notes.append(
            f"Probe radius raised from {probe:.2f} to {PROBE_RADIUS_MIN_ANG:.2f} Å "
            f"(the water probe). Below it the surface starts breaking into "
            f"separate pieces between atoms — at 1.0 Å it stops being watertight "
            f"altogether. It would not have made the parts any smaller either: "
            f"on an outward-facing patch the probe radius cancels out of the "
            f"surface definition. To shrink a part use Surface padding; to stop "
            f"two chains colliding, the fit pass already carves them apart."
        )
        probe = PROBE_RADIUS_MIN_ANG
    elif probe > PROBE_RADIUS_MAX_ANG:
        notes.append(
            f"Probe radius lowered from {probe:.2f} to {PROBE_RADIUS_MAX_ANG:.2f} Å "
            "— beyond that the surface is a featureless blob."
        )
        probe = PROBE_RADIUS_MAX_ANG

    spacing = float(params.grid_spacing_mm)
    probe_mm = probe * float(params.scale_mm_per_angstrom)
    needed = probe_mm / PROBE_VOXELS_MIN

    if spacing > needed:
        capped = needed
        if extent_mm is not None:
            span = np.asarray(extent_mm, float) + 2.0 * probe_mm
            floor = float((np.prod(span) / SURFACE_VOXEL_BUDGET) ** (1.0 / 3.0))
            if floor > needed:
                capped = min(spacing, floor)
        if capped < spacing:
            notes.append(
                f"Grid spacing refined from {spacing:.2f} to {capped:.2f} mm so the "
                f"{probe_mm:.2f} mm probe erosion spans at least one voxel."
            )
            spacing = capped
        if spacing > needed:
            notes.append(
                f"Grid spacing {spacing:.2f} mm is still coarse for a "
                f"{probe_mm:.2f} mm probe (memory budget); the surface may be "
                "under-resolved — raise the probe radius or lower the scale."
            )

    return SurfaceGrid(probe_ang=probe, spacing_mm=spacing, notes=notes)


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
