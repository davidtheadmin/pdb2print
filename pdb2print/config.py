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
    # ChimeraX-style ribbon cartoon: Carson-Bugg guide frame swept with
    # SSE-dependent cross-sections (twisting helix/strand ribbons, β-arrowheads,
    # coil tube) into one watertight solid.  See ``representations/cartoon.py``.
    CARTOON = "cartoon"        # secondary-structure ribbon cartoon
    #: Ball-and-stick, used for bound ligands.  Deliberately *not* offered as a
    #: choice in the UI and not registered in ``geometry._BUILDERS``: it is the
    #: only way a ligand is ever drawn, so there is nothing to select.  It exists
    #: as a :class:`Representation` member so a ligand mesh still carries a
    #: representation in its metadata and so ``needs_min_wall`` can exempt it.
    BALL_STICK = "ball_stick"  # ligand ball-and-stick


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
    #: A bound small molecule — a drug, a cofactor, a substrate — exported as its
    #: own object, one per residue, and drawn ball-and-stick.  It is a *molecule
    #: type* rather than a chain type on purpose: everything downstream (mesh
    #: dispatch, interference, export) keys off ``mtype``, so adding a member here
    #: is what lets a ligand travel through the pipeline as a first-class object
    #: without a parallel code path.
    LIGAND = "ligand"
    #: A piece of the display stand — base plate, columns, plaque, legend swatch.
    #: Not molecular at all, and it never enters the mesh/interference/connection
    #: passes: the stand is generated *after* a build, from the finished meshes,
    #: and joins the object list only at export.  It is a member here for the same
    #: reason ``LIGAND`` is — export, naming and colouring all branch on ``mtype``
    #: — and having its own value keeps a stand part from ever being mistaken for
    #: something with atoms in it.
    STAND = "stand"


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


class LigandStyle(str, Enum):
    """How a bound ligand is drawn.

    Kept separate from :class:`Representation` because these are not
    interchangeable with the polymer styles: a ligand has no backbone to trace
    and no secondary structure to ribbon, so the only thing the two lists share
    is the word "style".  A ligand does have atoms, though, which is why the
    surface builder works on one unchanged.
    """

    #: Spheres at the atoms, sticks between them. Reads as *that molecule* —
    #: the ring count, the linker, the substituent — and is the default.
    BALL_STICK = "ball_stick"
    #: Van der Waals spheres, fused. The most printable by a distance: nothing
    #: thin to snap, no supports, and it sits beside a surface-rendered protein
    #: as the same kind of object.
    SPACEFILL = "spacefill"
    #: The solvent-excluded surface, from the same builder the protein uses, so
    #: a drug in a surface model's pocket matches the thing it is bound to.
    SURFACE = "surface"
    #: Bonds only, at a uniform radius — a licorice model. The clearest read of
    #: the chemistry and the most fragile thing on the plate.
    STICKS = "sticks"


class ColumnShape(str, Enum):
    """Cross-section — and profile — of a display-stand column.

    Square by default.  A round column reads as a laboratory clamp; a square one
    reads as furniture, which is what a display stand is.  It also prints with
    two flat faces to the build plate instead of a tangent.

    The other two are the same idea carried further.  ``TAPER`` is a square shaft
    that narrows continuously rather than standing on a plinth — an obelisk,
    which turns the fact that a column must be thick at the bottom into something
    that looks intended.  ``FLUTED`` is the classical answer to the same problem:
    a round shaft scalloped down its length, which breaks a plain cylinder into
    something the eye reads as an object rather than as a rod.  Flutes also print
    well — all of it is convex vertical detail with no overhang anywhere.
    """

    SQUARE = "square"
    ROUND = "round"
    TAPER = "taper"
    FLUTED = "fluted"


class PlaqueRelief(str, Enum):
    """How the plaque's lettering meets the surface it is on.

    Three answers to one question, because there are three printers asking it.

    ``RAISED`` is the default: it stands the lettering off the surface, where it
    reads by touch and by shadow as well as by colour.

    ``FLUSH`` sinks the lettering, and the white tile under it, are
    sunk into the apron so every top surface is one continuous plane. Nothing
    stands proud and nothing is hollowed out — the colour changes *within* a
    layer, which is precisely what a multi-material printer does best and what a
    flat top surface prints most cleanly as. It is also the only one of the
    three with no overhang, no loose part and no cavity anywhere.

    ``ENGRAVED`` cuts the lettering away and produces no separate objects at
    all — including the legend's colour dots, which become cut circles. That is
    the point of it: it is the mode for a printer with one filament, where a
    separate object is either fused to its neighbour or lying loose in a hole,
    and a colour dot is a lie either way.
    """

    FLUSH = "flush"
    RAISED = "raised"
    ENGRAVED = "engraved"


class PlaqueFont(str, Enum):
    """Which typeface the plaque is set in.

    ``LINE`` is the hand-written stroke font: 64 glyphs as centre lines, swept
    to whatever width the nozzle can draw.  It has no thickness of its own,
    which is what makes it dependable at two millimetres, and no lowercase —
    it sets it as small capitals.

    ``SANS`` and ``SERIF`` are real typefaces, subset from DejaVu Sans Bold and
    DejaVu Serif Bold and triangulated into filled solids.  They have proper
    lowercase, real counters and real letterfitting, and at a five-millimetre
    headline the difference is the difference between typography and a plotter.
    Both are **bold** deliberately: a regular weight at two millimetres has
    stems a fifth of a millimetre wide, which a 0.4 mm nozzle does not draw.
    """

    LINE = "line"
    SANS = "sans"
    SERIF = "serif"


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
    #: Fraction of the socket's flat *back face* that may stand in open air before
    #: anything is done about it.  This is the disc you see when a cylinder is
    #: stuck onto an uneven surface and the surface falls away under part of it —
    #: the side wall of the socket is not the concern and is left alone.
    #: Everything below is skipped entirely when the back is this well covered,
    #: so a joint that already sits in solid material is never touched.
    socket_cap_exposed_max: float = 0.05
    #: First remedy: how far the socket may be *lengthened* so its walls carry on
    #: down and reach material further in, as a multiple of its nominal depth
    #: (1.0 = may double). The socket is not moved and the mating face does not
    #: shift — only its back is carried deeper, and it is kept at the shortest
    #: length that closes, so a joint never grows more than it has to.
    socket_extend_max: float = 1.0
    #: Second remedy, when no length within that budget closes the back: cap the
    #: flat disc with a cone built **onto** it.  Strictly additive — the socket
    #: keeps its full length and full radius and the cone is stacked on the end.
    #:
    #: It must be additive.  Chamfering the back edge instead removes the
    #: material the joint is made of, and since the socket runs only about
    #: ``socket_wall_mm`` deeper than the magnet pocket, a chamfer of any useful
    #: size starts biting before the bottom of the pocket: it thins the wall
    #: around the magnet and then undercuts it, leaving the magnet standing proud
    #: of a socket carved away from behind.
    #:
    #: The cone is 45°, so its height is simply how far the radius has to come in,
    #: held within the same extension budget above.  Its flat top is sized so its
    #: area falls under ``socket_cap_exposed_max`` — the same threshold that
    #: triggered the work — so what stays visible is below the level considered
    #: worth acting on in the first place.
    socket_back_taper: bool = True
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
    #: How much a seat is rewarded for there being *material depth* here, as
    #: opposed to the collar merely being surrounded.
    #:
    #: ``fill`` and ``embedding`` are both fractions **of the collar**, so both
    #: saturate as soon as the material is about one socket thick and stop
    #: telling the two cases apart.  Measured with the stock 7.2 mm socket: an
    #: arm 7.2 mm across reads 0.86 embedded and a 60 mm block reads 0.98 — a
    #: difference worth five points in a score whose main term spans a hundred.
    #: That is why joints landed on thin protruding arms and nothing objected.
    #:
    #: This reads the material itself: the volume the local probe ball actually
    #: found, against the volume the collar needs.  Saturates at three times the
    #: collar, which is roughly where an arm stops being an arm.
    seat_depth_weight: float = 25.0
    #: How much a joint is rewarded for connecting the two parts *as wholes*.
    #:
    #: Everything else in this score is local — contact points, a probe ball a
    #: few millimetres across, the material immediately around a seat. None of
    #: it knows where the bulk of one chain sits relative to the other, so a
    #: joint on a peripheral loop that happens to graze its neighbour scores as
    #: well as one on the interface the two objects actually meet across. That
    #: is the "the magnets do not connect the parts logically" complaint.
    #:
    #: Widening the local probe ball was the other candidate and it is the same
    #: idea, but measured to fail: past about 2.5x a local feature the ball
    #: reaches around it and the answer flips by 30-80 degrees. Taking the whole
    #: chain's centroid is that idea at its limit, with none of the wrap.
    #:
    #: ``axis`` rewards a joint that pulls apart along the line between the two
    #: chains; ``line`` rewards one that sits near that line rather than out on
    #: a flank.
    global_axis_weight: float = 20.0
    global_line_weight: float = 15.0
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
    #: Now read against a census normalised to 0-100 rather than a raw point
    #: count, so 25 means a fully buried joint is worth a quarter of a clean
    #: candidate's whole census — enough to decide between two assemblable
    #: tilts, not enough to rescue an axis that is genuinely obstructed. The
    #: point of the normalisation is that this ratio no longer depends on how
    #: finely the model happens to be meshed; the same 25 used to be 1-5% of
    #: the decision on a fine mesh and much more on a coarse one.
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
class StandParams:
    """Display stand: a base plate, columns into a cradle, and a plaque.

    The stand is generated **after** a build rather than during one.  Everything
    it needs is in the finished meshes, and the orientation it is built around is
    the one the user has just arranged in the viewer — which does not exist until
    there is something to look at.  So this block is inert during a normal build
    and only read by :mod:`pdb2print.stand`.

    **Orientation.**  ``orbit_theta_deg``/``orbit_phi_deg`` are model-viewer's
    camera orbit, in its own spherical convention (theta azimuthal about +Y, phi
    polar from +Y).  The stand is built so that "down on the screen" becomes
    down in the print — see :func:`stand.view_basis`.
    """

    #: Master switch.  Off for every ordinary build, which is what keeps the
    #: whole block out of the cache key (``cache.canonical_params``).
    enabled: bool = False

    #: Number of columns, or 0 to let :func:`stand.recommend_columns` decide.
    columns: int = 0

    # --- orientation (model-viewer camera orbit, degrees) ----------------
    orbit_theta_deg: float = 0.0
    orbit_phi_deg: float = 75.0
    #: Roll about the view axis, degrees clockwise as seen on screen.
    #:
    #: The orbit gives two degrees of freedom and an orientation needs three, so
    #: without this there are poses that simply cannot be asked for: the camera
    #: can be pointed at any face of the model, but the picture cannot be spun,
    #: and spinning the picture is what decides which way the model leans once it
    #: is standing.  This is the missing third.
    roll_deg: float = 0.0

    # --- base plate ------------------------------------------------------
    #: Margin (mm) from the model's footprint to the edge of the plate.
    plate_margin_mm: float = 7.0
    #: Plate thickness (mm).  Structural, not decorative: it also carries the
    #: bending load of every column.
    plate_thickness_mm: float = 4.0
    #: Corner radius (mm) of the plate.
    plate_corner_mm: float = 4.0

    # --- columns ---------------------------------------------------------
    #: Cross-section of the columns.
    column_shape: ColumnShape = ColumnShape.SQUARE
    #: Column diameter (mm) at the top, where it meets the model — the across-
    #: flats width for a square one.
    column_diameter_mm: float = 8.0
    #: How far inside the model's plan-view silhouette a column must sit (mm),
    #: measured from the column's own outer edge to the silhouette boundary.
    #:
    #: Left to itself the search puts columns on the outermost points it can
    #: find, because spreading them apart is what makes the model stable — and
    #: the extreme edge of a molecular silhouette is a thin, glancing tangent
    #: that looks precarious even when it is not, and gives the cradle almost
    #: nothing to bite.
    #:
    #: Measured against the **footprint**, not against distance from the centre.
    #: Those agree only for a round model: on anything elongated a radial rule
    #: keeps points that are central lengthways while still hanging off the
    #: narrow sides — which is the perched look it was meant to prevent.
    column_edge_margin_mm: float = 5.0
    #: The same rule expressed relatively: a column must sit at least this
    #: fraction of the *deepest available* inset away from the silhouette edge.
    #:
    #: Whichever of the two is smaller applies. The absolute margin is what bites
    #: on a large complex, where a fraction would be needlessly deep; the
    #: fraction is what bites on a small model, where the absolute margin exceeds
    #: the whole underside and would reject every candidate — taking the columns
    #: with it. Higher pulls them further in at the cost of how far apart they
    #: can get.
    column_edge_frac: float = 0.45
    #: Prefer to stand a mixed structure on its **protein**.  A nucleic backbone
    #: is a thin tube: a column under it meets a cylinder tangentially, and the
    #: cradle either grips almost nothing or swallows the strand. Protein
    #: presents a broad surface that a seat can actually sit in. DNA is used only
    #: when there is no protein candidate to be had.
    column_prefer_protein: bool = True
    #: How much wider each column is at the plate than at the model.  A slight
    #: taper is stiffer for the same material and reads as intentional.
    column_flare: float = 1.45
    #: Whether the flare is applied at all.  Off gives a column of one thickness
    #: from plate to cradle, which on a small model is the quieter object: the
    #: foot exists to stiffen a long column against bending, and a 20 mm one is
    #: not bending.  It is a look, and the look should be a switch.
    column_flared: bool = True
    #: A short wider pad where the column meets the model.  Structurally it
    #: spreads the cradle's contact over more of the shaft; visually it is what
    #: stops a column looking like it was cut off where it happened to end.
    #: Never wider than the foot, so it cannot reach past what the column search
    #: already proved was clear.
    column_capital: bool = False
    #: Clearance (mm) cut into the cradle so the model actually seats.  A plain
    #: boolean gives a geometrically perfect zero-clearance mate, which on an FDM
    #: part binds — the same lesson the magnet pockets already encode.
    cradle_clearance_mm: float = 0.35
    #: How deep (mm) the cradle wraps the model.  Deeper resists the model
    #: rocking about the line through the columns; too deep and the pocket needs
    #: supports of its own.
    cradle_depth_mm: float = 4.0
    #: Air gap (mm) between the plate top and the lowest point of the model.
    stand_off_mm: float = 6.0
    # --- assembly pins ---------------------------------------------------
    #: Print the plate and the columns as separate parts, joined by a pin.
    #:
    #: The reason is the plaque.  Lettering printed *upward* is as good as the
    #: top surface of an FDM part, which is not very; lettering printed
    #: **downward against a smooth build sheet** is as good as the sheet, which
    #: is excellent.  Turning the plate over to get that is only possible if the
    #: columns are not welded to it — so this splits them off and gives each one
    #: a peg into a socket.  Everything else follows: the plate prints upside
    #: down with a glass-smooth plaque, the columns print upright with no
    #: support, and the two push together afterwards.
    column_pins: bool = False
    #: Pin diameter (mm).  Comfortably inside the column's own width, and never
    #: so large that the socket eats the plate.
    pin_diameter_mm: float = 4.0
    #: How deep the pin goes (mm).  Clamped so the socket cannot break through
    #: the underside of the plate, which on a plate being printed upside down is
    #: the surface everybody is looking at.
    pin_depth_mm: float = 3.0
    #: Radial clearance (mm) cut into the socket.  Same lesson as every other
    #: mating pair here: a geometrically exact fit binds on an FDM part.
    pin_clearance_mm: float = 0.15

    #: A support point is only usable where the surface faces downward.  This is
    #: the cosine limit: 0.62 accepts anything within about 52 degrees of
    #: straight down, past which a cradle becomes a knife edge that neither
    #: prints nor holds.
    column_normal_min: float = 0.62

    # --- plaque ----------------------------------------------------------
    plaque: bool = True
    #: Show the four-character PDB ID (or the uploaded file's name).
    plaque_pdb_id: bool = True
    #: The structure's name, as printed.
    #:
    #: A plain string rather than a switch over something read from the header,
    #: because the header's version is a starting point and not an authority: it
    #: is sometimes absent, sometimes in shouting capitals, and sometimes a
    #: sentence nobody would put on a label. The front end prefills it from
    #: whatever the header gave and thereafter it belongs to the user. Empty
    #: means no name on the plaque, which is also how it is switched off.
    plaque_title_text: str = ""

    #: A line of the user's own — a lab, a date, a name.  Printed under the
    #: title, at the size of the scale line, and folded through the stroke
    #: font's substitution table like every other string on the plaque.
    plaque_note: str = ""
    #: Show a physical scale bar with its length in ångström.
    plaque_scalebar: bool = True
    #: Show one row per chain: a colour dot and the chain's name.  The dot is
    #: exported as a **separate object carrying that chain's colour**, so the
    #: slicer can be told to print it in the same filament as the chain itself.
    plaque_legend: bool = True
    #: Per-chain legend labels supplied by the user, as ``index<TAB>label``
    #: lines. A chain with no entry keeps whatever :func:`stand.legend_label`
    #: works out for it. Kept as the raw string rather than a mapping so it
    #: canonicalises into the cache key by itself, the way every other field
    #: here does.
    #:
    #: Editable because the generated label is a guess and sometimes a bad one:
    #: COMPND records carry class words instead of names, translated titles, and
    #: the occasional typo, and the person holding the print knows what it is.
    plaque_legend_labels: str = ""
    #: Put a white tile behind the lettering — **both blocks, one switch**.
    #:
    #: This governed only the left-hand block at first, on the reasoning that a
    #: colour dot wants a neutral field around it and so the legend's tile should
    #: not be optional.  That was wrong twice over: it made a control labelled
    #: "white tile" leave a white tile on the plate when it was switched off,
    #: which reads as a bug however it is justified, and the justification was
    #: only ever an argument for a good *default*.
    plaque_tile: bool = True
    #: How far (mm) the white backing tile stands off the plate.  Lower than the
    #: lettering on it, so the text still reads as raised.
    plaque_tile_mm: float = 0.45
    #: Cap height (mm) of the largest line of plaque text.  Everything else is a
    #: fixed fraction of it, and it shrinks automatically to fit the panel width.
    plaque_text_mm: float = 5.0
    #: How far (mm) the raised text stands off the plate — or, engraved, how
    #: deep it is cut into it.
    plaque_emboss_mm: float = 0.7
    #: How the lettering meets the apron.  See :class:`PlaqueRelief`.
    plaque_relief: PlaqueRelief = PlaqueRelief.RAISED
    #: Tilt (degrees) of the plaque face toward the viewer.
    #:
    #: A plaque lying flat is read at a glancing angle from anywhere but
    #: directly above, which is the one place nobody looks at a display stand
    #: from.  Raking it is what a lectern does.  Built as a wedge *added* to the
    #: plate — thickest at the back of the apron, tapering to a lip at the front
    #: edge — rather than as a cut into it, because the cut version is limited
    #: to the plate thickness and gives up at about five degrees.  Every face of
    #: the wedge is an upward slope, so it prints with no support.
    apron_rake_deg: float = 0.0
    #: Width (mm) of the left-hand information block, or 0 to split the plate
    #: evenly between it and the legend.
    #:
    #: A plain number of millimetres, after two rounds of trying to be clever
    #: about it. First a fraction of the plate, then a fraction interpolated
    #: between three measured anchors — both were harder to explain than they
    #: were worth, and neither could be checked by looking at the result. The
    #: rule now is one sentence: the left block is this wide, the legend keeps
    #: whatever its names need, and the plate grows if the two do not fit side
    #: by side.
    plaque_info_mm: float = 0.0

    #: Which typeface the lettering is set in.  See :class:`PlaqueFont`.
    plaque_font: PlaqueFont = PlaqueFont.SANS
    #: The thinnest part of a letter, in millimetres, that is worth printing.
    #:
    #: Only reads on a real typeface, which unlike the stroke font has a stem
    #: width of its own that shrinks with the type size.  Where a face's stem
    #: falls below this at the size asked for, the outline is grown by half the
    #: shortfall on every side — which thickens the letter without changing its
    #: shape much, and is the difference between lettering and a slicer quietly
    #: dropping every stroke thinner than one bead.  Default is one nozzle plus
    #: a little; raise it for a 0.6 mm nozzle.
    plaque_min_stroke_mm: float = 0.45
    #: Width (mm) of a text stroke.  Still wider than a single extrusion — a
    #: 0.4 mm nozzle drawing a 0.4 mm line is one bead with nothing either side
    #: of it to hold it down — but no wider than it has to be: heavier strokes
    #: closed up the counters and made the lettering read as blocky.
    plaque_stroke_mm: float = 0.6


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

    # --- bound ligands --------------------------------------------------
    #: Export each bound ligand as its own object (ball-and-stick), and carve the
    #: pocket it sits in out of its host so it lifts out and drops back in.
    #:
    #: **Off by default**, and the reason is that switching it on changes the
    #: *protein*, not only what is in the box: the host gains a drug-shaped void,
    #: which on a solvent-excluded surface is usually a fully enclosed cavity that
    #: a slicer packs with support material.  Someone who came to print a protein
    #: should get the protein they asked for.  Two lesser reasons it is opt-in
    #: rather than a sensible default: what counts as a ligand is a judgement call
    #: that :data:`LIGAND_BLOCKLIST` can only mostly automate, and every ligand is
    #: another object and another filament change in the slicer.
    #:
    #: When it *is* on it is often the whole point of the structure — 9YMP's
    #: inhibitor, 2HHB's four haems — so the switch is one click and the geometry
    #: behind it is not a compromise.
    #:
    #: Water and lone ions are never included whatever this is set to — see
    #: :data:`LIGAND_MIN_HEAVY_ATOMS` and :data:`LIGAND_BLOCKLIST`.
    include_ligands: bool = False
    #: How the ligand is drawn.  See :class:`LigandStyle`.
    ligand_style: LigandStyle = LigandStyle.BALL_STICK
    #: **Ligand atom size** — the *diameter* (mm) of the balls in ball-and-stick,
    #: and the bead size the spacefill style scales its van der Waals radii from.
    #:
    #: It is a diameter and not a radius because that is what the control means to
    #: the person moving it, and the halving belongs in the builder rather than in
    #: the front end — ``cartoon_coil_radius_mm`` is halved in ``index.html``
    #: instead, and the cost of that is a derived value the preset table has to
    #: reproduce exactly or the cache silently stops hitting.
    #:
    #: **Why this is a control at all.**  Every dimension in this file is absolute
    #: print millimetres while atom *positions* scale with
    #: ``scale_mm_per_angstrom``, so the two only agree at one scale.  On a protein
    #: that hardly shows: a surface has no internal spacing to disagree with.  On a
    #: ligand it is the whole appearance — the atoms are 1.4–1.6 Å apart, which is
    #: 2.1 mm at 1.5 mm/Å and 8.4 mm at 6 mm/Å, so a bead size that reads as a
    #: molecule at one scale is a lattice of disconnected pinheads at the other.
    #:
    #: Deriving it from the scale automatically was the obvious alternative and is
    #: wrong: the reason to raise the scale is usually that some *other* part of the
    #: model is too small to print, and silently inflating the ligand to match
    #: takes away the one adjustment that fixes what you were actually looking at.
    ligand_atom_mm: float = 2.2
    #: **Ligand bond size** — the *diameter* (mm) of the sticks, and the whole
    #: thickness of the ``STICKS`` style.
    #:
    #: Independent of the atom size rather than a fixed fraction of it. The ratio
    #: was fine as long as one number scaled the whole molecule, but it is the
    #: ratio itself that decides whether the thing reads as atoms joined by bonds
    #: or as a smooth worm — and at print scale the right answer moves, because
    #: the bonds are what snap and the beads are what hold it together.
    #: 1.4 because that is what the front end's slider ships, and this is what
    #: a request that omits the field gets.  They disagreed -- this said 1.2,
    #: the slider said 1.4 -- and it was invisible because canonical_params drops
    #: the field entirely when ligands are off.  Turn ligands on and the CLI, the
    #: presets and pregenerate.py all built a different model from the website,
    #: under a different cache key, so a pre-generated ligand entry could never
    #: be served to a real request.
    ligand_bond_mm: float = 1.4
    #: Multiplier on the van der Waals radii in the ``SPACEFILL`` style.  Below 1
    #: the atoms separate and the molecule reads as beads; at 1 they fuse into the
    #: solid lump that prints best.
    ligand_vdw_scale: float = 1.0

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
    # --- cartoon (ChimeraX-style ribbon) -------------------------------
    # Three "size" sliders drive the look.  Each ribbon's thickness is a *fixed
    # fraction* of its width (see ``cartoon._RIBBON_ASPECT``), so a size slider
    # scales the whole ribbon — wider and proportionally thicker — and the ribbon
    # stays a flat plank at any size instead of rounding into a tube.
    #: **Helix size** — width (mm) of the twisting helix ribbon.
    cartoon_helix_width_mm: float = 4.5
    #: **Sheet size** — width (mm) of the β-strand ribbon (arrowheads scale with it).
    cartoon_strand_width_mm: float = 4.0
    #: **Tube thickness** — radius (mm) of the round tube used for coil / loops.
    cartoon_coil_radius_mm: float = 0.9
    # Internal cartoon shape constants (not surfaced in the UI).  Path/twist
    # smoothing is fixed in ``cartoon._SMOOTH`` — it was briefly a slider and is
    # not one any more.
    #: How much wider the β-strand arrowhead barbs are than the strand body.
    cartoon_arrow_width_factor: float = 1.7
    #: Length of the strand arrowhead, in residues, from the strand C-terminus.
    cartoon_arrow_residues: float = 1.5
    #: Spline samples per residue for the cartoon sweep (higher = smoother twist).
    cartoon_samples_per_residue: int = 10
    slab_thickness_mm: float = 1.2       # base-slab (or rod) thickness
    slab_scale: float = 1.0              # scale factor on the in-plane base size
    connector_radius_mm: float = 0.6     # strut fusing each base to the backbone
    spline_samples_per_residue: int = 6  # backbone smoothness

    # --- nucleic base / backbone style ---------------------------------
    base_style: BaseStyle = BaseStyle.SLAB
    backbone_style: BackboneStyle = BackboneStyle.TUBE
    #: Sphere radius (mm) for atoms in the **base** "molecule" style.
    atom_radius_mm: float = 1.0
    #: Cylinder radius (mm) for the bonds ("sticks") in the **base** molecule style.
    bond_radius_mm: float = 0.5
    #: Sphere radius (mm) for atoms in the **backbone** "molecule" style.  Kept
    #: separate from the base pair above: the sugar-phosphate backbone and the
    #: base rings are drawn at the same time and want different weights — one
    #: shared slider could only ever be right for one of them.
    backbone_atom_radius_mm: float = 1.0
    #: Cylinder radius (mm) for the bonds in the **backbone** molecule style.
    backbone_bond_radius_mm: float = 0.5

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

    # --- display stand --------------------------------------------------
    #: Generated after the build from the finished meshes; inert unless
    #: ``stand.enabled``.  See :class:`StandParams`.
    stand: StandParams = field(default_factory=StandParams)

    def representation_for(self, mtype: MoleculeType) -> Representation:
        if mtype == MoleculeType.PROTEIN:
            return self.protein_representation
        # A ligand has no choice of representation — ball-and-stick is the only
        # thing that reads as a small molecule — but it must still answer this
        # question, because callers outside the geometry core ask it about a
        # chain's style (``connections._inflate_growth``, for one) and would
        # otherwise be handed the *nucleic* representation and act on it.
        if mtype == MoleculeType.LIGAND:
            return Representation.BALL_STICK
        # A stand part is not built by a representation at all — it arrives
        # already meshed — but the same outside callers ask, so answer with
        # something inert rather than handing back the nucleic style.
        if mtype == MoleculeType.STAND:
            return Representation.BALL_STICK
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


# --- bound ligands ---------------------------------------------------------
#
# A "ligand" here is any residue that is not an amino acid, not a nucleotide and
# not water.  That definition on its own is far too generous: a crystal structure
# is full of things that are in the file because of how it was grown, not because
# they are part of the molecule, and printing those gives you a bag of anonymous
# blobs floating around the protein.  Two filters cut it down, and they are
# deliberately different in kind so they catch different mistakes:
#
# * a **size** floor, below which something cannot be a drug or a cofactor;
# * a **name** blocklist, for additives big enough to pass the size floor.
#
#: Minimum heavy (non-hydrogen) atom count for a residue to count as a ligand.
#:
#: Six is the point where the two filters meet.  Everything smaller is a lone ion
#: or a fragment — a zinc, a chloride, a sulfate (5 atoms), a phosphate (5), an
#: acetate (4), a formate (3), a DMSO (4), an imidazole (5), an ethylene glycol
#: (4) — so the blocklist does not need to name any of those and stays short.  It
#: is also below every real ligand worth printing: the smallest interesting ones
#: (a nucleotide fragment, a short peptide-like inhibitor, a haem's porphyrin) run
#: well into double figures.
LIGAND_MIN_HEAVY_ATOMS: int = 6

#: Residue names that are never treated as ligands however big they are.
#:
#: These are the reagents of crystallography rather than the biology: cryo- and
#: precipitant molecules, buffers, reducing agents, detergents.  They pass the
#: size floor above (glycerol has exactly 6 heavy atoms, PEG fragments and MPD
#: more) which is precisely why the list has to exist.
#:
#: What is *deliberately absent* matters as much as what is here.  Sugars are not
#: blocked — glucose, maltose and acarbose are amylase's substrate, mannose is
#: what a lectin structure is *for* — so the only glycosylation codes listed are
#: the two N-linked GlcNAc stubs, which are essentially never the point of a
#: structure and otherwise litter a glycoprotein with a dozen tiny objects.
#: Nucleotides and cofactors (ATP, ADP, GTP, NAD, FAD, SAM, haem, ...) are not
#: blocked either: they are usually exactly what someone wants to see bound.
LIGAND_BLOCKLIST = frozenset({
    # polyols / cryoprotectants
    "GOL", "EDO", "PGO", "PGR", "PDO", "MPD", "MRD", "BU3", "BU1", "HEZ",
    "TBU", "IPA", "IPH", "MOH", "EOH", "DIO", "12P", "15P",
    # polyethylene glycols (the many CCD codes for "some length of PEG")
    "PEG", "PGE", "PG4", "PG5", "PG6", "P6G", "P33", "1PE", "2PE", "XPE",
    "PE3", "PE4", "PE5", "PE8", "7PE", "M2M", "TOE",
    # buffers
    "TRS", "EPE", "MES", "PIN", "BTB", "TAM", "NHE", "CXS", "IMD", "MPO",
    # salts / small anions big enough to reach the size floor
    "SO4", "PO4", "PO3", "NO3", "CO3", "CAC", "MLI", "MLA", "TLA", "TAR",
    "SIN", "OXL", "FLC", "CIT", "ACT", "ACY", "FMT",
    # reducing agents / thiols
    "BME", "DTT", "DTU", "DTV", "TCE",
    # solvents
    "DMS", "DMF", "ACN", "ETA", "GAI",
    # detergents (membrane-protein crystallography)
    "LDA", "LMT", "BOG", "BNG", "HTG", "OGA", "C8E", "P4C", "LI1",
    # N-linked glycosylation stubs — a modification, not a bound ligand
    "NAG", "NDG",
})


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
# Every current representation already owns its wall thickness *at build time*:
#   • SURFACE    — a solvent-excluded surface is thick everywhere by construction.
#   • TUBE_SLAB  — min-wall is now a parametric offset applied to the analytic
#     primitives before the mesh boolean (see ``tube_slab.build``), so its tube,
#     slabs and connectors are already ≥ min-wall.  Re-voxelising either here
#     would only reintroduce the grid stairstep we moved off of.
#   • BALL_STICK — same deal, and with more at stake: a ligand is the smallest
#     object in the file, so a voxel pass at the grid spacing chosen for a whole
#     protein would blur its rings shut and lose the one thing it is there for.
# The pass therefore stays in ``meshops`` only as a fallback for hypothetical
# future representations that build a thin shell and cannot self-thicken.
MIN_WALL_EXEMPT = frozenset({
    Representation.SURFACE, Representation.TUBE_SLAB, Representation.CARTOON,
    Representation.BALL_STICK,
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
