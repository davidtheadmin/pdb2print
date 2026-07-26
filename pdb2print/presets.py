"""Named parameter bundles — the one definition the UI and the cache share.

The front end shows these as the "Style presets" chips; ``scripts/pregenerate.py``
uses them to build the shipped cache.  Both must agree *exactly*: a cached entry
is found by hashing the parameters, so a preset that differs between the two by
so much as 0.1 mm produces a different key and the cache silently never hits.

Mirroring the UI, not inventing a parallel set
----------------------------------------------
Everything here reproduces what ``frontend/index.html`` actually submits for a
given chip, including the parts a preset does *not* set — the protein style, the
surface probe, the joinery switches — because those still reach the server and
still land in the cache key.  Two places worth knowing about:

* ``slab_thickness_mm`` / ``slab_scale`` are **derived** in the front end.  The
  UI exposes ``rod_thickness`` and ``plate_size``; which one is submitted depends
  on the base style, and the plate style scales its thickness with its footprint
  through ``PLATE_THICKNESS_PER_SIZE`` so a plate can never come out wide and
  paper-thin.  :func:`_slab` reproduces that.
* The cartoon dimensions are the UI's slider defaults rather than the dataclass
  defaults, and "tube thickness" is a *diameter* to the user but a radius to the
  builder — hence the halving.

Magnet diameter, magnet thickness, press-fit clearance and collar wall are
deliberately absent: they describe the magnets you bought and the printer you
own, so they must survive a preset change.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from .config import (
    PrintParams, ConnectionParams, Representation, BaseStyle, BackboneStyle,
)

#: The chip the UI opens on, and the one the cache is warmest for.
DEFAULT_PRESET = "Clean ladder"

#: Must match ``PLATE_THICKNESS_PER_SIZE`` in ``frontend/index.html``.
PLATE_THICKNESS_PER_SIZE = 1.2


def _slab(base_style: BaseStyle, rod_thickness: float, plate_size: float) -> dict:
    """The ``slab_thickness_mm`` / ``slab_scale`` pair the UI would submit.

    A rod is round: it reuses the thickness field as its diameter and has no
    in-plane footprint to scale.  Everything else goes down the plate path.
    """
    if base_style == BaseStyle.ROD:
        return dict(slab_thickness_mm=round(rod_thickness, 3), slab_scale=1.0)
    return dict(slab_thickness_mm=round(plate_size * PLATE_THICKNESS_PER_SIZE, 3),
                slab_scale=round(plate_size, 3))


#: The UI's "chunky" dimension set — the numbers that survive an FDM print
#: without supports.  Shared by every chip.  The backbone tube carries the whole
#: strand and is what snaps if it is skinny, so it runs thicker than the rungs
#: hanging off it.
_CHUNKY = dict(
    scale_mm_per_angstrom=1.5,
    grid_spacing_mm=0.5,
    min_wall_mm=1.0,
    nucleic_radius_mm=3.5,
    connector_radius_mm=1.4,
    atom_radius_mm=2.2,
    bond_radius_mm=1.2,
    backbone_atom_radius_mm=2.2,
    backbone_bond_radius_mm=1.2,
)

#: Controls a chip does not touch, but which the form still submits.  Held here
#: so the cache key a preset produces matches the key a real request produces.
_UI_UNTOUCHED = dict(
    protein_representation=Representation.SURFACE,
    nucleic_representation=Representation.TUBE_SLAB,
    probe_radius_ang=1.4,
    surface_atom_padding_ang=0.0,
    protein_tube_radius_mm=1.2,
    cartoon_helix_width_mm=4.5,
    cartoon_strand_width_mm=4.0,
    cartoon_coil_radius_mm=0.9,     # UI "tube thickness" 1.8 mm diameter / 2
    # The "Include ligands" checkbox, which loads checked.  This happens to equal
    # the ``PrintParams`` default, so omitting it would work today — and would be
    # a trap the first time the form's default and the dataclass default part
    # company, because the two sides would then hash differently and the cache
    # would stop hitting without anything appearing to be wrong.  Every field the
    # form submits is stated here for exactly that reason.
    include_ligands=False,          # the "Include ligands" switch, off as it loads
    # Still listed although the switch above makes ``canonical_params`` drop it:
    # the form submits it regardless of the switch, and if the mirror only held the
    # fields that happen to reach the key today, turning ligands on would be the
    # moment the two sides silently diverged.
    ligand_atom_mm=2.2,             # the "Ligand atoms" slider's default
    ligand_bond_mm=1.2,             # the "Ligand bonds" slider's default
)

#: Joinery is off until the user asks for it, exactly as the form loads.
_NO_JOINERY = ConnectionParams(connect=False, basepair_connect=False)


def _clean_ladder() -> PrintParams:
    """Smooth backbone tube, one round rod per base.  The default."""
    return PrintParams(
        backbone_style=BackboneStyle.TUBE,
        base_style=BaseStyle.ROD,
        connections=_NO_JOINERY,
        **_slab(BaseStyle.ROD, rod_thickness=2.7, plate_size=1.5),
        **_CHUNKY, **_UI_UNTOUCHED,
    )


def _molecular() -> PrintParams:
    """Ball-and-stick throughout — backbone atoms and base rings alike."""
    return PrintParams(
        backbone_style=BackboneStyle.MOLECULE,
        base_style=BaseStyle.MOLECULE,
        connections=_NO_JOINERY,
        **_slab(BaseStyle.MOLECULE, rod_thickness=2.7, plate_size=1.5),
        **_CHUNKY, **_UI_UNTOUCHED,
    )


def _tube_molecule_bases() -> PrintParams:
    """Smooth backbone tube with ball-and-stick bases hanging off it.

    Slightly finer balls and sticks than :func:`_molecular`: here they sit
    against a solid tube rather than against each other, so they can be smaller
    without looking spindly.
    """
    chunky = dict(_CHUNKY, atom_radius_mm=2.0, bond_radius_mm=1.1)
    return PrintParams(
        backbone_style=BackboneStyle.TUBE,
        base_style=BaseStyle.MOLECULE,
        connections=_NO_JOINERY,
        **_slab(BaseStyle.MOLECULE, rod_thickness=2.7, plate_size=1.5),
        **chunky, **_UI_UNTOUCHED,
    )


#: name -> zero-argument builder.  Builders rather than instances so no caller
#: can mutate a shared ``PrintParams`` (and its nested ``ConnectionParams``) out
#: from under everyone else.
_BUILDERS = {
    "Clean ladder": _clean_ladder,
    "Molecular": _molecular,
    "Tube + molecule bases": _tube_molecule_bases,
}

PRESET_NAMES = tuple(_BUILDERS)


def params_for(name: str, **overrides) -> PrintParams:
    """Return a fresh :class:`PrintParams` for the named preset.

    ``overrides`` are applied on top, which is how the pre-generation spec pins a
    single structure that needs, say, a different scale or the joinery switched
    on without inventing a whole preset for it.
    """
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown preset {name!r}. Known presets: {', '.join(PRESET_NAMES)}."
        ) from None
    params = builder()
    return replace(params, **overrides) if overrides else params


def all_params() -> Dict[str, PrintParams]:
    """Every preset, freshly built."""
    return {name: params_for(name) for name in PRESET_NAMES}
