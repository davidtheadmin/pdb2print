"""Representation dispatch: turn a chain into a mesh for a chosen representation.

This is the single extension point of the geometry core.  ``generate_chain_mesh``
looks up the representation the user selected for the chain's molecule type and
delegates to the matching builder.  A new representation is added by writing a
``build(chain, params)`` function in ``representations/`` and registering it in
``_BUILDERS`` — nothing else changes.

Ligands are the exception to "selected from params": a bound small molecule is
always ball-and-stick, so it is dispatched on its *molecule type* and never
appears in ``_BUILDERS``, which is the set of styles the UI offers.
"""

from __future__ import annotations

import trimesh

from .config import PrintParams, Representation, MoleculeType
from .chains import Chain
from .representations import surface, tube_slab, cartoon, ligand


# representation -> builder callable(chain, params) -> trimesh.Trimesh
#
# CARTOON is a real ChimeraX-style ribbon: a Carson-Bugg guide frame swept with
# SSE-dependent cross-sections (twisting helix/strand ribbons, arrowheads, coil
# tube) into one watertight solid — see ``representations/cartoon.py``.
_BUILDERS = {
    Representation.SURFACE: surface.build,
    Representation.TUBE_SLAB: tube_slab.build,
    Representation.CARTOON: cartoon.build,
}


def available_representations():
    return list(_BUILDERS.keys())


def generate_chain_mesh(chain: Chain, params: PrintParams) -> trimesh.Trimesh:
    """Build the (pre-repair) mesh for one object given the print parameters.

    The representation is selected from ``params`` by the chain's molecule type,
    so callers never hardcode a representation here.  A ligand bypasses that
    lookup — there is nothing to select — and goes straight to ball-and-stick.
    """
    if chain.mtype == MoleculeType.LIGAND:
        rep = Representation.BALL_STICK
        builder = ligand.build
    else:
        rep = params.representation_for(chain.mtype)
        builder = _BUILDERS.get(rep)
    if builder is None:
        raise ValueError(f"No builder registered for representation {rep!r}.")
    mesh = builder(chain, params)
    mesh.metadata["chain_id"] = chain.chain_id
    mesh.metadata["molecule_type"] = chain.mtype.value
    mesh.metadata["representation"] = rep.value
    return mesh
