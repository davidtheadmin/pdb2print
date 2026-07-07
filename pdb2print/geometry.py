"""Representation dispatch: turn a chain into a mesh for a chosen representation.

This is the single extension point of the geometry core.  ``generate_chain_mesh``
looks up the representation the user selected for the chain's molecule type and
delegates to the matching builder.  A new representation is added by writing a
``build(chain, params)`` function in ``representations/`` and registering it in
``_BUILDERS`` — nothing else changes.
"""

from __future__ import annotations

import trimesh

from .config import PrintParams, Representation
from .chains import Chain
from .representations import surface, tube_slab


# representation -> builder callable(chain, params) -> trimesh.Trimesh
_BUILDERS = {
    Representation.SURFACE: surface.build,
    Representation.TUBE_SLAB: tube_slab.build,
}


def available_representations():
    return list(_BUILDERS.keys())


def generate_chain_mesh(chain: Chain, params: PrintParams) -> trimesh.Trimesh:
    """Build the (pre-repair) mesh for one chain given the print parameters.

    The representation is selected from ``params`` by the chain's molecule type,
    so callers never hardcode a representation here.
    """
    rep = params.representation_for(chain.mtype)
    builder = _BUILDERS.get(rep)
    if builder is None:
        raise ValueError(f"No builder registered for representation {rep!r}.")
    mesh = builder(chain, params)
    mesh.metadata["chain_id"] = chain.chain_id
    mesh.metadata["molecule_type"] = chain.mtype.value
    mesh.metadata["representation"] = rep.value
    return mesh
