"""Representation implementations.

Each module exposes ``build(chain, params) -> trimesh.Trimesh`` and is dispatched
from :func:`pdb2print.geometry.generate_chain_mesh`.  Adding a representation is
a matter of dropping a new module here and registering it in ``geometry.py``.

``ligand`` is the one that is not user-selectable: it is chosen by molecule type
rather than by a setting, because ball-and-stick is the only sensible way to draw
a bound small molecule.
"""
