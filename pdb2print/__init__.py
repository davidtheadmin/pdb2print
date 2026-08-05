"""pdb2print — convert PDB structures into 3D-printable, per-chain-colored meshes.

The package is deliberately independent of any web framework (Gradio lives in
``app.py``) so the geometry core can later be ported to a client-side WASM build.

Typical use::

    from pdb2print import config, io, chains, geometry, meshops, export

    params = config.PrintParams()
    atoms = io.load_any("1ubq")            # PDB ID or file path
    built = []
    for ch in chains.split_chains(atoms):
        mesh = geometry.generate_chain_mesh(ch, params)
        mesh = meshops.repair(mesh)
        mesh = meshops.enforce_min_wall(mesh, params)
        built.append((ch, mesh))
    export.write_3mf(built, "out.3mf")
"""

from . import config, io, chains, geometry, meshops, export  # noqa: F401

__all__ = ["config", "io", "chains", "geometry", "meshops", "export"]

#: Bump together with ``CITATION.cff`` and an entry in ``CHANGELOG.md``.
#:
#: The minor version is the one to watch: the exported mesh is only comparable
#: within a release line, because the geometry depends on the pinned
#: ``manifold3d``.  A version that changes the mesh invalidates the disk cache.
__version__ = "1.3.0"
