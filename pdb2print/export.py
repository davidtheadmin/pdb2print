"""Exporters: multi-object 3MF (primary), per-chain STL (fallback), GLB preview.

The 3MF is written with lib3mf — Prusa's own library — as one object per chain
with a distinct colour, which is what lets PrusaSlicer assign a filament per
chain for the Core One's multi-material setup.
"""

from __future__ import annotations

import os
import zipfile
from typing import List, Tuple

import numpy as np
import trimesh

from .config import color_for_index, MoleculeType
from .chains import Chain

# A built chain is the (Chain, repaired trimesh) pair produced by the pipeline.
BuiltChain = Tuple[Chain, trimesh.Trimesh]


# --------------------------------------------------------------------------
# 3MF (primary output)
# --------------------------------------------------------------------------
def object_colors(built: List[BuiltChain]) -> List[tuple]:
    """The colour for each built object, in palette order.

    Chains and ligands take the next palette entry by position, which is what
    keeps a protein the same colour whether or not a drug was added after it.
    An object that carries its own ``color`` — the display-stand parts — keeps
    that instead, because a legend swatch is only useful if it is *the same
    colour as the chain it names*, and the palette index of a stand part has
    nothing to do with the chain it refers to.
    """
    colors = []
    for i, (chain, _mesh) in enumerate(built):
        own = getattr(chain, "color", None)
        colors.append(tuple(own) if own is not None else color_for_index(i))
    return colors


def write_3mf(built: List[BuiltChain], path: str) -> str:
    """Write a multi-object, per-chain-coloured 3MF. Returns ``path``.

    Falls back to a GLB with the same basename if lib3mf is unavailable, so the
    pipeline still yields a coloured multi-object file.
    """
    # Hard watertight gate (defence in depth): the pipeline already gates each
    # chain, but never let a non-manifold object into a 3MF regardless of caller.
    broken = [chain.label() for chain, mesh in built if not mesh.is_watertight]
    if broken:
        raise RuntimeError(
            "Refusing to write 3MF: non-watertight chain(s) "
            + ", ".join(broken) + ". Slicers reject non-manifold geometry."
        )

    try:
        import lib3mf
    except Exception:
        alt = os.path.splitext(path)[0] + ".glb"
        write_glb(built, alt)
        raise RuntimeError(
            "lib3mf is not installed; wrote a GLB fallback instead at "
            f"{alt}. Install lib3mf for true 3MF output."
        )

    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()
    color_group = model.AddColorGroup()
    palette = object_colors(built)

    # The display stand is several solids — plate, tiles, lettering, one object
    # per legend colour — that are one *thing*. Left as loose build items they
    # arrive in the slicer as a dozen peers of the protein, and dragging the
    # stand aside means selecting all of them without catching a chain. Collected
    # into a components object they arrive as a single object with parts: one
    # click moves the whole stand, and each part still takes its own filament.
    stand_components = []

    for i, (chain, mesh) in enumerate(built):
        mesh_obj = model.AddMeshObject()
        mesh_obj.SetName(chain.object_name())

        vertices = []
        for v in mesh.vertices:
            pos = lib3mf.Position()
            pos.Coordinates = (float(v[0]), float(v[1]), float(v[2]))
            vertices.append(pos)
        triangles = []
        for f in mesh.faces:
            tri = lib3mf.Triangle()
            tri.Indices = (int(f[0]), int(f[1]), int(f[2]))
            triangles.append(tri)
        mesh_obj.SetGeometry(vertices, triangles)

        r, g, b = palette[i]
        color = wrapper.FloatRGBAToColor(float(r), float(g), float(b), 1.0)
        color_id = color_group.AddColor(color)
        mesh_obj.SetObjectLevelProperty(color_group.GetResourceID(), color_id)

        if getattr(chain, "mtype", None) == MoleculeType.STAND:
            stand_components.append(mesh_obj)
        else:
            model.AddBuildItem(mesh_obj, wrapper.GetIdentityTransform())

    if stand_components:
        try:
            group = model.AddComponentsObject()
            group.SetName("Display stand")
            for mesh_obj in stand_components:
                group.AddComponent(mesh_obj, wrapper.GetIdentityTransform())
            model.AddBuildItem(group, wrapper.GetIdentityTransform())
        except Exception:
            # Older lib3mf builds without components support: fall back to loose
            # items, which is what shipped before. Worse to handle, not broken.
            for mesh_obj in stand_components:
                model.AddBuildItem(mesh_obj, wrapper.GetIdentityTransform())

    writer = model.QueryWriter("3mf")
    writer.WriteToFile(path)
    return path


# --------------------------------------------------------------------------
# STL (per-chain fallback)
# --------------------------------------------------------------------------
def write_stls(built: List[BuiltChain], out_dir: str) -> List[str]:
    """Write one STL per chain into ``out_dir``; return the list of paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for chain, mesh in built:
        p = os.path.join(out_dir, f"{chain.label()}.stl")
        mesh.export(p)
        paths.append(p)
    return paths


def write_stl_zip(built: List[BuiltChain], path: str) -> str:
    """Bundle per-chain STLs into a single zip for easy download."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="pdb2print_stl_")
    paths = write_stls(built, tmp)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))
    return path


# --------------------------------------------------------------------------
# GLB (interactive preview)
# --------------------------------------------------------------------------
# Bright magnet-marker colour for the preview (stands out from any chain colour).
_MAGNET_MARKER_RGBA = np.array([255, 0, 230, 255], dtype=np.uint8)


def _marker_geoms(markers):
    """Bright placeholder solids showing where magnets sit (preview only)."""
    geoms = []
    for mk in markers or []:
        center = np.asarray(mk["center"], float)
        axis = np.asarray(mk["axis"], float)
        height = 2.0 * float(mk["thickness"])
        transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis)
        transform[:3, 3] = center
        if mk.get("shape") == "square":
            d = float(mk["diameter"])
            g = trimesh.creation.box(extents=(d, d, height), transform=transform)
        else:
            g = trimesh.creation.cylinder(
                radius=float(mk["diameter"]) / 2.0, height=height, transform=transform)
        g.visual = trimesh.visual.ColorVisuals(mesh=g, face_colors=_MAGNET_MARKER_RGBA)
        geoms.append(g)
    return geoms


def build_scene(built: List[BuiltChain], markers=None) -> trimesh.Scene:
    """Assemble a coloured trimesh.Scene (one node per chain) for preview.

    ``markers`` (magnet placements from the connections pass) are added as bright
    highlight solids so the user can see where magnets will sit — preview only,
    never part of the printable 3MF/STL.
    """
    scene = trimesh.Scene()
    palette = object_colors(built)
    for i, (chain, mesh) in enumerate(built):
        m = mesh.copy()
        r, g, b = palette[i]
        rgba = (np.array([r, g, b, 1.0]) * 255).astype(np.uint8)
        m.visual = trimesh.visual.ColorVisuals(mesh=m, face_colors=rgba)
        scene.add_geometry(m, node_name=chain.label(), geom_name=chain.label())
    for k, geom in enumerate(_marker_geoms(markers)):
        name = f"magnet_{k}"
        scene.add_geometry(geom, node_name=name, geom_name=name)
    return scene


def write_glb(built: List[BuiltChain], path: str, markers=None) -> str:
    """Export a coloured GLB (used by the preview), with optional magnet markers."""
    scene = build_scene(built, markers=markers)
    scene.export(path)
    return path
