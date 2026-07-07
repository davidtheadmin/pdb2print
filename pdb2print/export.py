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

from .config import color_for_index
from .chains import Chain

# A built chain is the (Chain, repaired trimesh) pair produced by the pipeline.
BuiltChain = Tuple[Chain, trimesh.Trimesh]


# --------------------------------------------------------------------------
# 3MF (primary output)
# --------------------------------------------------------------------------
def write_3mf(built: List[BuiltChain], path: str) -> str:
    """Write a multi-object, per-chain-coloured 3MF. Returns ``path``.

    Falls back to a GLB with the same basename if lib3mf is unavailable, so the
    pipeline still yields a coloured multi-object file.
    """
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

    for i, (chain, mesh) in enumerate(built):
        mesh_obj = model.AddMeshObject()
        mesh_obj.SetName(chain.label())

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

        r, g, b = color_for_index(i)
        color = wrapper.FloatRGBAToColor(float(r), float(g), float(b), 1.0)
        color_id = color_group.AddColor(color)
        mesh_obj.SetObjectLevelProperty(color_group.GetResourceID(), color_id)

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
def build_scene(built: List[BuiltChain]) -> trimesh.Scene:
    """Assemble a coloured trimesh.Scene (one node per chain) for preview."""
    scene = trimesh.Scene()
    for i, (chain, mesh) in enumerate(built):
        m = mesh.copy()
        r, g, b = color_for_index(i)
        rgba = (np.array([r, g, b, 1.0]) * 255).astype(np.uint8)
        m.visual = trimesh.visual.ColorVisuals(mesh=m, face_colors=rgba)
        scene.add_geometry(m, node_name=chain.label(), geom_name=chain.label())
    return scene


def write_glb(built: List[BuiltChain], path: str) -> str:
    """Export a coloured GLB (used by the Gradio Model3D preview)."""
    scene = build_scene(built)
    scene.export(path)
    return path
