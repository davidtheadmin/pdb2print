"""Gradio front-end for pdb2print.

This module is intentionally thin: it collects parameters, calls
``pdb2print.pipeline.build_all``, and wires the results to download widgets.
All geometry lives in the ``pdb2print`` package so this file can be swapped for
a WASM shim later without touching the core.
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr

from pdb2print.config import PrintParams, Representation, MinWallMode
from pdb2print.pipeline import build_all
from pdb2print import export


REPRESENTATION_CHOICES = [Representation.SURFACE.value, Representation.TUBE_SLAB.value]

EXAMPLE_IDS = ["1UBQ", "1BNA", "1ZAA"]


def _run(source_id, uploaded_file, scale, grid_spacing, min_wall, min_wall_mode,
         protein_rep, nucleic_rep, nucleic_radius, slab_thickness,
         progress=gr.Progress()):
    source = None
    if uploaded_file is not None:
        source = uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file
    elif source_id and source_id.strip():
        source = source_id.strip()
    if not source:
        raise gr.Error("Enter a PDB ID or upload a .pdb/.cif file.")

    params = PrintParams(
        scale_mm_per_angstrom=float(scale),
        grid_spacing_mm=float(grid_spacing),
        min_wall_mm=float(min_wall),
        min_wall_mode=MinWallMode(min_wall_mode),
        protein_representation=Representation(protein_rep),
        nucleic_representation=Representation(nucleic_rep),
        nucleic_radius_mm=float(nucleic_radius),
        slab_thickness_mm=float(slab_thickness),
    )

    report = build_all(source, params, progress=lambda f, m: progress(f, desc=m))

    out_dir = tempfile.mkdtemp(prefix="pdb2print_out_")
    stem = os.path.join(out_dir, "pdb2print")

    glb_path = export.write_glb(report.built, stem + ".glb")
    stl_zip = export.write_stl_zip(report.built, stem + "_chains_stl.zip")

    threemf_path = stem + ".3mf"
    threemf_msg = ""
    try:
        export.write_3mf(report.built, threemf_path)
    except RuntimeError as exc:
        threemf_path = None
        threemf_msg = f"\n\n⚠ {exc}"

    return (
        glb_path,
        threemf_path,
        stl_zip,
        report.summary() + threemf_msg,
    )


def build_ui():
    with gr.Blocks(title="pdb2print") as demo:
        gr.Markdown(
            "# pdb2print\n"
            "Convert a PDB/CIF structure into a 3D-printable, per-chain-coloured "
            "**multi-object 3MF** for PrusaSlicer (plus per-chain STL fallback)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                source_id = gr.Textbox(
                    label="PDB ID", placeholder="e.g. 1ZAA", value="1UBQ"
                )
                uploaded = gr.File(
                    label="…or upload a structure",
                    file_types=[".pdb", ".ent", ".cif", ".mmcif", ".bcif"],
                )
                gr.Examples(EXAMPLE_IDS, inputs=source_id, label="Try an example")

                gr.Markdown("### Representation")
                protein_rep = gr.Radio(
                    REPRESENTATION_CHOICES, value=Representation.SURFACE.value,
                    label="Protein chains",
                )
                nucleic_rep = gr.Radio(
                    REPRESENTATION_CHOICES, value=Representation.TUBE_SLAB.value,
                    label="Nucleic-acid chains",
                )

                gr.Markdown("### Print parameters")
                scale = gr.Slider(0.1, 2.0, value=0.5, step=0.05,
                                  label="Scale (mm per Å)")
                grid_spacing = gr.Slider(0.2, 1.5, value=0.5, step=0.05,
                                         label="Grid spacing (mm) — mesh resolution")
                min_wall = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                     label="Minimum wall thickness (mm)")
                min_wall_mode = gr.Radio(
                    [MinWallMode.UNIFORM.value, MinWallMode.SELECTIVE.value],
                    value=MinWallMode.UNIFORM.value, label="Min-wall mode",
                )
                nucleic_radius = gr.Slider(0.4, 3.0, value=1.2, step=0.1,
                                           label="Nucleic backbone/tube radius (mm)")
                slab_thickness = gr.Slider(0.4, 3.0, value=1.2, step=0.1,
                                           label="Base slab thickness (mm)")
                run_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                preview = gr.Model3D(label="Preview (per-chain colours)")
                threemf_out = gr.File(label="Download 3MF (multi-object)")
                stl_out = gr.File(label="Download per-chain STL (zip)")
                log = gr.Textbox(label="Report", lines=8)

        run_btn.click(
            _run,
            inputs=[source_id, uploaded, scale, grid_spacing, min_wall,
                    min_wall_mode, protein_rep, nucleic_rep, nucleic_radius,
                    slab_thickness],
            outputs=[preview, threemf_out, stl_out, log],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
