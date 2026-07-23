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

from pdb2print.config import (
    PrintParams, Representation, MinWallMode, BaseStyle, BackboneStyle,
)
from pdb2print.pipeline import build_all
from pdb2print import export


REPRESENTATION_CHOICES = [Representation.SURFACE.value, Representation.TUBE_SLAB.value]
BASE_STYLE_CHOICES = [BaseStyle.SLAB.value, BaseStyle.ROD.value, BaseStyle.MOLECULE.value]
BACKBONE_STYLE_CHOICES = [BackboneStyle.TUBE.value, BackboneStyle.MOLECULE.value]

EXAMPLE_IDS = ["1UBQ", "1BNA", "1ZAA", "1TUP", "2HHB"]


def _run(source_id, uploaded_file, scale, grid_spacing, min_wall, min_wall_mode,
         protein_rep, nucleic_rep, nucleic_radius, slab_thickness,
         base_style, backbone_style, connector_radius, base_width, atom_radius,
         bond_radius, progress=gr.Progress()):
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
        base_style=BaseStyle(base_style),
        backbone_style=BackboneStyle(backbone_style),
        connector_radius_mm=float(connector_radius),
        slab_scale=float(base_width),
        atom_radius_mm=float(atom_radius),
        bond_radius_mm=float(bond_radius),
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


# Force a dark theme on load by setting the ?__theme=dark query param.
FORCE_DARK_JS = """
() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark');
    window.location.replace(u.href);
  }
}
"""

# Self-contained dark palette + compact control styling.  We hardcode the
# surface colours so the look is stable regardless of Gradio's theme internals;
# the accent is a dark green, with dark red reserved for warnings/errors.
CSS = """
:root {
  --p2p-bg:#0e0f11; --p2p-s1:#17191c; --p2p-s2:#1d2024; --p2p-border:#2a2e33;
  --p2p-text:#e6e8ea; --p2p-text2:#9aa0a6; --p2p-muted:#6b7178;
  --p2p-accent:#2f9e6f; --p2p-accent-d:#1f6b4a; --p2p-red:#b3453f;
}
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;
  background: var(--p2p-bg) !important;}
.gradio-container, .gradio-container * {border-color: var(--p2p-border);}

/* Top bar */
.p2p-topbar {align-items: center !important; gap: 12px !important;
  padding: 4px 2px 10px !important; border-bottom: 0.5px solid var(--p2p-border);
  margin-bottom: 12px;}
.p2p-word h1 {font-size: 1.15rem !important; margin: 0 !important;
  color: var(--p2p-text) !important; font-weight: 600;}
.p2p-word p {display: none;}

/* Cards / panels */
.p2p-card {border: 0.5px solid var(--p2p-border) !important; border-radius: 12px !important;
  padding: 14px 16px !important; background: var(--p2p-s1) !important;}
.p2p-head {font-size: 0.82rem !important; letter-spacing: .04em; text-transform: uppercase;
  color: var(--p2p-text2) !important; font-weight: 600 !important; margin: 0 0 4px !important;}
.p2p-cap {font-size: 0.8rem !important; color: var(--p2p-text2) !important;
  margin: 10px 0 4px !important;}
.p2p-cap:first-child {margin-top: 0 !important;}

/* Segmented toggles (styled radios) */
.p2p-seg .wrap, .p2p-seg fieldset {display: flex !important; gap: 0 !important;
  border: 0.5px solid var(--p2p-border); border-radius: 8px; overflow: hidden;
  background: var(--p2p-s2);}
.p2p-seg label {flex: 1; margin: 0 !important; padding: 7px 6px !important;
  justify-content: center; text-align: center; font-size: 12px !important;
  color: var(--p2p-text2) !important; background: transparent !important;
  border: none !important; border-left: 0.5px solid var(--p2p-border) !important;
  border-radius: 0 !important; cursor: pointer;}
.p2p-seg label:first-child {border-left: none !important;}
.p2p-seg label.selected, .p2p-seg label:has(input:checked) {
  background: var(--p2p-accent) !important; color: #fff !important;}
.p2p-seg input {display: none !important;}
.p2p-seg .wrap::before {display: none !important;}

/* Nested DNA block */
.p2p-nested {border-left: 2px solid var(--p2p-accent) !important;
  padding-left: 12px !important; margin-top: 8px;}

/* Preset chips */
.p2p-presets {gap: 6px !important;}
.p2p-presets button {border-radius: 999px !important; font-size: 12px !important;
  min-height: 30px !important; background: var(--p2p-s2) !important;
  color: var(--p2p-text2) !important; border: 0.5px solid var(--p2p-border) !important;}
.p2p-presets button:hover {border-color: var(--p2p-accent) !important;
  color: var(--p2p-text) !important;}

/* Sliders: compact */
.gradio-container input[type=range] {accent-color: var(--p2p-accent);}
.p2p-card .label-wrap, .gr-slider label {font-size: 12px !important;}

/* Generate button */
.p2p-generate button {background: var(--p2p-accent) !important; color: #fff !important;
  border: none !important; font-weight: 600 !important;}
.p2p-generate button:hover {background: var(--p2p-accent-d) !important;}

/* Preview stays put on scroll */
.p2p-preview {position: sticky; top: 8px;}

/* Report reads calmly; error emphasis handled by content */
.p2p-report textarea {font-family: var(--font-mono, monospace) !important;
  font-size: 12px !important; background: var(--p2p-s1) !important;}
"""

# (grid_spacing, min_wall, nucleic_radius, slab_thickness, base_width,
#  connector_radius, atom_radius, bond_radius)
_PRESETS = {
    "Balanced":    (0.5, 1.0, 1.2, 1.2, 1.0, 0.6, 1.0, 0.50),
    "Chunky":      (0.5, 1.5, 2.2, 2.2, 1.4, 1.4, 1.8, 1.10),
    "Fine detail": (0.3, 0.8, 0.9, 0.9, 0.9, 0.6, 0.8, 0.45),
}


def _molecule_active(base_style_val, backbone_style_val):
    """Ball-and-stick sizing only matters when a molecule style is selected."""
    return (base_style_val == BaseStyle.MOLECULE.value
            or backbone_style_val == BackboneStyle.MOLECULE.value)


def build_ui():
    with gr.Blocks(title="pdb2print") as demo:

        # ---- top bar: wordmark · PDB id · generate --------------------------
        with gr.Row(elem_classes="p2p-topbar"):
            gr.Markdown("# pdb2print", elem_classes="p2p-word")
            source_id = gr.Textbox(
                placeholder="PDB ID, e.g. 1ZAA", value="1UBQ",
                show_label=False, container=False, scale=3,
            )
            run_btn = gr.Button("Generate", variant="primary", scale=1,
                                elem_classes="p2p-generate")

        with gr.Row(equal_height=False):
            # ================= Preview (hero) ==============================
            with gr.Column(scale=5, min_width=340):
                with gr.Column(elem_classes="p2p-preview"):
                    preview = gr.Model3D(label="Preview — per-chain colours",
                                         height=440)
                    with gr.Group(elem_classes="p2p-card"):
                        gr.Markdown("Downloads", elem_classes="p2p-head")
                        with gr.Row():
                            threemf_out = gr.File(label="3MF (PrusaSlicer)")
                            stl_out = gr.File(label="STL (zip)")
                    log = gr.Textbox(label="Report", lines=5,
                                     elem_classes="p2p-report")

            # ================= Settings ====================================
            with gr.Column(scale=4, min_width=300):

                # --- Appearance: representation + DNA style (one panel) ----
                with gr.Group(elem_classes="p2p-card"):
                    gr.Markdown("Appearance", elem_classes="p2p-head")
                    gr.Markdown("Protein", elem_classes="p2p-cap")
                    protein_rep = gr.Radio(
                        REPRESENTATION_CHOICES, value=Representation.SURFACE.value,
                        show_label=False, container=False, elem_classes="p2p-seg",
                    )
                    with gr.Group(elem_classes="p2p-nested"):
                        gr.Markdown("DNA / RNA representation", elem_classes="p2p-cap")
                        nucleic_rep = gr.Radio(
                            REPRESENTATION_CHOICES,
                            value=Representation.TUBE_SLAB.value,
                            show_label=False, container=False, elem_classes="p2p-seg",
                        )
                        gr.Markdown("Backbone", elem_classes="p2p-cap")
                        backbone_style = gr.Radio(
                            BACKBONE_STYLE_CHOICES, value=BackboneStyle.TUBE.value,
                            show_label=False, container=False, elem_classes="p2p-seg",
                        )
                        gr.Markdown("Base (rung)", elem_classes="p2p-cap")
                        base_style = gr.Radio(
                            BASE_STYLE_CHOICES, value=BaseStyle.SLAB.value,
                            show_label=False, container=False, elem_classes="p2p-seg",
                        )

                # --- Print: presets + headline knobs + drawers -------------
                with gr.Group(elem_classes="p2p-card"):
                    gr.Markdown("Print", elem_classes="p2p-head")
                    gr.Markdown("Preset", elem_classes="p2p-cap")
                    with gr.Row(elem_classes="p2p-presets"):
                        preset_btns = [
                            gr.Button(name, size="sm", variant="secondary")
                            for name in _PRESETS
                        ]
                    scale = gr.Slider(0.2, 6.0, value=1.5, step=0.1,
                                      label="Scale (mm/Å)")
                    min_wall = gr.Slider(0.0, 5.0, value=1.0, step=0.1,
                                         label="Min wall (mm)")

                    with gr.Accordion("Advanced dimensions", open=False):
                        nucleic_radius = gr.Slider(0.4, 8.0, value=1.2, step=0.1,
                                                   label="Tube radius (mm)")
                        slab_thickness = gr.Slider(0.4, 8.0, value=1.2, step=0.1,
                                                   label="Base thickness (mm)")
                        base_width = gr.Slider(0.3, 3.0, value=1.0, step=0.05,
                                               label="Base width scale")
                        connector_radius = gr.Slider(
                            0.2, 8.0, value=0.6, step=0.1,
                            label="Connector radius (mm) — the ladder spokes")
                        with gr.Group(visible=False) as ball_stick_group:
                            gr.Markdown("Ball-and-stick sizing",
                                        elem_classes="p2p-cap")
                            atom_radius = gr.Slider(0.3, 8.0, value=1.0, step=0.1,
                                                    label="Atom radius (mm)")
                            bond_radius = gr.Slider(0.2, 8.0, value=0.5, step=0.1,
                                                    label="Bond radius (mm)")
                        grid_spacing = gr.Slider(
                            0.2, 1.5, value=0.5, step=0.05,
                            label="Grid spacing (mm) — mesh resolution")
                        min_wall_mode = gr.Radio(
                            [MinWallMode.UNIFORM.value, MinWallMode.SELECTIVE.value],
                            value=MinWallMode.UNIFORM.value, label="Min-wall mode",
                        )

                # --- Upload (tucked away) ----------------------------------
                with gr.Accordion("Upload a structure instead of a PDB ID",
                                  open=False):
                    uploaded = gr.File(
                        show_label=False,
                        file_types=[".pdb", ".ent", ".cif", ".mmcif", ".bcif"],
                    )
                gr.Examples(EXAMPLE_IDS, inputs=source_id, label="Examples")

        # ---- interactivity -------------------------------------------------
        bs_inputs = [base_style, backbone_style]
        for comp in bs_inputs:
            comp.change(
                lambda b, bb: gr.update(visible=_molecule_active(b, bb)),
                inputs=bs_inputs, outputs=ball_stick_group,
            )

        preset_targets = [grid_spacing, min_wall, nucleic_radius, slab_thickness,
                          base_width, connector_radius, atom_radius, bond_radius]
        for btn, name in zip(preset_btns, _PRESETS):
            btn.click(lambda vals=_PRESETS[name]: vals, outputs=preset_targets)

        run_btn.click(
            _run,
            inputs=[source_id, uploaded, scale, grid_spacing, min_wall,
                    min_wall_mode, protein_rep, nucleic_rep, nucleic_radius,
                    slab_thickness, base_style, backbone_style, connector_radius,
                    base_width, atom_radius, bond_radius],
            outputs=[preview, threemf_out, stl_out, log],
        )
    return demo


if __name__ == "__main__":
    theme = gr.themes.Base(
        primary_hue="emerald", secondary_hue="red", neutral_hue="slate",
    )
    build_ui().launch(theme=theme, css=CSS, js=FORCE_DARK_JS)
