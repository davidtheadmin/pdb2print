"""End-to-end orchestration, framework-free.

``build_all`` is the one call the UI (or a script, or a future WASM shim) makes:
input -> chains -> per-chain mesh -> repair -> min-wall.  It returns the built
chains plus a small report so callers can surface progress and warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from . import io, chains as chains_mod, geometry, meshops
from .config import PrintParams
from .export import BuiltChain


@dataclass
class BuildReport:
    built: List[BuiltChain] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Built {len(self.built)} chain(s):"]
        for chain, mesh in self.built:
            wt = "watertight" if mesh.is_watertight else "NOT watertight"
            lines.append(
                f"  • {chain.label()}: {chain.n_residues} residues, "
                f"{len(mesh.faces)} faces, {wt}"
            )
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def build_all(source: str, params: PrintParams,
              progress=None) -> BuildReport:
    """Run the full pipeline for a PDB ID or file path.

    ``progress`` is an optional callable ``(fraction, message)`` for UI feedback.
    """
    def report(frac, msg):
        if progress is not None:
            progress(frac, msg)

    report(0.05, "Loading structure…")
    atoms = io.load_any(source)

    report(0.15, "Identifying chains…")
    chain_list = chains_mod.split_chains(atoms)
    if not chain_list:
        raise ValueError("No protein or nucleic-acid chains found in the input.")

    out = BuildReport()
    not_watertight = []
    n = len(chain_list)
    for i, chain in enumerate(chain_list):
        base = 0.15 + 0.8 * (i / n)
        report(base, f"Meshing {chain.label()} ({i + 1}/{n})…")
        try:
            # Each representation now owns its wall thickness at build time
            # (surface: thick by construction; tube-slab: parametric offset on
            # its primitives), so enforce_min_wall is a no-op for both and only
            # exists as a fallback for future thin representations.  Repair keeps
            # all sizeable components as a safety net.
            mesh = geometry.generate_chain_mesh(chain, params)
            mesh = meshops.enforce_min_wall(mesh, params)
            mesh = meshops.repair(mesh)
        except Exception as exc:  # keep going; report the bad chain
            out.warnings.append(f"Skipped {chain.label()}: {exc}")
            continue
        out.built.append((chain, mesh))
        if not mesh.is_watertight:
            not_watertight.append(chain.label())

    # Hard watertight gate: a non-watertight mesh must never reach the 3MF
    # exporter (slicers reject non-manifold geometry), so fail loudly here
    # rather than silently shipping a broken object.
    if not_watertight:
        raise RuntimeError(
            "Watertight gate failed — these chain(s) meshed but are not "
            "watertight/manifold: " + ", ".join(not_watertight)
            + ". Refusing to export. Try a finer grid spacing or report this "
            "structure as a bug."
        )

    if not out.built:
        raise ValueError("Every chain failed to mesh. See warnings.\n"
                         + "\n".join(out.warnings))
    report(1.0, "Done.")
    return out
