"""End-to-end orchestration, framework-free.

``build_all`` is the one call the UI (or a script, or a future WASM shim) makes:
input -> chains -> per-chain mesh -> repair -> min-wall.  It returns the built
chains plus a small report so callers can surface progress and warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from . import io, chains as chains_mod, geometry, meshops, connections
from .config import PrintParams, InterferenceRule
from .export import BuiltChain


class BuildCancelled(RuntimeError):
    """Raised inside :func:`build_all` when the caller asks it to stop.

    Kept distinct from every other failure so the per-chain error handler can
    re-raise it rather than swallowing it into a "skipped this chain" warning and
    carrying on — a cancel has to unwind the whole build, not one chain of it.
    """


@dataclass
class BuildReport:
    built: List[BuiltChain] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: Applied connectors (each a plain dict from ``connections.apply``).
    connections: List[dict] = field(default_factory=list)
    #: Magnet marker placements (center/axis/size) for the preview highlight.
    connection_markers: List[dict] = field(default_factory=list)

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
              progress=None, should_cancel=None) -> BuildReport:
    """Run the full pipeline for a PDB ID or file path.

    ``progress`` is an optional callable ``(fraction, message)`` for UI feedback.

    ``should_cancel`` is an optional zero-argument predicate polled between
    phases (and between chains); when it first returns true the build raises
    :class:`BuildCancelled`.  Cancellation is cooperative rather than pre-emptive
    — a single long boolean cannot be interrupted — so the effective granularity
    is one chain, which is what makes abandoning a large complex cheap without
    the geometry core needing to know anything about threads or HTTP.
    """
    def report(frac, msg):
        if progress is not None:
            progress(frac, msg)

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise BuildCancelled("Build cancelled.")

    check_cancel()
    report(0.05, "Loading structure…")
    atoms, names = io.load_with_names(source)

    report(0.15, "Identifying chains…")
    chain_list = chains_mod.split_chains(atoms)
    if not chain_list:
        raise ValueError("No protein or nucleic-acid chains found in the input.")
    # Attach the subunit name parsed from the header (falls back to the chain id
    # at display time); geometry never depends on it.
    for chain in chain_list:
        chain.name = names.get(chain.chain_id)

    out = BuildReport()
    not_watertight = []
    n = len(chain_list)
    for i, chain in enumerate(chain_list):
        check_cancel()
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
        except BuildCancelled:      # never downgrade a cancel to a skipped chain
            raise
        except Exception as exc:  # keep going; report the bad chain
            out.warnings.append(f"Skipped {chain.label()}: {exc}")
            continue
        # Builders record any parameter they had to clamp to stay meshable
        # (probe radius / grid spacing) so the user hears about it.
        for note in mesh.metadata.get("notes", ()):
            if note not in out.warnings:
                out.warnings.append(note)
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

    # Fit + connector pass.  This runs whenever *either* is wanted: even with no
    # connectors at all, chains meshed independently interpenetrate at every
    # binding interface, and two objects that simply get printed and handed over
    # still have to fit together.
    check_cancel()
    needs_fit = params.resolve_interference != InterferenceRule.NONE
    if params.connections.enabled() or needs_fit:
        report(0.9, "Fitting and connecting objects…"
               if params.connections.enabled() else "Fitting objects together…")
        try:
            # The pass is the slowest part of a large complex, so its own
            # progress is forwarded rather than leaving the bar parked at 0.9 —
            # otherwise a build that is working and a build that is stuck are
            # indistinguishable, which is exactly how it looked on a five-chain
            # structure.
            def _conn_progress(f, m):
                # The connector pass is the slowest part of a large complex, so
                # cancellation is polled here too rather than only before it.
                check_cancel()
                report(0.90 + 0.06 * f, m)

            out.built, out.connections, out.connection_markers, fit_notes = \
                connections.apply(out.built, params, progress=_conn_progress)
            out.warnings.extend(fit_notes)
        except BuildCancelled:      # a cancel unwinds; it is not a skipped pass
            raise
        except Exception as exc:  # never let the connector pass sink a good build
            out.warnings.append(f"Connections skipped: {exc}")
        # Re-gate: a connector must never break watertightness.
        broke = [c.label() for c, m in out.built if not m.is_watertight]
        if broke:
            raise RuntimeError(
                "Watertight gate failed after the connections pass for: "
                + ", ".join(broke) + ". Refusing to export."
            )

    report(1.0, "Done.")
    return out
