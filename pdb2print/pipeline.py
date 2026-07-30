"""End-to-end orchestration, framework-free.

``build_all`` is the one call the UI (or a script, or a future WASM shim) makes:
input -> chains -> per-chain mesh -> repair -> min-wall.  It returns the built
chains plus a small report so callers can surface progress and warnings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from . import io, chains as chains_mod, geometry, meshops, connections
from .config import PrintParams, InterferenceRule, MoleculeType
from .export import BuiltChain


class BuildCancelled(RuntimeError):
    """Raised inside :func:`build_all` when the caller asks it to stop.

    Kept distinct from every other failure so the per-chain error handler can
    re-raise it rather than swallowing it into a "skipped this chain" warning and
    carrying on — a cancel has to unwind the whole build, not one chain of it.
    """


#: Rough peak memory one large chain needs while its surface is meshed.  Measured
#: 2026-07-30 at ``scale=1.5, spacing=0.35`` on a 1535-atom chain: ~780 MiB of
#: RSS, almost all of it inside the EDT and marching cubes.  Finer settings cost
#: more, so this is a floor, not a guarantee — which is why ``auto`` stays
#: conservative and why the flag is off by default.
_WORKER_PEAK_BYTES = 1024 ** 3


def _auto_workers() -> int:
    """Worker count from free RAM, not from core count.

    Four cores are useless if the fourth worker is the one that gets OOM-killed
    halfway through a build, so memory is the binding constraint and the one
    this reads.  Falls back to a cautious 2 wherever free memory cannot be
    queried (Windows has no ``sysconf``), because guessing high here is the
    expensive mistake.
    """
    cpus = os.cpu_count() or 1
    try:
        free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return max(1, min(cpus, 2))
    return max(1, min(cpus, int(free // _WORKER_PEAK_BYTES)))


def _worker_count() -> int:
    """How many processes to mesh chains in — ``PDB2PRINT_WORKERS``.

    Unset, ``0`` or ``1`` means the serial loop, byte for byte the code path that
    shipped before this existed.  An integer is taken as given (so a box with
    known headroom can be told what to do); ``auto`` sizes it from free RAM.
    """
    raw = os.environ.get("PDB2PRINT_WORKERS", "").strip().lower()
    if not raw or raw in {"0", "1", "off", "false", "no"}:
        return 1
    if raw == "auto":
        return _auto_workers()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def mesh_chain(chain, params: PrintParams):
    """Mesh one chain: generate, apply the min-wall fallback, repair.

    Module-level and taking only picklable arguments, because it is also the
    function a worker process runs.  ``enforce_min_wall`` is a no-op for every
    representation that currently exists (each owns its wall thickness at build
    time) and stays only as a fallback for future thin ones; ``repair`` keeps all
    sizeable components as a safety net.
    """
    mesh = geometry.generate_chain_mesh(chain, params)
    mesh = meshops.enforce_min_wall(mesh, params)
    return meshops.repair(mesh)


@dataclass
class BuildReport:
    built: List[BuiltChain] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: Applied connectors (each a plain dict from ``connections.apply``).
    connections: List[dict] = field(default_factory=list)
    #: Magnet marker placements (center/axis/size) for the preview highlight.
    connection_markers: List[dict] = field(default_factory=list)
    #: Where the structure was actually read from, and what its header called
    #: it.  Read here because here is the one moment both are certainly
    #: available: an uploaded file is deleted when the build finishes, and a
    #: fetched one costs a download to see again.  The display stand's plaque
    #: wants the title and used to go and find it for itself, twice.
    source_path: Optional[str] = None
    title: Optional[str] = None

    def summary(self) -> str:
        lines = [f"Built {len(self.built)} object(s):"]
        for chain, mesh in self.built:
            wt = "watertight" if mesh.is_watertight else "NOT watertight"
            # A ligand is one residue by definition, so "1 residues" tells you
            # nothing; its atom count is the number that describes it.
            size = (f"{chain.n_atoms} atoms" if chain.mtype == MoleculeType.LIGAND
                    else f"{chain.n_residues} residues")
            lines.append(
                f"  • {chain.label()}: {size}, {len(mesh.faces)} faces, {wt}"
            )
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def _accept_mesh(out: "BuildReport", chain, mesh, not_watertight: List[str]) -> None:
    """Record one successfully meshed chain on the report.

    Shared by the serial and parallel paths so the two cannot drift: the order
    warnings are appended in, and which failures are fatal, is the same code
    either way.
    """
    # Builders record any parameter they had to clamp to stay meshable (probe
    # radius / grid spacing) so the user hears about it.
    for note in mesh.metadata.get("notes", ()):
        if note not in out.warnings:
            out.warnings.append(note)
    # A ligand that will not mesh cleanly is *dropped*, not fatal.  A chain is
    # load-bearing — a complex missing a subunit is the wrong model and the user
    # has to know — but a ligand is an addition, and taking the whole build down
    # over one awkward cofactor would mean a structure that used to export fine
    # stops exporting the moment ligands are switched on.  The warning says what
    # was lost.
    if not mesh.is_watertight:
        if chain.mtype == MoleculeType.LIGAND:
            out.warnings.append(
                f"Left out {chain.display_name()} (chain "
                f"{chain.chain_id}): its mesh did not come out watertight, so "
                f"it could not be exported. The rest of the model is "
                f"unaffected."
            )
            return
        not_watertight.append(chain.label())
    out.built.append((chain, mesh))


def _mesh_chains_parallel(chain_list, params: PrintParams, out: "BuildReport",
                          not_watertight: List[str], report, check_cancel,
                          workers: int) -> None:
    """Mesh every chain across ``workers`` processes.

    **Processes, not threads.**  The hot kernels here — ``distance_transform_edt``,
    ``marching_cubes``, the manifold booleans — hold the GIL, so threads would
    serialise on exactly the work worth spreading.

    **Spawn, not fork**, so a Windows dev box and a Linux container behave the
    same way.  It costs a fresh interpreter and a fresh import of the geometry
    stack per worker, which is why this is off unless asked for: on a small
    build that start-up is more than the meshing it saves.

    Results are reassembled **in chain order**, never completion order, so the
    exported objects and the warning list do not depend on which worker happened
    to finish first.  Only the progress messages arrive out of order, which is
    honest — that is the order the work actually finished in.
    """
    import concurrent.futures as cf
    import multiprocessing as mp

    n = len(chain_list)
    meshes: dict = {}
    failures: dict = {}
    pool = cf.ProcessPoolExecutor(max_workers=workers,
                                  mp_context=mp.get_context("spawn"))
    try:
        futures = {pool.submit(mesh_chain, chain, params): i
                   for i, chain in enumerate(chain_list)}
        pending, done = set(futures), 0
        while pending:
            # A short timeout rather than blocking on completion: a cancel must
            # be noticed while the workers are still busy, not only when one of
            # them happens to hand something back.  Nothing can interrupt a
            # boolean mid-flight, so a worker already inside one runs to the end
            # of its chain either way — but the build stops waiting for the rest.
            finished, pending = cf.wait(pending, timeout=0.5,
                                        return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                i = futures[fut]
                done += 1
                report(0.15 + 0.8 * (done / n),
                       f"Meshed {chain_list[i].label()} ({done}/{n})…")
                try:
                    meshes[i] = fut.result()
                except Exception as exc:      # keep going; report the bad chain
                    failures[i] = exc
            check_cancel()
    except BaseException:
        # Includes BuildCancelled.  Do not wait: the point of cancelling is to
        # stop waiting.  Queued chains are dropped outright and the workers are
        # torn down as they come free.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)

    for i, chain in enumerate(chain_list):
        if i in failures:
            out.warnings.append(f"Skipped {chain.label()}: {failures[i]}")
            continue
        _accept_mesh(out, chain, meshes[i], not_watertight)


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

    out = BuildReport()

    check_cancel()
    report(0.05, "Loading structure…")
    path = io.resolve_source(source)
    out.source_path = path
    atoms, names = io.load_with_names(path)
    try:
        from .names import structure_title
        out.title = structure_title(path)
    except Exception:
        out.title = None

    report(0.15, "Identifying chains…")
    chain_list = chains_mod.split_chains(
        atoms, include_ligands=params.include_ligands)
    if not chain_list:
        raise ValueError("No protein or nucleic-acid chains found in the input.")
    # Attach the subunit name parsed from the header (falls back to the chain id
    # at display time); geometry never depends on it.  A ligand already carries
    # its own name (built from the residue), and the header name belongs to the
    # *polymer* of that chain id, so overwriting it would label the drug with the
    # protein it is bound to.
    for chain in chain_list:
        if chain.mtype != MoleculeType.LIGAND:
            chain.name = names.get(chain.chain_id)

    not_watertight = []
    n = len(chain_list)
    workers = min(_worker_count(), n)
    if workers > 1:
        _mesh_chains_parallel(chain_list, params, out, not_watertight,
                              report, check_cancel, workers)
    else:
        for i, chain in enumerate(chain_list):
            check_cancel()
            base = 0.15 + 0.8 * (i / n)
            report(base, f"Meshing {chain.label()} ({i + 1}/{n})…")
            try:
                mesh = mesh_chain(chain, params)
            except BuildCancelled:  # never downgrade a cancel to a skipped chain
                raise
            except Exception as exc:  # keep going; report the bad chain
                out.warnings.append(f"Skipped {chain.label()}: {exc}")
                continue
            _accept_mesh(out, chain, mesh, not_watertight)

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
        # Re-gate: a connector must never break watertightness.  Same asymmetry
        # as above — a broken ligand is dropped with a note, a broken chain is
        # fatal.  Dropping one here is safe: it is the *host* that was carved, so
        # the pocket is already correct and the model is simply the one without
        # the drug in it.
        kept = []
        for chain, mesh in out.built:
            if mesh.is_watertight or chain.mtype != MoleculeType.LIGAND:
                kept.append((chain, mesh))
                continue
            out.warnings.append(
                f"Left out {chain.display_name()} (chain {chain.chain_id}): the "
                f"fit pass left its mesh non-watertight. Its pocket in the "
                f"surrounding object is still there."
            )
        out.built = kept
        broke = [c.label() for c, m in out.built if not m.is_watertight]
        if broke:
            raise RuntimeError(
                "Watertight gate failed after the connections pass for: "
                + ", ".join(broke) + ". Refusing to export."
            )

    report(1.0, "Done.")
    return out
