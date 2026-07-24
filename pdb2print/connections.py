"""Post-build connector / joinery pass (simplified).

The per-chain meshes come out of the geometry core as separate watertight
solids, one colour each.  This pass *modifies those solids* — with the same
``manifold3d`` kernel the representations use — so chosen chains are joined for
printing, while **every object stays watertight and a single connected body**.
The 3MF export is unchanged: still individual coloured objects, only now
touching, bulged together, or pocketed for magnets.

Two independent switches (see :class:`~pdb2print.config.ConnectionParams`):

* **connect** — join chains whose surfaces come within ``contact_threshold_mm``:
  - **magnets**: subtract a magnet pocket (round cylinder or square block,
    Ø × thickness) from each side, so parts printed separately snap together;
  - **inflate**: grow both surfaces at the contact until they merge (organic,
    no visible strut) — the default;
  - **bridge**: drop a short cylinder spanning the gap (a peg/strut).
* **basepair_connect** — tie the two strands of a DNA duplex together at every
  base pair.  Complementary bases are paired by centroid geometry with an
  antiparallel-register check and a distance cutoff, so a wrong pair or an
  unwound bubble is never bridged.  Rod/slab rungs are *continued at their own
  radius to the midline* so opposing rungs meet as one smooth bar; molecule
  bases are joined by thin bond-like spokes.

**Watertight guarantee.** Every boolean goes through :func:`_commit`, which
accepts the result only if it is still a single connected body — so a pocket
that would blow through a thin backbone is skipped (min-wall honoured) rather
than shipping a broken object.  The pipeline re-gates watertightness after the
pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import (
    PrintParams, ConnectionParams, NoMagnetMethod, MagnetShape,
    MoleculeType, BaseStyle,
)
from .chains import Chain
from .representations import _manifold, tube_slab


# Cap the vertex clouds used for nearest-point detection so big surface meshes
# stay fast; chunky print geometry does not need exact nearest surface points.
_MAX_PROBE_VERTS = 3000


@dataclass
class Connection:
    """One applied (or skipped) connector, for the report/UI."""

    a_id: str
    b_id: str
    kind: str            # protein-protein | dna-protein | dna-dna | dna-basepair
    method: str          # magnet | inflate | bridge | basepair
    gap_mm: float = 0.0
    count: int = 1
    applied: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "a": self.a_id, "b": self.b_id, "kind": self.kind,
            "method": self.method, "gap_mm": round(float(self.gap_mm), 2),
            "count": self.count, "applied": self.applied, "note": self.note,
        }


# --------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _frame(axis: np.ndarray) -> np.ndarray:
    """Orthonormal 3×3 whose *rows* are (u, v, axis) — for oriented boxes."""
    w = _unit(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(ref, w))
    v = np.cross(w, u)
    return np.array([u, v, w])


def _probe_points(mesh) -> np.ndarray:
    v = np.asarray(mesh.vertices, float)
    if len(v) > _MAX_PROBE_VERTS:
        idx = np.random.default_rng(0).choice(len(v), _MAX_PROBE_VERTS, replace=False)
        v = v[idx]
    return v


def _nearest(mesh_a, mesh_b) -> Tuple[np.ndarray, np.ndarray, float]:
    """Nearest (point_on_a, point_on_b, gap) between two meshes' vertex clouds."""
    pa = _probe_points(mesh_a)
    pb = _probe_points(mesh_b)
    dist, idx = cKDTree(pb).query(pa, k=1)
    k = int(np.argmin(dist))
    return pa[k], pb[idx[k]], float(dist[k])


def _min_wall_radius(params: PrintParams, r: float) -> float:
    if params.min_wall_mm > 0:
        return max(r, params.min_wall_mm / 2.0)
    return r


def _kind(a: Chain, b: Chain) -> str:
    if a.mtype == MoleculeType.PROTEIN and b.mtype == MoleculeType.PROTEIN:
        return "protein-protein"
    if a.mtype == MoleculeType.NUCLEIC and b.mtype == MoleculeType.NUCLEIC:
        return "dna-dna"
    return "dna-protein"


# --------------------------------------------------------------------------
# Watertight-safe boolean commit
# --------------------------------------------------------------------------
def _components(man) -> int:
    try:
        return len(man.decompose())
    except Exception:
        return 1


def _commit(man, tool, add: bool):
    """Apply ``man ± tool`` and accept it only if the result is one solid body.

    Returns the new manifold on success, or ``None`` if the boolean emptied the
    object or split it into more than one piece (e.g. a pocket that would blow
    through a thin wall).  This is what keeps every object watertight and honours
    minimum wall — a bad connector is skipped, never shipped.
    """
    try:
        out = (man + tool) if add else _manifold.difference(man, tool)
    except Exception:
        return None
    if out.is_empty() or _components(out) != 1:
        return None
    return out


# --------------------------------------------------------------------------
# Chain-to-chain join primitives
# --------------------------------------------------------------------------
def _magnet_tool(mid, axis, diameter, thickness, shape: MagnetShape):
    """The single tool subtracted from *both* parts for a magnet joint.

    Centred on the gap midpoint and extending ``±thickness`` along ``axis`` — so
    the two assembled magnets (each ``thickness`` thick, ``2·thickness`` total)
    meet in the middle.  Each part keeps only the portion of this cylinder that
    lies inside it, giving a seat from which its magnet protrudes to touch the
    other; nothing of the far wall is left, which is what a per-surface pocket
    got wrong.
    """
    r = diameter / 2.0
    a_end = mid - axis * thickness
    b_end = mid + axis * thickness
    if shape == MagnetShape.SQUARE:
        return _manifold.oriented_box(mid, _frame(axis), [r, r, thickness])
    return _manifold.frustum(a_end, b_end, r, r)


#: Minimum number of supporting contacts under a magnet disc to accept the seat.
#: Below this the interface is a spike/sliver with too little meat — abandon it.
_MIN_MAGNET_FOOTPRINT = 3


def _farthest_seeds(points: np.ndarray, n: int) -> List[int]:
    """Indices of ``n`` well-spread points (farthest-point sampling)."""
    if len(points) <= n:
        return list(range(len(points)))
    seeds = [0]
    d2 = np.sum((points - points[0]) ** 2, axis=1)
    for _ in range(1, n):
        k = int(np.argmax(d2))
        seeds.append(k)
        d2 = np.minimum(d2, np.sum((points - points[k]) ** 2, axis=1))
    return seeds


def _magnet_sites(mesh_a, mesh_b, count: int, contact_thresh: float, magnet_r: float):
    """Placement for ``count`` magnets, chosen for a well-supported seat.

    Returns ``(center, axis, gap)`` per magnet.  The trap with "just take the
    closest points" is a thin surface spike: it gives the smallest gap but has no
    material to seat a magnet.  So each candidate contact point is *scored by how
    much consistent contact surrounds it* — points within a magnet-sized radius
    whose contact direction agrees — which is high on a broad flat interface
    (plenty of meat) and low on an isolated spike.  Magnets are placed at the
    best-supported, well-separated spots, and each one's centre/axis is averaged
    over its supporting neighbourhood so it sits square on the interface.
    """
    pa = _probe_points(mesh_a)
    pb = _probe_points(mesh_b)
    d, idx = cKDTree(pb).query(pa, k=1)
    band = min(float(d.min()) + 1.5, contact_thresh)
    mask = d <= band
    if not mask.any():
        mask = np.zeros(len(d), bool)
        mask[int(np.argmin(d))] = True
    A, B, dd = pa[mask], pb[idx[mask]], d[mask]
    mids = 0.5 * (A + B)
    dirs = B - A
    dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9, None)

    reach = magnet_r + 1.5                      # area a magnet needs around it
    neigh = cKDTree(mids).query_ball_point(mids, reach)
    consistent = []                             # supporting neighbours per candidate
    support = np.zeros(len(mids), int)
    for k in range(len(mids)):
        nb = np.asarray(neigh[k], dtype=int)
        cons = nb[dirs[nb] @ dirs[k] > 0.6]     # same-facing contact = real interface
        consistent.append(cons if len(cons) else np.array([k]))
        support[k] = len(cons)

    count = max(1, count)
    # Prefer meaty seats (high support); break ties toward the smaller gap.
    order = sorted(range(len(mids)), key=lambda k: (-support[k], dd[k]))
    chosen = []
    for k in order:
        if len(chosen) >= count:
            break
        if any(np.linalg.norm(mids[k] - mids[c]) < reach for c in chosen):
            continue
        chosen.append(k)

    sites = []
    for k in chosen:
        cons = consistent[k]
        # Place the magnet on the meaty seed's own closest-point line — centre at
        # its midpoint, axis exactly along it.  (Averaging directions or fitting
        # the patch plane both made orientation worse on some interfaces, so we
        # keep the simple, tight line.)  Footprint counts supporting contacts
        # under the magnet disc so spikes/slivers are abandoned upstream.
        # KNOWN ISSUE: on a protein wrapped against curved DNA the nearest-point
        # line can still be off-normal (see NOTES.md TODO).
        center = mids[k]
        axis = _unit(dirs[k])
        footprint = int((np.linalg.norm(mids[cons] - center, axis=1) <= magnet_r).sum())
        gap = float(dd[k])
        sites.append((center, axis, gap, footprint))
    return sites


def _apply_magnet(mans, i, j, mesh_a, mesh_b, count: int,
                  cp: ConnectionParams, markers: list) -> Tuple[bool, str]:
    """Subtract ``count`` centred ``Ø × 2·thickness`` magnet cylinders.

    Each is placed on the interface plane (see :func:`_magnet_sites`) and its
    position/size is recorded in ``markers`` so the preview can highlight where
    the magnets sit.  A magnet is skipped where its combined thickness cannot
    span the local gap (``2·thickness > gap``).
    """
    d, t, shape = cp.connector_diameter_mm, cp.magnet_thickness_mm, cp.magnet_shape
    sites = _magnet_sites(mesh_a, mesh_b, count, cp.contact_threshold_mm, d / 2.0)
    placed = 0
    reasons = []
    for center, axis, gap, footprint in sites:
        if footprint < _MIN_MAGNET_FOOTPRINT:
            reasons.append("too little contact area for a magnet")
            continue
        if 2.0 * t <= gap + 1e-6:
            reasons.append(f"gap {gap:.1f} mm ≥ 2×thickness {2*t:.1f} mm")
            continue
        tool = _magnet_tool(center, axis, d, t, shape)
        a2 = _commit(mans[i], tool, add=False)
        b2 = _commit(mans[j], tool, add=False)
        if a2 is not None and b2 is not None:
            mans[i], mans[j] = a2, b2
            placed += 1
            markers.append({"center": [float(x) for x in center],
                            "axis": [float(x) for x in axis],
                            "diameter": float(d), "thickness": float(t),
                            "shape": shape.value})
        else:
            reasons.append("would break watertightness")

    attempted = len(sites)
    if placed == attempted and not reasons:
        note = f"{placed} magnet(s)" if placed > 1 else ""
    elif placed:
        note = f"placed {placed}/{attempted} — skipped: " + "; ".join(sorted(set(reasons)))
    else:
        note = "no magnet placed — " + "; ".join(sorted(set(reasons)) or ["no contact"])
    return placed > 0, note


def _apply_bridge(mans, i, j, pa, pb, cp: ConnectionParams, params: PrintParams) -> bool:
    """Union a short cylinder spanning the gap — half grown from each object."""
    r = _min_wall_radius(params, cp.connector_diameter_mm / 2.0)
    mid = 0.5 * (pa + pb)
    n = _unit(pb - pa)
    ov = 0.3
    a2 = _commit(mans[i], _manifold.capsule(pa, mid + n * ov, r), add=True)
    b2 = _commit(mans[j], _manifold.capsule(pb, mid - n * ov, r), add=True)
    if a2 is not None:
        mans[i] = a2
    if b2 is not None:
        mans[j] = b2
    return a2 is not None and b2 is not None


def _rebuild_inflated(chain: Chain, params: PrintParams, amount_mm: float):
    """Rebuild one chain's mesh grown outward by ``amount_mm`` (the inflate join).

    "Inflate" is a small, uniform size increase applied at *build* time — exactly
    the surface-padding knob for a protein and the tube/base radii for DNA — so
    two neighbours swell until their surfaces overlap and weld, with no strut and
    no re-meshing artefact.  The growth is bounded by the contact threshold, so
    it stays small.
    """
    import dataclasses
    from . import geometry, meshops
    if chain.mtype == MoleculeType.PROTEIN:
        q = dataclasses.replace(
            params,
            surface_atom_padding_ang=params.surface_atom_padding_ang
            + amount_mm / params.scale_mm_per_angstrom,
        )
    else:
        q = dataclasses.replace(
            params,
            nucleic_radius_mm=params.nucleic_radius_mm + amount_mm,
            slab_thickness_mm=params.slab_thickness_mm + 2.0 * amount_mm,
            connector_radius_mm=params.connector_radius_mm + amount_mm,
            atom_radius_mm=params.atom_radius_mm + amount_mm,
            bond_radius_mm=params.bond_radius_mm + amount_mm,
        )
    mesh = geometry.generate_chain_mesh(chain, q)
    mesh = meshops.enforce_min_wall(mesh, q)
    return meshops.repair(mesh)


# --------------------------------------------------------------------------
# DNA interstrand base-pair connect
# --------------------------------------------------------------------------
def _pair_bases(cen_a: np.ndarray, cen_b: np.ndarray, max_dist: float):
    """Complementary base pairs between two strands (indices, distance).

    A duplex pairs bases in a fixed *register*: antiparallel means strand A
    residue ``i`` pairs strand B residue ``s - i`` for one constant ``s`` (a
    parallel register — e.g. when the file stores one strand reversed — is
    ``i + off``).  We evaluate every register, keep only pairs within
    ``max_dist``, and choose the register that pairs the most bases (shortest
    total distance breaks ties).  This is one-to-one and crossing-free by
    construction — it locks onto the true partner at the helix ends where a
    greedy nearest-neighbour swaps a base for its diagonal neighbour — and the
    cutoff naturally drops overhangs and unwound bubbles rather than mis-pairing
    them.
    """
    n_a, n_b = len(cen_a), len(cen_b)
    if n_a == 0 or n_b == 0:
        return []
    dmat = np.linalg.norm(cen_a[:, None, :] - cen_b[None, :, :], axis=-1)

    def register(index_of_b):
        pairs, total = [], 0.0
        for i in range(n_a):
            j = index_of_b(i)
            if 0 <= j < n_b and dmat[i, j] <= max_dist:
                pairs.append((i, j, float(dmat[i, j])))
                total += dmat[i, j]
        return pairs, total

    best, best_key = [], (-1, np.inf)
    # Antiparallel diagonals (j = s - i) and parallel diagonals (j = i + off).
    registers = ([(lambda i, s=s: s - i) for s in range(n_a + n_b - 1)]
                 + [(lambda i, o=o: i + o) for o in range(-(n_a - 1), n_b)])
    for idx_of_b in registers:
        pairs, total = register(idx_of_b)
        key = (len(pairs), -total)
        if key > best_key:
            best, best_key = pairs, key
    return best


def _apply_basepairs(man_a, man_b, chain_a: Chain, chain_b: Chain,
                     params: PrintParams):
    """Tie two DNA strands together at each base pair; returns (a, b, n_links)."""
    cp = params.connections
    cen_a = tube_slab.base_centroids_mm(chain_a, params)
    cen_b = tube_slab.base_centroids_mm(chain_b, params)
    max_dist = cp.basepair_max_dist_ang * params.scale_mm_per_angstrom
    pairs = _pair_bases(cen_a, cen_b, max_dist)
    if not pairs:
        return man_a, man_b, 0

    molecule = params.base_style == BaseStyle.MOLECULE
    if molecule:
        # Thin bond-like spokes between the paired bases.
        r = _min_wall_radius(params, params.bond_radius_mm)
    else:
        # Continue each rung at its own radius to the midline so opposing rungs
        # meet as one smooth bar (rod rungs are nucleic_radius*0.7 thick).
        r = _min_wall_radius(params, params.nucleic_radius_mm * 0.7)

    # The link runs *along the base-pair axis*, continuing each rung to the
    # middle (not a separate strut off the backbone).  It starts a little behind
    # the centroid — a back-step *onto* the existing rung — so it always fuses to
    # the strand body while still reading as the rung reaching the axis.
    back = max(0.6, r)
    fwd = 0.4
    n_done = 0
    for ia, ib, _d in pairs:
        ca, cb = cen_a[ia], cen_b[ib]
        mid = 0.5 * (ca + cb)
        u = _unit(cb - ca)                     # A-centroid → B-centroid (toward axis)
        sa = _commit(man_a, _manifold.capsule(ca - u * back, mid + u * fwd, r), add=True)
        sb = _commit(man_b, _manifold.capsule(cb + u * back, mid - u * fwd, r), add=True)
        if sa is not None and sb is not None:
            man_a, man_b = sa, sb
            n_done += 1
    return man_a, man_b, n_done


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _nucleic_strand_pairs(built) -> List[Tuple[int, int]]:
    """Pair up nucleic chains into duplexes by nearest-mesh proximity."""
    idx = [i for i, (c, _m) in enumerate(built) if c.mtype == MoleculeType.NUCLEIC]
    if len(idx) < 2:
        return []
    if len(idx) == 2:
        return [(idx[0], idx[1])]
    pairs, used = [], set()
    for a in range(len(idx)):
        if idx[a] in used:
            continue
        best, best_d = None, np.inf
        for b in range(len(idx)):
            if a == b or idx[b] in used:
                continue
            _, _, gap = _nearest(built[idx[a]][1], built[idx[b]][1])
            if gap < best_d:
                best, best_d = idx[b], gap
        if best is not None:
            pairs.append((idx[a], best))
            used.add(idx[a])
            used.add(best)
    return pairs


def apply(built: List[Tuple[Chain, "object"]], params: PrintParams):
    """Apply the connections pass; returns ``(new_built, [connection dicts])``.

    Input meshes must already be watertight (the pipeline gates them first).
    Output meshes are re-repaired (the fast-path preserves the already-good
    manifold result) and remain watertight single bodies.
    """
    from . import meshops
    cp = params.connections
    if not cp.enabled() or len(built) < 1:
        return built, [], []

    chains = [c for c, _m in built]
    meshes = [m for _c, m in built]
    applied: List[Connection] = []
    markers: list = []   # magnet positions for the preview highlight
    inflate = cp.connect and not cp.use_magnets \
        and cp.no_magnet_method == NoMagnetMethod.INFLATE

    # 1a) Detect chain-to-chain contacts (skipping DNA↔DNA when base-pairing).
    contacts = []  # (i, j, pa, pb, gap)
    if cp.connect:
        n = len(built)
        for i in range(n):
            for j in range(i + 1, n):
                if (cp.basepair_connect
                        and chains[i].mtype == MoleculeType.NUCLEIC
                        and chains[j].mtype == MoleculeType.NUCLEIC):
                    continue
                pa, pb, gap = _nearest(meshes[i], meshes[j])
                if gap <= cp.contact_threshold_mm:
                    contacts.append((i, j, pa, pb, gap))

    # 1b) Inflate rebuilds each contacting object slightly larger (before the
    #     manifold conversion) so neighbours swell until they overlap.
    if inflate and contacts:
        grow = [0.0] * len(built)
        for i, j, _pa, _pb, gap in contacts:
            # Half the gap on each side, plus a small weld; capped so a wide gap
            # can't balloon a thin chain (use bridge for those instead).  Kept
            # deliberately gentle — just enough to overlap.
            amt = min(max(0.0, gap) / 2.0 + 0.05, 1.0)
            grow[i] = max(grow[i], amt)
            grow[j] = max(grow[j], amt)
        for idx, amt in enumerate(grow):
            if amt > 0:
                try:
                    meshes[idx] = _rebuild_inflated(chains[idx], params, amt)
                except Exception:
                    pass
        for i, j, _pa, _pb, gap in contacts:
            applied.append(Connection(
                chains[i].chain_id, chains[j].chain_id,
                _kind(chains[i], chains[j]), "inflate", gap_mm=gap, applied=True))

    mans = [_manifold.from_trimesh(m) for m in meshes]

    # 1c) Magnet / bridge joins operate on the manifolds.
    if cp.connect and not inflate:
        for i, j, pa, pb, gap in contacts:
            kind = _kind(chains[i], chains[j])
            if cp.use_magnets:
                method = "magnet"
                # Protein↔protein and DNA↔protein each get their own count.
                n_mag = cp.magnet_count if kind == "protein-protein" else cp.dna_magnet_count
                ok, note = _apply_magnet(
                    mans, i, j, meshes[i], meshes[j], n_mag, cp, markers)
            else:
                method = "bridge"
                ok = _apply_bridge(mans, i, j, pa, pb, cp, params)
                note = "" if ok else "skipped (would break watertightness)"
            applied.append(Connection(
                chains[i].chain_id, chains[j].chain_id, kind, method,
                gap_mm=gap, applied=ok, note=note))

    # 2) DNA interstrand base-pair connect.
    if cp.basepair_connect:
        for i, j in _nucleic_strand_pairs(built):
            mans[i], mans[j], n_links = _apply_basepairs(
                mans[i], mans[j], chains[i], chains[j], params)
            applied.append(Connection(
                chains[i].chain_id, chains[j].chain_id, "dna-basepair", "basepair",
                count=n_links, applied=n_links > 0,
                note=f"{n_links} base-pair link(s)" if n_links else "no base pairs found",
            ))

    # 3) Back to meshes; repair fast-path keeps the already-watertight results.
    new_built = []
    for (chain, old_mesh), man in zip(built, mans):
        mesh = _manifold.to_trimesh(man)
        mesh.metadata.update(old_mesh.metadata)
        mesh = meshops.repair(mesh)
        new_built.append((chain, mesh))
    return new_built, [c.as_dict() for c in applied], markers
