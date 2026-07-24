"""Secondary-structure cartoon representation for proteins.

A print-oriented cartoon (not a smooth ChimeraX ribbon): the Cα trace is turned
into three kinds of exact analytic solid, fused with one manifold boolean into a
single watertight body:

* **helix** (SSE ``a``, runs of ≥ 4 residues) — a straight cylinder from the run
  start to the run end (the classic "cylinder helix" cartoon; very printable);
* **sheet** (SSE ``b``) — flat, wide planks (oriented boxes) following the strand;
* **everything else** — a thin tube along the smoothed Cα spline.

A continuous thin **spine** tube is always laid through the whole Cα spline first,
so the chain is guaranteed to be one connected, watertight solid regardless of
which helix/sheet solids sit on top of it.

Secondary structure is assigned with biotite's ``annotate_sse`` (P-SEA).  If that
is unavailable or disagrees with the residue count, every residue falls back to
coil, so the build degrades to a smooth tube rather than failing.

Chunkiness is controlled by ``params.cartoon_thickness_mm`` (helix radius; sheets
and the spine derive their sizes from it), and grown to satisfy ``min_wall_mm``.
"""

from __future__ import annotations

import numpy as np

from ..config import PrintParams
from ._common import catmull_rom
from . import _manifold
from .tube_slab import _residue_iter, _atom_coord


def _ca_points(chain) -> np.ndarray:
    """Per-residue Cα coordinate (ångström), falling back to the residue centroid."""
    pts = []
    for _res_name, res in _residue_iter(chain.atoms):
        ca = _atom_coord(res, "CA")
        if ca is None:
            ca = res.coord.mean(axis=0).astype(float)
        pts.append(ca)
    return np.asarray(pts, float)


def _sse(chain, n: int):
    """Per-residue secondary structure as a list of 'a'/'b'/'c'.

    Falls back to all-coil ('c') if biotite is missing, errors, or returns a
    length that does not line up with our residue list.
    """
    try:
        import biotite.structure as struc
        sse = struc.annotate_sse(chain.atoms)
        if len(sse) == n:
            return [str(x) for x in sse]
    except Exception:
        pass
    return ["c"] * n


def _runs(sse):
    """Group consecutive equal SSE labels into ``(label, start, end)`` (inclusive)."""
    runs = []
    i = 0
    n = len(sse)
    while i < n:
        j = i
        while j + 1 < n and sse[j + 1] == sse[i]:
            j += 1
        runs.append((sse[i], i, j))
        i = j + 1
    return runs


def _perp(v: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to ``v`` (v assumed non-zero)."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    p = np.cross(v, ref)
    n = np.linalg.norm(p)
    return p / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _plank(a, b, prev, nxt, half_w: float, half_t: float):
    """A flat, wide oriented box (sheet slab) spanning Cα points ``a``→``b``.

    ``prev``/``nxt`` are neighbouring Cα points (or ``None``) used to estimate the
    strand's local plane so the plank lies flat and widens perpendicular to the
    strand direction.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return None
    t = axis / length
    # Estimate the sheet-plane normal from the local curvature; fall back to any
    # perpendicular when the three points are colinear.
    w = None
    if nxt is not None:
        w = np.asarray(nxt, float) - b
    elif prev is not None:
        w = a - np.asarray(prev, float)
    if w is None or np.linalg.norm(np.cross(t, w)) < 1e-6:
        normal = _perp(t)
    else:
        normal = np.cross(t, w)
        normal /= np.linalg.norm(normal)
    width_dir = np.cross(normal, t)
    width_dir /= np.linalg.norm(width_dir)
    axes = np.array([t, width_dir, normal])           # rows = local axes
    half_extents = [length / 2.0, half_w, half_t]
    center = 0.5 * (a + b)
    return _manifold.oriented_box(center, axes, half_extents)


def build(chain, params: PrintParams):
    """Return a watertight trimesh cartoon of a protein ``chain``."""
    s = params.scale_mm_per_angstrom
    ca = _ca_points(chain) * s
    n = len(ca)
    if n == 0:
        raise ValueError("No residues to build a cartoon for.")

    # Sizes (mm), grown to honour the minimum wall.
    half = params.min_wall_mm / 2.0 if params.min_wall_mm > 0 else 0.0
    helix_r = max(params.cartoon_thickness_mm, half)
    coil_r = max(params.cartoon_thickness_mm * 0.55, half)
    plank_t = max(params.cartoon_thickness_mm * 0.6, half)
    plank_w = max(params.cartoon_thickness_mm * 1.7, coil_r)

    solids = []

    # 1) Continuous spine along the smoothed Cα trace — guarantees connectivity.
    if n >= 2:
        spline = catmull_rom(ca, params.spline_samples_per_residue)
        for i in range(len(spline) - 1):
            solids.append(_manifold.capsule(spline[i], spline[i + 1], coil_r))
    else:
        solids.append(_manifold.sphere(ca[0], coil_r))

    # 2) Helix cylinders and sheet planks on top.
    sse = _sse(chain, n)
    for kind, a, b in _runs(sse):
        if kind == "a" and (b - a) >= 3:
            solids.append(_manifold.capsule(ca[a], ca[b], helix_r))
        elif kind == "b" and (b - a) >= 1:
            for k in range(a, b):
                prev = ca[k - 1] if k - 1 >= 0 else None
                nxt = ca[k + 2] if k + 2 < n else None
                box = _plank(ca[k], ca[k + 1], prev, nxt, plank_w, plank_t)
                if box is not None:
                    solids.append(box)

    fused = _manifold.union(solids)
    return _manifold.to_trimesh(fused)
