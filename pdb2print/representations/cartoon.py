"""Secondary-structure cartoon representation for proteins (ChimeraX-style).

A real ribbon cartoon — the kind ChimeraX/PyMOL draw — rebuilt so the result is
a single watertight solid you can print:

* the Cα trace carries a **Carson–Bugg guide frame**: a per-residue orientation
  taken from the *peptide plane* (the carbonyl C=O direction), **not** from the
  curve geometry.  This is the crux.  A Frenet frame from the spline tangent is
  undefined exactly where a β-strand is straight, so a tangent-derived ribbon
  corkscrews; the carbonyl-derived frame makes strands lie flat and consistently
  oriented.  Adjacent carbonyls point ~180° apart along a strand, so every other
  guide vector is flipped to remove that alternation, then the vectors are
  lightly smoothed (Carson & Bugg, *J. Mol. Graphics* 1986).
* helix and strand residues have their Cα control points smoothed toward the
  local run centre-line (fixed strength, :data:`_SMOOTH`), the same trick
  ChimeraX uses so a helix reads as a clean spiral instead of following every
  backbone wobble.
* the path *and* the frame are interpolated with a Catmull–Rom spline to a smooth
  centre-line carrying a smoothly-varying frame at every sample.
* a closed cross-section is swept along it, its shape set by secondary structure:
  a flat rounded rectangle for **helices** (a twisting ribbon) and **strands**
  (a flat plank widening into an **arrowhead** at the C-terminal end), and a
  round tube for **coil / loops**.  The profile morphs smoothly across SSE
  boundaries so segments stay continuous.

Secondary structure is assigned with biotite's ``annotate_sse`` (P-SEA).  If that
is unavailable, errors, or disagrees with the residue count, every residue falls
back to coil, so the build degrades to a smooth tube rather than failing.

Sizes are in print-mm and grown to honour ``min_wall_mm`` **before** the sweep —
the ribbon minor axis and the arrow tip are the thin, fragile features — so the
solid is printable without the voxel min-wall pass.  ``Representation.CARTOON``
therefore stays in :data:`pdb2print.config.MIN_WALL_EXEMPT`.

The sweep is watertight by construction: a regular ``M × K`` grid of ring
vertices closed with two end-cap fans, so every edge is shared by exactly two
faces.  Self-intersection where a helix ribbon twists tightly does not affect
that topological watertightness (slicers resolve it by winding); keeping enough
samples per residue keeps the twist per segment small in any case.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache as _lru_cache

from ..config import PrintParams
from ._common import _catmull_pos_tan  # analytic Catmull-Rom position + tangent
from .tube_slab import _residue_iter, _atom_coord


# Cross-section vertex count.  A little higher than strictly needed so the
# semicircular ribbon edges and the coil tube stay smooth.
_SECTION_VERTS = 20
# A CA–CA step longer than this (ångström, pre-scale) is a chain break; the tube
# still bridges it so the chain prints as one piece, but the guide frame is not
# carried across the gap.
_CHAIN_BREAK_ANG = 4.5
# Ribbon thickness as a fraction of its width — fixed so a ribbon always reads as
# a flat plank.  One "size" slider scales the whole ribbon (wider *and*
# proportionally thicker); it can never round into a tube the way an independent
# thickness knob did.
_RIBBON_ASPECT = 0.30
# Runs shorter than these are demoted to coil, so a stub of β never becomes a
# lone arrowhead and a one-turn blip never becomes a ribbon.
_MIN_HELIX_LEN = 4
_MIN_STRAND_LEN = 3
# Path/twist regularisation strength.  Fixed, not user-facing: exposing it as a
# slider was tried and removed — the useful range was narrow enough that the
# default was the only sensible setting, so it was a knob with nothing to say.
# This is that default; strands are smoothed at full strength and helices only
# lightly (see :func:`_smooth_control_points`).
_SMOOTH = 0.7


def _ca_backbone(chain):
    """Per-residue ``(CA, C, O)`` coordinates in ångström.

    ``CA`` always resolves (residue centroid as a last resort); ``C`` and ``O``
    are the carbonyl atoms and may be ``None`` when absent, in which case the
    guide frame is filled in by transport from the neighbours.
    """
    ca, c, o = [], [], []
    for _res_name, res in _residue_iter(chain.atoms):
        p = _atom_coord(res, "CA")
        if p is None:
            p = res.coord.mean(axis=0).astype(float)
        ca.append(p)
        c.append(_atom_coord(res, "C"))
        o.append(_atom_coord(res, "O"))
    return np.asarray(ca, float), c, o


def _sse(chain, n: int):
    """Per-residue secondary structure as 'a'/'b'/'c'; all-coil on any failure."""
    try:
        import biotite.structure as struc
        sse = struc.annotate_sse(chain.atoms)
        if len(sse) == n:
            return [str(x) for x in sse]
    except Exception:
        pass
    return ["c"] * n


def _runs(sse):
    """Group consecutive equal SSE labels into ``(label, start, end)`` inclusive."""
    runs = []
    i, n = 0, len(sse)
    while i < n:
        j = i
        while j + 1 < n and sse[j + 1] == sse[i]:
            j += 1
        runs.append((sse[i], i, j))
        i = j + 1
    return runs


def _clean_sse(sse):
    """Demote too-short helix/strand runs to coil.

    A β-strand shorter than :data:`_MIN_STRAND_LEN` cannot carry both a shaft and
    an arrowhead, so it would print as a bare arrow; a helix shorter than
    :data:`_MIN_HELIX_LEN` is barely a turn.  Both read better as coil.
    """
    out = list(sse)
    for label, a, b in _runs(sse):
        length = b - a + 1
        if (label == "a" and length < _MIN_HELIX_LEN) or \
           (label == "b" and length < _MIN_STRAND_LEN):
            for i in range(a, b + 1):
                out[i] = "c"
    return out


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between unit vectors ``a`` and ``b``.

    Rotates at a constant angular rate, so a ribbon's twist advances uniformly
    between residues instead of snapping through the linear chord — which is what
    made tightly-twisting helices look kinked.
    """
    d = float(np.clip(a @ b, -1.0, 1.0))
    if d > 0.9995:                       # nearly parallel → linear is fine
        v = a + t * (b - a)
    else:
        theta = np.arccos(d) * t
        c = b - a * d
        cn = np.linalg.norm(c)
        if cn < 1e-9:
            return a
        v = a * np.cos(theta) + (c / cn) * np.sin(theta)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-9 else a


def _perp(v: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to ``v`` (``v`` assumed non-zero)."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    p = np.cross(v, ref)
    nrm = np.linalg.norm(p)
    return p / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])


def _tangents(ca: np.ndarray) -> np.ndarray:
    """Per-residue unit tangents by central difference of the Cα trace."""
    n = len(ca)
    t = np.zeros_like(ca)
    for i in range(n):
        lo = ca[max(i - 1, 0)]
        hi = ca[min(i + 1, n - 1)]
        d = hi - lo
        nrm = np.linalg.norm(d)
        t[i] = d / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])
    return t


def _guide_widths(ca, c_atoms, o_atoms, tan):
    """Per-residue ribbon **width** unit vectors (the Carson–Bugg guide vector).

    The width vector is the carbonyl direction ``C→O`` projected perpendicular to
    the local tangent.  The 180°-per-residue alternation along a strand is
    removed by flipping any vector that opposes its predecessor, and residues
    with no carbonyl inherit a parallel-transported vector from the last good
    one.  Returns an ``(N, 3)`` array of unit vectors.
    """
    n = len(ca)
    raw = [None] * n
    for i in range(n):
        if c_atoms[i] is not None and o_atoms[i] is not None:
            co = np.asarray(o_atoms[i], float) - np.asarray(c_atoms[i], float)
        elif o_atoms[i] is not None:
            co = np.asarray(o_atoms[i], float) - ca[i]
        else:
            continue
        w = co - tan[i] * float(co @ tan[i])
        nrm = np.linalg.norm(w)
        if nrm > 1e-6:
            raw[i] = w / nrm

    # Flip to remove the alternation, carrying the running reference forward.
    prev = None
    for i in range(n):
        if raw[i] is None:
            continue
        if prev is not None and float(raw[i] @ prev) < 0.0:
            raw[i] = -raw[i]
        prev = raw[i]

    # Fill gaps (and a possible all-missing chain) by transport along the tangent.
    ref = None
    for i in range(n):
        if raw[i] is not None:
            ref = raw[i]
            break
    if ref is None:
        ref = _perp(tan[0])
    for i in range(n):
        if raw[i] is None:
            w = ref - tan[i] * float(ref @ tan[i])
            nrm = np.linalg.norm(w)
            raw[i] = w / nrm if nrm > 1e-6 else _perp(tan[i])
        ref = raw[i]
    return np.asarray(raw, float)


def _smooth_widths(width, tan, smoothing, passes=2):
    """Lightly average neighbouring guide vectors, re-orthonormalised.

    Damps carbonyl jitter so the ribbon's twist varies smoothly.  Scaled by the
    same ``smoothing`` master and re-projected perpendicular to each tangent so
    the frame stays valid.
    """
    if smoothing <= 0.0:
        return width
    lam = min(smoothing, 0.8)
    w = width.copy()
    n = len(w)
    for _ in range(passes):
        prev = w.copy()
        for i in range(1, n - 1):
            avg = prev[i] + lam * (0.5 * (prev[i - 1] + prev[i + 1]) - prev[i])
            avg = avg - tan[i] * float(avg @ tan[i])
            nrm = np.linalg.norm(avg)
            if nrm > 1e-6:
                w[i] = avg / nrm
    return w


def _smooth_control_points(ca, sse, smoothing):
    """Regularise helix/strand Cα points, per structure type.

    A single Laplacian step (pull each point toward its neighbours' midpoint),
    but with a **structure-dependent strength**:

    * **strand** — full strength.  A β-strand is nearly straight, so smoothing
      flattens its backbone pleat into a clean plane without shortening it.
    * **helix** — light strength, hard-capped.  Laplacian smoothing pulls a
      helix toward its own axis and, applied strongly, collapses the spiral into
      a straight twisted stick.  Kept gentle so the ribbon keeps winding around
      the (invisible) helix axis — the "ribbon wrapped around a rod" look.
    * **coil** — untouched; loops should follow the true backbone.

    ``smoothing`` in ``[0, 1]`` is the master scale; ``0`` returns the trace.
    """
    if smoothing <= 0.0:
        return ca.copy()
    lam = np.zeros(len(sse))
    for i, s in enumerate(sse):
        if s == "a":
            lam[i] = min(smoothing * 0.25, 0.2)
        elif s == "b":
            lam[i] = smoothing
    out = ca.copy()
    prev = out.copy()
    n = len(ca)
    for i in range(1, n - 1):
        if lam[i] > 0.0:
            mid = 0.5 * (prev[i - 1] + prev[i + 1])
            out[i] = prev[i] + lam[i] * (mid - prev[i])
    return out


def _section(hw: float, ht: float, k: int) -> np.ndarray:
    """Memoised wrapper around :func:`_section_uncached`.

    ``(hw, ht)`` takes only a handful of distinct values along a chain -- within
    an SSE run the two are constant, so the interpolation upstream is a no-op and
    only arrowhead samples vary -- but this was called once per swept sample,
    about 3,000 times on a 300-residue chain, at ~48us each.  The copy is so a
    caller that decides to write into its ring cannot corrupt every later one; it
    costs about a microsecond against the 48 it saves.
    """
    return _section_cached(float(hw), float(ht), int(k)).copy()


@_lru_cache(maxsize=512)
def _section_cached(hw: float, ht: float, k: int) -> np.ndarray:
    return _section_uncached(hw, ht, k)


def _section_uncached(hw: float, ht: float, k: int) -> np.ndarray:
    """A closed **rounded-rectangle** cross-section, ``(k, 2)`` in (width, normal).

    The corner radius is the smaller half-axis, so a wide ribbon (``hw > ht``)
    is a *stadium*: genuinely flat top and bottom faces with semicircular thin
    edges — it reads as a flat plank and never as a rounded square, however thick
    it gets.  When ``hw == ht`` the flats vanish and it is a circle, so a coil
    swept with equal half-axes is a round tube and the profile morphs smoothly
    between the two.

    Sampled by **arc length** from a fixed start so consecutive rings stay in
    correspondence (stable, even triangles) and no two vertices coincide.
    """
    r = min(hw, ht)
    fw = max(hw - r, 0.0)   # half-length of the flat face along the width axis
    fh = max(ht - r, 0.0)   # half-length of the flat face along the normal axis
    q = np.pi / 2.0
    # Boundary pieces, CCW from (-fw, -ht): straight edges + quarter-circle corners.
    pieces = [
        ("edge", ((-fw, -ht), (fw, -ht)), 2.0 * fw),
        ("arc", ((fw, -fh), r, -q, 0.0), q * r),
        ("edge", ((hw, -fh), (hw, fh)), 2.0 * fh),
        ("arc", ((fw, fh), r, 0.0, q), q * r),
        ("edge", ((fw, ht), (-fw, ht)), 2.0 * fw),
        ("arc", ((-fw, fh), r, q, 2.0 * q), q * r),
        ("edge", ((-hw, fh), (-hw, -fh)), 2.0 * fh),
        ("arc", ((-fw, -fh), r, 2.0 * q, 3.0 * q), q * r),
    ]
    pieces = [p for p in pieces if p[2] > 1e-12]   # drop zero-length edges
    total = sum(p[2] for p in pieces) or 1.0

    pts = np.empty((k, 2))
    for i in range(k):
        s = (i / k) * total
        acc = 0.0
        for kind, params, length in pieces:
            if s <= acc + length or (kind, params, length) is pieces[-1]:
                t = (s - acc) / length if length > 1e-12 else 0.0
                if kind == "edge":
                    a = np.asarray(params[0]); b = np.asarray(params[1])
                    pts[i] = a + (b - a) * t
                else:
                    (cx, cy), rr, a0, a1 = params
                    ang = a0 + (a1 - a0) * t
                    pts[i] = (cx + rr * np.cos(ang), cy + rr * np.sin(ang))
                break
            acc += length
    return pts


def _profile_arrays(sse, dims):
    """Per-residue base ``(half_width, half_thick)`` from the SSE.

    Helix and strand are flat ribbons whose thickness is a fixed fraction of
    their width (so they stay flat at any size); coil is a round tube (equal
    half-axes → circular :func:`_section`).
    """
    hw = np.empty(len(sse))
    ht = np.empty(len(sse))
    for i, s in enumerate(sse):
        if s == "a":
            hw[i], ht[i] = dims["helix_hw"], dims["helix_ht"]
        elif s == "b":
            hw[i], ht[i] = dims["strand_hw"], dims["strand_ht"]
        else:
            hw[i], ht[i] = dims["coil_r"], dims["coil_r"]
    return hw, ht


def _arrow_halfwidth(dist_to_end, strand_hw, tip_half, factor, arrow_len):
    """Ribbon half-width inside a strand arrowhead.

    ``dist_to_end`` is residues from the sample to the C-terminal end of the
    strand run.  At the barb base (``dist == arrow_len``) the width steps up to
    ``factor × strand_hw`` (the barbs), then tapers linearly to ``tip_half`` at
    the point (``dist == 0``).
    """
    frac = max(0.0, min(1.0, dist_to_end / arrow_len))
    return tip_half + (factor * strand_hw - tip_half) * frac


def _loft(centers, w_hat, n_hat, sections):
    """Close a swept cross-section into a watertight ``trimesh.Trimesh``.

    ``sections[i]`` is an ``(K, 2)`` ring in the local (width, normal) frame at
    ``centers[i]``.  Rings are joined by quad bands and both ends capped with a
    fan to a centroid vertex.
    """
    import trimesh

    m = len(centers)
    k = sections[0].shape[0]
    verts = np.empty((m * k + 2, 3))
    for i in range(m):
        ring = (centers[i]
                + np.outer(sections[i][:, 0], w_hat[i])
                + np.outer(sections[i][:, 1], n_hat[i]))
        verts[i * k:(i + 1) * k] = ring
    c0 = m * k          # centroid of the first ring
    c1 = m * k + 1      # centroid of the last ring
    verts[c0] = centers[0]
    verts[c1] = centers[-1]

    faces = []
    for s in range(m - 1):
        base0, base1 = s * k, (s + 1) * k
        for kk in range(k):
            k2 = (kk + 1) % k
            a, b = base0 + kk, base0 + k2
            c, d = base1 + k2, base1 + kk
            faces.append((a, b, c))
            faces.append((a, c, d))
    # End caps (winding is normalised by meshops.repair.fix_normals downstream).
    for kk in range(k):
        k2 = (kk + 1) % k
        faces.append((c0, k2, kk))
        faces.append((c1, (m - 1) * k + kk, (m - 1) * k + k2))

    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, np.int64),
                           process=False)


def _dims(params: PrintParams):
    """Effective cross-section half-sizes (print-mm) after min-wall growth.

    The **Sheet size** and **Helix size** sliders set each ribbon's *width*; its
    thickness is a fixed fraction of that (:data:`_RIBBON_ASPECT`), so the ribbon
    stays a flat plank however large the slider goes.  The **Tube thickness**
    slider sets the coil radius.  Every dimension is floored at the minimum wall.
    """
    half_wall = params.min_wall_mm / 2.0 if params.min_wall_mm > 0 else 0.0
    grow = lambda x: max(x, half_wall)
    helix_hw = grow(params.cartoon_helix_width_mm * 0.5)
    strand_hw = grow(params.cartoon_strand_width_mm * 0.5)
    helix_ht = grow(params.cartoon_helix_width_mm * 0.5 * _RIBBON_ASPECT)
    strand_ht = grow(params.cartoon_strand_width_mm * 0.5 * _RIBBON_ASPECT)
    coil_r = grow(params.cartoon_coil_radius_mm)
    # The arrow tip is blunted to a printable minimum rather than a knife edge.
    tip_half = max(strand_ht, half_wall)
    return {
        "helix_hw": helix_hw, "strand_hw": strand_hw,
        "helix_ht": helix_ht, "strand_ht": strand_ht,
        "coil_r": coil_r, "tip_half": tip_half,
    }


def build(chain, params: PrintParams):
    """Return a watertight ``trimesh.Trimesh`` cartoon of a protein ``chain``."""
    s = params.scale_mm_per_angstrom
    ca_ang, c_atoms, o_atoms = _ca_backbone(chain)
    n = len(ca_ang)
    if n == 0:
        raise ValueError("No residues to build a cartoon for.")

    dims = _dims(params)
    if n < 3:
        # Too short for a frame/spline — a plain coil tube keeps it watertight.
        from . import _manifold
        ca = ca_ang * s
        r = dims["coil_r"]
        if n == 1:
            return _manifold.to_trimesh(_manifold.sphere(ca[0], r))
        return _manifold.to_trimesh(_manifold.capsule(ca[0], ca[-1], r))

    sse = _clean_sse(_sse(chain, n))

    # --- guide frame (in ångström, then scaled) -----------------------------
    tan_raw = _tangents(ca_ang)
    width = _guide_widths(ca_ang, c_atoms, o_atoms, tan_raw)
    width = _smooth_widths(width, tan_raw, _SMOOTH)
    ctrl = _smooth_control_points(ca_ang, sse, _SMOOTH) * s

    # Per-residue profile + run bookkeeping for the arrowheads.
    hw_res, ht_res = _profile_arrays(sse, dims)
    run_label = [""] * n
    run_start = [0] * n
    run_end = [0] * n
    for label, a, b in _runs(sse):
        for i in range(a, b + 1):
            run_label[i] = label
            run_start[i] = a
            run_end[i] = b

    spr = max(2, int(params.cartoon_samples_per_residue))
    arrow_len = max(0.5, float(params.cartoon_arrow_residues))
    factor = float(params.cartoon_arrow_width_factor)

    centers, w_hats, n_hats, sections = [], [], [], []

    def add_sample(pos, tangent, w_lin, res_coord):
        t_hat = tangent / (np.linalg.norm(tangent) or 1.0)
        wp = w_lin - t_hat * float(w_lin @ t_hat)
        nrm = np.linalg.norm(wp)
        w_hat = wp / nrm if nrm > 1e-6 else _perp(t_hat)
        n_hat = np.cross(t_hat, w_hat)

        i0 = int(np.floor(res_coord))
        i0 = min(max(i0, 0), n - 1)
        i1 = min(i0 + 1, n - 1)
        frac = res_coord - i0
        hw = (1 - frac) * hw_res[i0] + frac * hw_res[i1]
        ht = (1 - frac) * ht_res[i0] + frac * ht_res[i1]
        # Arrowhead: overrides the width inside a strand's C-terminal tail.  The
        # arrow is clamped to leave at least one residue of shaft, so a strand is
        # never all arrow (short strands are already demoted to coil upstream).
        if run_label[i0] == "b":
            run_len = run_end[i0] - run_start[i0] + 1
            eff_arrow = min(arrow_len, max(1.0, run_len - 1.0))
            dist = run_end[i0] - res_coord
            if dist <= eff_arrow:
                hw = _arrow_halfwidth(dist, dims["strand_hw"], dims["tip_half"],
                                      factor, eff_arrow)
                ht = dims["strand_ht"]

        centers.append(pos)
        w_hats.append(w_hat)
        n_hats.append(n_hat)
        sections.append(_section(hw, ht, _SECTION_VERTS))

    for i in range(n - 1):
        broken = np.linalg.norm(ca_ang[i + 1] - ca_ang[i]) > _CHAIN_BREAK_ANG
        p0 = ctrl[i - 1] if i - 1 >= 0 else ctrl[i]
        p1, p2 = ctrl[i], ctrl[i + 1]
        p3 = ctrl[i + 2] if i + 2 < n else ctrl[i + 1]
        for j in range(spr):
            t = j / spr
            pos, tan = _catmull_pos_tan(p0, p1, p2, p3, t)
            # Slerp the (already flipped) width vectors so the ribbon twist
            # advances uniformly; the frame does not carry meaningfully across a
            # chain break, so hold it steady there.
            w_lin = width[i] if broken else _slerp(width[i], width[i + 1], t)
            add_sample(pos, tan, w_lin, i + t)
    # Final endpoint sample.
    add_sample(ctrl[-1], ctrl[-1] - ctrl[-2], width[-1], float(n - 1))

    # Drop any zero-length step so no degenerate ring band is emitted.
    keep = [0]
    for i in range(1, len(centers)):
        if np.linalg.norm(centers[i] - centers[keep[-1]]) > 1e-6:
            keep.append(i)
    centers = [centers[i] for i in keep]
    w_hats = [w_hats[i] for i in keep]
    n_hats = [n_hats[i] for i in keep]
    sections = [sections[i] for i in keep]

    if len(centers) < 2:
        from . import _manifold
        return _manifold.to_trimesh(_manifold.sphere(centers[0], dims["coil_r"]))

    return _loft(np.asarray(centers), np.asarray(w_hats), np.asarray(n_hats),
                 sections)
