"""Tests for the v2 connectors + UX features.

Offline end-to-end coverage using bundled structures:

* ``1bna``        — B-DNA dodecamer (two strands) → interstrand base-pair connect.
* ``mini_complex``— 1bna's DNA plus a small poly-ALA peptide placed in contact
  (chain ``P``) → protein↔DNA whole-object joins.  A synthetic stand-in for the
  1ZAA protein, whose chain could not be retrieved offline (the sandbox web-fetch
  truncates large PDBs, a limitation the project brief already documents).
* ``1zaa``        — DNA strands + protein header (names parsing).

The through-line of every connection test is the definition of done: **each
object stays watertight and a single connected body** after the pass.
"""

from __future__ import annotations

import json
import os

import pytest

from pdb2print import names, io, chains as chains_mod
from pdb2print.config import (
    PrintParams, ConnectionParams, NoMagnetMethod, MagnetShape,
    BaseStyle, BackboneStyle,
)
from pdb2print.pipeline import build_all

HERE = os.path.dirname(__file__)
D = os.path.join(HERE, "data")
BNA = os.path.join(D, "1bna.pdb")
ZAA = os.path.join(D, "1zaa.pdb")
COMPLEX = os.path.join(D, "mini_complex.pdb")


def _params(**conn):
    p = PrintParams(scale_mm_per_angstrom=0.6, grid_spacing_mm=0.9, min_wall_mm=1.0,
                    base_style=BaseStyle.ROD, backbone_style=BackboneStyle.TUBE)
    for k, v in conn.items():
        setattr(p.connections, k, v)
    return p


def _all_watertight_single(report):
    return all(m.is_watertight and m.body_count == 1 for _c, m in report.built)


# --------------------------------------------------------------------------
# Feature A — subunit names
# --------------------------------------------------------------------------
def test_pdb_compnd_names():
    assert names.chain_names(BNA) == {"A": "DNA", "B": "DNA"}


def test_pdb_compnd_multi_molecule_block():
    # 1ZAA: two DNA strands (A,B) and a protein (C) — distinct molecule blocks.
    got = names.chain_names(ZAA)
    assert got["A"] == "DNA" and got["B"] == "DNA"
    assert "ZIF268" in got["C"]


def test_prettify_rules():
    from pdb2print.names import _prettify
    assert _prettify("HEMOGLOBIN (ALPHA CHAIN)") == "Hemoglobin (Alpha Chain)"
    assert _prettify("DNA (5'-D(*CP*GP*CP*G)-3')") == "DNA"
    assert _prettify("DNA-BINDING DOMAIN") == "DNA-Binding Domain"
    assert _prettify("PROTEIN (ZIF268)") == "Protein (ZIF268)"
    assert _prettify(None) is None
    assert _prettify("   ") is None


def test_noncontiguous_chain_not_duplicated():
    """A chain interrupted in the file (protein + a later same-id HETATM) must
    build exactly once — the 1TUP zinc-bound p53 doubling bug.

    ``biotite.get_chains`` reports a chain id once per contiguous block, so a
    protein chain followed later by its bound metal under the same id was meshed
    and exported twice.  ``split_chains`` now de-duplicates ids.
    """
    import numpy as np
    from pdb2print.config import MoleculeType
    atoms, _ = io.load_with_names(os.path.join(D, "1ubq.pdb"))
    other = atoms.copy()
    other.coord = other.coord + np.array([30.0, 0.0, 0.0])
    other.chain_id = np.array(["B"] * other.array_length())
    zn = atoms[:1].copy()
    zn.chain_id = np.array(["A"]); zn.res_name = np.array(["ZN"])
    zn.atom_name = np.array(["ZN"]); zn.hetero = np.array([True])
    zn.element = np.array(["ZN"]); zn.res_id = np.array([200])
    zn.coord = np.array([[10.0, 10.0, 10.0]])
    combined = atoms + other + zn          # chain-id order A … B … A
    got = chains_mod.split_chains(combined)
    assert [c.chain_id for c in got] == ["A", "B"]
    assert all(c.mtype == MoleculeType.PROTEIN for c in got)


def test_names_plumbed_to_chain_and_fallback():
    report = build_all(BNA, _params())
    for chain, _mesh in report.built:
        assert chain.name == "DNA"
        assert chain.display_name() == "DNA"
    # Fallback when a chain has no header name.
    ch = chains_mod.Chain(chain_id="Z", atoms=report.built[0][0].atoms,
                          mtype=report.built[0][0].mtype, name=None)
    assert ch.display_name() == "Chain Z"


# --------------------------------------------------------------------------
# Feature D — connections keep every object watertight
# --------------------------------------------------------------------------
def test_baseline_no_connections_watertight():
    report = build_all(BNA, _params())
    assert _all_watertight_single(report)
    assert report.connections == []


def test_basepair_pairs_all_correctly():
    """A clean dodecamer must pair every base in the true antiparallel register."""
    report = build_all(BNA, _params(basepair_connect=True))
    assert _all_watertight_single(report)
    (con,) = report.connections
    assert con["kind"] == "dna-basepair" and con["applied"]
    assert con["count"] >= 11  # 12 pairs; at most one end rung may self-skip


def test_basepair_register_is_correct_and_cutoff_drops_bubbles():
    """The pairing itself: correct antiparallel register + bubble exclusion."""
    import numpy as np
    from pdb2print.representations import tube_slab
    from pdb2print import connections as cx
    p = _params()
    atoms, _ = io.load_with_names(BNA)
    cs = chains_mod.split_chains(atoms)
    ca = tube_slab.base_centroids_mm(cs[0], p)
    cb = tube_slab.base_centroids_mm(cs[1], p)
    max_d = p.connections.basepair_max_dist_ang * p.scale_mm_per_angstrom
    pairs = cx._pair_bases(ca, cb, max_d)
    n = len(ca)
    assert len(pairs) == n
    assert all(j == n - 1 - i for i, j, _ in pairs)      # true WC register
    # Push a couple of bases far apart (a bubble) — they must be left unpaired.
    cb2 = cb.copy(); cb2[5:7] += np.array([40.0, 0.0, 0.0])
    pairs2 = cx._pair_bases(ca, cb2, max_d)
    used = {j for _, j, _ in pairs2} | {i for i, _, _ in pairs2}
    assert 5 not in used and 6 not in used
    assert len(pairs2) == n - 2


def test_basepair_grows_volume():
    base = build_all(BNA, _params())
    linked = build_all(BNA, _params(basepair_connect=True))
    for (_ca, ma), (_cb, mb) in zip(base.built, linked.built):
        assert mb.volume > ma.volume


@pytest.mark.parametrize("style", [BaseStyle.ROD, BaseStyle.MOLECULE])
def test_basepair_watertight_per_style(style):
    p = _params(basepair_connect=True)
    p.base_style = style
    report = build_all(BNA, p)
    assert _all_watertight_single(report)


def test_inflate_join_watertight():
    report = build_all(COMPLEX, _params(
        connect=True, no_magnet_method=NoMagnetMethod.INFLATE,
        contact_threshold_mm=3.5))
    assert _all_watertight_single(report)
    kinds = {c["kind"] for c in report.connections if c["applied"]}
    assert "dna-protein" in kinds


def test_bridge_join_watertight():
    report = build_all(COMPLEX, _params(
        connect=True, no_magnet_method=NoMagnetMethod.BRIDGE,
        contact_threshold_mm=3.5, connector_diameter_mm=3.0))
    assert _all_watertight_single(report)
    assert any(c["method"] == "bridge" and c["applied"] for c in report.connections)


@pytest.mark.parametrize("shape", [MagnetShape.ROUND, MagnetShape.SQUARE])
def test_magnet_join_watertight(shape):
    report = build_all(COMPLEX, _params(
        connect=True, use_magnets=True, magnet_shape=shape,
        contact_threshold_mm=3.5, connector_diameter_mm=3.0,
        magnet_thickness_mm=1.5))
    assert _all_watertight_single(report)
    assert any(c["method"] == "magnet" for c in report.connections)


def test_inflate_grows_and_closes_gap():
    """Inflate must grow the contacting objects and bring them into contact."""
    from pdb2print import connections as cx
    base = build_all(COMPLEX, _params())
    inflated = build_all(COMPLEX, _params(
        connect=True, no_magnet_method=NoMagnetMethod.INFLATE,
        contact_threshold_mm=3.5))
    bvol = {c.chain_id: m.volume for c, m in base.built}
    for c, m in inflated.built:
        assert m.volume > bvol[c.chain_id]          # every contacting object grew
    # Gaps shrink, and a near-contact interface welds (overlaps) after inflating.
    import itertools
    before, after = [], []
    for i, j in itertools.combinations(range(len(base.built)), 2):
        before.append(cx._nearest(base.built[i][1], base.built[j][1])[2])
        after.append(cx._nearest(inflated.built[i][1], inflated.built[j][1])[2])
    assert all(a <= b + 1e-6 for a, b in zip(after, before))   # every gap shrank
    assert min(after) < 0.5                                     # closest pair welds


def test_magnet_skips_when_gap_exceeds_two_thickness_without_socket():
    """Bare magnets can only meet if 2×thickness spans the gap; else skip + explain.

    Only applies with the socket off — a socket collar closes the gap itself, so
    the thickness-vs-gap test is not a limit there (see the next test).
    """
    report = build_all(COMPLEX, _params(
        connect=True, use_magnets=True, socket=False, contact_threshold_mm=4.0,
        connector_diameter_mm=3.0, magnet_thickness_mm=1.0))  # 2T = 2.0 mm
    assert _all_watertight_single(report)
    # The A↔P contact (~3 mm gap) exceeds 2×1.0 mm and must be reported skipped.
    skipped = [c for c in report.connections
               if c["method"] == "magnet" and not c["applied"]
               and "thickness" in c["note"]]
    assert skipped, report.connections


def test_socket_bridges_a_gap_that_bare_magnets_cannot():
    """The collar spans the gap, so the same joint that was skipped now builds."""
    kw = dict(connect=True, use_magnets=True, contact_threshold_mm=4.0,
              connector_diameter_mm=3.0, magnet_thickness_mm=1.0)
    bare = build_all(COMPLEX, _params(socket=False, **kw))
    socketed = build_all(COMPLEX, _params(socket=True, **kw))
    assert _all_watertight_single(socketed)
    placed = lambda r: sum(c["applied"] for c in r.connections if c["method"] == "magnet")
    assert placed(socketed) > placed(bare)


def test_socket_adds_material_and_pocket_removes_it():
    """A socketed magnet joint nets out as a collar (added) minus a bore (cut)."""
    base = build_all(COMPLEX, _params())
    bvol = {c.chain_id: m.volume for c, m in base.built}
    joined = build_all(COMPLEX, _params(
        connect=True, use_magnets=True, socket=True, contact_threshold_mm=3.5,
        connector_diameter_mm=3.0, magnet_thickness_mm=1.5))
    assert _all_watertight_single(joined)
    # At least one object must have changed volume — the joint is real geometry.
    assert any(abs(m.volume - bvol[c.chain_id]) > 1e-6 for c, m in joined.built)


def test_magnet_bore_is_oversize_for_press_fit():
    """The cut bore must be wider and deeper than the nominal magnet, not equal.

    An FDM hole printed to a magnet's exact size comes out undersize and will not
    accept it, so a nominal-sized pocket is a bug, not a tolerance choice.
    """
    from pdb2print import connections as cx
    cp = ConnectionParams(connector_diameter_mm=4.0, magnet_thickness_mm=2.0)
    assert cp.magnet_fit_clearance_mm > 0
    assert cp.magnet_depth_clearance_mm > 0
    bore_r = cp.connector_diameter_mm / 2 + cp.magnet_fit_clearance_mm / 2
    assert bore_r > cp.connector_diameter_mm / 2
    # And the chamfer never eats more than a third of a thin magnet's grip.
    assert min(cp.magnet_chamfer_mm, 0.3 * 0.5) <= 0.15
    assert cx._MIN_SEAT_FILL > 0


def test_seat_faces_are_coplanar_on_the_mid_plane():
    """Both halves of a joint must end on exactly the same plane, or it isn't flush."""
    import numpy as np
    from pdb2print import connections as cx
    center = np.array([1.0, 2.0, 3.0])
    axis = cx._unit(np.array([0.3, -0.5, 0.8]))
    a = cx._seat_solid(center, -axis, 4.0, 2.0)
    b = cx._seat_solid(center, +axis, 4.0, 2.0)
    # Project every vertex onto the axis; the two solids must meet at 0 and
    # extend to opposite sides only.
    for man, sign in ((a, -1.0), (b, +1.0)):
        v = np.asarray(cx._manifold.to_trimesh(man).vertices, float)
        t = (v - center) @ (axis * sign)
        assert abs(t.min()) < 1e-5        # the flat face sits on the mid-plane
        assert abs(t.max() - 4.0) < 1e-5  # and the solid extends one way only


def test_mass_axis_beats_the_contact_line_on_a_tilted_interface():
    """The centroid line must point along the material, not along a surface spike.

    Two offset boxes overlapping in x: the true interface normal is +x, but the
    nearest surface points are diagonal.  The mass-centroid axis should recover
    something much closer to +x than the raw contact direction does.
    """
    import numpy as np
    import trimesh
    from pdb2print import connections as cx

    a = trimesh.creation.box(extents=(10, 10, 10))
    b = trimesh.creation.box(extents=(10, 10, 10))
    b.apply_translation([10.5, 4.0, 0.0])          # offset along y as well as x
    ma = cx._manifold.from_trimesh(a)
    mb = cx._manifold.from_trimesh(b)

    center = np.array([5.25, 2.0, 0.0])
    _va, ca = cx._local_mass(ma, center, 6.0)
    _vb, cb = cx._local_mass(mb, center, 6.0)
    assert ca is not None and cb is not None
    mass_axis = cx._unit(cb - ca)
    assert mass_axis[0] > 0.7, mass_axis        # dominated by +x, the real normal


def _rod_and_slab():
    """A DNA-like rod above a protein-like slab, clipped asymmetrically.

    The rod is sampled over x in [-2, 9] while the joint sits at x = 0, so its
    centre of mass is dragged well along +x — the exact effect that swings the
    magnet axis toward the helix when the DNA tube is thick.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 4000)
    rod = np.stack([rng.uniform(-2.0, 9.0, 4000),
                    4.0 + 2.0 * np.cos(theta), 2.0 * np.sin(theta)], 1)
    slab = np.stack([rng.uniform(-10, 10, 4000), np.zeros(4000),
                     rng.uniform(-10, 10, 4000)], 1)
    strip = np.stack([rng.uniform(-8, 8, 200), rng.uniform(-0.6, 0.6, 200),
                      np.zeros(200)], 1)
    return rod, slab, strip


def _axis_error_deg(axis):
    import numpy as np
    return float(np.degrees(np.arccos(min(1.0, abs(axis[1])))))


def test_patch_pca_tells_a_strip_from_a_disc():
    import numpy as np
    from pdb2print import connections as cx
    _rod, _slab, strip = _rod_and_slab()
    direction, elongation = cx._patch_long_axis(strip)
    assert elongation > 5.0
    assert abs(direction[0]) > 0.99            # the strip runs along x
    rng = np.random.default_rng(1)
    disc = np.column_stack([rng.uniform(-4, 4, 200), rng.uniform(-4, 4, 200),
                            np.zeros(200)])
    assert cx._patch_long_axis(disc)[1] < 2.0  # round patch: no long direction


def test_rod_shaped_blob_does_not_swing_the_axis_90_degrees():
    """The DNA case: a thick backbone tube must not drag the magnet off-normal.

    The raw centroid line is ~40° off here because the probe ball clips an
    asymmetric length of rod.  Whatever axis is chosen must still be close to the
    true interface normal (+y).
    """
    import numpy as np
    from pdb2print.config import ConnectionParams
    from pdb2print import connections as cx

    rod, slab, strip = _rod_and_slab()
    center = np.array([0.0, 1.0, 0.0])
    cen_a, cen_b = slab.mean(0), rod.mean(0)
    raw = cx._unit(cen_b - cen_a)
    assert _axis_error_deg(raw) > 30           # the failure this test guards

    seat = cx.Seat(center=center, axis=cx._unit(np.array([0.0, 1.0, 0.0])),
                   gap=2.0, footprint=20, patch=strip)
    axis, _label, _agree, _blocked = cx._choose_axis(
        seat, cen_a, cen_b, slab, rod, ConnectionParams(), 3.0, 5.0)
    assert _axis_error_deg(axis) < 5.0, _axis_error_deg(axis)


def test_strip_projection_rescues_a_noisy_contact_line():
    """When the contact line is itself tilted, the de-elongated mass axis wins."""
    import numpy as np
    from pdb2print.config import ConnectionParams
    from pdb2print import connections as cx

    rod, slab, strip = _rod_and_slab()
    seat = cx.Seat(center=np.array([0.0, 1.0, 0.0]),
                   axis=cx._unit(np.array([0.34, 0.94, 0.0])),   # 20° off
                   gap=2.0, footprint=20, patch=strip)
    axis, label, _agree, _blocked = cx._choose_axis(
        seat, slab.mean(0), rod.mean(0), slab, rod, ConnectionParams(), 3.0, 5.0)
    assert label == "mass-flat", label
    assert _axis_error_deg(axis) < 5.0


def test_path_census_counts_material_in_the_way():
    """An axis driven along the rod must register as heavily blocked."""
    import numpy as np
    from pdb2print import connections as cx
    rod, slab, _strip = _rod_and_slab()
    center = np.array([0.0, 1.0, 0.0])
    across, _ = cx._path_census(slab, rod, center, np.array([0.0, 1.0, 0.0]),
                                3.0, 5.0)
    along, _ = cx._path_census(slab, rod, center, np.array([1.0, 0.0, 0.0]),
                               3.0, 5.0)
    assert across == 0                 # nothing in the way across the interface
    assert along > 50, along           # driving along the rod hits the slab


def test_wrapped_interface_falls_back_to_the_contact_line():
    """A mass axis that disagrees badly with the contact line must be rejected.

    This is the protein-wrapped-around-DNA case: the probe ball reaches around
    the far side and the centroid line flips.  The guard keeps the contact line.
    """
    import numpy as np
    from pdb2print.config import ConnectionParams
    from pdb2print import connections as cx

    cp = ConnectionParams()
    contact = cx._unit(np.array([1.0, 0.0, 0.0]))
    for mass in (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        agreement = float(np.dot(cx._unit(mass), contact))
        assert agreement < cp.axis_agreement_min      # would be rejected


def test_bridge_joint_is_a_flat_faced_cylinder_not_a_capsule():
    """The bridge must reuse the seat path, so it has a flat mating face."""
    report = build_all(COMPLEX, _params(
        connect=True, no_magnet_method=NoMagnetMethod.BRIDGE,
        contact_threshold_mm=3.5, connector_diameter_mm=3.0))
    assert _all_watertight_single(report)
    bridges = [c for c in report.connections if c["method"] == "bridge"]
    assert bridges and any(c["applied"] for c in bridges)


def test_overhang_in_the_joint_path_is_cut_away():
    """Neither part may keep material on the other's side of the mating face.

    A lobe hanging over the socket makes the joint look right in the preview and
    then refuse to close, so the approach path is cleared on both sides.
    """
    import numpy as np
    report = build_all(COMPLEX, _params(
        connect=True, use_magnets=True, socket=True, contact_threshold_mm=3.5,
        connector_diameter_mm=3.0, magnet_thickness_mm=1.5))
    assert _all_watertight_single(report)
    by_id = {c.chain_id: m for c, m in report.built}
    ids = [c.chain_id for c, _m in report.built]
    for mark in report.connection_markers:
        center = np.asarray(mark["center"], float)
        axis = np.asarray(mark["axis"], float)
        radius = 0.5 * (mark.get("socket_diameter") or mark["diameter"])
        for cid in ids:
            v = np.asarray(by_id[cid].vertices, float)
            rel = v - center
            t = rel @ axis
            radial = np.linalg.norm(rel - np.outer(t, axis), axis=1)
            near = radial <= radius - 0.15          # inside the footprint
            if not near.any():
                continue
            # A part may live on one side of the face or the other, but no part
            # may straddle it inside the footprint.
            assert not (np.any(t[near] > 0.15) and np.any(t[near] < -0.15)), (
                f"chain {cid} still straddles the mating face at {center}")


def test_two_magnets_land_on_different_patches():
    """Asking for 2 magnets must not stack them on top of each other."""
    import numpy as np
    report = build_all(COMPLEX, _params(
        connect=True, use_magnets=True, contact_threshold_mm=3.5,
        connector_diameter_mm=2.5, magnet_thickness_mm=1.5,
        magnet_count=2, dna_magnet_count=2))
    assert _all_watertight_single(report)
    marks = report.connection_markers
    if len(marks) >= 2:
        centers = np.array([m["center"] for m in marks])
        d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        off_diag = d[~np.eye(len(d), dtype=bool)]
        assert off_diag.min() > 2.5      # at least a socket diameter apart


def test_connect_plus_basepair_combined_watertight():
    report = build_all(COMPLEX, _params(
        connect=True, contact_threshold_mm=3.5, basepair_connect=True))
    assert _all_watertight_single(report)
    kinds = {c["kind"] for c in report.connections}
    assert "dna-protein" in kinds and "dna-basepair" in kinds


def test_connection_off_leaves_meshes_untouched():
    off = build_all(COMPLEX, _params())          # connect off, basepair off
    assert off.connections == []


# --------------------------------------------------------------------------
# Server: SSE progress stream + detect endpoint
# --------------------------------------------------------------------------
def _sse_events(text):
    events = []
    for block in text.split("\n\n"):
        ev, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if data:
            events.append((ev, json.loads(data)))
    return events


def _client():
    from fastapi.testclient import TestClient
    import server
    return TestClient(server.app)


def test_sse_stream_progress_then_result():
    c = _client()
    with open(BNA, "rb") as fh:
        r = c.post("/api/generate",
                   files={"file": ("1bna.pdb", fh, "chemical/x-pdb")},
                   data={"scale": "0.6", "grid_spacing": "0.9", "min_wall": "1.0",
                         "base_style": "rod", "backbone_style": "tube",
                         "basepair_connect": "true"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    events = _sse_events(r.text)
    progress = [d for e, d in events if e == "progress"]
    results = [d for e, d in events if e == "result"]
    assert len(progress) >= 2
    assert progress[0]["frac"] < progress[-1]["frac"]
    assert len(results) == 1
    res = results[0]
    assert res["ok"] and res["glb_url"] and res["stl_url"]
    assert [ch["name"] for ch in res["chains"]] == ["DNA", "DNA"]
    assert any(cn["kind"] == "dna-basepair" for cn in res["connections"])


def test_generate_with_connections_via_server():
    c = _client()
    with open(COMPLEX, "rb") as fh:
        r = c.post("/api/generate",
                   files={"file": ("mini_complex.pdb", fh, "chemical/x-pdb")},
                   data={"scale": "0.6", "grid_spacing": "0.9", "min_wall": "1.0",
                         "base_style": "rod", "backbone_style": "tube",
                         "connect": "true", "use_magnets": "false",
                         "no_magnet_method": "inflate", "basepair_connect": "true"})
    res = [d for e, d in _sse_events(r.text) if e == "result"][0]
    assert res["ok"]
    methods = {cn["method"] for cn in res["connections"]}
    assert "inflate" in methods and "basepair" in methods


def test_generate_validation_error_is_json_not_stream():
    c = _client()
    r = c.post("/api/generate", data={"scale": "0.6"})  # no source
    assert r.status_code == 400
    assert "application/json" in r.headers.get("content-type", "")
    assert r.json()["ok"] is False
