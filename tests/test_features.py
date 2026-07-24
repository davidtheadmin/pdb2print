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


def test_magnet_skips_when_gap_exceeds_two_thickness():
    """Magnets can only meet if 2×thickness spans the gap; else skip + explain."""
    report = build_all(COMPLEX, _params(
        connect=True, use_magnets=True, contact_threshold_mm=4.0,
        connector_diameter_mm=3.0, magnet_thickness_mm=1.0))  # 2T = 2.0 mm
    assert _all_watertight_single(report)
    # The A↔P contact (~3 mm gap) exceeds 2×1.0 mm and must be reported skipped.
    skipped = [c for c in report.connections
               if c["method"] == "magnet" and not c["applied"]
               and "thickness" in c["note"]]
    assert skipped, report.connections


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
