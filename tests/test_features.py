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
    # Every pair that started *apart* must close.  A pair already touching is
    # exempt: it has no gap left to shrink, and `_nearest` measures against a
    # random vertex subsample (`_probe_points`), so re-meshing reshuffles which
    # vertices are compared and moves the reading by a few microns either way.
    # Asserting a strict decrease on an already-welded pair is therefore a test
    # of the sampling noise, not of the inflate pass.
    assert all(a <= b + 1e-6 or b < 0.5 for a, b in zip(after, before))
    assert min(after) < 0.5                                     # closest pair welds


# --------------------------------------------------------------------------
# Inflate must not deform a nucleic base
# --------------------------------------------------------------------------
#: Base-shape parameters.  Inflate grew all of these, which is not a size
#: increase but a distortion: the slab's *footprint* is fixed at 4.5 x 3.0 Å and
#: only its thickness is a parameter, so growing it fattened the plate
#: through-plane without moving its edges any closer to the neighbour; and in
#: the molecule style the ring atoms sit ~1.4 Å apart, so a grown atom radius
#: swallowed the hole in the middle of the ring.
_BASE_SHAPE_PARAMS = (
    "slab_thickness_mm", "connector_radius_mm", "atom_radius_mm",
    "bond_radius_mm", "slab_scale",
)


def _nucleic_chain(path=BNA):
    from pdb2print import io, chains as chains_mod
    from pdb2print.config import MoleculeType
    atoms = io.load_any(path)
    return [c for c in chains_mod.split_chains(atoms)
            if c.mtype == MoleculeType.NUCLEIC][0]


def test_inflate_leaves_base_shape_parameters_alone():
    """A nucleic chain inflates its *backbone*; every base dimension is frozen."""
    from pdb2print.connections import _inflate_growth
    p = PrintParams(scale_mm_per_angstrom=0.6, min_wall_mm=1.0,
                    base_style=BaseStyle.ROD, backbone_style=BackboneStyle.TUBE)
    grown = _inflate_growth(_nucleic_chain(), p, 1.0)
    for name in _BASE_SHAPE_PARAMS:
        assert getattr(grown, name) == getattr(p, name), name
    # ...and the backbone did move, so the join can still weld.
    assert grown.nucleic_radius_mm > p.nucleic_radius_mm
    assert grown.backbone_atom_radius_mm > p.backbone_atom_radius_mm
    assert grown.backbone_bond_radius_mm > p.backbone_bond_radius_mm


@pytest.mark.parametrize("style", [BaseStyle.SLAB, BaseStyle.ROD, BaseStyle.MOLECULE])
def test_inflate_does_not_fatten_the_bases(style):
    """Measured, not asserted by construction: how much solid sits at a base.

    A ball at the base-ring centre is intersected with the built mesh.  Growing
    the base parameters filled it in — the rod rung went from 9% solid to 74%,
    and the molecule style's ring hole closed completely — which is what "weird
    stuff to the DNA" looked like.  Inflating the backbone alone leaves the
    reading essentially where it started.
    """
    import numpy as np
    from pdb2print import geometry, meshops
    from pdb2print.representations import tube_slab, _manifold
    from pdb2print.connections import _inflate_growth

    chain = _nucleic_chain()
    p = PrintParams(scale_mm_per_angstrom=0.6, grid_spacing_mm=0.9, min_wall_mm=1.0,
                    base_style=style, backbone_style=BackboneStyle.TUBE)
    radius = 2.0
    centres = tube_slab.base_link_frames_mm(chain, p)[0][2:10]
    ball = 4.0 / 3.0 * np.pi * radius ** 3

    def fill(params):
        mesh = geometry.generate_chain_mesh(chain, params)
        mesh = meshops.repair(meshops.enforce_min_wall(mesh, params))
        man = _manifold.from_trimesh(mesh)
        return np.mean([
            _manifold.to_trimesh(man ^ _manifold.sphere(c, radius)).volume / ball
            for c in centres
        ])

    before = fill(p)
    after = fill(_inflate_growth(chain, p, 1.0))
    # A fatter backbone tube reaches a little way into the probe ball, so allow
    # a small rise; the old behaviour was several times the original reading.
    assert after < before + 0.03, f"bases fattened: {before:.3f} -> {after:.3f}"


def test_inflate_puts_the_growth_on_the_protein():
    """A DNA↔protein contact is closed by growing the protein, not the DNA."""
    from pdb2print import connections as cx
    from pdb2print.config import MoleculeType

    class _Stub:
        def __init__(self, mtype):
            self.mtype = mtype

    p = PrintParams()
    chains = [_Stub(MoleculeType.NUCLEIC), _Stub(MoleculeType.PROTEIN)]
    # An ordinary close contact: the protein carries all of it.
    shares = dict(cx._inflate_shares(0, 1, 0.4, chains, p))
    assert set(shares) == {1}
    assert shares[1] == pytest.approx(0.4 + cx._INFLATE_WELD_MM)
    # A gap wider than one chain may grow spills onto the DNA backbone, so the
    # join still reaches exactly as far as it did before.
    wide = dict(cx._inflate_shares(0, 1, 3.0, chains, p))
    assert wide[1] == cx._INFLATE_MAX_MM
    assert 0 < wide[0] <= cx._INFLATE_MAX_MM
    # Protein↔protein still swells symmetrically from both sides.
    both = dict(cx._inflate_shares(
        0, 1, 0.6, [_Stub(MoleculeType.PROTEIN)] * 2, p))
    assert both[0] == both[1] == pytest.approx((0.6 + cx._INFLATE_WELD_MM) / 2)


def test_inflate_growth_is_the_same_on_every_nucleic_chain():
    """The backbone is one gauge: no strand may come out fatter than another."""
    from pdb2print import connections as cx
    from pdb2print.config import MoleculeType

    class _Stub:
        def __init__(self, mtype):
            self.mtype = mtype

    chains = [_Stub(MoleculeType.NUCLEIC), _Stub(MoleculeType.NUCLEIC),
              _Stub(MoleculeType.PROTEIN), _Stub(MoleculeType.PROTEIN)]
    grow = cx._unify_nucleic_growth([1.0, 0.0, 0.7, 0.2], chains)
    assert grow[0] == grow[1] == 1.0        # both strands levelled up
    assert grow[2:] == [0.7, 0.2]           # proteins stay per-chain
    # Nothing to level when no strand grew at all.
    assert cx._unify_nucleic_growth([0.0, 0.0, 0.5, 0.0], chains)[:2] == [0.0, 0.0]


def test_inflate_grows_both_strands_of_a_duplex_equally():
    """Regression: with base-pairing on, one strand inflated and the other did not.

    ``basepair_connect`` suppresses the strand↔strand contact, so each strand's
    growth came only from its own protein contacts.  In ``mini_complex`` those
    gaps are 3.44 mm and 0.06 mm, so chain A grew by the full cap while chain B
    barely moved — the duplex printed with two different backbone gauges.
    """
    base = build_all(COMPLEX, _params())
    inflated = build_all(COMPLEX, _params(
        connect=True, no_magnet_method=NoMagnetMethod.INFLATE,
        contact_threshold_mm=3.5, basepair_connect=True))
    before = {c.chain_id: m.volume for c, m in base.built}
    ratio = {c.chain_id: m.volume / before[c.chain_id]
             for c, m in inflated.built if c.mtype.value == "nucleic"}
    assert len(ratio) == 2
    a, b = ratio["A"], ratio["B"]
    assert a > 1.05, "the duplex did not inflate at all"
    # Was 3.07 vs 1.09.  The strands differ slightly in length, so compare the
    # growth *ratio* rather than demanding identical volumes.
    assert abs(a - b) / max(a, b) < 0.02, f"strands inflated unequally: {a} vs {b}"


def test_inflate_reports_a_cartoon_only_joint_as_skipped():
    """Two cartoon ribbons cannot be offset, so the joint is reported, not faked."""
    from pdb2print import connections as cx
    from pdb2print.config import MoleculeType, Representation

    class _Stub:
        mtype = MoleculeType.PROTEIN

    p = PrintParams(protein_representation=Representation.CARTOON)
    assert cx._inflate_shares(0, 1, 0.5, [_Stub(), _Stub()], p) == []


def _facing_blocks(gap_mm: float, size: float = 14.0):
    """Two solid blocks facing each other across a gap of exactly ``gap_mm``.

    The gap rule below is about a *gap*, so the fixture has to contain one, and
    a structure fixture does not reliably provide it: the probe-volume search
    deliberately seats a joint where the two surfaces come closest, so on
    mini_complex it lands on a contact with essentially no gap at all and the
    rule under test never fires. Carve clearance does not manufacture one
    either -- the seat gap is measured between the two surface crossings along
    the joint axis, not set by the clearance. Two blocks a known distance apart
    is the honest way to build the case, and it exercises the same code path:
    ``_apply_magnet`` on a real pair of manifolds.

    Subdivided rather than left as eight corners, because the search reads the
    mesh vertices as its surface cloud and a bare box has none anywhere near
    the joint axis.
    """
    import numpy as np
    import trimesh
    from pdb2print.representations import _manifold

    meshes, mans = [], []
    for sign in (-1.0, 1.0):
        m = trimesh.creation.box(extents=(size, size, size))
        m.apply_translation([sign * (size / 2.0 + gap_mm / 2.0), 0.0, 0.0])
        m = m.subdivide_to_size(1.0)
        m.merge_vertices()
        meshes.append(m)
        mans.append(_manifold.from_trimesh(m))
    return mans, meshes


def _blocks_join(gap_mm: float, socket: bool):
    """Try one magnet joint across ``gap_mm``; returns ``(ok, note, markers, parts)``."""
    from pdb2print import connections as cx
    from pdb2print.representations import _manifold

    mans, meshes = _facing_blocks(gap_mm)
    cp = ConnectionParams(socket=socket, use_magnets=True,
                          connector_diameter_mm=3.0,
                          magnet_thickness_mm=0.5,      # 2T = 1.0 mm
                          contact_threshold_mm=4.0)
    markers: list = []
    # The third value is how many magnets really went in — the number the
    # joints list shows, which used to exist only inside ``note``.
    ok, note, placed = cx._apply_magnet(mans, 0, 1, meshes[0], meshes[1], 1, cp,
                                        PrintParams(), markers)
    assert placed == len(markers)
    return ok, note, markers, [_manifold.to_trimesh(m) for m in mans]


def test_magnet_skips_when_gap_exceeds_two_thickness_without_socket():
    """Bare magnets can only meet if 2×thickness spans the gap; else skip + explain.

    Only applies with the socket off — a socket collar closes the gap itself, so
    the thickness-vs-gap test is not a limit there (see the next test).
    """
    ok, note, markers, parts = _blocks_join(1.5, socket=False)
    assert not ok, note
    assert "thickness" in note, note
    assert markers == []
    # Refusing must leave both parts exactly as they were.
    assert all(p.is_watertight and p.body_count == 1 for p in parts)


def test_socket_bridges_a_gap_that_bare_magnets_cannot():
    """The collar spans the gap, so the same joint that was skipped now builds."""
    ok, note, markers, parts = _blocks_join(1.5, socket=True)
    assert ok, note
    assert len(markers) == 1
    assert all(p.is_watertight and p.body_count == 1 for p in parts)


def test_socket_off_places_the_magnet_in_the_same_spot_as_socket_on():
    """Turning the collar off must not move the magnet.

    The socket radius is not only the collar's: it sets the probe ball's floor,
    the acceptance gate, the spacing between candidates and the approach cut.
    While it collapsed to the bare pocket with the socket off, toggling the
    control silently re-placed every joint.
    """
    import numpy as np
    kw = dict(connect=True, use_magnets=True, contact_threshold_mm=3.5,
              connector_diameter_mm=3.0, magnet_thickness_mm=1.5)
    on = build_all(COMPLEX, _params(socket=True, **kw))
    off = build_all(COMPLEX, _params(socket=False, **kw))
    assert on.connection_markers and off.connection_markers
    assert len(on.connection_markers) == len(off.connection_markers)
    for a, b in zip(on.connection_markers, off.connection_markers):
        assert (a["a"], a["b"]) == (b["a"], b["b"])
        assert np.allclose(a["center"], b["center"]), (a["center"], b["center"])
        assert np.allclose(a["axis"], b["axis"])


def _stepped_block_pair(step: float = 2.8, size: float = 14.0, gap: float = 0.4):
    """A block whose face falls away under half the joint, facing a flat one."""
    import trimesh
    from pdb2print.representations import _manifold

    half = size / 2.0
    a = trimesh.creation.box(extents=(size, size, size))
    a.apply_translation([-(half + gap / 2.0), 0.0, 0.0])
    # Cut the y > 0 half of A back by ``step``, so half the joint footprint has
    # solid material right up to the face and half has nothing for ``step`` mm.
    notch = trimesh.creation.box(extents=(2 * step, size, size))
    notch.apply_translation([-gap / 2.0, half, 0.0])
    man_a = _manifold.difference(_manifold.from_trimesh(a),
                                 _manifold.from_trimesh(notch))
    b = trimesh.creation.box(extents=(size, size, size))
    b.apply_translation([half + gap / 2.0, 0.0, 0.0])
    return man_a, _manifold.from_trimesh(b)


def test_the_mating_plane_slides_to_even_out_the_two_sides():
    """One collar buried and one standing proud is worth moving the plane for.

    Midway across the gap is a rule about the *gap*; it says nothing about how
    much of either collar is buried. Where one surface falls away under the
    joint and the other is solid, the two sides end up wildly uneven, and
    ``seat.hidden`` -- which the score ranks on -- is the worse of them.
    """
    import numpy as np
    from pdb2print import connections as cx

    man_a, man_b = _stepped_block_pair()
    socket_r, depth, probe_r = 3.1, 3.2, 4.65
    seat = cx.Seat(center=np.zeros(3), axis=np.array([1.0, 0.0, 0.0]),
                   gap=0.4, footprint=50)
    seat.probe_r = probe_r

    def buried(where):
        ba = cx._local_solid(man_a, where, 6.5)
        bb = cx._local_solid(man_b, where, 6.5)
        return (cx._embedding(ba, where, -seat.axis, socket_r, depth),
                cx._embedding(bb, where, +seat.axis, socket_r, depth))

    before = buried(seat.center)
    assert abs(before[0] - before[1]) > 0.3, before      # the case under test

    cx._balance_burial(seat, man_a, man_b, socket_r, depth, probe_r)

    assert seat.center[0] < -1e-6, "the plane should move into the shy side"
    after = buried(seat.center)
    assert abs(after[0] - after[1]) < abs(before[0] - before[1])
    assert min(after) > min(before) + 0.05, (before, after)
    # It must be reversible: a plane that cannot be cut without severing the
    # part has to be able to go back where the search found it.
    assert cx._unbalance(seat)
    assert np.allclose(seat.center, 0.0)


def test_an_even_joint_is_left_where_the_search_put_it():
    """No gain, no move. A shift that buys nothing still costs a cut."""
    import numpy as np
    from pdb2print import connections as cx

    mans, _meshes = _facing_blocks(0.4)
    seat = cx.Seat(center=np.zeros(3), axis=np.array([1.0, 0.0, 0.0]),
                   gap=0.4, footprint=50)
    seat.probe_r = 4.65
    cx._balance_burial(seat, mans[0], mans[1], 3.1, 3.2, 4.65)
    assert np.allclose(seat.center, 0.0)
    assert seat.home is None


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
    # The local solid is cut once and then reused, so the probe radius is
    # applied here rather than inside.
    ca = cx._blob_centre(cx._local_solid(ma, center, 6.0))
    cb = cx._blob_centre(cx._local_solid(mb, center, 6.0))
    assert ca is not None and cb is not None
    mass_axis = cx._unit(cb - ca)
    assert mass_axis[0] > 0.7, mass_axis        # dominated by +x, the real normal


def test_embedding_measures_how_buried_a_socket_would_be():
    """The anti-'sticking out' measure, against geometry with known answers.

    ``fill`` cannot serve this purpose: it starts at each part's own surface, so
    the stub of collar spanning the half-gap — the piece with nothing behind it —
    is excluded from it by construction.  ``_embedding`` is taken on the collar as
    built, from the mid-plane, so that stub counts against it.
    """
    import numpy as np
    import trimesh
    from pdb2print import connections as cx

    r, depth = 3.6, 3.7
    block = cx._manifold.from_trimesh(trimesh.creation.box(extents=(40, 40, 40)))
    down, up = np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
    blob = lambda c: cx._local_solid(block, np.asarray(c, float), 8.0)

    face = np.array([0.0, 0.0, 20.0])
    assert cx._embedding(blob(face), face, down, r, depth) > 0.99   # buried
    assert cx._embedding(blob(face), face, up, r, depth) < 0.01     # in open air

    # Started 1 mm off the face: that millimetre of collar is unsupported.
    off = np.array([0.0, 0.0, 21.0])
    assert 0.65 < cx._embedding(blob(off), off, down, r, depth) < 0.80

    # Half over the edge of the block: half the collar hangs in the air.
    edge = np.array([20.0, 0.0, 20.0])
    assert 0.40 < cx._embedding(blob(edge), edge, down, r, depth) < 0.60

    # A rod thinner than the socket can only ever bury the area ratio of it.
    rod = cx._manifold.from_trimesh(
        trimesh.creation.cylinder(radius=1.5, height=40))
    here = np.zeros(3)
    thin = cx._embedding(cx._local_solid(rod, here, 8.0), here, down, r, depth)
    assert 0.10 < thin < 0.25, thin

    assert cx._embedding(None, face, down, r, depth) == 0.0   # nothing here


def test_embedding_is_maximised_by_the_true_surface_normal():
    """Why it is a sound *orientation* signal, not just a seat score.

    The axis search scores candidates on counts of obstructing surface points,
    which says nothing about how deeply the collar buries itself — so a tilt that
    leaves half the socket in open air scored level with one that sinks it in.
    Embedding peaks, smoothly and symmetrically, exactly on the real normal.
    """
    import numpy as np
    import trimesh
    from pdb2print import connections as cx

    tilt = np.radians(30.0)
    rot = np.array([[np.cos(tilt), 0, np.sin(tilt)],
                    [0, 1, 0],
                    [-np.sin(tilt), 0, np.cos(tilt)]])
    box = trimesh.creation.box(extents=(40, 40, 40))
    box.apply_transform(np.block([[rot, np.zeros((3, 1))],
                                  [np.zeros((1, 3)), np.ones((1, 1))]]))
    face = rot @ np.array([0.0, 0.0, 20.0])
    inward = -(rot @ np.array([0.0, 0.0, 1.0]))
    blob = cx._local_solid(cx._manifold.from_trimesh(box), face, 8.0)

    scores = {}
    for deg in (-40, -20, -10, 0, 10, 20, 40):
        th = np.radians(deg)
        spin = np.array([[np.cos(th), 0, np.sin(th)],
                         [0, 1, 0],
                         [-np.sin(th), 0, np.cos(th)]])
        scores[deg] = cx._embedding(blob, face, spin @ inward, 3.6, 3.7)

    assert max(scores, key=scores.get) == 0, scores      # peaks on the normal
    assert scores[0] > 0.99
    for deg in (10, 20, 40):                             # and falls off either way
        assert scores[deg] < scores[0]
        assert scores[-deg] < scores[0]


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
    for mark in report.connection_markers:
        center = np.asarray(mark["center"], float)
        axis = np.asarray(mark["axis"], float)
        radius = 0.5 * (mark.get("socket_diameter") or mark["diameter"])
        # The two parts this magnet joins, and only those. A joint clears its
        # own approach path; a third chain that happens to lie on the same line
        # is a real and separate problem -- nothing cuts it, and nothing here
        # pretends otherwise.
        for cid in (mark["a"], mark["b"]):
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


# --------------------------------------------------------------------------
# Feature E — parts that actually fit: interference resolution
# --------------------------------------------------------------------------
#
# Chains are meshed independently, so at a binding interface both solids claim
# the same volume.  The pass that fixes that is also what makes magnets land in
# the right place, so the two are tested together here.

import numpy as np

from pdb2print import interference
from pdb2print.chains import Chain
from pdb2print.config import (
    InterferenceRule, MoleculeType, PROBE_RADIUS_MIN_ANG, resolve_surface_grid,
)
from pdb2print.representations import _manifold

OVERLAP = os.path.join(D, "overlap_complex.pdb")


def _fake(chain_id, mtype):
    """A Chain carrying only what the interference pass looks at."""
    return Chain(chain_id=chain_id, atoms=None, mtype=mtype)


def _two_overlapping_balls(r_a=6.0, r_b=4.0, offset=7.0):
    """Two solids sharing a lens of volume, big one first."""
    return (_manifold.sphere(np.zeros(3), r_a, 48),
            _manifold.sphere(np.array([offset, 0.0, 0.0]), r_b, 48))


def _shared(mans):
    return interference.residual_overlap(mans)


def test_dilate_grows_a_solid_by_the_requested_amount():
    """The translate-union stand-in for a Minkowski sum must not undershoot.

    The clearance is a guarantee, so what matters is the *thinnest* direction,
    not the average one.
    """
    r, grow = 5.0, 0.3
    ball = _manifold.sphere(np.zeros(3), r, 64)
    grown = interference.dilate(ball, grow)
    # Compare effective radii by volume; the union of translates can only add.
    r_grown = (3.0 * abs(_manifold.volume(grown)) / (4.0 * np.pi)) ** (1.0 / 3.0)
    assert r_grown >= r + grow * 0.9
    assert r_grown <= r + grow * 1.35        # and does not run away either


def test_probe_radius_does_not_change_the_size_of_the_part():
    """The misconception this whole feature exists to correct.

    ``EDT(atoms grown by p) − p`` cancels on a convex patch, so turning the
    probe radius down does *not* shrink a chain and cannot stop two chains
    colliding — it only carves into concave pockets.  If this test ever fails,
    the probe radius has become a size knob and the guidance in the UI is wrong.
    """
    from pdb2print.representations import surface
    atoms, _names = io.load_with_names(BNA)
    chain = chains_mod.split_chains(atoms)[0]

    def extent(probe):
        p = PrintParams(scale_mm_per_angstrom=0.6, grid_spacing_mm=0.9,
                        probe_radius_ang=probe)
        m = surface.build(chain, p)
        return m.bounds[1] - m.bounds[0]

    small, large = extent(1.4), extent(2.0)
    # A 43% bigger probe must not move the outer surface by more than a voxel.
    assert np.allclose(small, large, atol=1.0), (small, large)


def test_probe_radius_is_clamped_to_the_water_probe():
    """Below 1.4 Å the surface sheds pieces, so the slider floor is enforced."""
    low = resolve_surface_grid(PrintParams(probe_radius_ang=1.0))
    assert low.probe_ang == pytest.approx(PROBE_RADIUS_MIN_ANG)
    assert low.notes and "1.40" in low.notes[0]
    # The known-good default must pass through completely untouched.
    default = resolve_surface_grid(PrintParams())
    assert default.probe_ang == pytest.approx(1.4)
    assert default.spacing_mm == pytest.approx(0.5)
    assert default.notes == []


def test_auto_rule_carves_the_protein_and_spares_the_nucleic():
    """DNA sits in a groove, so the protein is the one that gives way."""
    dna, protein = _two_overlapping_balls()
    chains = [_fake("A", MoleculeType.NUCLEIC), _fake("P", MoleculeType.PROTEIN)]
    before = [_manifold.volume(dna), _manifold.volume(protein)]
    out, overlaps, notes = interference.resolve(
        [dna, protein], chains, PrintParams(), None)
    after = [_manifold.volume(m) for m in out]

    assert overlaps and notes
    assert after[0] == pytest.approx(before[0], rel=1e-6)   # DNA untouched
    assert after[1] < before[1] * 0.999                     # protein carved
    assert _shared(out) == pytest.approx(0.0, abs=1e-6)


def test_auto_rule_keeps_the_larger_of_two_same_type_chains():
    big, small = _two_overlapping_balls()
    chains = [_fake("A", MoleculeType.PROTEIN), _fake("B", MoleculeType.PROTEIN)]
    before = [_manifold.volume(big), _manifold.volume(small)]
    out, _ov, _n = interference.resolve([big, small], chains, PrintParams(), None)
    after = [_manifold.volume(m) for m in out]
    assert after[0] == pytest.approx(before[0], rel=1e-6)
    assert after[1] < before[1] * 0.999


def test_symmetric_rule_trims_both_parts():
    a, b = _two_overlapping_balls()
    chains = [_fake("A", MoleculeType.NUCLEIC), _fake("P", MoleculeType.PROTEIN)]
    before = [_manifold.volume(a), _manifold.volume(b)]
    out, _ov, _n = interference.resolve(
        [a, b], chains, PrintParams(resolve_interference=InterferenceRule.SYMMETRIC),
        None)
    after = [_manifold.volume(m) for m in out]
    assert after[0] < before[0] * 0.999
    assert after[1] < before[1] * 0.999
    assert _shared(out) == pytest.approx(0.0, abs=1e-6)


def test_none_rule_leaves_the_overlap_alone():
    a, b = _two_overlapping_balls()
    chains = [_fake("A", MoleculeType.NUCLEIC), _fake("P", MoleculeType.PROTEIN)]
    out, overlaps, notes = interference.resolve(
        [a, b], chains, PrintParams(resolve_interference=InterferenceRule.NONE), None)
    assert overlaps == [] and notes == []
    assert _shared(out) > 1.0


def test_carve_leaves_a_real_clearance_not_a_zero_fit():
    """Parts must not merely stop overlapping — they need room to be assembled."""
    dna, protein = _two_overlapping_balls()
    chains = [_fake("A", MoleculeType.NUCLEIC), _fake("P", MoleculeType.PROTEIN)]
    gap = 0.4
    out, _ov, _n = interference.resolve(
        [dna, protein], chains, PrintParams(fit_clearance_mm=gap), None)
    # Growing the untouched part by *less* than the clearance must still not
    # reach the carved one; growing it by more must.
    assert _shared([interference.dilate(out[0], gap * 0.5), out[1]]) \
        == pytest.approx(0.0, abs=1e-6)
    assert _shared([interference.dilate(out[0], gap * 2.0), out[1]]) > 0.0


def test_overlapping_chains_are_made_printable_end_to_end():
    """The whole point: nothing exported may share space with anything else."""
    report = build_all(OVERLAP, _params())
    assert all(m.is_watertight for _c, m in report.built)
    mans = [_manifold.from_trimesh(m) for _c, m in report.built]
    assert _shared(mans) == pytest.approx(0.0, abs=1e-3)
    assert any("Interference at" in w for w in report.warnings)


def test_fit_runs_with_connections_switched_off():
    """Two parts that are merely printed and handed over still have to fit."""
    loose = build_all(OVERLAP, _params())
    stuck = build_all(OVERLAP, PrintParams(
        scale_mm_per_angstrom=0.6, grid_spacing_mm=0.9, min_wall_mm=1.0,
        base_style=BaseStyle.ROD, backbone_style=BackboneStyle.TUBE,
        resolve_interference=InterferenceRule.NONE))
    assert loose.connections == [] and stuck.connections == []
    assert _shared([_manifold.from_trimesh(m) for _c, m in stuck.built]) > 1.0
    assert _shared([_manifold.from_trimesh(m) for _c, m in loose.built]) \
        == pytest.approx(0.0, abs=1e-3)


def test_magnets_seat_on_an_interface_that_was_interpenetrating():
    """The hard case: two parts that overlapped, carved apart, then joined.

    This used to asserted that such joints came from a dedicated
    interference-lobe seat source and carried its axis. That source no longer
    exists -- the probe-volume search finds the same places on their own merits,
    because a former deep overlap is now a broad close contact with material on
    both sides, which is exactly what it scores. So what is pinned here is the
    outcome rather than the mechanism: joints are found, they are real geometry,
    and the parts genuinely come apart afterwards.
    """
    report = build_all(OVERLAP, _params(connect=True, use_magnets=True))
    assert _all_watertight_single(report)
    marks = report.connection_markers
    assert marks, "no magnets placed at all"

    # Every joint sits where there is material on both sides -- the measure the
    # search is built on, and the one that stops a magnet landing on a spike.
    assert all(min(m["probe_a"], m["probe_b"]) > 0.0 for m in marks)

    # And the parts still fit together: carving plus connectors must not have
    # reintroduced interference.
    from pdb2print import interference
    from pdb2print.representations import _manifold
    mans = [_manifold.from_trimesh(m) for _c, m in report.built]
    assert interference.residual_overlap(mans) < 1.0

def test_basepair_rungs_weld_and_survive_the_closing_sweep():
    """Opposing rungs must share a real volume, and still share it at the end.

    Each half runs a short way past the midline on purpose: two round ends that
    merely touch share a single tangent point, which is nothing for the slicer
    to weld. That overlap is between two *objects*, so the closing sweep after
    connecting found it, carved it out, and added a fit clearance on top — the
    deliberate weld came out of the build as a gap of air in the middle of every
    rung, which is exactly "the helix does not hold together".

    Pinned from both sides. The weld has to exist, and it has to be a weld
    rather than the ``2r`` crossing an unclamped capsule used to produce: a
    link aimed at the midline once overshot it by a full radius, invisible on
    screen and a collision in the print.
    """
    report = build_all(BNA, _params(basepair_connect=True))
    assert _all_watertight_single(report)
    assert report.connections[0]["count"] > 0
    mans = [_manifold.from_trimesh(m) for _c, m in report.built]
    shared = _shared(mans)
    assert shared > 0.0, "the rungs were carved apart again"
    smallest = min(m.volume for _c, m in report.built)
    assert shared < 0.05 * smallest, f"{shared:.1f} mm3 is a bulge, not a weld"
    # And nothing reports the weld as interference, because it is not.
    assert not any("still share" in w for w in report.warnings)


# --------------------------------------------------------------------------
# Cartoon representation (ChimeraX-style ribbon)
# --------------------------------------------------------------------------
def test_cartoon_builds_watertight_single_body():
    """1UBQ (mixed α/β) as a cartoon must mesh to one watertight body and vary
    its cross-section with secondary structure (a plain tube would not)."""
    from pdb2print.config import Representation
    p = PrintParams(protein_representation=Representation.CARTOON,
                    scale_mm_per_angstrom=1.5, min_wall_mm=1.0)
    report = build_all(os.path.join(D, "1ubq.pdb"), p)
    assert report.built, report.summary()
    assert _all_watertight_single(report)
    _chain, mesh = report.built[0]
    assert mesh.metadata["representation"] == "cartoon"
    # The ribbon (helix/strand) reaches wider than the coil tube, so the mesh is
    # meaningfully broader than a bare coil-radius tube would be.
    span = (mesh.bounds[1] - mesh.bounds[0]).max()
    assert span > 2 * p.cartoon_coil_radius_mm * 3


def test_cartoon_degrades_to_tube_without_sse():
    """With no assignable secondary structure the builder must still produce a
    watertight solid (a coil tube), never raise."""
    from pdb2print.representations import cartoon
    from pdb2print.config import Representation
    from pdb2print import meshops
    chain = chains_mod.split_chains(io.load_with_names(os.path.join(D, "1ubq.pdb"))[0])[0]
    p = PrintParams(protein_representation=Representation.CARTOON)
    mesh = meshops.repair(cartoon.build(chain, p))
    assert mesh.is_watertight


# --------------------------------------------------------------------------
# DNA↔DNA is never magnetised; builds can be cancelled
# --------------------------------------------------------------------------
def test_dna_dna_is_never_joined_unless_base_pairing_asked_for_it():
    """Two strands get nothing. Protein↔DNA is unaffected and still magnetises.

    They used to be silently bridged, on the reasoning that a magnet pocket is
    wider than a backbone tube so a magnet could not be seated there. True, but
    the conclusion was wrong: a joint nobody asked for appeared between two
    strands, and the base-pair pass is the feature that *is* meant to link them.
    So it is now the only thing that does.
    """
    report = build_all(BNA, _params(
        connect=True, use_magnets=True, contact_threshold_mm=3.5,
        connector_diameter_mm=3.0, magnet_thickness_mm=1.5))
    assert _all_watertight_single(report)
    assert not [c for c in report.connections if c["kind"] == "dna-dna"]

    mixed = build_all(COMPLEX, _params(
        connect=True, use_magnets=True, contact_threshold_mm=3.5,
        connector_diameter_mm=3.0, magnet_thickness_mm=1.5))
    assert not [c for c in mixed.connections if c["kind"] == "dna-dna"]
    assert any(c["method"] == "magnet"
               for c in mixed.connections if c["kind"] == "dna-protein")


def test_build_can_be_cancelled():
    """``should_cancel`` unwinds the whole build rather than skipping a chain."""
    from pdb2print.pipeline import BuildCancelled
    with pytest.raises(BuildCancelled):
        build_all(BNA, _params(), should_cancel=lambda: True)
    # A predicate that never fires leaves the build completely unaffected.
    report = build_all(BNA, _params(), should_cancel=lambda: False)
    assert _all_watertight_single(report)


def test_backbone_and_base_ball_stick_sizes_are_independent():
    """The backbone molecule style has its own atom/bond radii, not the base's."""
    import dataclasses
    from pdb2print import geometry, meshops
    from pdb2print.config import BaseStyle, BackboneStyle
    chain = chains_mod.split_chains(io.load_with_names(BNA)[0])[0]
    base_p = dataclasses.replace(
        _params(), base_style=BaseStyle.SLAB,
        backbone_style=BackboneStyle.MOLECULE, min_wall_mm=0.0)

    def vol(**kw):
        p = dataclasses.replace(base_p, **kw)
        return meshops.repair(geometry.generate_chain_mesh(chain, p)).volume

    thin = vol(backbone_atom_radius_mm=0.6)
    thick = vol(backbone_atom_radius_mm=1.6)
    assert thick > thin * 1.2, "backbone atom radius must drive the backbone"


def test_rod_rung_honours_its_thickness_setting():
    """The rod rung is sized by the rung-thickness control, not the tube radius."""
    import dataclasses
    from pdb2print import geometry, meshops
    from pdb2print.config import BaseStyle
    chain = chains_mod.split_chains(io.load_with_names(BNA)[0])[0]
    base_p = dataclasses.replace(_params(), base_style=BaseStyle.ROD,
                                 min_wall_mm=0.0)

    def vol(**kw):
        p = dataclasses.replace(base_p, **kw)
        return meshops.repair(geometry.generate_chain_mesh(chain, p)).volume

    assert vol(slab_thickness_mm=2.5) > vol(slab_thickness_mm=1.0) * 1.2


# --------------------------------------------------------------------------
# Per-pair joint overrides
# --------------------------------------------------------------------------
def test_joint_override_parser_is_best_effort():
    """One bad line costs that line and nothing else."""
    from pdb2print.connections import joint_overrides
    got = joint_overrides(
        "0\t1\tnone\n"
        "3\t2\tjoin\n"        # written the other way round
        "x\ty\tnone\n"        # not numbers
        "4\t4\tnone\n"        # a pair with itself is not a joint
        "5\t6\tmagnet\n"      # not one of the two modes
        "7\t8\n"              # no mode at all
        "\n"
    )
    assert got == {(0, 1): "none", (2, 3): "join"}
    assert joint_overrides("") == {}
    assert joint_overrides(None) == {}


def test_joint_overrides_reach_the_cache_key_in_every_branch():
    """Two different override sets must never share one cache entry.

    ``canonical_params`` prunes the connections block in four branches and two
    of them replace the dict wholesale.  A ``join`` changes the interference
    carve, and that pass runs whether or not the connect switch is on, so the
    field has to survive even the branch that throws the joinery away.
    """
    from pdb2print import cache

    def key(**conn):
        return cache.key_for(BNA, _params(**conn))

    for branch in (
        dict(connect=False, basepair_connect=False),      # 1: dict replaced
        dict(connect=False, basepair_connect=True),       # 2: dict replaced
        dict(connect=True, use_magnets=False),            # 3: keys popped
        dict(connect=True, use_magnets=True),             # 4: keys popped
    ):
        plain = key(**branch)
        none = key(**branch, joint_overrides="0\t1\tnone")
        join = key(**branch, joint_overrides="0\t1\tjoin")
        assert len({plain, none, join}) == 3, branch


def test_empty_joint_overrides_do_not_move_the_cache_key():
    """Empty means today's behaviour, so it has to mean today's key.

    Otherwise adding the field would orphan every entry already in ``cache/``,
    including the pre-generated ones shipped in the repo.
    """
    from pdb2print import cache
    p = _params(connect=True, use_magnets=True)
    before = cache.key_for(BNA, p)
    p.connections.joint_overrides = ""
    assert cache.key_for(BNA, p) == before


def test_bridge_count_reaches_the_cache_key():
    """The bridge reads the two magnet counts, so they must be part of its key.

    They used to be dropped for the whole no-magnets branch, which was right for
    inflate — it never runs the seat search — and wrong for the bridge, which
    uses exactly those two fields to decide how many rods to drop.
    """
    from pdb2print import cache
    from pdb2print.config import NoMagnetMethod

    def key(n):
        return cache.key_for(BNA, _params(
            connect=True, use_magnets=False,
            no_magnet_method=NoMagnetMethod.BRIDGE, magnet_count=n))

    assert key(1) != key(3)


def test_none_veto_leaves_the_pair_unjoined_and_says_so():
    """A vetoed pair keeps its row, reports the plan, and gets no connector."""
    report = build_all(OVERLAP, _params(connect=True, use_magnets=True))
    magnets = [c for c in report.connections if c["method"] == "magnet"
               and c["applied"]]
    assert magnets, "nothing to veto in the baseline build"
    i, j = magnets[0]["ai"], magnets[0]["bi"]

    vetoed = build_all(OVERLAP, _params(
        connect=True, use_magnets=True,
        joint_overrides=f"{i}\t{j}\tnone"))
    row = [c for c in vetoed.connections if (c["ai"], c["bi"]) == (i, j)]
    assert len(row) == 1, "a vetoed joint must stay in the list to be un-vetoed"
    assert row[0]["method"] == "none"
    assert row[0]["count"] == 0
    # No refusal warning for something the user already decided.
    assert "refused" not in row[0]["note"]
    assert _all_watertight_single(vetoed)
    # And no magnet went in for that pair.
    for m in vetoed.connection_markers:
        assert {m.get("a"), m.get("b")} != {row[0]["a"], row[0]["b"]}


def test_zero_count_vetoes_every_pair_of_that_kind():
    """Zero is a real answer, and it is the same skip a hand veto takes."""
    report = build_all(OVERLAP, _params(
        connect=True, use_magnets=True, magnet_count=0, dna_magnet_count=0))
    joins = [c for c in report.connections if c["kind"] != "dna-basepair"]
    assert joins, "no interfaces found at all"
    assert all(c["method"] == "none" for c in joins)
    assert not report.connection_markers
    assert _all_watertight_single(report)


def test_join_keeps_a_pair_fused_through_the_closing_sweep():
    """The trap: the sweep after connecting must not carve a joined pair apart.

    That second ``interference.resolve`` takes no overlap list of its own, so it
    runs a full sweep and will undo the join unless the pair is excluded there
    too.  What is pinned here is the outcome — the two solids still share space
    at the end of the whole pass, and every other pair does not.
    """
    from pdb2print import interference
    from pdb2print.representations import _manifold

    baseline = build_all(OVERLAP, _params(connect=True, use_magnets=True))
    mans = [_manifold.from_trimesh(m) for _c, m in baseline.built]
    assert interference.residual_overlap(mans) < 1.0, "baseline already overlaps"

    # A pair the carve really had to pull apart, so there is an overlap to skip.
    carved = [c for c in baseline.connections
              if c["kind"] in ("protein-protein", "dna-protein")]
    assert carved
    i, j = carved[0]["ai"], carved[0]["bi"]

    fused = build_all(OVERLAP, _params(
        connect=True, use_magnets=True, joint_overrides=f"{i}\t{j}\tjoin"))
    row = [c for c in fused.connections if (c["ai"], c["bi"]) == (i, j)]
    assert len(row) == 1 and row[0]["method"] == "join"

    mans = [_manifold.from_trimesh(m) for _c, m in fused.built]
    shared = {(o.i, o.j): o.volume
              for o in interference.pair_overlaps(mans, want_pieces=False)}
    if row[0]["applied"]:
        assert shared.get((i, j), 0.0) > 0.0, "the join was carved back apart"
        # Nothing else was left interpenetrating on the way.
        assert all(v < 1.0 for k, v in shared.items() if k != (i, j))
        # And the request is not reported back as a fault.
        assert not any("still share" in w for w in fused.warnings)
    else:
        # The pair does not touch: nothing to skip carving, and saying so is
        # the whole of the answer.  No geometry is invented for it.
        assert "do not touch" in row[0]["note"]
        assert not shared

    assert all(m.is_watertight for _c, m in fused.built)


# --------------------------------------------------------------------------
# Chain exclusion, and the stable identity it needs
# --------------------------------------------------------------------------
def test_split_chains_numbers_every_chain_in_source_order():
    """The index is the only stable name a chain has."""
    atoms, _ = io.load_with_names(OVERLAP)
    got = chains_mod.split_chains(atoms)
    assert [c.index for c in got] == list(range(len(got)))


def test_excluded_chain_parser_is_best_effort():
    from pdb2print.chains import parse_excluded
    assert parse_excluded("0, 2 ,x,,3\n5") == {0, 2, 3, 5}
    assert parse_excluded("") == set()
    assert parse_excluded(None) == set()
    assert parse_excluded("-1") == set()          # no such chain


def test_excluding_a_chain_leaves_the_others_alone():
    """The point of the source index: nothing about the kept chains moves.

    Colour especially. Excluding a chain used to shift every chain after it onto
    the next palette entry, which is a bad surprise for anyone who has already
    printed half a model in matching filament.
    """
    import dataclasses
    from pdb2print import export

    full = build_all(OVERLAP, _params())
    assert len(full.built) >= 3
    colors = {c.chain_id: col for (c, _m), col
              in zip(full.built, export.object_colors(full.built))}
    drop = full.built[0][0].index

    cut = build_all(OVERLAP, dataclasses.replace(
        _params(), exclude_chains=str(drop)))
    assert len(cut.built) == len(full.built) - 1
    assert all(c.index != drop for c, _m in cut.built)
    assert _all_watertight_single(cut)

    kept = {c.chain_id: col for (c, _m), col
            in zip(cut.built, export.object_colors(cut.built))}
    for cid, col in kept.items():
        assert col == colors[cid], f"chain {cid} was recoloured by the exclusion"


def test_a_joint_veto_survives_excluding_another_chain():
    """Vetoes are keyed on the source pair, so a dropped chain cannot move one.

    Keyed on built position instead, removing a chain would silently slide every
    veto after it onto a different pair — the quietest possible way to get the
    wrong model.
    """
    import dataclasses
    base = build_all(OVERLAP, _params(connect=True, use_magnets=True))
    pairs = [(c["ai"], c["bi"]) for c in base.connections
             if c["method"] == "magnet"]
    assert len(pairs) >= 2

    # Drop a chain that is in neither of the first pair's ends, so the pair
    # itself survives and only the numbering could have moved.
    ends = set(pairs[0])
    spare = next(c.index for c, _m in base.built if c.index not in ends)
    i, j = pairs[0]

    cut = build_all(OVERLAP, dataclasses.replace(
        _params(connect=True, use_magnets=True),
        exclude_chains=str(spare),
        connections=dataclasses.replace(
            _params(connect=True, use_magnets=True).connections,
            joint_overrides=f"{i}\t{j}\tnone")))
    row = [c for c in cut.connections if (c["ai"], c["bi"]) == (i, j)]
    assert len(row) == 1, "the vetoed pair lost its identity when a chain went"
    assert row[0]["method"] == "none"
    assert _all_watertight_single(cut)


def test_excluding_every_chain_is_refused():
    import dataclasses
    atoms, _ = io.load_with_names(BNA)
    n = len(chains_mod.split_chains(atoms))
    with pytest.raises(ValueError):
        build_all(BNA, dataclasses.replace(
            _params(), exclude_chains=",".join(str(i) for i in range(n))))


def test_exclude_chains_reaches_the_cache_key():
    """It changes which objects exist, so it can never be merged away."""
    import dataclasses as dc
    from pdb2print import cache
    p = _params()
    plain = cache.key_for(BNA, p)
    assert cache.key_for(BNA, dc.replace(p, exclude_chains="0")) != plain
    assert cache.key_for(BNA, dc.replace(p, exclude_chains="1")) != \
        cache.key_for(BNA, dc.replace(p, exclude_chains="0"))
    # ...and an empty list is today's behaviour, so it has to be today's key.
    assert cache.key_for(BNA, dc.replace(p, exclude_chains="")) == plain


def test_cache_version_is_part_of_every_key():
    """Bumping it has to make every existing entry unreachable.

    5 → 6 was not a geometry change but a payload one: an entry stores the
    finished result dict verbatim, so an older one hands the front end a payload
    with no ``parts`` and no ``ai``/``bi`` — the chains-and-joints panel reads
    exactly those, and would simply not appear on the popular structures the
    shipped entries exist for.
    """
    from pdb2print import cache
    p = _params()
    before = cache.key_for(BNA, p)
    old = cache.CACHE_VERSION
    try:
        cache.CACHE_VERSION = old - 1
        assert cache.key_for(BNA, p) != before
    finally:
        cache.CACHE_VERSION = old
    assert cache.key_for(BNA, p) == before


def test_a_cache_hit_keeps_the_colours_the_build_gave_it():
    """A stand raised from a cache hit must not recolour the model.

    The palette follows ``Chain.index``, which a reopened object only has if the
    entry stored it. Without it the reopened objects fall back to their position
    in the built list — so a build with a chain excluded came back one colour
    off across the board, visibly, the moment the stand appeared.
    """
    import dataclasses
    from pdb2print import cache, export

    report = build_all(OVERLAP, dataclasses.replace(
        _params(), exclude_chains="0"))
    original = export.object_colors(report.built)

    reopened = cache.objects_from_meta(
        {"objects": cache.describe_objects(report.built)})
    assert len(reopened) == len(report.built)
    assert [o.index for o in reopened] == [c.index for c, _m in report.built]
    assert export.object_colors([(o, None) for o in reopened]) == original



def test_overlap_mode_fuses_what_touches_and_names_what_does_not():
    """One piece by leaving well alone: nothing grown, nothing cut, nothing added.

    The chains are built overlapping wherever they touch and the fit pass is the
    only thing that pulls them apart, so not running it is a fused model with no
    geometry invented for it. What it cannot do is join two parts that only come
    close — there is no overlap to keep — so those are reported rather than
    quietly left loose.
    """
    from pdb2print.config import NoMagnetMethod
    from pdb2print import interference
    from pdb2print.representations import _manifold

    report = build_all(OVERLAP, _params(
        connect=True, no_magnet_method=NoMagnetMethod.OVERLAP))
    assert all(m.is_watertight for _c, m in report.built)

    rows = [c for c in report.connections if c["method"] == "overlap"]
    assert rows, "nothing was reported at all"
    assert any(c["applied"] for c in rows), "nothing came out fused"

    # Fused means still sharing space at the end of the pass.
    mans = [_manifold.from_trimesh(m) for _c, m in report.built]
    shared = {(o.i, o.j) for o in interference.pair_overlaps(mans,
                                                             want_pieces=False)}
    assert shared, "the carve ran after all"

    # A pair that only comes close is named, not silently left loose.
    loose = [c for c in rows if not c["applied"]]
    if loose:
        assert any("come off the plate loose" in w for w in report.warnings)


def test_bridge_halves_weld_rather_than_meet_on_a_plane():
    """A bridge is a one-piece joint, so its two halves must share material.

    Both collars used to be built against the shared mid-plane, which is right
    for a magnet — that joint comes apart in the hand — and wrong for a bridge.
    Two flat discs meeting exactly on a plane have no contact area for the
    slicer to weld, and any sliver of numerical overlap between them was found
    by the closing sweep and carved out with a fit clearance on top: a one-piece
    model with a ring of air through the middle of every peg.

    The weld is entirely inside the peg's own cylinder, so the shape does not
    change — which is what the upper bound here is for.
    """
    from pdb2print import interference
    from pdb2print.representations import _manifold
    from pdb2print.config import NoMagnetMethod

    report = build_all(COMPLEX, _params(
        connect=True, no_magnet_method=NoMagnetMethod.BRIDGE,
        contact_threshold_mm=3.5, connector_diameter_mm=3.0))
    assert _all_watertight_single(report)
    bridged = [c for c in report.connections
               if c["method"] == "bridge" and c["applied"]]
    assert bridged, "no bridge was built at all"

    mans = [_manifold.from_trimesh(m) for _c, m in report.built]
    shared = {(o.i, o.j): o.volume
              for o in interference.pair_overlaps(mans, want_pieces=False)}
    assert shared, "the two halves meet on a plane and share nothing"

    # A weld, not a peg driven clean through its neighbour: the shared volume is
    # a disc of the peg's own radius, so it cannot be much of the model.
    smallest = min(m.volume for _c, m in report.built)
    assert sum(shared.values()) < 0.05 * smallest
    # And a deliberate weld is not reported back as interference.
    assert not any("still share" in w for w in report.warnings)


def test_share_table_matches_the_controls():
    """The share code is positional, so the table has to stay an exact mirror.

    Widen a slider or reorder a field and every link already in the wild decodes
    to different numbers — silently, because a positional format has no way to
    notice. This is the same failure ``presets.py`` had when it drifted from the
    front end; the difference is that this one fails a test instead of a print.
    """
    import subprocess
    import sys
    root = os.path.dirname(HERE)
    script = os.path.join(root, "scripts", "build_share_table.py")
    got = subprocess.run([sys.executable, script, "--check"],
                         capture_output=True, text=True)
    assert got.returncode == 0, got.stdout + got.stderr


# --------------------------------------------------------------------------
# Cartoon hydrogen-bond struts
# --------------------------------------------------------------------------
def _ubq_chain():
    return chains_mod.split_chains(
        io.load_with_names(os.path.join(D, "1ubq.pdb"))[0])[0]


def test_hbond_detector_finds_the_ubiquitin_sheet():
    """The detector must find the β-hairpin everyone knows is in 1UBQ, and must
    never let proline donate."""
    from pdb2print.representations import hbonds
    chain = _ubq_chain()
    pairs = hbonds.backbone_hbonds(chain)
    assert pairs, "no backbone hydrogen bonds found in 1UBQ"

    # The N-terminal hairpin: strand 1 (residues 1-7) pairs with strand 2
    # (10-17) antiparallel.  Indices are 0-based positions in the chain.
    long_range = {(i, j) for i, j in pairs if abs(i - j) > 5}
    hairpin = {(i, j) for i, j in long_range if i < 20 and j < 20}
    assert len(hairpin) >= 4, sorted(long_range)[:10]

    # Proline has no amide hydrogen, so it can accept but never donate.
    pro = {k for k, (name, _res) in enumerate(_residue_pairs(chain))
           if name == "PRO"}
    assert pro, "1UBQ has prolines; the fixture changed"
    assert not (pro & {i for i, _j in pairs}), "a proline donated"


def _residue_pairs(chain):
    from pdb2print.representations.tube_slab import _residue_iter
    for name, res in _residue_iter(chain.atoms):
        yield name, res


def test_hbond_modes_are_nested_subsets():
    """Helices and sheets are each part of Both, and Both is part of All.

    This is the promise the control makes on screen.  It is also what stops the
    two middle modes drifting apart from the third once someone edits the
    classifier.
    """
    from pdb2print.config import HBondMode, Representation
    from pdb2print.representations import cartoon
    chain = _ubq_chain()
    p = PrintParams(protein_representation=Representation.CARTOON)
    ca, _c, _o = cartoon._ca_backbone(chain)
    sse = cartoon._clean_sse(cartoon._sse(chain, len(ca)))

    sets = {m: set(cartoon._hbond_pairs(chain, sse, m)) for m in HBondMode}
    assert sets[HBondMode.NONE] == set()
    assert sets[HBondMode.HELIX] <= sets[HBondMode.BOTH]
    assert sets[HBondMode.SHEET] <= sets[HBondMode.BOTH]
    assert sets[HBondMode.BOTH] <= sets[HBondMode.ALL]
    assert sets[HBondMode.BOTH] == sets[HBondMode.HELIX] | sets[HBondMode.SHEET]
    assert sets[HBondMode.ALL], "1UBQ should offer something to brace"
    # One end is enough, so a bond out of a sheet into a loop is a sheet bond —
    # and a helix-to-strand bond is in both of the middle modes.
    for mode, letter in ((HBondMode.HELIX, "a"), (HBondMode.SHEET, "b")):
        for i, j in sets[mode]:
            assert letter in (sse[i], sse[j])
    loose = {(i, j) for (i, j) in sets[HBondMode.SHEET]
             if sse[i] != sse[j]}
    assert loose, "sheets that only bond to themselves — the filter is too tight"


def test_cartoon_hbonds_off_returns_the_bare_ribbon():
    """Off must be the mesh that shipped before struts existed — same vertices,
    same faces.  The cache key depends on it: ``canonical_params`` drops the
    field when it is off, so every cartoon entry already in ``cache/`` is served
    for this build."""
    from pdb2print.config import HBondMode, Representation
    from pdb2print.representations import cartoon
    chain = _ubq_chain()
    p = PrintParams(protein_representation=Representation.CARTOON)
    assert p.cartoon_hbonds == HBondMode.NONE
    plain = cartoon.build(chain, p)
    again = cartoon.build(chain, PrintParams(
        protein_representation=Representation.CARTOON,
        cartoon_hbonds=HBondMode.NONE))
    assert plain.vertices.shape == again.vertices.shape
    assert (plain.faces == again.faces).all()
    assert not plain.metadata.get("notes")


def test_cartoon_hbonds_off_is_dropped_from_the_cache_key():
    """The field must not appear in the canonical params when it is off, or
    every pre-generated cartoon entry stops being reachable."""
    from pdb2print.cache import canonical_params
    from pdb2print.config import HBondMode, Representation
    off = canonical_params(PrintParams(
        protein_representation=Representation.CARTOON))
    assert "cartoon_hbonds" not in off
    on = canonical_params(PrintParams(
        protein_representation=Representation.CARTOON,
        cartoon_hbonds=HBondMode.SHEET))
    assert on.get("cartoon_hbonds") == "sheet"
    # And it is dropped entirely when no cartoon is built, on or off.
    surf = canonical_params(PrintParams(
        protein_representation=Representation.SURFACE,
        cartoon_hbonds=HBondMode.ALL))
    assert "cartoon_hbonds" not in surf


def test_cartoon_hbonds_on_stays_watertight_and_adds_material():
    """Struts must fuse into one watertight solid, not sit beside it."""
    from pdb2print.config import HBondMode, Representation
    from pdb2print.representations import cartoon
    from pdb2print import meshops
    chain = _ubq_chain()
    base = PrintParams(protein_representation=Representation.CARTOON,
                       scale_mm_per_angstrom=1.5, min_wall_mm=1.0)
    off = meshops.repair(cartoon.build(chain, base))
    braced = meshops.repair(cartoon.build(chain, PrintParams(
        protein_representation=Representation.CARTOON,
        scale_mm_per_angstrom=1.5, min_wall_mm=1.0,
        cartoon_hbonds=HBondMode.ALL)))
    assert braced.is_watertight
    assert braced.body_count == 1
    assert braced.volume > off.volume
    # A strut is never thinner than half the minimum wall, so the exemption in
    # config.MIN_WALL_EXEMPT stays true with the struts on.
    dims = cartoon._dims(base)
    assert cartoon._strut_radius(base, dims) >= base.min_wall_mm / 2.0


def test_strut_end_lies_flat_against_the_ribbon_it_lands_on():
    """A strut running in the plane of a ribbon must be squashed to the ribbon's
    own half-thickness, and one leaving through the face must stay round."""
    import numpy as np
    from pdb2print.representations import cartoon
    hw, ht, r = 2.0, 0.6, 0.5
    normal = np.array([0.0, 0.0, 1.0])

    # In the ribbon's plane: fully flattened, so it cannot bulge out of a face.
    thin, wide, deep = cartoon._strut_end(
        np.array([1.0, 0.0, 0.0]), normal, hw, ht, r)
    assert abs(deep - ht) < 1e-9
    assert abs(wide - cartoon._STRUT_END_WIDTH * hw) < 1e-9
    assert abs(abs(float(thin @ normal)) - 1.0) < 1e-9
    assert wide > deep, "the end has to be wider than it is thick"

    # Straight out through the face: nothing to lie against, so it stays round.
    thin, wide, deep = cartoon._strut_end(normal, normal, hw, ht, r)
    assert abs(wide - r) < 1e-9 and abs(deep - r) < 1e-9

    # A round coil tube has no flat to lie against: its end must come out
    # circular, and the same size as the tube, not as an oval standing on end.
    _thin, wide, deep = cartoon._strut_end(
        np.array([1.0, 0.0, 0.0]), normal, 0.9, 0.9, r)
    assert abs(wide - deep) < 1e-9, (wide, deep)
    assert abs(wide - 0.9) < 1e-9


def test_strut_anchors_leave_a_ribbon_by_the_edge_facing_the_partner():
    """A sheet bond runs sideways, so it has to start at the side of the plank —
    and at the side the partner is on, not the other one."""
    import numpy as np
    from pdb2print.representations import cartoon
    centre = np.zeros(3)
    w = np.array([1.0, 0.0, 0.0])          # the ribbon's width axis
    hw, ht = 2.0, 0.6                      # a flat plank
    edge = hw - ht                         # centre of the rolled edge

    # Partner off to +x: leave by the +x edge. Off to -x: the other one.
    for sign in (+1.0, -1.0):
        got = cartoon._strut_anchor(centre, w, hw, ht, np.array([sign, 0.0, 0.0]))
        assert abs(got[0] - sign * edge) < 1e-9, got
    # Still inside the solid, with the whole rolled edge left to bury the end in.
    assert edge < hw

    # Straight out through the face: no sideways component, so no offset.
    face = cartoon._strut_anchor(centre, w, hw, ht, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(face, centre)
    # Halfway between: half the offset, so the anchor slides rather than snaps.
    d = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    assert abs(cartoon._strut_anchor(centre, w, hw, ht, d)[0]
               - edge / np.sqrt(2.0)) < 1e-9

    # A round tube has no edge to leave by: the anchor is the axis, whatever
    # direction the strut goes.
    for d in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        assert np.allclose(cartoon._strut_anchor(centre, w, 0.9, 0.9, d), centre)


def test_strut_solid_is_closed_and_stays_between_its_ends():
    """The ends sit on the centre-line points; nothing may reach past them."""
    import numpy as np
    from pdb2print.representations import cartoon
    a, b, r = np.zeros(3), np.array([0.0, 0.0, 10.0]), 0.5
    normal = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, 0.0, 1.0])
    end = cartoon._strut_end(d, normal, 2.0, 0.6, r)
    solid = cartoon._strut_solid(a, b, end, cartoon._strut_end(-d, normal, 2.0, 0.6, r), r)
    solid.fix_normals()
    assert solid.is_watertight and solid.body_count == 1
    assert solid.bounds[0][2] >= -1e-6 and solid.bounds[1][2] <= 10.0 + 1e-6
    # A zero-length strut is dropped rather than lofted into a degenerate band.
    assert cartoon._strut_solid(a, a, end, end, r) is None


# --------------------------------------------------------------------------
# Extended PDB IDs (pdb_00001ubq) alongside the 4-character ones
# --------------------------------------------------------------------------
def test_pdb_ids_are_accepted_in_both_formats():
    """Both spellings are valid PDB IDs and always will be. The 4-character form
    is the one that comes out, because an entry that has one keeps it."""
    from pdb2print.io import canonical_pdb_id, looks_like_pdb_id

    # 4-character, however it was typed.
    assert canonical_pdb_id("1ubq") == "1UBQ"
    assert canonical_pdb_id("  6UV8 ") == "6UV8"

    # Extended, collapsed back to the entry's real name. The pdb_0000 block is
    # reserved for exactly this, so nothing is guessed.
    assert canonical_pdb_id("pdb_00001ubq") == "1UBQ"
    assert canonical_pdb_id("PDB_00001UBQ") == "1UBQ"
    assert canonical_pdb_id("pdb_00006UV8") == "6UV8"

    # An ID from the new block has no 4-character name, so it keeps its own —
    # and the prefix comes out lowercase, which is how the wwPDB specifies it.
    assert canonical_pdb_id("pdb_1000axyz") == "pdb_1000axyz"
    assert canonical_pdb_id("PDB_1000AXYZ") == "pdb_1000axyz"

    # Inside the reserved block but not shaped like a 4-character ID (those
    # start with a digit): left alone rather than silently truncated.
    assert canonical_pdb_id("pdb_0000zzzz") == "pdb_0000zzzz"

    for junk in ("", "   ", "x", "12345", "ubq1", "pdb_123", "pdb_00001ub",
                 "1ubq.pdb", "pdb-00001ubq", None):
        assert canonical_pdb_id(junk) is None, junk
        assert not looks_like_pdb_id(junk or "")

    # cache.is_cacheable rides on this: an upload must never look like an ID.
    from pdb2print.cache import is_cacheable
    assert is_cacheable("1UBQ") and is_cacheable("pdb_1000axyz")
    assert not is_cacheable("/tmp/something/mystructure.pdb")


def test_both_spellings_of_an_entry_share_one_cache_key():
    """Otherwise the same model is built twice, stored twice, and neither copy
    is ever served to someone who spelled it the other way."""
    from pdb2print.cache import key_for
    from pdb2print.config import Representation
    p = PrintParams(protein_representation=Representation.CARTOON)

    same = {key_for(s, p) for s in ("1UBQ", "1ubq", "pdb_00001ubq",
                                    "PDB_00001UBQ", "  pdb_00001UBQ  ")}
    assert len(same) == 1, same

    # A different entry is still a different key, extended or not.
    assert key_for("6UV8", p) not in same
    assert key_for("pdb_1000axyz", p) not in same


def test_existing_cache_keys_do_not_move():
    """The 2.2 GB of pre-generated entries were keyed on the upper-cased source.
    Canonicalising must reproduce that byte for byte, or accepting a new ID
    format would silently orphan every build the project has ever cached."""
    import hashlib
    import json
    from pdb2print.cache import key_for, canonical_params, CACHE_VERSION
    from pdb2print.config import Representation

    p = PrintParams(protein_representation=Representation.CARTOON)
    # The expression key_for used before extended IDs existed.
    payload = {
        "v": CACHE_VERSION,
        "source": "1UBQ".strip().upper(),
        "params": canonical_params(p),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    assert key_for("1UBQ", p) == hashlib.sha256(blob).hexdigest()[:20]


def test_a_bad_id_says_what_a_good_one_looks_like():
    """The error is the only place a user finds out the extended form is
    accepted, so it has to name both."""
    import pytest as _pytest
    with _pytest.raises(ValueError) as err:
        io.resolve_source("not-an-id")
    text = str(err.value)
    assert "1UBQ" in text and "pdb_00001ubq" in text
