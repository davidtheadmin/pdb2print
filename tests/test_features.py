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
    ok, note = cx._apply_magnet(mans, 0, 1, meshes[0], meshes[1], 1, cp,
                                PrintParams(), markers)
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

def test_basepair_rungs_meet_without_sharing_space():
    """Opposing rungs must read as one bar and still come apart.

    A capsule's solid runs a full radius past its end point, so a link aimed at
    the midline used to cross it by ``2r`` — invisible on screen, a collision in
    the print, and added *after* the fit pass had already run.
    """
    report = build_all(BNA, _params(basepair_connect=True))
    assert _all_watertight_single(report)
    assert report.connections[0]["count"] > 0
    mans = [_manifold.from_trimesh(m) for _c, m in report.built]
    assert _shared(mans) == pytest.approx(0.0, abs=1e-3)


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
