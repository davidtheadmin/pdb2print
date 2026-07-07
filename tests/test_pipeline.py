"""End-to-end pipeline tests.

The default run uses a bundled tiny structure so tests work offline.  Pass
``--rcsb`` (or set ``PDB2PRINT_TEST_RCSB=1``) to additionally exercise the RCSB
fetch path on the recommended test IDs: 1UBQ (protein), 1BNA (DNA), 1ZAA
(protein-DNA complex).
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from pdb2print.config import PrintParams, Representation, MinWallMode
from pdb2print import chains as chains_mod, geometry, meshops, export
from pdb2print.pipeline import build_all


HERE = os.path.dirname(__file__)
TINY_PDB = os.path.join(HERE, "data", "tiny.pdb")

RUN_RCSB = os.environ.get("PDB2PRINT_TEST_RCSB") == "1"


def _fast_params(**kw):
    # Coarse grid keeps unit tests quick.
    base = dict(scale_mm_per_angstrom=0.5, grid_spacing_mm=0.8, min_wall_mm=1.0)
    base.update(kw)
    return PrintParams(**base)


def test_tiny_surface_is_watertight():
    report = build_all(TINY_PDB, _fast_params())
    assert report.built, report.summary()
    for _, mesh in report.built:
        assert mesh.is_watertight, report.summary()
        assert mesh.is_winding_consistent
        assert len(mesh.faces) > 0


def test_min_wall_thickens_when_representation_opts_in():
    """A representation that keeps min-wall gets thicker with a larger target."""
    params = _fast_params(min_wall_mm=2.0)
    chain = chains_mod.split_chains(_load(TINY_PDB))[0]
    base = meshops.repair(geometry.generate_chain_mesh(chain, params))
    # Tag as a representation that keeps min-wall to exercise the pass (the
    # tiny fixture is protein-only, whose surface would otherwise decline it).
    tagged = base.copy()
    tagged.metadata["representation"] = Representation.TUBE_SLAB.value
    thick = meshops.enforce_min_wall(tagged, params)
    assert thick.volume > base.volume * 1.05


def test_surface_declines_min_wall():
    """A Gaussian surface must be returned unchanged by the min-wall pass."""
    params = _fast_params(min_wall_mm=2.0)
    chain = chains_mod.split_chains(_load(TINY_PDB))[0]
    base = meshops.repair(geometry.generate_chain_mesh(chain, params))
    assert base.metadata.get("representation") == Representation.SURFACE.value
    same = meshops.enforce_min_wall(base, params)
    assert same.volume == pytest.approx(base.volume)


def test_exports_write_files(tmp_path):
    report = build_all(TINY_PDB, _fast_params())
    glb = export.write_glb(report.built, str(tmp_path / "out.glb"))
    stlzip = export.write_stl_zip(report.built, str(tmp_path / "out.zip"))
    assert os.path.getsize(glb) > 0
    assert os.path.getsize(stlzip) > 0
    # 3MF is best-effort (lib3mf may be absent in CI).
    try:
        p = export.write_3mf(report.built, str(tmp_path / "out.3mf"))
        assert os.path.getsize(p) > 0
    except RuntimeError:
        pytest.skip("lib3mf not installed")


@pytest.mark.skipif(not RUN_RCSB, reason="set PDB2PRINT_TEST_RCSB=1 to run")
@pytest.mark.parametrize("pdb_id", ["1UBQ", "1BNA", "1ZAA"])
def test_rcsb_ids(pdb_id):
    report = build_all(pdb_id, _fast_params(grid_spacing_mm=1.0))
    assert report.built, report.summary()
    for _, mesh in report.built:
        assert mesh.is_watertight, f"{pdb_id}: {report.summary()}"


def _load(path):
    from pdb2print import io
    return io.load_file(path)
