"""
Stage 1 regression test against REAL (not synthetic) UIUC airfoils.

`test_stage1_mesh.py` uses analytically-generated NACA 4-digit fixtures,
which are clean and cosine-clustered by construction. That's good for a
fast, deterministic core suite, but it can't catch problems that only
show up on real digitized data -- which is exactly what happened here:
this test's fixtures (already fetched, checked into
tests/fixtures/real_uiuc/, no network needed at test time) originally
surfaced two real bugs that the synthetic suite completely missed:

  - `goe398.dat` (33 raw points, coarse spacing): checkMesh reported max
    skewness 13.7 (hard fail) and even hung under `-allTopology
    -allGeometry`. Fixed by cosine-resampling the airfoil boundary for
    meshing purposes in `generate_mesh` (see `_resample_cosine`), and by
    dropping those extra checkMesh flags entirely (not part of the spec's
    gate, and the cause of the hang).
  - `e387.dat` (thin, low-Re): failed the same skewness check for the
    same underlying reason. Same fix.

Runs against every real airfoil checked into fixtures/real_uiuc/ (35 as of
writing -- discovered dynamically, not a hardcoded subset, so a fixture
added later is automatically covered). That set spans thin low-Re
sailplane sections, very-high-camber high-lift sections, older/coarsely
digitized sections (as few as 33 raw points), thick sections, NACA 4-digit
/ 5-digit / 6-series families, a transonic section (rae2822), and several
airfoils with a genuine (not synthetic) open TE gap in the raw data
(largest: m6, ~0.5% chord).
"""

import os

import pytest

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage1_mesh import run_stage1

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "real_uiuc")

REAL_AIRFOILS = sorted(f[:-4] for f in os.listdir(FIXTURES) if f.endswith(".dat"))


@pytest.fixture(scope="module")
def stage1_real_results(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage1_real")
    results = {}
    for name in REAL_AIRFOILS:
        dat_path = os.path.join(FIXTURES, f"{name}.dat")
        coords = load_airfoil(dat_path)
        results[name] = run_stage1(coords, name, str(out_dir))
    return results


@pytest.mark.parametrize("name", REAL_AIRFOILS)
def test_real_uiuc_checkmesh_gate(stage1_real_results, name):
    check = stage1_real_results[name]["check"]
    assert check["negative_volume_cells"] == 0, check["raw_output"][-2000:]
    assert check["non_orthogonality_ok"], check["raw_output"][-2000:]
    assert check["skewness_ok"], check["raw_output"][-2000:]
    assert check["passed"]


@pytest.mark.parametrize("name", REAL_AIRFOILS)
def test_real_uiuc_full_mesh_ok(stage1_real_results, name):
    # Stronger than the spec's own Gate #1 (which ignores aspect
    # ratio/determinant) -- current implementation achieves a fully clean
    # checkMesh verdict on all 5, so hold it to that bar as a regression
    # guard. If this ever regresses without the Gate #1 test above also
    # failing, it's a determinant/aspect-ratio-only regression -- see the
    # note in check_mesh() before treating that as blocking.
    check = stage1_real_results[name]["check"]
    assert check["mesh_ok"], check["raw_output"][-2000:]
