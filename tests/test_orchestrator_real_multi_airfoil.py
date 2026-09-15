"""
Real multi-airfoil orchestrator validation -- no mocks, real WSL/gmsh/ccx.

Written before running the actual parallel batch, per this project's own
convention: the real 3-airfoil sequential run (naca0012/2412/0021, full
5-AoA sweep) already confirmed the composed chain and `run_batch`'s
sequential path for real (see STATUS.md). What's still unconfirmed is
`max_workers>1` under genuine concurrent WSL calls -- STATUS.md flags
this explicitly as unvalidated, and this project's own failure history
("each stage passes alone, composition breaks") is exactly why a mocked
test can't stand in for this one.

These tests are slow (several minutes, real solves) and gated by the
same real-hardware dependency as test_stage*_real_uiuc.py (WSL +
OpenFOAM + gmsh + CalculiX) -- run explicitly with:
    pytest tests/test_orchestrator_real_multi_airfoil.py -v -s
not swept up by default full-suite `pytest tests/` runs given the extra
wall-clock cost on top of the already-real per-stage suites.
"""

import os
import time

import pytest

from pipeline import orchestrator

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
REAL_UIUC = os.path.join(FIXTURES, "real_uiuc")

pytestmark = pytest.mark.skipif(
    os.environ.get("ADE_SKIP_REAL_TESTS") == "1",
    reason="real WSL/gmsh/ccx orchestrator run explicitly disabled",
)

# 4 airfoils, none previously run through the orchestrator (the earlier
# real confirmation used naca0012/2412/0021) -- 2 from tests/fixtures/,
# 2 from the broader real_uiuc corpus, so this also doubles as evidence
# the orchestrator isn't accidentally special-cased to the 3 airfoils
# already exercised.
PARALLEL_VALIDATION_AIRFOILS = [
    {"name": "naca0006", "dat_path": os.path.join(FIXTURES, "naca0006.dat")},
    {"name": "naca4412", "dat_path": os.path.join(FIXTURES, "naca4412.dat")},
    {"name": "clarky", "dat_path": os.path.join(REAL_UIUC, "clarky.dat")},
    {"name": "e387", "dat_path": os.path.join(REAL_UIUC, "e387.dat")},
]


def test_real_parallel_batch_converges_with_no_corruption_and_beats_serial(tmp_path):
    """
    4 real airfoils, full spec-locked 5-AoA sweep, max_workers=2. One
    real run checks two things (kept in one test rather than two,
    since each real run costs several minutes of actual WSL/gmsh/ccx
    work and both checks need the same run):

    1. **No WSL/gmsh cross-process corruption.** STATUS.md's "FIXED:
       gmsh + WSL subprocess" section fixed *in-process* gmsh->WSL
       ordering (run_isolated's spawned subprocess) but never validated
       genuinely concurrent WSL calls from two separate OS processes at
       once. If that contends or corrupts, it won't necessarily show up
       as a clean pytest failure with a pointing stack trace -- it can
       show up as a case that would have converged sequentially failing
       to converge, or a slow/garbled WSL call. So the assertions check
       the real physics outcome (every AoA converged), not just "did
       not raise".
    2. **Real speedup.** The 3-airfoil sequential run already measured
       ~116s/airfoil for a full 5-AoA sweep (349s / 3, see STATUS.md)
       -- 4 airfoils serially would be ~464s. The threshold below is
       set well above the optimistic ~232s (2x116s) parallel case and
       well below the serial estimate, so it only fails if
       max_workers=2 isn't actually overlapping real work.
    """
    out_dir = str(tmp_path / "parallel_validation")
    start = time.monotonic()
    manifest = orchestrator.run_batch(
        PARALLEL_VALIDATION_AIRFOILS, out_dir, max_workers=2, resume=False,
    )
    elapsed = time.monotonic() - start

    for spec in PARALLEL_VALIDATION_AIRFOILS:
        name = spec["name"]
        entry = manifest[name]
        assert entry["status"] == "success", (
            f"{name}: failed under max_workers=2 "
            f"(category={entry.get('failure_category')}): {entry.get('error')}"
        )
        assert entry["n_converged"] == entry["n_total"] == 5, (
            f"{name}: expected all 5 AoAs to converge (as they do "
            f"sequentially for well-behaved real UIUC geometry at this "
            f"Reynolds number) -- got {entry['n_converged']}/{entry['n_total']}, "
            f"statuses {entry['per_aoa_status']} -- possible WSL/gmsh "
            "cross-process contention under concurrency"
        )
        assert os.path.exists(entry["h5_path"]), (
            f"{name}: manifest claims success but {entry['h5_path']} is missing"
        )

    serial_estimate = 4 * 116
    assert elapsed < serial_estimate * 0.75, (
        f"parallel batch took {elapsed:.0f}s, not meaningfully under the "
        f"~{serial_estimate}s serial estimate -- max_workers=2 may not be "
        "overlapping real WSL/gmsh work"
    )
