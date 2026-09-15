"""
Verification tests for Stage 3 -- CFD execution + convergence check.

Contract (from .claude/airfoil_pipeline_build_spec.md, lines 95-104):
  Input:  a Stage 2 case dir
  Output: converged: bool, status: "converged"|"non_converged"|"diverged"
          |"crashed", final residuals per field
  Convergence logic:
    - Track "Initial residual" (not "Final residual") for Ux, Uy, p, nuTilda.
    - Converged: max(initial residuals) < 1e-5 at the final iteration.
    - Diverged: residuals trending upward, or solver throws (NaN,
      Foam::error, floating point exception).
    - Non-converged: hit endTime without crossing threshold -- then check
      slope of max-residual over last ~50 iterations: still decreasing
      meaningfully -> extend endTime and rerun; flat/plateaued -> flag
      for review (do NOT just rerun longer).

Crashed vs. diverged (a genuine spec ambiguity, resolved here): the spec
lists "solver throws (NaN, Foam::error, floating point exception)" under
the *Diverged* bullet, but the output schema also lists "crashed" as a
distinct status. Resolved as: "diverged" = the physics blew up, whether
detected via residual trend or via the solver's own throw (a throw IS
evidence of divergence -- it's the same underlying failure, just detected
differently); "crashed" = the solver never produced a single real
iteration at all (an execution/config failure, e.g. a missing
`solver` entry in controlDict) -- distinct from anything physical.

Fixture provenance (tests/fixtures/solver_logs/) -- every log here is
either a REAL captured `foamRun` run or a real run's prefix with only the
divergent tail mutated (never fabricated from scratch), so the parser is
validated against OpenFOAM's actual log format, not this test suite's own
idea of it:

  - real_airFoil2D_converged.log: the bundled, spec-named baseline
    tutorial (tutorials/incompressibleFluid/airFoil2D), run as-is.
    Converges in 313 iterations; ends with a genuine
    "SIMPLE solution converged in 313 iterations" line.
  - real_ours_naca0012_aoa4_converged.log: THIS pipeline's own Stage
    0->1->2 output (naca0012, AoA=4 deg, Re=5e5) run through foamRun.
    Converges in 370 iterations. Getting this to converge at all
    surfaced a real Stage 2 bug (see test_stage2_case_gen.py's
    test_control_dict_specifies_solver, and stage2_case_gen.py's
    nu_tilda_inf comment) -- this log is the proof the fix worked.
  - real_ours_naca0012_aoa0_oscillating.log: an earlier real run of this
    pipeline's own output (before the nu_tilda_inf fix) at AoA=0.
    Genuinely never converges -- nuTilda gets stuck in an exact two-value
    oscillation (4.44906e-05 / 2.50650e-05, alternating every iteration)
    for the full 1000-iteration run. A real plateaued/non-converged case,
    kept as fixture data regardless of the since-fixed root cause,
    because the classifier still needs to recognize this shape correctly.
  - real_truncated_still_decreasing.log: real_ours_naca0012_aoa4_converged.log
    truncated at iteration 300 (real data, just cut short, as if endTime
    had been 300 instead of continuing to convergence at 370) -- residuals
    are still above 1e-5 but clearly still dropping.
  - mutated_diverged_via_trend.log: 15 real early-transient iterations,
    unmodified, followed by 30 iterations with the same real solver-line
    format/solver names but residual values multiplied by ~1.6x per
    iteration (geometric growth) -- simulates the spec's "trending
    upward" criterion without needing to reproduce a genuine numerical
    blowup, which would be slow and non-reproducible on demand.
  - mutated_diverged_via_throw.log: 15 real early-transient iterations
    (real data), followed by a FOAM FATAL ERROR / floating point
    exception banner in OpenFOAM's real banner structure (verified
    against a genuine fatal-error capture from this session -- see
    real_crashed_missing_solver.log below), reworded for an FPE since we
    don't have a genuine FPE trace on hand.
  - real_crashed_missing_solver.log: a REAL captured foamRun failure from
    this session (an earlier version of stage2_case_gen.py's controlDict
    was missing the `solver` entry) -- zero solver iterations ever ran.

Thresholds (DIVERGE_SLOPE_THRESHOLD, DECREASING_SLOPE_THRESHOLD in
stage3_run.py) were derived FROM these exact fixtures' measured slopes,
not guessed -- see stage3_run.py's module docstring for the numbers.
"""

import os

import numpy as np
import pytest

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage1_mesh import run_stage1
from pipeline.stage2_case_gen import generate_case
from pipeline.stage3_run import parse_convergence_log, run_case, TRACKED_FIELDS

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
LOGS = os.path.join(FIXTURES, "solver_logs")


def _log(name):
    with open(os.path.join(LOGS, name)) as f:
        return f.read()


# --- 1. Converged classification --------------------------------------------


@pytest.mark.parametrize("fname", [
    "real_airFoil2D_converged.log",
    "real_ours_naca0012_aoa4_converged.log",
])
def test_converged_log_classified_correctly(fname):
    result = parse_convergence_log(_log(fname))
    assert result["status"] == "converged"
    assert result["converged"] is True
    assert result["crash_reason"] is None
    assert set(result["final_residuals"]) == set(TRACKED_FIELDS)
    assert max(result["final_residuals"].values()) < 1e-5


# --- 2. Non-converged, still decreasing -> should extend + rerun -----------


def test_still_decreasing_log_classified_correctly():
    result = parse_convergence_log(_log("real_truncated_still_decreasing.log"))
    assert result["status"] == "non_converged"
    assert result["converged"] is False
    assert result["trend"] == "still_decreasing"
    assert max(result["final_residuals"].values()) >= 1e-5


# --- 3. Non-converged, plateaued -> flag for review, don't just rerun ------


def test_plateaued_log_classified_correctly():
    result = parse_convergence_log(_log("real_ours_naca0012_aoa0_oscillating.log"))
    assert result["status"] == "non_converged"
    assert result["converged"] is False
    assert result["trend"] == "plateaued"
    assert max(result["final_residuals"].values()) >= 1e-5


# --- 4. Diverged, via trending-upward residuals -----------------------------


def test_diverged_via_trend_log_classified_correctly():
    result = parse_convergence_log(_log("mutated_diverged_via_trend.log"))
    assert result["status"] == "diverged"
    assert result["converged"] is False


# --- 5. Diverged, via solver throw (FPE/Foam::error) after real iterations -


def test_diverged_via_throw_log_classified_correctly():
    result = parse_convergence_log(_log("mutated_diverged_via_throw.log"))
    assert result["status"] == "diverged"
    assert result["converged"] is False
    assert result["crash_reason"] is not None
    # Real iterations DID happen before the throw -- distinguishes this
    # from "crashed" (see module docstring's crashed/diverged reconciliation).
    assert result["iterations"] > 0


# --- 6. Crashed: no real iteration ever completed ---------------------------


def test_crashed_log_classified_correctly():
    result = parse_convergence_log(_log("real_crashed_missing_solver.log"))
    assert result["status"] == "crashed"
    assert result["converged"] is False
    assert result["iterations"] == 0
    assert result["final_residuals"] is None
    assert result["crash_reason"] is not None


def test_empty_log_is_crashed_not_an_exception():
    result = parse_convergence_log("")
    assert result["status"] == "crashed"
    assert result["iterations"] == 0


# --- 7. Residuals are read from "Initial residual", not "Final residual" ---


def test_reads_initial_residual_not_final():
    """
    A log line like 'Initial residual = 1, Final residual = 0.06' must
    parse to 1, not 0.06 -- confirmed against the real converged log's
    very first iteration, where the two differ by more than an order of
    magnitude (real_airFoil2D_converged.log's first Ux line: Initial
    residual = 1, Final residual = 0.0611362).
    """
    text = _log("real_airFoil2D_converged.log")
    result = parse_convergence_log(text)
    # Not directly assertable from final_residuals (that's the LAST
    # iteration's initial residual) -- so parse the series directly via
    # the same regex path and check its very first value.
    from pipeline.stage3_run import _parse_residual_series
    ux_series = _parse_residual_series(text, "Ux")
    assert ux_series[0] == pytest.approx(1.0)
    assert ux_series[0] != pytest.approx(0.0611362)


# --- 8. Real end-to-end run: the pipeline actually executes -----------------


@pytest.fixture(scope="module")
def real_run_result(tmp_path_factory):
    """
    Run the full Stage 0 -> 1 -> 2 -> 3 chain for real (naca0012, AoA=4,
    Re=5e5 -- the same configuration validated in
    real_ours_naca0012_aoa4_converged.log above).

    This does NOT assert a specific convergence outcome beyond "the
    solver ran and produced a classifiable result" -- pinning today's
    exact behavior (e.g. asserting non-convergence) would enshrine a
    defect as expected behavior the moment it's fixed, or start failing
    for unrelated reasons (mesh/solver nondeterminism, timing) the moment
    it's not. What this DOES guarantee is the actual target of this
    exercise: Stage 3 can take a real Stage 2 case and run it to a real,
    parseable, valid classification -- i.e. the pipeline is genuinely
    runnable end-to-end, not merely well-formed.
    """
    out_dir = tmp_path_factory.mktemp("stage3_real")
    coords = load_airfoil(os.path.join(FIXTURES, "naca0012.dat"))
    stage1 = run_stage1(coords, "naca0012", str(out_dir))
    stage2 = generate_case(
        mesh_case_dir=stage1["case_dir"], name="naca0012", aoa_deg=4.0,
        reynolds=5e5, output_dir=str(out_dir), nu=1.5e-5,
    )
    return run_case(stage2["case_dir"], timeout=280)


def test_real_run_executes_and_produces_valid_classification(real_run_result):
    assert real_run_result["returncode"] == 0
    assert real_run_result["status"] in (
        "converged", "non_converged", "diverged", "crashed"
    )
    assert os.path.exists(real_run_result["log_path"])
    assert os.path.getsize(real_run_result["log_path"]) > 0
    # The one assertion that actually proves a solve happened, not just
    # that something producing a classifiable (even trivially crashed)
    # result ran -- a loose "status in (...)" check alone would pass on a
    # silent crash just as easily as on a real solve.
    assert real_run_result["iterations"] > 0, (
        f"foamRun produced zero iterations (status={real_run_result['status']!r}, "
        f"crash_reason={real_run_result['crash_reason']!r}) -- this is not "
        "the pipeline actually running, see log_path for details"
    )


def test_real_run_produces_residuals_for_all_tracked_fields(real_run_result):
    assert set(real_run_result["final_residuals"]) == set(TRACKED_FIELDS)
    for value in real_run_result["final_residuals"].values():
        assert np.isfinite(value)
        assert value >= 0


def test_real_run_writes_fvschemes_and_fvsolution(real_run_result):
    system_dir = os.path.join(real_run_result["case_dir"], "system")
    assert os.path.exists(os.path.join(system_dir, "fvSchemes"))
    assert os.path.exists(os.path.join(system_dir, "fvSolution"))


# --- Error handling ----------------------------------------------------------


def test_missing_case_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_case(str(tmp_path / "does_not_exist"))
