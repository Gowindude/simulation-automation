"""
Stage 3 -- CFD execution + convergence check.

Input:  a Stage 2 case dir (has 0/{U,p,nuTilda,nut}, constant/polyMesh,
        constant/{physicalProperties,momentumTransport}, system/controlDict).
Output: {"converged": bool, "status": "converged"|"non_converged"|
        "diverged"|"crashed", "final_residuals": {...}|None, ...}

Stage 2 does not write system/fvSchemes or system/fvSolution (they're
solver numerics, not case-generation concerns -- see stage2_case_gen.py's
module docstring). Stage 3 writes them here if missing, copied verbatim
from the spec's own named validated baseline
(tutorials/incompressibleFluid/airFoil2D) -- confirmed to reference no
patch names (no 'inlet'/'outlet'/'walls'), so they apply unmodified to
this pipeline's farfield/airfoil/front/back patches.

Convergence logic (spec lines 96-104), thresholds calibrated against real
solver logs (see tests/fixtures/solver_logs/ and their generation notes
in test_stage3_run.py) rather than guessed:
  - Track "Initial residual" (not "Final residual") for Ux, Uy, p, nuTilda.
  - Converged: max(initial residuals) < 1e-5 at the final iteration.
  - Diverged: residuals trending upward (checked via a linear fit of
    log10(max residual) over the last ~50 iterations), or the solver
    throws (NaN/Inf residual value, or a FOAM FATAL ERROR / floating
    point exception in the log).
  - Non-converged: hit endTime without crossing threshold -- sub-classify
    by the same slope: still meaningfully decreasing -> extend endTime
    and rerun; flat/plateaued -> flag for review, do NOT just rerun
    longer (this is the spec's own explicit instruction, not a
    convenience default).
  - Crashed: the solver never produced a usable iteration at all (e.g.
    "solver not specified", a missing dictionary entry) -- an execution
    failure distinct from a physical divergence.
"""

import os
import re
import subprocess

import numpy as np

from pipeline._shell import run_shell as _run_wsl_raw
from pipeline._shell import to_linux_path as _to_wsl_path

TRACKED_FIELDS = ("Ux", "Uy", "p", "nuTilda")
CONVERGENCE_THRESHOLD = 1e-5

# Calibrated against tests/fixtures/solver_logs/ (see that file's
# docstring for how each fixture was obtained): real non-diverging cases
# never exceed a last-50-iteration log-slope of about +0.0044, while a
# genuinely divergent run (geometric residual growth) measures +0.105 --
# an order of magnitude of headroom either side of 0.05.
DIVERGE_SLOPE_THRESHOLD = 0.05

# Real "still decreasing" cases measured -0.009 to -0.014; real
# "plateaued" cases (including a genuine oscillating limit cycle) measured
# -0.0003 to +0.0044. -0.005 sits with margin in the gap between them.
DECREASING_SLOPE_THRESHOLD = -0.005

RESIDUAL_WINDOW = 50

_CRASH_PATTERNS = [
    re.compile(r"FOAM FATAL ERROR", re.IGNORECASE),
    # Exclude "...floating point exception trapping..." -- OpenFOAM prints
    # that on EVERY run at startup (sigFpe enabling its own trap handler),
    # not on a crash; confirmed as a false positive against a real
    # converged log before adding this exclusion.
    re.compile(r"floating point exception(?!\s+trapping)", re.IGNORECASE),
    re.compile(r"Foam::error", re.IGNORECASE),
    re.compile(r"Segmentation fault", re.IGNORECASE),
]

_FV_SCHEMES = """FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,nuTilda) bounded Gauss linearUpwind grad(nuTilda);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method meshWave;
}
"""

_FV_SOLUTION = """FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-08;
        relTol          0.1;
    }

    nuTilda
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-08;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;

    residualControl
    {
        p               1e-5;
        U               1e-5;
        nuTilda         1e-5;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        nuTilda         0.7;
    }
}
"""


def _run_wsl(bash_cmd: str, timeout: int) -> subprocess.CompletedProcess:
    return _run_wsl_raw(bash_cmd, timeout=timeout, source_openfoam=True)


def _ensure_numerics(case_dir: str) -> None:
    """Write fvSchemes/fvSolution if the case doesn't already have them."""
    system_dir = os.path.join(case_dir, "system")
    os.makedirs(system_dir, exist_ok=True)
    schemes_path = os.path.join(system_dir, "fvSchemes")
    solution_path = os.path.join(system_dir, "fvSolution")
    if not os.path.exists(schemes_path):
        with open(schemes_path, "w", encoding="utf-8") as f:
            f.write(_FV_SCHEMES)
    if not os.path.exists(solution_path):
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(_FV_SOLUTION)


def _detect_crash(log_text: str) -> str | None:
    for pattern in _CRASH_PATTERNS:
        match = pattern.search(log_text)
        if match:
            # A few lines of context around the match, trimmed, as the reason.
            start = log_text.rfind("\n", 0, match.start())
            end = log_text.find("\n\n", match.end())
            snippet = log_text[max(start, 0):end if end != -1 else match.end() + 200]
            return snippet.strip()
    return None


def _parse_residual_series(log_text: str, field: str) -> list:
    raw = re.findall(rf"Solving for {field}, Initial residual = ([\w.+\-]+)", log_text)
    return [float(v) for v in raw]


def _log_slope(values) -> float | None:
    """Slope of log10(values) vs. index via a linear fit; None if <2 points."""
    if len(values) < 2:
        return None
    y = np.log10(np.asarray(values, dtype=np.float64))
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def parse_convergence_log(log_text: str) -> dict:
    """
    Classify a foamRun log's convergence status.

    Returns:
        {
            "status": "converged" | "non_converged" | "diverged" | "crashed",
            "converged": bool,
            "iterations": int,
            "final_residuals": {"Ux": float, "Uy": float, "p": float, "nuTilda": float} | None,
            "trend": "still_decreasing" | "plateaued" | None,
            "crash_reason": str | None,
        }
    """
    crash_reason = _detect_crash(log_text)
    residuals = {f: _parse_residual_series(log_text, f) for f in TRACKED_FIELDS}
    iterations = min((len(v) for v in residuals.values()), default=0)

    # No real iteration ever completed: an execution failure (bad
    # controlDict, missing dictionary entry, etc.), not a physical
    # divergence -- "crashed", regardless of whether a fatal-error banner
    # was present (a log with zero iterations and no banner at all is
    # just as unusable).
    if iterations == 0:
        return {
            "status": "crashed",
            "converged": False,
            "iterations": 0,
            "final_residuals": None,
            "trend": None,
            "crash_reason": crash_reason or "no solver iterations found in log",
        }

    final_residuals = {f: residuals[f][iterations - 1] for f in TRACKED_FIELDS}

    # The solver DID produce real iterations before throwing -- that's the
    # physics blowing up (NaN/Inf, floating point exception, Foam::error
    # mid-solve), which is "diverged", not "crashed": crashed is reserved
    # for the case where the solver never got a solve going at all.
    if crash_reason or any(not np.isfinite(v) for v in final_residuals.values()):
        return {
            "status": "diverged",
            "converged": False,
            "iterations": iterations,
            "final_residuals": final_residuals,
            "trend": None,
            "crash_reason": crash_reason,
        }

    if max(final_residuals.values()) < CONVERGENCE_THRESHOLD:
        return {
            "status": "converged",
            "converged": True,
            "iterations": iterations,
            "final_residuals": final_residuals,
            "trend": None,
            "crash_reason": None,
        }

    max_series = [
        max(residuals[f][i] for f in TRACKED_FIELDS) for i in range(iterations)
    ]
    window = max_series[-min(RESIDUAL_WINDOW, iterations):]
    slope = _log_slope(window)

    if slope is not None and slope > DIVERGE_SLOPE_THRESHOLD:
        return {
            "status": "diverged",
            "converged": False,
            "iterations": iterations,
            "final_residuals": final_residuals,
            "trend": None,
            "crash_reason": None,
        }

    trend = "still_decreasing" if (slope is not None and slope < DECREASING_SLOPE_THRESHOLD) else "plateaued"
    return {
        "status": "non_converged",
        "converged": False,
        "iterations": iterations,
        "final_residuals": final_residuals,
        "trend": trend,
        "crash_reason": None,
    }


def run_case(case_dir: str, timeout: int = 600, log_filename: str = "log.foamRun") -> dict:
    """
    Run foamRun against a Stage 2 case dir and classify convergence.

    Writes system/fvSchemes and system/fvSolution first if the case
    doesn't already have them (see module docstring). Never raises on a
    solver failure -- crashed/diverged/non_converged are reported, not
    thrown -- but does raise if `case_dir` itself doesn't exist or the
    WSL invocation itself fails to run (as opposed to the solver running
    and failing).

    Returns:
        {"case_dir": str, "log_path": str, **parse_convergence_log(...)}

    Raises:
        FileNotFoundError: if `case_dir` doesn't exist.
    """
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(f"No such case dir: {case_dir}")

    _ensure_numerics(case_dir)

    log_path = os.path.join(case_dir, log_filename)
    case_wsl = _to_wsl_path(case_dir)
    result = _run_wsl(f'cd "{case_wsl}" && foamRun', timeout=timeout)
    log_text = result.stdout + result.stderr

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)

    parsed = parse_convergence_log(log_text)
    # Surfaced separately from `status`/`crash_reason`: a nonzero
    # returncode with zero parsed iterations means the WSL invocation
    # itself failed (bad path, foamRun not found) -- indistinguishable
    # from a genuine OpenFOAM config failure by log content alone, so
    # callers need the raw exit code to tell the two apart.
    return {
        "case_dir": case_dir,
        "log_path": log_path,
        "returncode": result.returncode,
        **parsed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: run foamRun and classify convergence.")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    result = run_case(args.case_dir, timeout=args.timeout)
    print(f"status: {result['status']}")
    print(f"converged: {result['converged']}")
    print(f"iterations: {result['iterations']}")
    print(f"final_residuals: {result['final_residuals']}")
