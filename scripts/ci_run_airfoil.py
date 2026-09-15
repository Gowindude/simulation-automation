"""
CI matrix worker: run ONE named airfoil (from tests/fixtures/real_uiuc/)
through the full Stage 0-9 pipeline, natively on a Linux runner.

Usage: python scripts/ci_run_airfoil.py <airfoil_name>

Deliberately does NOT fail the job just because this particular airfoil
didn't fully converge or was rejected at Stage 0/1 -- run_single_airfoil's
own contract is "never raises for a real pipeline failure" (see
pipeline/orchestrator.py), and a geometry_mesh rejection or a
non-converged AoA is expected, real production data (STATUS.md's own
precedent: 8/41 airfoils in an earlier local run had a non-5/5 AoA
outcome and that was correct, not a bug). Failing the CI job on every
such case would bury 91 jobs' worth of real signal under red X's for
outcomes that aren't infra problems. This job is red only if something
actually broke: an unhandled exception, or the result dict itself is
malformed.

Follows the two lessons from the single-airfoil smoke test incident
(2026-09-15): run as a real file (never piped via stdin -- breaks
multiprocessing's spawn-context re-import in pipeline/_gmsh_isolation.py),
and sys.path must include the repo root explicitly since `scripts/` is
not it.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import run_single_airfoil


def main():
    if len(sys.argv) != 2:
        print("usage: python scripts/ci_run_airfoil.py <airfoil_name>", file=sys.stderr)
        sys.exit(2)

    name = sys.argv[1]
    dat_path = os.path.join("tests", "fixtures", "real_uiuc", f"{name}.dat")
    if not os.path.exists(dat_path):
        print(f"no such fixture: {dat_path}", file=sys.stderr)
        sys.exit(2)

    spec = {"name": name, "dat_path": dat_path}
    result = run_single_airfoil(spec, os.path.join(".orchestrator_runs", "ci_matrix"))

    summary = {k: v for k, v in result.items() if k != "per_aoa_results"}
    print(json.dumps(summary, indent=2, default=str))

    os.makedirs("ci_results", exist_ok=True)
    with open(os.path.join("ci_results", f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Infra correctness only -- NOT "did this airfoil converge". A
    # well-formed result dict (regardless of its own status field) means
    # the harness itself worked; see module docstring.
    assert isinstance(result, dict) and "status" in result, f"malformed result: {result!r}"


if __name__ == "__main__":
    main()
