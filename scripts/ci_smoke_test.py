"""
CI smoke test: one real airfoil through the full Stage 0-9 pipeline.

Must be run as a real file (`python scripts/ci_smoke_test.py`), never
piped via `python - <<HEREDOC` (stdin) or `python -c`. Real incident
(2026-09-15, first GitHub Actions run): running this as a stdin-piped
script sets `__main__.__file__` to `<stdin>`, and pipeline/_gmsh_isolation
.run_isolated's spawned multiprocessing child crashes trying to
re-import that non-existent path -- before it can even report the
failure, which (before pipeline/_gmsh_isolation.py's own timeout fix
that same night) hung the parent for the rest of the job's timeout
rather than failing fast. Both fixes matter: this file must be real,
AND run_isolated must not hang forever even if something similar
recurs.
"""

import json
import os
import sys

# `python scripts/ci_smoke_test.py` puts scripts/ on sys.path, not the
# repo root -- `pipeline` wouldn't be importable otherwise. Same fix as
# scripts/build_dashboard_data.py already uses for this exact problem.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import run_single_airfoil


def main():
    spec = {"name": "naca0012", "dat_path": "tests/fixtures/naca0012.dat"}
    result = run_single_airfoil(spec, ".orchestrator_runs/ci_smoke")

    summary = {k: v for k, v in result.items() if k != "per_aoa_results"}
    print(json.dumps(summary, indent=2, default=str))

    assert result["status"] == "success", f"pipeline failed: {result.get('error')}"
    assert result["n_converged"] >= 1, "no AoA converged -- 0/5 is not a passing smoke test"
    assert result["h5_path"], "no .h5 written"


if __name__ == "__main__":
    main()
