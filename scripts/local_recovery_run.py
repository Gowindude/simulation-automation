"""
Local recovery pass: re-run any GitHub Actions matrix-run-failed airfoil
LOCALLY with the Stage 1 troubleshooter agent enabled.

Why local, not in CI: the troubleshooter shells out to the local `claude`
CLI, authenticated against this machine's Claude subscription -- a fresh
GH Actions runner has neither the CLI installed nor any auth session
(would need a paid ANTHROPIC_API_KEY secret, a different billing model,
to run there instead). So the deployed shape is: the cheap, fully
parallel GH Actions matrix run does the deterministic-only bulk pass (as
now), and whatever it can't mesh gets a local second pass here, for free
against the existing subscription -- exactly the manual workflow that
recovered 4/5 real failures the night this was built (STATUS.md,
2026-09-15), now a repeatable tool instead of a throwaway script.

Usage:
    # After downloading a matrix run's per-airfoil result artifacts
    # (gh run download <run-id> -D ci_results_downloaded):
    python scripts/local_recovery_run.py --ci-results-dir ci_results_downloaded

    # Or name specific airfoils directly:
    python scripts/local_recovery_run.py --airfoils ah93w480b as5048 whitcomb
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import run_single_airfoil


def find_failed_airfoils(ci_results_dir: str) -> list[str]:
    """Every airfoil whose downloaded result has status != "success" --
    both geometry_mesh rejections and any other failure category, since
    either is worth a real recovery attempt.

    Recursive glob: `gh run download <run-id> --pattern "result-*"`
    nests each per-job artifact in its own `result-<name>/<name>.json`
    subdirectory (confirmed against a real download, 2026-09-15), not a
    flat directory of .json files."""
    failed = []
    for path in sorted(glob.glob(os.path.join(ci_results_dir, "**", "*.json"), recursive=True)):
        with open(path) as f:
            r = json.load(f)
        if r.get("status") != "success":
            failed.append(r["name"])
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-results-dir", default=None, help="Directory of downloaded ci_results/*.json")
    parser.add_argument("--airfoils", nargs="*", default=None, help="Airfoil names to retry directly")
    parser.add_argument("--dat-dir", default=os.path.join("tests", "fixtures", "real_uiuc"))
    parser.add_argument("--out-dir", default=os.path.join(".orchestrator_runs", "local_recovery"))
    args = parser.parse_args()

    if args.airfoils:
        names = args.airfoils
    elif args.ci_results_dir:
        names = find_failed_airfoils(args.ci_results_dir)
    else:
        parser.error("must pass --ci-results-dir or --airfoils")

    if not names:
        print("Nothing to recover -- no failed airfoils found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "troubleshooter_log.jsonl")
    print(f"Recovering {len(names)} airfoil(s) locally with the troubleshooter enabled: {names}")

    results = {}
    for name in names:
        dat_path = os.path.join(args.dat_dir, f"{name}.dat")
        if not os.path.exists(dat_path):
            print(f"  {name}: SKIPPED -- no .dat fixture at {dat_path}")
            results[name] = {"status": "skipped", "reason": "no fixture"}
            continue
        spec = {
            "name": name, "dat_path": dat_path,
            "enable_troubleshooter": True,
            "troubleshooter_log_path": log_path,
        }
        result = run_single_airfoil(spec, args.out_dir)
        results[name] = {
            "status": result["status"],
            "n_converged": result.get("n_converged"),
            "n_total": result.get("n_total"),
            "failure_category": result.get("failure_category"),
        }
        print(f"  {name}: {results[name]}")

    n_recovered = sum(1 for r in results.values() if r["status"] == "success")
    print(f"\n{n_recovered}/{len(names)} recovered")
    with open(os.path.join(args.out_dir, "recovery_summary.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
