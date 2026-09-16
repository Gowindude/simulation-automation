"""
Verification tests for scripts/local_recovery_run.py's pure logic
(finding which airfoils a CI matrix run actually failed on, from
downloaded ci_results/*.json). The actual recovery pass itself
(run_single_airfoil with the troubleshooter enabled) reuses
orchestrator.run_single_airfoil directly -- already covered by
tests/test_orchestrator.py's troubleshooter-forwarding tests -- so this
suite only covers the new selection logic.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.local_recovery_run import find_failed_airfoils


def _write_result(dir_path, name, status, failure_category=None):
    with open(os.path.join(dir_path, f"{name}.json"), "w") as f:
        json.dump({"name": name, "status": status, "failure_category": failure_category}, f)


def test_finds_only_non_success_airfoils(tmp_path):
    _write_result(tmp_path, "good1", "success")
    _write_result(tmp_path, "bad1", "failed", "geometry_mesh")
    _write_result(tmp_path, "good2", "success")
    _write_result(tmp_path, "bad2", "failed", "other")

    failed = find_failed_airfoils(str(tmp_path))

    assert set(failed) == {"bad1", "bad2"}


def test_empty_dir_returns_empty_list(tmp_path):
    assert find_failed_airfoils(str(tmp_path)) == []


def test_all_success_returns_empty_list(tmp_path):
    _write_result(tmp_path, "good1", "success")
    _write_result(tmp_path, "good2", "success")

    assert find_failed_airfoils(str(tmp_path)) == []


def test_finds_results_nested_in_per_artifact_subdirectories(tmp_path):
    """Real structure from `gh run download <id> --pattern "result-*"`
    (confirmed 2026-09-15): each per-job artifact lands in its own
    result-<name>/<name>.json subdirectory, not a flat directory."""
    good_dir = tmp_path / "result-good1"
    good_dir.mkdir()
    _write_result(good_dir, "good1", "success")

    bad_dir = tmp_path / "result-bad1"
    bad_dir.mkdir()
    _write_result(bad_dir, "bad1", "failed", "geometry_mesh")

    failed = find_failed_airfoils(str(tmp_path))

    assert failed == ["bad1"]
