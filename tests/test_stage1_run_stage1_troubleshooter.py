"""
Verification tests for run_stage1's checkMesh-quality-gate troubleshooter
retry (Change B, 2026-09-15 -- the second half of the A/B test the user
asked for alongside generate_mesh's escalation fix).

Contract (locked before implementation, per .claude/airfoil_pipeline_build_spec.md's
"propose the verification tests ... then agree on them before you build"):
  - Off by default (enable_troubleshooter=False): a checkMesh failure is
    returned as-is, exactly like before this change -- no retry, no
    agent call.
  - enable_troubleshooter=True + checkMesh fails: the agent is called
    with failure_kind="checkmesh_quality_gate" (not "gmsh_crash" --
    these are different failure classes) and the mesh is regenerated
    with its proposed params, then re-checked.
  - If a retry's new mesh passes checkMesh, the loop stops early and
    that passing check is returned.
  - If checkmesh_troubleshooter_max_retries is exhausted without a pass,
    the last (failing) check is returned -- never raises, matching
    run_stage1's existing "checkMesh failing is reported, not raised"
    contract.
  - An agent error (CLI unavailable, bad output) stops the retry loop
    and returns the last real check result, rather than crashing.

All gmsh/WSL calls are mocked -- no real solve needed for this suite.
"""

import os

import numpy as np
import pytest

import pipeline.stage1_mesh as stage1_mesh
from pipeline.stage1_mesh import run_stage1


def _naca0012_coords(n=41):
    x_upper = np.linspace(1, 0, n // 2 + 1)
    x_lower = np.linspace(0, 1, n - len(x_upper))
    t = 0.12
    y_upper = 5 * t * (0.2969 * np.sqrt(x_upper) - 0.1260 * x_upper - 0.3516 * x_upper**2
                       + 0.2843 * x_upper**3 - 0.1015 * x_upper**4)
    y_lower = -5 * t * (0.2969 * np.sqrt(x_lower) - 0.1260 * x_lower - 0.3516 * x_lower**2
                        + 0.2843 * x_lower**3 - 0.1015 * x_lower**4)
    x = np.concatenate([x_upper, x_lower])
    y = np.concatenate([y_upper, y_lower])
    return np.stack([x, y], axis=1)


def _failing_check(max_skewness=7.5):
    return {
        "passed": False, "negative_volume_cells": 0,
        "non_orthogonality_ok": False, "max_non_orthogonality_deg": 102.5,
        "skewness_ok": False, "max_skewness": max_skewness,
        "mesh_ok": False, "raw_output": "mock",
    }


def _passing_check():
    return {
        "passed": True, "negative_volume_cells": 0,
        "non_orthogonality_ok": True, "max_non_orthogonality_deg": 40.0,
        "skewness_ok": True, "max_skewness": 1.5,
        "mesh_ok": True, "raw_output": "mock",
    }


@pytest.fixture(autouse=True)
def _mock_mesh_pipeline(monkeypatch, tmp_path):
    """generate_mesh/convert_to_openfoam are mocked -- this suite tests
    run_stage1's own retry orchestration, not real gmsh/WSL behavior
    (already covered elsewhere)."""
    monkeypatch.setattr(stage1_mesh, "generate_mesh", lambda *a, **k: str(tmp_path / "mock.msh"))
    monkeypatch.setattr(stage1_mesh, "convert_to_openfoam", lambda *a, **k: None)


def test_disabled_by_default_returns_failing_check_as_is(monkeypatch, tmp_path):
    monkeypatch.setattr(stage1_mesh, "check_mesh", lambda *a, **k: _failing_check())
    diagnose_calls = []
    monkeypatch.setattr(stage1_mesh, "diagnose_mesh_failure", lambda *a, **k: diagnose_calls.append(1) or {})

    result = run_stage1(_naca0012_coords(), "t1", str(tmp_path))

    assert result["check"]["passed"] is False
    assert not diagnose_calls


def test_enabled_calls_agent_with_checkmesh_failure_kind(monkeypatch, tmp_path):
    checks = [_failing_check(), _passing_check()]
    monkeypatch.setattr(stage1_mesh, "check_mesh", lambda *a, **k: checks.pop(0))

    calls = []

    def fake_diagnose(desc, params, geometry_stats, timeout=90, failure_kind="gmsh_crash", previous_attempts=None):
        calls.append({"failure_kind": failure_kind, "desc": desc, "previous_attempts": previous_attempts})
        return {"reasoning": "test", "bl_size": 5e-4, "bl_layers": 8, "bl_ratio": 1.12}

    monkeypatch.setattr(stage1_mesh, "diagnose_mesh_failure", fake_diagnose)

    result = run_stage1(_naca0012_coords(), "t2", str(tmp_path), enable_troubleshooter=True)

    assert len(calls) == 1
    assert calls[0]["failure_kind"] == "checkmesh_quality_gate"
    assert "skewness" in calls[0]["desc"].lower()
    assert result["check"]["passed"] is True


def test_stops_early_once_a_retry_passes(monkeypatch, tmp_path):
    checks = [_failing_check(), _failing_check(), _passing_check(), _failing_check()]
    monkeypatch.setattr(stage1_mesh, "check_mesh", lambda *a, **k: checks.pop(0))
    monkeypatch.setattr(
        stage1_mesh, "diagnose_mesh_failure",
        lambda *a, **k: {"reasoning": "t", "bl_size": 5e-4, "bl_layers": 8, "bl_ratio": 1.12},
    )

    result = run_stage1(
        _naca0012_coords(), "t3", str(tmp_path),
        enable_troubleshooter=True, checkmesh_troubleshooter_max_retries=5,
    )

    assert result["check"]["passed"] is True
    assert len(checks) == 1  # the 4th, never-consumed check proves it stopped early


def test_exhausts_retries_and_returns_last_failing_check(monkeypatch, tmp_path):
    monkeypatch.setattr(stage1_mesh, "check_mesh", lambda *a, **k: _failing_check())
    call_count = {"n": 0}

    def fake_diagnose(desc, params, geometry_stats, timeout=90, failure_kind="gmsh_crash", previous_attempts=None):
        call_count["n"] += 1
        return {"reasoning": "t", "bl_size": 5e-4, "bl_layers": 8, "bl_ratio": 1.12}

    monkeypatch.setattr(stage1_mesh, "diagnose_mesh_failure", fake_diagnose)

    result = run_stage1(
        _naca0012_coords(), "t4", str(tmp_path),
        enable_troubleshooter=True, checkmesh_troubleshooter_max_retries=3,
    )

    assert result["check"]["passed"] is False
    assert call_count["n"] == 3


def test_agent_error_stops_loop_without_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(stage1_mesh, "check_mesh", lambda *a, **k: _failing_check())

    def raising_diagnose(*a, **k):
        raise RuntimeError("agent unavailable")

    monkeypatch.setattr(stage1_mesh, "diagnose_mesh_failure", raising_diagnose)

    result = run_stage1(
        _naca0012_coords(), "t5", str(tmp_path),
        enable_troubleshooter=True, checkmesh_troubleshooter_max_retries=3,
    )

    assert result["check"]["passed"] is False  # returned, not raised


def test_regenerate_mesh_raising_during_retry_does_not_crash_run_stage1(monkeypatch, tmp_path):
    """Real bug found 2026-09-15 by running against actual failing
    geometries (ah79100b): the agent's proposed params can themselves
    cause generate_mesh to raise a gmsh crash on the retry attempt --
    that must be treated as "this retry attempt failed," not propagate
    up and crash run_stage1 entirely."""
    checks = [_failing_check(), _failing_check()]  # initial check + 1st retry's check; 2nd retry: no more mesh to check (crashed) or exhausted -> passing
    monkeypatch.setattr(stage1_mesh, "check_mesh", lambda *a, **k: checks.pop(0) if checks else _passing_check())

    call_count = {"n": 0}

    def flaky_generate_mesh(*a, **k):
        call_count["n"] += 1
        if call_count["n"] == 2:  # 1st call = initial mesh (succeeds); 2nd = first retry (crashes)
            raise RuntimeError("Failed to mesh 'x' after 1 attempts.\nEdge not recovered")
        return str(tmp_path / "mock.msh")

    monkeypatch.setattr(stage1_mesh, "generate_mesh", flaky_generate_mesh)
    monkeypatch.setattr(
        stage1_mesh, "diagnose_mesh_failure",
        lambda *a, **k: {"reasoning": "t", "bl_size": 5e-4, "bl_layers": 8, "bl_ratio": 1.12},
    )

    result = run_stage1(
        _naca0012_coords(), "t7", str(tmp_path),
        enable_troubleshooter=True, checkmesh_troubleshooter_max_retries=3,
    )

    # Must not raise -- the first retry's crash is absorbed, and the
    # loop continues to later retries that eventually succeed.
    assert result["check"]["passed"] is True
    assert call_count["n"] == 4  # initial + crashed retry + 2 more successful retries


def test_attempt_history_grows_across_retries(monkeypatch, tmp_path):
    checks = [_failing_check(), _failing_check(), _passing_check()]
    monkeypatch.setattr(stage1_mesh, "check_mesh", lambda *a, **k: checks.pop(0))

    seen_history_lengths = []

    def fake_diagnose(desc, params, geometry_stats, timeout=90, failure_kind="gmsh_crash", previous_attempts=None):
        seen_history_lengths.append(len(previous_attempts or []))
        return {"reasoning": "t", "bl_size": 5e-4, "bl_layers": 8, "bl_ratio": 1.12}

    monkeypatch.setattr(stage1_mesh, "diagnose_mesh_failure", fake_diagnose)

    run_stage1(
        _naca0012_coords(), "t6", str(tmp_path),
        enable_troubleshooter=True, checkmesh_troubleshooter_max_retries=5,
    )

    # First call: nothing tried yet (empty history). Second call: one
    # real prior attempt (params + outcome), from the first retry.
    assert seen_history_lengths == [0, 1]
