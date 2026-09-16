"""
Orchestrator tests: looping over a real airfoil list, parallelization,
progress/failure tracking (build spec step 8: "loop over a handful of
additional airfoils for the demo", after single-airfoil 0-9 works).

Written before `pipeline/orchestrator.py` exists, per the spec's own
convention ("propose the verification tests ... before you build").

Two layers are tested separately, deliberately:

1. The *batch loop* (`run_batch`): manifest bookkeeping, resume,
   per-airfoil failure isolation, progress callbacks, sequential vs.
   parallel dispatch. Tested with an injected fake `run_one` so these
   tests run in milliseconds and don't touch WSL/gmsh at all.
2. The *single-airfoil chain and its failure categorization*
   (`run_single_airfoil`): Stage 0/1 (geometry/mesh) failures --
   including checkMesh's gate #1 reporting failure rather than raising
   -- must be distinguished from solver-side failures, per the spec's
   agent-scope split (.claude/airfoil_pipeline_build_spec.md lines
   11-13: only Stage 0/1 failures are agent-recoverable). Tested by
   monkeypatching the real stage functions `run_single_airfoil` calls,
   not by re-mocking the whole chain -- so these tests exercise the
   orchestrator's own control flow (which stage raised, was the mesh
   gate honored, does one bad AoA still let the airfoil complete)
   without needing a real WSL/gmsh environment.

One additional cheap, non-mocked test (`test_run_isolated_works_inside_a_process_pool_worker`)
directly verifies the nesting shape parallel batches depend on:
ProcessPoolExecutor workers (unlike multiprocessing.Pool workers) are
not daemonic, so they're allowed to spawn `run_isolated()`'s own child
process for Stage 5/6's gmsh work. This is the one place "all stages
individually pass, composition breaks" (this project's own failure
history, see STATUS.md) could bite a parallel batch silently.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pytest

from pipeline import orchestrator
from pipeline._gmsh_isolation import run_isolated


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture_path(name):
    return os.path.join(FIXTURES, name)


def make_spec(name, dat_name=None):
    return {"name": name, "dat_path": fixture_path(dat_name or f"{name}.dat")}


# --- run_batch: manifest / failure isolation / resume / progress ----------


def _fake_run_one_success(spec, output_dir):
    # Actually creates the h5 file it claims -- resume's "is this really
    # done" check looks at the filesystem, not just the manifest.
    airfoil_dir = os.path.join(output_dir, spec["name"])
    os.makedirs(airfoil_dir, exist_ok=True)
    h5_path = os.path.join(airfoil_dir, f"airfoil_{spec['name']}.h5")
    open(h5_path, "w").close()
    return {
        "name": spec["name"], "status": "success", "h5_path": h5_path,
        "failure_category": None, "error": None,
    }


def _fake_run_one_success_no_file(spec, output_dir):
    # Claims success but never writes the h5 -- simulates a manifest
    # entry that outlived its output.
    return {
        "name": spec["name"], "status": "success",
        "h5_path": os.path.join(output_dir, spec["name"], f"airfoil_{spec['name']}.h5"),
        "failure_category": None, "error": None,
    }


def _fake_run_one_fail_on_bad(spec, output_dir):
    if spec["name"] == "bad":
        return {
            "name": "bad", "status": "failed", "h5_path": None,
            "failure_category": "geometry_mesh", "error": "self-intersecting contour",
        }
    return _fake_run_one_success(spec, output_dir)


def _fake_run_one_raises_on_bad(spec, output_dir):
    if spec["name"] == "bad":
        raise RuntimeError("boom")
    return _fake_run_one_success(spec, output_dir)


def test_manifest_has_entry_for_every_airfoil_including_failures(tmp_path):
    specs = [make_spec("good1"), make_spec("bad"), make_spec("good2")]
    manifest = orchestrator.run_batch(
        specs, str(tmp_path), run_one=_fake_run_one_fail_on_bad,
    )
    assert set(manifest) == {"good1", "bad", "good2"}
    assert manifest["bad"]["status"] == "failed"
    assert manifest["bad"]["failure_category"] == "geometry_mesh"
    assert manifest["good1"]["status"] == "success"


def test_one_failing_airfoil_does_not_abort_batch(tmp_path):
    """A crashing run_one call for one airfoil must not stop later ones --
    matches the spec's "failed cases recorded explicitly, never silently
    dropped" applied at batch scope, not just per-AoA."""
    calls = []

    def run_one(spec, output_dir):
        calls.append(spec["name"])
        return _fake_run_one_raises_on_bad(spec, output_dir)

    specs = [make_spec("good1"), make_spec("bad"), make_spec("good2")]
    manifest = orchestrator.run_batch(specs, str(tmp_path), run_one=run_one)
    assert calls == ["good1", "bad", "good2"]
    assert manifest["bad"]["status"] == "failed"
    assert manifest["bad"]["failure_category"] == "other"
    assert "boom" in manifest["bad"]["error"]
    assert manifest["good1"]["status"] == "success"
    assert manifest["good2"]["status"] == "success"


def test_manifest_written_to_disk_incrementally(tmp_path):
    specs = [make_spec("good1"), make_spec("good2")]
    orchestrator.run_batch(specs, str(tmp_path), run_one=_fake_run_one_success)

    manifest_path = os.path.join(str(tmp_path), "batch_manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path) as f:
        on_disk = json.load(f)
    assert set(on_disk) == {"good1", "good2"}


def test_resume_skips_airfoils_already_recorded_successful(tmp_path):
    specs = [make_spec("good1"), make_spec("good2")]
    calls = []

    def counting_run_one(spec, output_dir):
        calls.append(spec["name"])
        return _fake_run_one_success(spec, output_dir)

    orchestrator.run_batch(specs, str(tmp_path), run_one=counting_run_one)
    assert calls == ["good1", "good2"]

    calls.clear()
    orchestrator.run_batch(specs, str(tmp_path), run_one=counting_run_one, resume=True)
    assert calls == [], "resume=True must skip airfoils the manifest already marks successful"


def test_resume_reruns_airfoil_whose_h5_is_missing(tmp_path):
    """The manifest is the completion authority, but a success entry
    pointing at a missing file must not be trusted blindly (e.g. the
    output dir was partially cleaned up) -- rerun rather than silently
    report success for data that isn't actually there."""
    specs = [make_spec("good1")]
    orchestrator.run_batch(specs, str(tmp_path), run_one=_fake_run_one_success_no_file)

    calls = []

    def counting_run_one(spec, output_dir):
        calls.append(spec["name"])
        return _fake_run_one_success_no_file(spec, output_dir)

    orchestrator.run_batch(specs, str(tmp_path), run_one=counting_run_one, resume=True)
    assert calls == ["good1"]


def test_resume_false_reruns_everything(tmp_path):
    specs = [make_spec("good1")]
    calls = []

    def counting_run_one(spec, output_dir):
        calls.append(spec["name"])
        return _fake_run_one_success(spec, output_dir)

    orchestrator.run_batch(specs, str(tmp_path), run_one=counting_run_one)
    orchestrator.run_batch(specs, str(tmp_path), run_one=counting_run_one, resume=False)
    assert calls == ["good1", "good1"]


def test_progress_callback_invoked_once_per_airfoil(tmp_path):
    specs = [make_spec("good1"), make_spec("bad"), make_spec("good2")]
    seen = []
    orchestrator.run_batch(
        specs, str(tmp_path), run_one=_fake_run_one_fail_on_bad,
        progress_callback=lambda result: seen.append(result["name"]),
    )
    assert sorted(seen) == ["bad", "good1", "good2"]


def test_max_workers_1_never_touches_process_pool(tmp_path, monkeypatch):
    def exploding_pool(*args, **kwargs):
        raise AssertionError("max_workers=1 must use a plain sequential loop, not a pool")

    monkeypatch.setattr(orchestrator, "ProcessPoolExecutor", exploding_pool)
    specs = [make_spec("good1"), make_spec("good2")]
    manifest = orchestrator.run_batch(
        specs, str(tmp_path), run_one=_fake_run_one_success, max_workers=1,
    )
    assert manifest["good1"]["status"] == "success"


def _fake_run_one_fat_result(spec, output_dir):
    # Mirrors what run_single_airfoil actually returns on success: a
    # per_aoa_results list carrying the real (large) per-AoA arrays,
    # plus a small diagnostic "error" string on the AoA that crashed
    # (exactly what run_single_airfoil's _crashed_cfd_record produces).
    airfoil_dir = os.path.join(output_dir, spec["name"])
    os.makedirs(airfoil_dir, exist_ok=True)
    h5_path = os.path.join(airfoil_dir, f"airfoil_{spec['name']}.h5")
    open(h5_path, "w").close()
    return {
        "name": spec["name"], "status": "success", "h5_path": h5_path,
        "failure_category": None, "error": None, "n_total": 2, "n_converged": 1,
        "per_aoa_results": [
            {
                "cfd": {"status": "converged", "pressure_vs_arc_length": [[0.0, 1.0]] * 5000},
                "fea": {"stress_field": {i: float(i) for i in range(5000)}},
            },
            {
                "cfd": {
                    "status": "crashed", "pressure_vs_arc_length": None,
                    "error": "RuntimeError: xfoil failed to run: timeout",
                },
                "fea": None,
            },
        ],
    }


def test_manifest_entry_excludes_bulky_per_aoa_arrays_but_keeps_error_text(tmp_path):
    """The manifest is a progress/failure log, not a second copy of the
    .h5's data -- a success result's per_aoa_results (thousands of
    pressure/stress entries per AoA, per STATUS.md) must not be
    serialized into the manifest on every airfoil completion, or a
    35-airfoil batch re-writes hundreds of MB of JSON.

    But a crashed AoA's small diagnostic error string is exactly the
    signal a future troubleshooter agent (or a human) needs to tell a
    transient WSL/concurrency crash apart from a real physics/mesh
    failure -- dropping it along with the bulky arrays would make the
    manifest useless for that, so it must survive the trim."""
    orchestrator.run_batch(
        [make_spec("good1")], str(tmp_path), run_one=_fake_run_one_fat_result,
    )
    with open(os.path.join(str(tmp_path), "batch_manifest.json")) as f:
        on_disk = json.load(f)
    entry = on_disk["good1"]
    assert "per_aoa_results" not in entry
    assert entry["per_aoa_status"] == ["converged", "crashed"]
    assert entry["per_aoa_errors"] == [None, "RuntimeError: xfoil failed to run: timeout"]
    assert entry["status"] == "success"


def test_discover_airfoils_builds_specs_from_directory():
    specs = orchestrator.discover_airfoils(FIXTURES)
    names = {s["name"] for s in specs}
    assert "naca0012" in names
    assert "naca2412" in names
    for s in specs:
        assert os.path.exists(s["dat_path"])
    # deterministic order matters for reproducible batch runs/resume logs
    assert names == {s["name"] for s in specs}
    assert [s["name"] for s in specs] == sorted(s["name"] for s in specs)


# --- run_batch: real parallel dispatch (module-level fake, picklable) -----


def _picklable_slow_success(spec, output_dir):
    time.sleep(1.5)
    return {
        "name": spec["name"], "status": "success", "h5_path": None,
        "failure_category": None, "error": None,
    }


def test_parallel_dispatch_actually_overlaps(tmp_path):
    # Compared against a real sequential run (not a fixed threshold):
    # ProcessPoolExecutor startup on Windows spawn is itself slow
    # (~1.5-2s observed just to spawn one worker and re-import
    # `pipeline.orchestrator`), so a tight absolute-time assertion is
    # flaky here. 1.5s of fake work per task swamps that fixed overhead
    # enough for the ratio to be a reliable signal instead.
    specs = [make_spec("a"), make_spec("b"), make_spec("c"), make_spec("d")]

    start = time.monotonic()
    orchestrator.run_batch(
        specs, str(tmp_path / "seq"), run_one=_picklable_slow_success, max_workers=1,
    )
    sequential_elapsed = time.monotonic() - start

    start = time.monotonic()
    manifest = orchestrator.run_batch(
        specs, str(tmp_path / "par"), run_one=_picklable_slow_success, max_workers=4,
    )
    parallel_elapsed = time.monotonic() - start

    assert manifest["a"]["status"] == "success"
    assert manifest["d"]["status"] == "success"
    assert parallel_elapsed < sequential_elapsed * 0.75, (
        f"parallel ({parallel_elapsed:.2f}s) not meaningfully faster than "
        f"sequential ({sequential_elapsed:.2f}s) -- looks like max_workers "
        "isn't actually overlapping tasks"
    )


# --- run_isolated nested inside a ProcessPoolExecutor worker --------------


def _trivial_isolated_fn(x):
    return x * 2


def _pool_worker_calls_run_isolated(x):
    return run_isolated(_trivial_isolated_fn, x)


def test_run_isolated_works_inside_a_process_pool_worker():
    """ProcessPoolExecutor workers are not daemonic (unlike
    multiprocessing.Pool workers), so they're allowed to spawn
    run_isolated()'s own child process. This is the real nesting shape
    a parallel batch uses: run_batch -> pool worker -> run_single_airfoil
    -> Stage 5/6 -> run_isolated(). If this test used
    multiprocessing.Pool instead, it would fail with
    "daemonic processes are not allowed to have children"."""
    with ProcessPoolExecutor(max_workers=1) as ex:
        result = ex.submit(_pool_worker_calls_run_isolated, 21).result(timeout=30)
    assert result == 42


# --- run_single_airfoil: failure categorization (monkeypatched stages) ----


def test_stage0_failure_is_categorized_as_geometry_mesh(monkeypatch, tmp_path):
    def raising_load_airfoil(dat_path):
        raise ValueError("self-intersecting contour")

    monkeypatch.setattr(orchestrator, "load_airfoil", raising_load_airfoil)
    result = orchestrator.run_single_airfoil(make_spec("naca0012"), str(tmp_path))
    assert result["status"] == "failed"
    assert result["failure_category"] == "geometry_mesh"
    assert "self-intersecting" in result["error"]


def test_checkmesh_gate_failure_halts_airfoil_as_geometry_mesh(monkeypatch, tmp_path):
    """run_stage1() reports a failing checkMesh gate in its return value,
    it does not raise -- the orchestrator must treat {"check": {"passed":
    False}} as a hard stop for that airfoil (Gate #1 is non-negotiable
    per the spec), not feed a bad mesh into Stage 2."""
    monkeypatch.setattr(orchestrator, "load_airfoil", lambda dat_path: object())

    def fake_run_stage1(coords, name, output_dir, **kwargs):
        return {"msh_path": "x.msh", "case_dir": "x_case", "check": {"passed": False, "negative_volume_cells": 3}}

    stage2_called = []
    monkeypatch.setattr(orchestrator, "run_stage1", fake_run_stage1)
    monkeypatch.setattr(orchestrator, "generate_case", lambda *a, **k: stage2_called.append(1))

    result = orchestrator.run_single_airfoil(make_spec("naca0012"), str(tmp_path))
    assert result["status"] == "failed"
    assert result["failure_category"] == "geometry_mesh"
    assert not stage2_called, "Stage 2 must never run against a mesh that failed checkMesh"


def test_enable_troubleshooter_is_forwarded_from_spec_to_run_stage1(monkeypatch, tmp_path):
    """Real gap found 2026-09-15: pipeline/troubleshooter.py's Stage 1
    agent (escalation + checkMesh-quality retries, both A/B tested for
    real that same night) was never actually reachable through
    run_single_airfoil/run_batch -- every real test of it called
    run_stage1 directly in a throwaway script. A spec that asks for it
    must actually receive it, or the yield improvement measured never
    reaches a real batch run."""
    monkeypatch.setattr(orchestrator, "load_airfoil", lambda dat_path: object())

    received_kwargs = {}

    def fake_run_stage1(coords, name, output_dir, **kwargs):
        received_kwargs.update(kwargs)
        return {"msh_path": "x.msh", "case_dir": "x_case", "check": {"passed": True}}

    monkeypatch.setattr(orchestrator, "run_stage1", fake_run_stage1)
    monkeypatch.setattr(orchestrator, "generate_case", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))

    spec = make_spec("naca0012")
    spec["enable_troubleshooter"] = True
    spec["troubleshooter_log_path"] = "some/log/path.jsonl"
    orchestrator.run_single_airfoil(spec, str(tmp_path))

    assert received_kwargs.get("enable_troubleshooter") is True
    assert received_kwargs.get("troubleshooter_log_path") == "some/log/path.jsonl"


def test_troubleshooter_defaults_to_disabled_when_not_in_spec(monkeypatch, tmp_path):
    """Opt-in, matching generate_mesh/run_stage1's own default -- a spec
    that doesn't ask for it must not silently enable real Claude usage."""
    monkeypatch.setattr(orchestrator, "load_airfoil", lambda dat_path: object())

    received_kwargs = {}

    def fake_run_stage1(coords, name, output_dir, **kwargs):
        received_kwargs.update(kwargs)
        return {"msh_path": "x.msh", "case_dir": "x_case", "check": {"passed": True}}

    monkeypatch.setattr(orchestrator, "run_stage1", fake_run_stage1)
    monkeypatch.setattr(orchestrator, "generate_case", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))

    orchestrator.run_single_airfoil(make_spec("naca0012"), str(tmp_path))

    assert received_kwargs.get("enable_troubleshooter") is False


def test_per_aoa_solver_crash_is_recorded_not_raised(monkeypatch, tmp_path):
    """A crash in Stage 2/3/4 for one AoA (solver-side, out of agent
    scope per the spec) must not abort the whole airfoil -- it's logged
    as that AoA's status and the sweep continues, per "failed cases
    recorded explicitly, never silently dropped."""
    monkeypatch.setattr(orchestrator, "load_airfoil", lambda dat_path: object())
    monkeypatch.setattr(
        orchestrator, "run_stage1",
        lambda coords, name, output_dir, **kwargs: {
            "msh_path": "x.msh", "case_dir": "x_case", "check": {"passed": True},
        },
    )

    calls = {"n": 0}

    def flaky_generate_case(mesh_case_dir, name, aoa_deg, reynolds, output_dir, nu=1.5e-5):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("foamRun WSL invocation failed")
        return {"case_dir": os.path.join(output_dir, "case"), "U_inf": 34.45}

    monkeypatch.setattr(orchestrator, "generate_case", flaky_generate_case)
    monkeypatch.setattr(
        orchestrator, "run_case",
        lambda case_dir, **k: {"status": "non_converged", "case_dir": case_dir},
    )
    monkeypatch.setattr(
        orchestrator, "build_cfd_record",
        lambda stage3_result, dat_path, name, aoa_deg, reynolds, U_inf: {
            "status": stage3_result["status"], "Cl": None, "Cd": None,
            "Cl_xfoil": None, "Cd_xfoil": None, "xfoil_converged": None,
            "pressure_vs_arc_length": None,
        },
    )
    monkeypatch.setattr(
        orchestrator, "aggregate_airfoil_record",
        lambda **kwargs: {"h5_path": os.path.join(kwargs["output_dir"], "airfoil_x.h5")},
    )

    spec = make_spec("naca0012")
    spec["aoa_sweep_deg"] = (0.0, 4.0)
    result = orchestrator.run_single_airfoil(spec, str(tmp_path))

    assert result["status"] == "success", "one bad AoA must not fail the whole airfoil"
    assert result["n_total"] == 2
    statuses = {r["cfd"]["status"] for r in result["per_aoa_results"]}
    assert "crashed" in statuses
    assert "non_converged" in statuses
