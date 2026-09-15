"""
Multi-airfoil orchestrator (build spec step 8): loop the full Stage 0-9
chain over a real airfoil list, in parallel where safe, with a durable
progress/failure manifest.

Two layers:

`run_single_airfoil(spec, output_dir)` -- the real Stage 0-9 chain for
one airfoil, already proven to work end-to-end for a single process (see
STATUS.md's single- and 3-airfoil confirmed runs). Composes the existing
stage modules directly; ordering no longer matters between WSL-dependent
stages and gmsh-dependent Stages 5/6, since those already isolate gmsh
in a spawned subprocess via `pipeline._gmsh_isolation.run_isolated()`.

Failure categorization follows the spec's own agent-scope split
(.claude/airfoil_pipeline_build_spec.md lines 7-14): Stage 0/1
(geometry/mesh) failures -- including checkMesh's Gate #1 reporting
failure rather than raising -- are the only category a future agent
layer is meant to recover from, so they're tagged "geometry_mesh"
distinctly from any other ("other") failure. A crash in one AoA's
CFD/FEA chain (solver-side, explicitly out of agent scope) does not
abort the airfoil -- it's recorded as that AoA's own "crashed" status
and the sweep continues, per the spec's "failed cases recorded
explicitly, never silently dropped."

`run_batch(airfoil_specs, output_dir, ...)` -- loops `run_single_airfoil`
(or an injected `run_one`, e.g. for testing) over a list of airfoils,
writing a manifest to `<output_dir>/batch_manifest.json` after every
airfoil so progress survives a crash. `max_workers=1` (default) runs a
plain sequential loop; `max_workers>1` dispatches via
`ProcessPoolExecutor`, one process per in-flight airfoil, each isolated
from the others' WSL/gmsh state the same way STATUS.md's "isolate each
airfoil's full pipeline run in its own subprocess" workaround already
validated -- and unlike `multiprocessing.Pool`, `ProcessPoolExecutor`
workers are not daemonic, so they're still allowed to spawn Stage 5/6's
own `run_isolated()` child process (see test_orchestrator.py's nesting
test). No concurrent-WSL production run has been validated yet in this
project, so treat `max_workers>1` as opt-in, not the default.
"""

import glob
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage1_mesh import run_stage1
from pipeline.stage2_case_gen import generate_case
from pipeline.stage3_run import run_case
from pipeline.stage4_postprocess import build_cfd_record, extract_surface_pressure
from pipeline.stage5_structural_geometry import generate_structural_geometry
from pipeline.stage6_structural_mesh import generate_structural_mesh
from pipeline.stage7_load_mapping import map_pressure_to_mesh
from pipeline.stage8_calculix_run import run_calculix_analysis
from pipeline.stage9_aggregation import aggregate_airfoil_record

DEFAULT_AOA_SWEEP_DEG = (-2.0, 2.0, 6.0, 10.0, 14.0)
DEFAULT_REYNOLDS = 5e5
DEFAULT_NU = 1.5e-5
DEFAULT_SPAN = 3.0
DEFAULT_SPAR_LOCATIONS = (0.2, 0.6)
DEFAULT_RIB_SPACING = 0.5


def discover_airfoils(dat_dir, **spec_defaults):
    """
    Build a list of airfoil specs from every `.dat` file in `dat_dir`,
    sorted by name for a reproducible, resumable batch order.
    """
    paths = sorted(glob.glob(os.path.join(dat_dir, "*.dat")))
    specs = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        spec = {"name": name, "dat_path": path}
        spec.update(spec_defaults)
        specs.append(spec)
    return specs


def _failure(name, category, exc):
    return {
        "name": name,
        "status": "failed",
        "h5_path": None,
        "failure_category": category,
        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
    }


def _crashed_cfd_record(exc):
    return {
        "status": "crashed",
        "Cl": None, "Cd": None, "Cl_xfoil": None, "Cd_xfoil": None,
        "xfoil_converged": None, "pressure_vs_arc_length": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_single_airfoil(spec, output_dir):
    """
    Run the full Stage 0-9 chain for one airfoil.

    spec: {"name", "dat_path", and optionally "reynolds", "nu",
           "aoa_sweep_deg", "span", "spar_locations", "rib_spacing"}.

    Returns a result dict, always -- never raises for a real pipeline
    failure (only for a broken spec, e.g. a missing "name"/"dat_path"):
        success: {"name", "status": "success", "h5_path", "n_converged",
                  "n_total", "per_aoa_results", "failure_category": None,
                  "error": None}
        failure: {"name", "status": "failed", "h5_path": None,
                  "failure_category": "geometry_mesh" | "other", "error"}
    """
    name = spec["name"]
    dat_path = spec["dat_path"]
    airfoil_dir = os.path.join(output_dir, name)
    os.makedirs(airfoil_dir, exist_ok=True)

    reynolds = spec.get("reynolds", DEFAULT_REYNOLDS)
    nu = spec.get("nu", DEFAULT_NU)
    aoa_sweep_deg = spec.get("aoa_sweep_deg", DEFAULT_AOA_SWEEP_DEG)
    span = spec.get("span", DEFAULT_SPAN)
    spar_locations = spec.get("spar_locations", DEFAULT_SPAR_LOCATIONS)
    rib_spacing = spec.get("rib_spacing", DEFAULT_RIB_SPACING)

    try:
        coords = load_airfoil(dat_path)
    except Exception as exc:
        return _failure(name, "geometry_mesh", exc)

    try:
        stage1 = run_stage1(coords, name, airfoil_dir)
    except Exception as exc:
        return _failure(name, "geometry_mesh", exc)
    if not stage1["check"]["passed"]:
        # Gate #1 is non-negotiable (spec Must-Pass Gates #1) -- a
        # failing checkMesh must never feed into Stage 2, so this is a
        # hard stop for the airfoil, not a per-AoA failure.
        return _failure(
            name, "geometry_mesh",
            RuntimeError(f"checkMesh gate failed: {stage1['check']}"),
        )

    struct_mesh = None  # built lazily, once, on the first converged AoA
    per_aoa_results = []

    for aoa_deg in aoa_sweep_deg:
        try:
            stage2 = generate_case(
                stage1["case_dir"], name, aoa_deg, reynolds, airfoil_dir, nu=nu,
            )
            stage3 = run_case(stage2["case_dir"])
            cfd_record = build_cfd_record(
                stage3, dat_path, name, aoa_deg, reynolds, stage2["U_inf"],
            )
        except Exception as exc:
            # Solver-side failure -- explicitly out of agent scope per
            # the spec (lines 11-13): logged and excluded from the
            # training set, not raised, so the rest of the sweep and
            # the rest of the batch keep going.
            per_aoa_results.append({"cfd": _crashed_cfd_record(exc), "fea": None})
            continue

        fea_record = None
        if cfd_record["status"] == "converged":
            try:
                if struct_mesh is None:
                    struct_geom = generate_structural_geometry(
                        coords, name, airfoil_dir, span, spar_locations, rib_spacing,
                    )
                    struct_mesh = generate_structural_mesh(
                        struct_geom["brep_path"], name, airfoil_dir,
                    )
                surface = extract_surface_pressure(stage3["case_dir"], stage2["U_inf"])
                loads = map_pressure_to_mesh(
                    surface, struct_mesh["inp_path"], name, airfoil_dir,
                    stage2["U_inf"], span,
                )
                fea_record = run_calculix_analysis(
                    struct_mesh["inp_path"], loads, name, airfoil_dir,
                )
            except Exception as exc:
                # Structural chain failed for a converged AoA -- keep the
                # cfd record (it's real), record no fea, don't abort the
                # sweep. Not "geometry_mesh": Stage 5/6 failing here is a
                # solve-time tool failure, not the Stage 0/1 airfoil-shape
                # class the spec scopes agent recovery to.
                cfd_record = dict(cfd_record)
                cfd_record["structural_error"] = f"{type(exc).__name__}: {exc}"

        per_aoa_results.append({"cfd": cfd_record, "fea": fea_record})

    try:
        agg = aggregate_airfoil_record(
            name=name, source_file=dat_path, reynolds=reynolds, span=span,
            spar_locations=spar_locations, rib_spacing=rib_spacing,
            aoa_sweep_deg=list(aoa_sweep_deg), per_aoa_results=per_aoa_results,
            output_dir=airfoil_dir,
        )
    except Exception as exc:
        return _failure(name, "other", exc)

    n_converged = sum(1 for r in per_aoa_results if r["cfd"]["status"] == "converged")
    return {
        "name": name,
        "status": "success",
        "h5_path": agg["h5_path"],
        "failure_category": None,
        "error": None,
        "n_total": len(per_aoa_results),
        "n_converged": n_converged,
        "per_aoa_results": per_aoa_results,
    }


def _run_and_wrap(run_one, spec, output_dir):
    """Guarantee a result dict even if run_one itself raises (e.g. a
    broken spec, or an uncaught bug in run_single_airfoil) -- a batch
    must never lose an airfoil's manifest entry to an exception."""
    try:
        return run_one(spec, output_dir)
    except Exception as exc:
        return _failure(spec["name"], "other", exc)


def _load_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def _write_manifest(manifest_path, manifest):
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp_path, manifest_path)


def _summarize_for_manifest(result):
    """
    The manifest is a progress/failure log, not a data store -- the
    `.h5` already holds the real per-AoA arrays (pressure curves,
    stress fields). A success result's `per_aoa_results` carries that
    same data (thousands of (s, Cp) pairs and, per STATUS.md, ~23k
    stress-field entries per converged AoA) -- keeping it in the
    manifest would mean re-serializing several MB of already-persisted
    data to JSON after every single airfoil in a batch. Reduce it to
    each AoA's status before writing.
    """
    summary = {k: v for k, v in result.items() if k != "per_aoa_results"}
    per_aoa_results = result.get("per_aoa_results")
    if per_aoa_results is not None:
        summary["per_aoa_status"] = [r["cfd"]["status"] for r in per_aoa_results]
        # Small diagnostic strings (a crashed AoA's exception message),
        # not the bulky pressure/stress arrays -- these are exactly
        # what a future troubleshooter agent (or a human triaging a
        # large batch) needs to tell a transient crash apart from a
        # real physics/mesh failure, so they're kept even though the
        # rest of per_aoa_results is dropped above.
        summary["per_aoa_errors"] = [
            r["cfd"].get("error") or r["cfd"].get("structural_error")
            for r in per_aoa_results
        ]
    return summary


def _is_done(manifest, name):
    entry = manifest.get(name)
    if entry is None or entry.get("status") != "success":
        return False
    h5_path = entry.get("h5_path")
    # A success entry pointing at a file that no longer exists is not
    # actually done -- the manifest is the completion authority, but
    # only in combination with the output it claims to have produced.
    return bool(h5_path) and os.path.exists(h5_path)


def run_batch(
    airfoil_specs, output_dir, run_one=run_single_airfoil,
    max_workers=1, resume=True, progress_callback=None,
):
    """
    Run `run_one` (default: the real `run_single_airfoil`) for every spec
    in `airfoil_specs`, tracking progress/failures in a manifest written
    to `<output_dir>/batch_manifest.json` after every completed airfoil.

    max_workers=1 (default): plain sequential loop, no process pool.
    max_workers>1: dispatches via ProcessPoolExecutor, one process per
    in-flight airfoil. Opt-in -- see module docstring.

    resume=True (default): airfoils the manifest already marks
    "success" (and whose h5_path still exists on disk) are skipped.

    Returns the full manifest dict, keyed by airfoil name.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "batch_manifest.json")
    manifest = _load_manifest(manifest_path) if resume else {}

    pending = [
        spec for spec in airfoil_specs
        if not (resume and _is_done(manifest, spec["name"]))
    ]

    def _record(result):
        manifest[result["name"]] = _summarize_for_manifest(result)
        _write_manifest(manifest_path, manifest)
        if progress_callback is not None:
            progress_callback(result)

    if max_workers <= 1:
        for spec in pending:
            _record(_run_and_wrap(run_one, spec, output_dir))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_and_wrap, run_one, spec, output_dir): spec
                for spec in pending
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    # Only reachable if the worker process itself died
                    # (e.g. segfault) rather than run_one raising --
                    # _run_and_wrap already catches ordinary exceptions.
                    result = _failure(spec["name"], "other", exc)
                _record(result)

    return manifest
