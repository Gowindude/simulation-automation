"""
Verification tests for Stage 1 -- CFD meshing (gmsh -> OpenFOAM).

Contract (from .claude/airfoil_pipeline_build_spec.md):
  Input:  Stage 0 coords array + domain sizing params (C-grid, far-field
          radius 15c, wake extent 20c)
  Output: a .msh file, converted to OpenFOAM polyMesh via gmshToFoam
  Must-Pass Gate #1: checkMesh reports zero negative-volume cells and
          passes OpenFOAM's own non-orthogonality/skewness checks
  Suggested check: mesh 5 geometrically diverse UIUC airfoils, confirm all
          pass checkMesh

These tests shell out to WSL (gmshToFoam, checkMesh) and are slower than
Stage 0's pure-Python tests.
"""

import os
import re

import numpy as np
import pytest

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage1_mesh import generate_mesh, convert_to_openfoam, check_mesh, run_stage1

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# 5 geometrically diverse airfoils, per the spec's suggested Stage 1 test.
DIVERSE_AIRFOILS = [
    "naca0006.dat",   # very thin symmetric
    "naca0021.dat",   # thick symmetric
    "naca0012.dat",   # baseline symmetric
    "naca2412.dat",   # moderate camber
    "naca6412.dat",   # high camber
]

FAR_FIELD_RADIUS = 15.0
WAKE_EXTENT = 20.0


def fixture_path(name):
    return os.path.join(FIXTURES, name)


@pytest.fixture(scope="module")
def stage1_results(tmp_path_factory):
    """Run the full Stage 1 pipeline once per airfoil, reused across tests."""
    out_dir = tmp_path_factory.mktemp("stage1")
    results = {}
    for fname in DIVERSE_AIRFOILS:
        name = fname.replace(".dat", "")
        coords = load_airfoil(fixture_path(fname))
        results[name] = run_stage1(
            coords, name, str(out_dir),
            far_field_radius=FAR_FIELD_RADIUS, wake_extent=WAKE_EXTENT,
        )
    return results


# --- 1. Mesh generation succeeds, non-empty output ---------------------------


@pytest.mark.parametrize("fname", DIVERSE_AIRFOILS)
def test_mesh_file_generated(stage1_results, fname):
    name = fname.replace(".dat", "")
    msh_path = stage1_results[name]["msh_path"]
    assert os.path.exists(msh_path)
    assert os.path.getsize(msh_path) > 0


# --- 2. gmshToFoam produces a valid polyMesh ---------------------------------


@pytest.mark.parametrize("fname", DIVERSE_AIRFOILS)
def test_polymesh_files_present(stage1_results, fname):
    name = fname.replace(".dat", "")
    case_dir = stage1_results[name]["case_dir"]
    poly_mesh = os.path.join(case_dir, "constant", "polyMesh")
    for required in ["points", "faces", "owner", "neighbour", "boundary"]:
        path = os.path.join(poly_mesh, required)
        assert os.path.exists(path), f"missing {required}"
        assert os.path.getsize(path) > 0


# --- 3. checkMesh passes the spec's Must-Pass Gate #1 ------------------------


@pytest.mark.parametrize("fname", DIVERSE_AIRFOILS)
def test_checkmesh_passes_gate(stage1_results, fname):
    name = fname.replace(".dat", "")
    check = stage1_results[name]["check"]
    assert check["negative_volume_cells"] == 0, check["raw_output"][-2000:]
    assert check["non_orthogonality_ok"], check["raw_output"][-2000:]
    assert check["skewness_ok"], check["raw_output"][-2000:]
    assert check["passed"]


# --- 4. Boundary patches are correctly typed ---------------------------------


def _read_boundary(case_dir):
    with open(os.path.join(case_dir, "constant", "polyMesh", "boundary")) as f:
        return f.read()


def _patch_type(boundary_text, name):
    match = re.search(rf"{name}\s*\{{[^}}]*?type\s+(\w+);", boundary_text, re.DOTALL)
    assert match, f"patch '{name}' not found"
    return match.group(1)


@pytest.mark.parametrize("fname", DIVERSE_AIRFOILS)
def test_boundary_patch_types(stage1_results, fname):
    name = fname.replace(".dat", "")
    case_dir = stage1_results[name]["case_dir"]
    boundary_text = _read_boundary(case_dir)

    assert _patch_type(boundary_text, "airfoil") == "wall"
    assert _patch_type(boundary_text, "farfield") == "patch"
    assert _patch_type(boundary_text, "front") == "empty"
    assert _patch_type(boundary_text, "back") == "empty"


# --- 5. Domain sizing sanity: bounding box reflects the locked params -------


@pytest.mark.parametrize("fname", DIVERSE_AIRFOILS)
def test_domain_bounding_box_matches_sizing(stage1_results, fname):
    name = fname.replace(".dat", "")
    check = stage1_results[name]["check"]
    match = re.search(
        r"Overall domain bounding box \(([-\d.]+) ([-\d.]+) [-\d.]+\) "
        r"\(([-\d.]+) ([-\d.]+) [-\d.]+\)",
        check["raw_output"],
    )
    assert match, "could not find bounding box in checkMesh output"
    x_min, y_min, x_max, y_max = (float(v) for v in match.groups())

    expected_x_max = 1.0 + WAKE_EXTENT  # wake extent measured from the TE (x=1)
    assert x_min == pytest.approx(-FAR_FIELD_RADIUS, abs=1e-3)
    assert y_min == pytest.approx(-FAR_FIELD_RADIUS, abs=1e-3)
    assert x_max == pytest.approx(expected_x_max, abs=1e-3)
    assert y_max == pytest.approx(FAR_FIELD_RADIUS, abs=1e-3)


# --- Single-case detail check: gmshToFoam raises on a bad path --------------


def test_convert_to_openfoam_raises_on_missing_mesh(tmp_path):
    with pytest.raises(RuntimeError):
        convert_to_openfoam(str(tmp_path / "does_not_exist.msh"), str(tmp_path / "case"))


# --- 6. Targeted retry on a diagnosable gmsh failure signature -------------
#
# Found probing Stage 1 for troubleshooter-agent scoping (2026-09-14/15):
# a real 40%-thick synthetic airfoil failed gmsh's boundary-layer
# extrusion with "Edge not recovered" / "intersections in the 1D mesh" --
# the BL offset self-overlapping near the LE's tight curvature. The
# existing blind retry ladder *increases* bl_size on every failure
# (`bl_size *= 2.0`), which makes this specific failure worse, not
# better -- confirmed empirically: reducing bl_size (1e-3 -> 3e-4) fixed
# the same geometry on the first attempt, and the resulting mesh passed
# checkMesh's Gate #1 cleanly (skewness 1.07, non-orthogonality 52.9 deg).
# These tests mock subprocess.run so they run in milliseconds -- no real
# gmsh/thick-airfoil regeneration needed to lock in the retry *direction*.


def _fake_gmsh_result(stdout, returncode=1):
    import subprocess as sp
    return sp.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_bl_self_intersection_signature_triggers_reduced_bl_size_retry(monkeypatch):
    import pipeline.stage1_mesh as stage1_mesh

    bl_self_intersection_output = (
        "Info    : [  0%] :-( There are 2 intersections in the 1D mesh "
        "(curves 444444 444444)\n"
        "MESH_ERROR: Edge not recovered: 98 118 444444\n"
    )
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        # cmd = [sys.executable, script_path] -- read back the bl_size
        # this attempt actually requested from the generated script.
        with open(cmd[1]) as f:
            script = f.read()
        match = re.search(r"bl_size\s*=\s*([\d.eE+-]+)", script)
        calls.append(float(match.group(1)))
        if len(calls) == 1:
            return _fake_gmsh_result(bl_self_intersection_output)
        return _fake_gmsh_result("MESH_SUCCESS", returncode=0)

    monkeypatch.setattr(stage1_mesh.subprocess, "run", fake_run)

    coords = load_airfoil(fixture_path("naca0021.dat"))
    stage1_mesh.generate_mesh(coords, "retry_test", str(_tmp_dir()), bl_size=1e-3, max_retries=2)

    assert len(calls) == 2
    assert calls[1] < calls[0], (
        f"BL self-intersection signature must reduce bl_size on retry "
        f"(got {calls[0]} -> {calls[1]}) -- the generic ladder increasing "
        f"it is exactly wrong for this failure class"
    )


def test_generic_failure_signature_keeps_existing_increase_ladder(monkeypatch):
    import pipeline.stage1_mesh as stage1_mesh

    generic_failure_output = "MESH_ERROR: some unrelated gmsh failure\n"
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        with open(cmd[1]) as f:
            script = f.read()
        match = re.search(r"bl_size\s*=\s*([\d.eE+-]+)", script)
        calls.append(float(match.group(1)))
        if len(calls) == 1:
            return _fake_gmsh_result(generic_failure_output)
        return _fake_gmsh_result("MESH_SUCCESS", returncode=0)

    monkeypatch.setattr(stage1_mesh.subprocess, "run", fake_run)

    coords = load_airfoil(fixture_path("naca0021.dat"))
    stage1_mesh.generate_mesh(coords, "retry_test2", str(_tmp_dir()), bl_size=1e-3, max_retries=2)

    assert len(calls) == 2
    assert calls[1] > calls[0], (
        "a failure signature that isn't the diagnosed BL self-intersection "
        "case must keep the existing (untargeted) increase-bl_size ladder"
    )


def _tmp_dir():
    import tempfile
    return tempfile.mkdtemp(prefix="stage1_retry_test_")
