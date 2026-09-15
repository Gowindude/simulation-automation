"""
Pipeline integration tests: Stage 0 -> Stage 1 -> Stage 2, chained.

Each stage's own test suite (test_stage0_*, test_stage1_*, test_stage2_*)
builds its own hand-controlled input, so none of them exercise the
*handoffs* between stages -- e.g. Stage 1 re-deriving structure from
Stage 0's output instead of trusting its contract, or Stage 2's hardcoded
patch names silently drifting from what Stage 1 actually names its
physical groups. This file runs the real chain end-to-end for a few
geometrically diverse airfoils and asserts specifically at those seams.

Scope note: this confirms the pipeline produces a *well-formed* case up
through Stage 2, not a *runnable* one -- Stage 2's case has no
system/fvSchemes or system/fvSolution (that's Stage 3's concern, per the
spec's build order), so `foamRun` cannot execute it yet. "Works" here
means every stage's contract is honored and consistent with the next
stage's assumptions, not "solves."
"""

import os

import numpy as np
import pytest

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage1_mesh import run_stage1, check_mesh
from pipeline.stage2_case_gen import generate_case
from tests.test_stage0_geometry_loader import polygon_has_self_intersections

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# Geometrically diverse but small enough to run every time (Stage 1's own
# real-UIUC suite already covers breadth at ~150s; this adds Stage 2 on
# top of Stage 1, so kept to 3 rather than all 35).
CHAIN_AIRFOILS = [
    "naca0006.dat",  # very thin symmetric
    "naca2412.dat",  # moderate camber
    "naca6412.dat",  # high camber
]

AOA_DEG = 4.0
REYNOLDS = 5e5
NU = 1.5e-5


def fixture_path(name):
    return os.path.join(FIXTURES, name)


def _boundary_field_keys(case_dir, field_path):
    import subprocess

    def to_wsl(p):
        p = os.path.abspath(p)
        drive, rest = os.path.splitdrive(p)
        return f"/mnt/{drive.rstrip(':').lower()}{rest.replace(chr(92), '/')}"

    result = subprocess.run(
        ["wsl.exe", "--", "bash", "-lc",
         f'source /opt/openfoam12/etc/bashrc && '
         f'foamDictionary -case "{to_wsl(case_dir)}" -entry boundaryField '
         f'-keywords "{field_path}"'],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return set(result.stdout.split())


@pytest.fixture(scope="module", params=CHAIN_AIRFOILS)
def chained_result(request, tmp_path_factory):
    """Run Stage 0 -> Stage 1 -> Stage 2 for one airfoil, once."""
    fname = request.param
    name = fname.replace(".dat", "")
    out_dir = tmp_path_factory.mktemp(f"chain_{name}")

    coords = load_airfoil(fixture_path(fname))
    stage1 = run_stage1(coords, name, str(out_dir))
    stage2 = generate_case(
        mesh_case_dir=stage1["case_dir"],
        name=name,
        aoa_deg=AOA_DEG,
        reynolds=REYNOLDS,
        output_dir=str(out_dir),
        nu=NU,
    )
    return {"name": name, "coords": coords, "stage1": stage1, "stage2": stage2}


# --- Seam 1: Stage 0 -> Stage 1 -- resampled boundary is still valid -------


def test_resampled_boundary_stays_closed_and_non_self_intersecting(chained_result):
    from pipeline.stage1_mesh import _resample_cosine

    coords = chained_result["coords"]
    resampled = _resample_cosine(coords)

    # Mirror generate_mesh()'s actual usage exactly: gmsh points come from
    # coords[:-1], with the loop closed by wraparound (index % n), not by
    # appending resampled[0] again. That distinction matters here: the
    # spline's endpoint evaluation leaves resampled[-1] only
    # floating-point-close to resampled[0] (~1e-19), not bit-identical,
    # and appending resampled[0] as an *extra* point creates a spurious
    # near-zero-length edge that the intersection check's 1e-12
    # collinearity epsilon misreads as crossing its neighbors -- a false
    # positive from over-closing an already-closed loop, not a real
    # defect in the resampled geometry (confirmed empirically: it
    # disappears once the check matches the real gmsh usage below).
    mesh_pts = resampled[:-1]
    closed = np.vstack([mesh_pts, mesh_pts[0]])
    assert not polygon_has_self_intersections(closed), (
        f"{chained_result['name']}: cosine resampling for meshing introduced "
        "a self-intersection that Stage 0's own (already-passing) check "
        "never sees, since it only ever checks Stage 0's raw output"
    )


# --- Seam 2: Stage 1 -> Stage 2 -- patch names agree in both directions ---


def test_stage2_field_patches_match_stage1_mesh_patches(chained_result):
    from tests.test_stage2_case_gen import _list_boundary_patches

    case_dir = chained_result["stage2"]["case_dir"]
    mesh_patches = set(_list_boundary_patches(case_dir))

    for field in ["U", "p", "nuTilda", "nut"]:
        field_patches = _boundary_field_keys(case_dir, f"0/{field}")
        assert field_patches == mesh_patches, (
            f"{chained_result['name']}: 0/{field}'s boundaryField patches "
            f"{field_patches} don't match the mesh's actual patches "
            f"{mesh_patches} -- Stage 2's hardcoded patch list "
            "(pipeline/stage2_case_gen.py: _BOUNDARY_PATCHES) has drifted "
            "from what Stage 1 actually names its physical groups. "
            "OpenFOAM would not catch this until Stage 3 tries to solve."
        )


# --- Seam 3: the copied mesh is still a valid mesh, not a partial copy ----


def test_copied_mesh_still_passes_checkmesh_gate(chained_result):
    check = check_mesh(chained_result["stage2"]["case_dir"])
    assert check["negative_volume_cells"] == 0, check["raw_output"][-2000:]
    assert check["non_orthogonality_ok"], check["raw_output"][-2000:]
    assert check["skewness_ok"], check["raw_output"][-2000:]
    assert check["passed"], (
        f"{chained_result['name']}: constant/polyMesh no longer passes "
        "checkMesh after Stage 2's shutil.copytree -- byte-hashing just "
        "'points' (as test_stage2_case_gen.py does) would not catch a "
        "truncated/partial copy of faces/owner/neighbour/boundary"
    )


# --- Seam 4: Stage 2's Re -> U_inf conversion assumes unit chord ----------


def test_chord_is_unit_before_reynolds_conversion(chained_result):
    coords = chained_result["coords"]
    chord = coords[:, 0].max() - coords[:, 0].min()
    assert chord == pytest.approx(1.0, abs=1e-9), (
        f"{chained_result['name']}: Stage 0 output chord is {chord}, not 1.0 -- "
        "Stage 2's U_inf = Reynolds * nu is only correct because chord is "
        "assumed to be exactly 1.0 (Re = U*c/nu). If this ever fails, "
        "Stage 2's Reynolds derivation is silently wrong, not just this test."
    )

    U_inf = chained_result["stage2"]["U_inf"]
    assert U_inf == pytest.approx(REYNOLDS * NU, rel=1e-9)
