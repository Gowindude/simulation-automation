"""
Verification tests for Stage 8 -- CalculiX execution.

Contract (from .claude/airfoil_pipeline_build_spec.md, Stage 8, verbatim):
    Input: shell mesh + loads (from Stage 7) + material properties (pick
           a placeholder -- e.g. generic aluminum -- and flag as
           configurable)
    Output: .frd result file -> parsed stress field (von Mises, or
            whatever's most presentable for the demo)

Must-Pass Gate #4 (verbatim): "Static equilibrium (Stage 8): CalculiX
reaction forces balance the applied load resultant. A first-principles
check independent of material properties or mesh quality -- if it fails,
the boundary condition or load setup is wrong." -- `test_gate4_static_equilibrium`.

Design decisions locked before writing this suite:
  - Material: generic aluminum (E=70 GPa, nu=0.33, rho=2700 kg/m^3),
    uniform 2mm shell thickness across skin/spar/rib -- placeholders per
    the spec's own explicit allowance, both configurable parameters.
  - BC: cantilever -- full 6-DOF fixity at every node with z=0 (root),
    tip free. Standard, physically obvious choice for a spar/rib wing
    box "growing out of" a fuselage/wall; not a genuinely ambiguous call.
  - Per-element *DLOAD sign: CalculiX's shell pressure label "P" is
    positive ALONG the element's own connectivity (node-order) normal --
    confirmed empirically via a single-element probe deck's `.frd`
    displacement output (D3 came out positive for positive P, i.e. the
    same direction as the RH-rule connectivity normal). An earlier
    verbal read of that same probe output had the sign backwards; caught
    for real by Gate #4 itself failing (reaction ~197% off, matching a
    clean 2x/opposite-sign error, not a real physics discrepancy) before
    this suite was declared passing -- exactly the kind of "plausible-
    looking wrong stress field" the spec warns Gate #4 exists to catch.
    Stage 8 recomputes Stage 7's own outward-vs-raw-normal check and
    solves for the P value that reproduces Stage 7's intended physical
    force regardless of the mesh's (arbitrary, gmsh-assigned) per-
    element node ordering: P = -pressure_pa * sign.
  - A real, confirmed CalculiX behavior investigated empirically before
    writing this suite: a load applied directly at an already-BOUNDARY-
    constrained node/DOF does not fully appear in that node's printed
    reaction (verified via 6 independent minimal decks, cross-checked
    against .frd's own FORC record -- not a reporting bug, a real
    property of how CalculiX solves a system with a prescribed DOF).
    This only matters for elements with a corner exactly at the root
    (z=0); for a real structural mesh (thousands of elements) this is a
    small fraction of the total load, unlike the toy decks used to
    investigate it (where the root edge was most of the whole mesh).
    Gate #4's tolerance accounts for this rather than assuming it's
    exactly zero.
  - CalculiX invocation via WSL (`ccx`, confirmed installed: v2.17),
    same `wsl.exe` pattern as Stages 1/3/4.

Verification approach: a real (coarse, for speed) Stage 5->6 mesh +
Stage 7 run against a synthetic, hand-controlled Cp(s) curve (same
practice as Stage 7's own suite) so Gate #4 has a deterministic,
independently-known applied resultant to check CalculiX's reaction
against -- not a real WSL/OpenFOAM CFD solve, which Stage 7's tests
already establish is unnecessary for verifying the mapping/solve
mechanism itself.
"""

import os

import numpy as np
import pytest

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage5_structural_geometry import generate_structural_geometry
from pipeline.stage6_structural_mesh import generate_structural_mesh, parse_inp
from pipeline.stage7_load_mapping import map_pressure_to_mesh
from pipeline.stage8_calculix_run import run_calculix_analysis

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

SPAN = 3.0
SPAR_LOCATIONS = (0.2, 0.6)
RIB_SPACING = 0.5
MESH_SIZE = 0.2  # coarse, for fast CalculiX solves in this suite
U_INF = 7.5
RHO_AIR = 1.225

E_ALU = 70e9
NU_ALU = 0.33
RHO_ALU = 2700.0
THICKNESS = 0.002


def fixture_path(name):
    return os.path.join(FIXTURES, name)


def _cp_suction_peak(s):
    return -1.5 * np.exp(-3.0 * s)


def _make_synthetic_surface(coords, cp_func):
    x, y = coords[:-1, 0], coords[:-1, 1]
    seg = np.diff(coords[:, :2], axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seglen)])[:-1]
    Cp = np.array([cp_func(si) for si in s])
    return {"s": s, "x": x, "y": y, "Cp": Cp}


@pytest.fixture(scope="module")
def naca0012_coords():
    return load_airfoil(fixture_path("naca0012.dat"))


@pytest.fixture(scope="module")
def stage6_mesh(naca0012_coords, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage8_mesh")
    geom = generate_structural_geometry(
        naca0012_coords, "naca0012", str(out_dir),
        span=SPAN, spar_locations=SPAR_LOCATIONS, rib_spacing=RIB_SPACING,
    )
    return generate_structural_mesh(
        geom["brep_path"], "naca0012", str(out_dir), mesh_size=MESH_SIZE,
    )


@pytest.fixture(scope="module")
def stage7_loads(stage6_mesh, naca0012_coords, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage8_loads")
    surface = _make_synthetic_surface(naca0012_coords, _cp_suction_peak)
    return map_pressure_to_mesh(
        surface, stage6_mesh["inp_path"], "naca0012", str(out_dir),
        U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
    )


@pytest.fixture(scope="module")
def default_result(stage6_mesh, stage7_loads, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage8_run")
    return run_calculix_analysis(
        stage6_mesh["inp_path"], stage7_loads, "naca0012", str(out_dir),
        E=E_ALU, nu=NU_ALU, rho=RHO_ALU, thickness=THICKNESS,
    )


# --- 1. Output artifact validity --------------------------------------------


def test_output_artifacts_exist(default_result):
    assert os.path.exists(default_result["frd_path"])
    assert os.path.getsize(default_result["frd_path"]) > 0
    assert os.path.exists(default_result["dat_path"])
    assert os.path.getsize(default_result["dat_path"]) > 0


def test_max_von_mises_is_positive_finite(default_result):
    assert np.isfinite(default_result["max_von_mises_pa"])
    assert default_result["max_von_mises_pa"] > 0.0


def test_stress_field_covers_full_mesh_and_contains_the_max(default_result):
    """Stage 9's schema needs a full per-node stress array, not just the
    max -- confirm it's genuinely populated and self-consistent with the
    reported max/location, not a stub."""
    field = default_result["stress_field"]
    assert len(field) > 100, f"stress_field has only {len(field)} entries -- looks like a stub"
    assert all(v >= 0.0 for v in field.values())
    assert field[default_result["max_stress_node"]] == pytest.approx(
        default_result["max_von_mises_pa"], rel=1e-9
    )


# --- 2. Gate #4 -- static equilibrium, non-negotiable -----------------------


def test_gate4_static_equilibrium(default_result, stage7_loads):
    reaction = np.array(default_result["reaction_force_n"])
    applied = np.array(stage7_loads["resultant_force_n"])

    residual = reaction + applied
    residual_mag = float(np.linalg.norm(residual))
    applied_mag = float(np.linalg.norm(applied))

    assert applied_mag > 0.0, "test setup produced zero applied load"
    rel_residual = residual_mag / applied_mag
    assert rel_residual < 0.05, (
        f"Gate #4 failed: reaction {reaction} does not balance applied load "
        f"{-applied} (relative residual {rel_residual*100:.2f}%, expected "
        "near-zero, allowing a small margin for load fractions landing "
        "directly on root-constrained nodes -- see module docstring)"
    )


def test_reaction_force_is_nontrivial(default_result):
    """Sanity companion to Gate #4: the reaction shouldn't be
    coincidentally near-zero (which would make Gate #4 pass vacuously)."""
    reaction_mag = float(np.linalg.norm(default_result["reaction_force_n"]))
    assert reaction_mag > 1.0, (
        f"reaction magnitude {reaction_mag} N is suspiciously small -- "
        "Gate #4 could be passing vacuously rather than for a real reason"
    )


# --- 3. Physical sanity ------------------------------------------------------


def test_max_von_mises_physically_reasonable(default_result):
    # Generic aluminum yields around 100-500 MPa depending on alloy; a
    # generous upper bound (not a tight one) just to catch a gross
    # units/scaling error, not to validate real structural margins.
    assert default_result["max_von_mises_pa"] < 1e10, (
        f"max von Mises {default_result['max_von_mises_pa']:.3e} Pa is "
        "implausibly high for this load magnitude -- likely a units bug"
    )


def test_thinner_shell_increases_stress(stage6_mesh, stage7_loads, tmp_path_factory):
    """Halving the shell thickness (same load, same material) should
    increase peak stress -- a basic beam/shell-theory sanity check that
    the thickness parameter is actually wired into the section, not a
    dead knob."""
    out_dir = tmp_path_factory.mktemp("stage8_thickness")
    thick_result = run_calculix_analysis(
        stage6_mesh["inp_path"], stage7_loads, "naca0012_thick", str(out_dir),
        E=E_ALU, nu=NU_ALU, rho=RHO_ALU, thickness=THICKNESS,
    )
    thin_result = run_calculix_analysis(
        stage6_mesh["inp_path"], stage7_loads, "naca0012_thin", str(out_dir),
        E=E_ALU, nu=NU_ALU, rho=RHO_ALU, thickness=THICKNESS / 2.0,
    )
    assert thin_result["max_von_mises_pa"] > thick_result["max_von_mises_pa"], (
        "halving shell thickness did not increase peak stress -- "
        "thickness parameter may not be reaching the *SHELL SECTION cards"
    )


# --- Error handling ------------------------------------------------------------


def test_missing_mesh_path_raises(stage7_loads, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_calculix_analysis(
            str(tmp_path / "no_such.inp"), stage7_loads, "naca0012", str(tmp_path),
        )


def test_nonpositive_thickness_raises(stage6_mesh, stage7_loads, tmp_path):
    with pytest.raises(ValueError):
        run_calculix_analysis(
            stage6_mesh["inp_path"], stage7_loads, "naca0012", str(tmp_path),
            thickness=0.0,
        )


def test_nonpositive_youngs_modulus_raises(stage6_mesh, stage7_loads, tmp_path):
    with pytest.raises(ValueError):
        run_calculix_analysis(
            stage6_mesh["inp_path"], stage7_loads, "naca0012", str(tmp_path),
            E=0.0,
        )
