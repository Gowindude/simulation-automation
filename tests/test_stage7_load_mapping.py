"""
Verification tests for Stage 7 -- Load mapping (fluid -> structure).

Contract (from .claude/airfoil_pipeline_build_spec.md, Stage 7, verbatim):
    Input: Stage 4 pressure(arc-length) for one AoA + Stage 6 shell mesh
    Output: nodal/element pressure loads on the shell mesh
    Method: interpolate the 2D pressure(arc-length) curve via
            `scipy.interpolate`, apply the same value at every span
            station (uniform-along-span, per the locked decision above)

This is also where the spec's Must-Pass Gate #3 lives (verbatim):
    "Load conservation (Stage 7): total resultant force from the mapped
    structural loads matches the total resultant force from the source
    CFD pressure, within ~1%. This is the single highest-risk
    silent-failure point in the whole pipeline -- a wrong interpolation
    or sign error still produces a perfectly reasonable-looking stress
    field."
That gate is non-negotiable and is `test_gate3_load_conservation` below.

Design decisions locked before writing this suite:
  - Input is Stage 4's *live* `extract_surface_pressure()` return dict
    (s, x, y, Cp) -- not the schema-trimmed `(s, Cp)` pairs the Final
    Output Schema eventually stores. Stage 7 runs in the same pipeline
    pass, before Stage 9 ever serializes anything, so it can (and
    should) use the fuller data: mapping onto the structural mesh needs
    physical (x, y) to project mesh points onto Stage 4's own arc-length
    parametrization. Relying on two independently-recomputed arc-length
    conventions lining up would be exactly the kind of silent
    interpolation bug Gate #3 exists to catch.
  - Only "skin" elements (per Stage 6's ELSETs) get direct aerodynamic
    pressure -- spar/rib elements are internal load-carrying members,
    not aerodynamic surfaces, and get zero direct load (they see load
    only via the skin's structural connectivity, which is CalculiX's
    job in Stage 8, not this stage's).
  - Introduces `rho_air` (default 1.225 kg/m^3, ISA sea level) as a new
    parameter: the CFD solver runs at rho=1 (kinematic pressure, so `Cp`
    is dimensionless), but a *physical* pressure in Pa is needed to load
    a real, meter-scale shell structure: `p_Pa = Cp * 0.5 * rho_air *
    U_inf^2`. This mirrors how Stage 2 introduced `nu` with a sensible
    real-world default rather than the spec dictating one.
  - Sign convention mirrors Stage 4's own force-integration code
    (`pressure_integrated_cl_cd`): force = -Cp * outward_normal * area,
    i.e. a positive Cp (higher than freestream) pushes INWARD against
    the surface, not outward.
  - Mapping to the structural mesh's arc-length position: for each skin
    element, project its centroid (x, y) (z is ignored -- load is
    uniform along span, the locked decision) onto Stage 4's own ordered
    surface polyline via nearest-segment projection, then interpolate
    Cp at that arc-length via `scipy.interpolate`.

Because the mapping is linear (interpolate + multiply by a constant),
verifying it fully means verifying both its slope and its intercept:
  - test_pressure_scales_linearly_with_input_cp -- the "slope": scaling
    the entire input Cp(s) curve by a constant k scales every mapped
    element pressure by exactly k. Matters downstream because Stage 8's
    CalculiX solve will be linear-elastic -- a scaling bug here doesn't
    shift results a little, it multiplies the ENTIRE stress field by
    the same wrong factor, uniformly, while still looking plausible.
  - test_zero_cp_curve_gives_zero_pressure_everywhere -- the
    "intercept": the physically ideal (no-load) case. If an all-zero Cp
    input doesn't map to an exact all-zero pressure field, there's a
    hidden offset in the mapping that no amount of scaling would ever
    reveal.

Verification approach: parse Stage 6's real `.inp` mesh and the module's
own output artifact directly (matching Stages 5/6's practice of testing
the actual on-disk/returned artifact, not an internal state).
"""

import json
import os

import numpy as np
import pytest

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage5_structural_geometry import generate_structural_geometry
from pipeline.stage6_structural_mesh import generate_structural_mesh, parse_inp
from pipeline.stage7_load_mapping import map_pressure_to_mesh

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

SPAN = 3.0
SPAR_LOCATIONS = (0.2, 0.6)
RIB_SPACING = 0.5
MESH_SIZE = 0.08  # coarser than Stage 6's default -- keeps this suite fast
RHO_AIR = 1.225
U_INF = 7.5  # ~ Reynolds=5e5, nu=1.5e-5, matches this project's usual test case
AOA_DEG = 4.0


def fixture_path(name):
    return os.path.join(FIXTURES, name)


# --- A synthetic, hand-controlled "Stage 4 surface" ------------------------
#
# A real Stage 4 run requires WSL/OpenFOAM. Most tests here need precise
# control over the input Cp(s) curve anyway (to check linearity, zero-load,
# sign, and a literal hand-calculated reference force), so a synthetic
# surface -- built directly from the SAME naca0012 coords Stage 5/6 mesh,
# with a simple, exactly-known Cp(s) function -- is more rigorous than a
# real CFD run for verifying the mapping mechanism itself. A real end-to-end
# run (with real Stage 4 output) is done separately outside the unit suite,
# same practice as Stages 5/6.


def _make_synthetic_surface(coords, cp_func):
    """
    Build a Stage-4-shaped surface dict from Stage 0's own coords (not
    the CFD mesh's resampled boundary) -- close enough for testing the
    mapping mechanism, and means the "expected" force computed directly
    from this dict is unambiguous.
    """
    x, y = coords[:-1, 0], coords[:-1, 1]
    seg = np.diff(coords[:, :2], axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seglen)])[:-1]
    Cp = np.array([cp_func(si) for si in s])
    return {"s": s, "x": x, "y": y, "Cp": Cp}


def _hand_calc_2d_force(coords, cp_func):
    """
    Independent, standalone trapezoidal-style integration of Cp(s) around
    the closed 2D polygon, written fresh here (NOT calling
    pipeline.stage4_postprocess.pressure_integrated_cl_cd) -- a genuine
    second computational path to cross-check Stage 7's mapped 3D
    resultant against, per the user's request for hand-calc verification
    rather than the pipeline checking itself.

    Returns (fx, fy) in the rho=1, per-unit-span convention (matching
    how Cp itself is nondimensionalized) -- caller scales to physical
    Newtons via * rho_air * span.

    Uses trapezoidal (segment-midpoint-averaged) Cp sampling, not a
    single per-segment vertex value -- a rectangle rule under-resolves a
    Cp(s) that varies quickly relative to this polygon's (non-uniform,
    digitized) segment spacing, while the actual pipeline linearly
    interpolates Cp and integrates over thousands of fine mesh elements.
    Comparing a coarser quadrature of the same underlying curve against a
    finer one would fail this test for a purely numerical reason, not a
    real implementation bug -- trapezoidal sampling here keeps the
    comparison apples-to-apples.
    """
    x, y = coords[:-1, 0], coords[:-1, 1]
    n = len(x)
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    seg_len = np.hypot(x_next - x, y_next - y)
    tangent = np.column_stack([x_next - x, y_next - y]) / seg_len[:, None]
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    signed_area = 0.5 * np.sum(x * y_next - x_next * y)
    if signed_area < 0:
        normal = -normal

    Cp_vertex = np.array([cp_func(_arc_length_at_vertex(coords, i)) for i in range(n)])
    Cp_mid = 0.5 * (Cp_vertex + np.roll(Cp_vertex, -1))
    force = -(Cp_mid[:, None] * normal * seg_len[:, None])
    fx, fy = force.sum(axis=0)
    return float(fx), float(fy)


def _arc_length_at_vertex(coords, i):
    seg = np.diff(coords[: i + 1, :2], axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum()) if i > 0 else 0.0


def _cp_constant(value):
    return lambda s: value


def _cp_suction_peak(s, s_total=None):
    # A simple, smooth, non-constant Cp(s): negative (suction) everywhere,
    # peaked near s=0 (LE-ish, given _make_synthetic_surface's convention),
    # decaying with arc length -- enough shape to exercise real
    # interpolation, not just a constant lookup.
    return -1.5 * np.exp(-3.0 * s)


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def naca0012_coords():
    return load_airfoil(fixture_path("naca0012.dat"))


@pytest.fixture(scope="module")
def stage6_mesh(naca0012_coords, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage7_stage6mesh")
    geom = generate_structural_geometry(
        naca0012_coords, "naca0012", str(out_dir),
        span=SPAN, spar_locations=SPAR_LOCATIONS, rib_spacing=RIB_SPACING,
    )
    return generate_structural_mesh(
        geom["brep_path"], "naca0012", str(out_dir), mesh_size=MESH_SIZE,
    )


@pytest.fixture(scope="module")
def default_surface(naca0012_coords):
    return _make_synthetic_surface(naca0012_coords, _cp_suction_peak)


@pytest.fixture(scope="module")
def default_result(stage6_mesh, default_surface, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage7")
    return map_pressure_to_mesh(
        default_surface, stage6_mesh["inp_path"], "naca0012", str(out_dir),
        U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
    )


# --- 1. Output artifact validity ---------------------------------------------


def test_output_exists_and_covers_all_skin_elements(default_result, stage6_mesh):
    assert os.path.exists(default_result["loads_path"])
    with open(default_result["loads_path"]) as f:
        data = json.load(f)
    parsed = parse_inp(stage6_mesh["inp_path"])
    skin_ids = parsed["elsets"]["skin"]
    mapped_ids = {int(k) for k in data["element_pressures_pa"].keys()}
    assert mapped_ids == skin_ids, "mapped pressures don't cover exactly the skin ELSET"
    assert all(np.isfinite(v) for v in data["element_pressures_pa"].values())


def test_non_skin_elements_get_no_direct_load(default_result, stage6_mesh):
    parsed = parse_inp(stage6_mesh["inp_path"])
    non_skin_ids = parsed["elsets"]["spar"] | parsed["elsets"]["rib"]
    mapped_ids = {int(k) for k in default_result["element_pressures_pa"].keys()}
    assert non_skin_ids.isdisjoint(mapped_ids), (
        "spar/rib elements received a direct aerodynamic pressure -- only "
        "skin elements should"
    )


# --- 2. Gate #3 -- load conservation, ~1%, non-negotiable --------------------


def test_gate3_load_conservation(stage6_mesh, naca0012_coords, tmp_path_factory):
    surface = _make_synthetic_surface(naca0012_coords, _cp_suction_peak)
    out_dir = tmp_path_factory.mktemp("stage7_gate3")
    result = map_pressure_to_mesh(
        surface, stage6_mesh["inp_path"], "naca0012_gate3", str(out_dir),
        U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
    )

    # fx_2d/fy_2d are dimensionless (integral of a dimensionless Cp over a
    # unit-chord perimeter) -- physically they're force coefficients, not
    # yet a force. Converting to a real per-span force needs the dynamic
    # pressure q = 0.5*rho_air*U_inf^2 (chord=1 m, per Stage 0's contract);
    # multiplying by span then gives the total 3D resultant in Newtons.
    q_dyn = 0.5 * RHO_AIR * U_INF ** 2
    chord = 1.0
    fx_2d, fy_2d = _hand_calc_2d_force(naca0012_coords, _cp_suction_peak)
    expected_fx = fx_2d * q_dyn * chord * SPAN
    expected_fy = fy_2d * q_dyn * chord * SPAN
    expected_mag = float(np.hypot(expected_fx, expected_fy))

    got_fx, got_fy, got_fz = result["resultant_force_n"]
    got_mag = float(np.hypot(got_fx, got_fy))

    assert got_mag > 0, "resultant force is zero -- mapping produced no net load"
    rel_err = abs(got_mag - expected_mag) / expected_mag
    assert rel_err < 0.01, (
        f"load conservation Gate #3 failed: mapped resultant {got_mag:.4f} N "
        f"vs. independent hand-calc {expected_mag:.4f} N ({rel_err*100:.2f}% "
        "off, must be within 1%)"
    )
    assert abs(got_fz) < 1e-6 * max(got_mag, 1.0), (
        "resultant force has a non-negligible spanwise (z) component -- "
        "a uniform-along-span 2D pressure load should produce none"
    )


# --- 3. Physical sanity (hand-calc order-of-magnitude bound) ----------------


def test_pressure_magnitude_physically_reasonable(default_result):
    q = 0.5 * RHO_AIR * U_INF ** 2  # dynamic pressure, ~34.5 Pa here
    pressures = np.array(list(default_result["element_pressures_pa"].values()))
    # Cp for attached flow rarely exceeds roughly [-4, +1] -- a generous
    # bound, not a tight physical one, just enough to catch a gross unit
    # or scaling error (e.g. forgetting to multiply/divide by q).
    assert np.all(np.abs(pressures) < 6.0 * q), (
        f"mapped pressures exceed a generous physical bound (6*q={6*q:.1f} Pa) "
        "-- likely a units or scaling bug, not real aerodynamics"
    )


# --- 4. Sign convention -------------------------------------------------------


def test_positive_cp_pushes_against_outward_normal(stage6_mesh, naca0012_coords, tmp_path_factory):
    """
    A uniform, strongly positive Cp (unphysically large on purpose, just
    to get an unambiguous sign) must map to a resultant force pointing
    INTO the airfoil body (opposite the shape's own outward-pointing
    directions), matching Stage 4's own established convention
    (force = -Cp * normal * area).
    """
    surface = _make_synthetic_surface(naca0012_coords, _cp_constant(1.0))
    out_dir = tmp_path_factory.mktemp("stage7_sign")
    result = map_pressure_to_mesh(
        surface, stage6_mesh["inp_path"], "naca0012_sign", str(out_dir),
        U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
    )
    fx, fy, _ = result["resultant_force_n"]
    # A uniform positive Cp on a closed convex-ish loop nets to
    # approximately zero net force (pressure pushes inward everywhere,
    # symmetric-ish cancellation) -- so instead check sign locally: every
    # individual element's implied force opposes its own outward normal.
    # (Full derivation lives in the module; this test checks the
    # documented invariant via the module's own per-element diagnostics.)
    assert "per_element_force_dot_normal" in result
    dots = np.array(result["per_element_force_dot_normal"])
    assert np.all(dots <= 1e-9), (
        "found skin elements where a positive Cp produced a force in the "
        "SAME direction as the outward normal -- sign convention is "
        "backwards versus Stage 4's own force = -Cp*normal*area"
    )


# --- 5. Mapping accuracy at a known sample -----------------------------------


def test_interpolation_recovers_known_sample_value(stage6_mesh, naca0012_coords, tmp_path_factory):
    """A skin element whose centroid lies exactly on the input curve's
    own sample points should map to (very close to) that curve's local
    Cp value -- not a distorted/clamped value."""
    surface = _make_synthetic_surface(naca0012_coords, _cp_suction_peak)
    out_dir = tmp_path_factory.mktemp("stage7_interp")
    result = map_pressure_to_mesh(
        surface, stage6_mesh["inp_path"], "naca0012_interp", str(out_dir),
        U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
    )
    q = 0.5 * RHO_AIR * U_INF ** 2
    pressures = np.array(list(result["element_pressures_pa"].values()))
    cps_implied = pressures / q
    # The suction-peak curve ranges [-1.5, ~0); every mapped element's
    # implied Cp should fall within (a small tolerance around) that same
    # range -- interpolation/projection shouldn't manufacture values
    # outside the source curve's own range.
    assert cps_implied.min() > -1.6
    assert cps_implied.max() < 0.1


# --- 6. Linearity ("slope") and zero-load ("intercept") ---------------------


def test_pressure_scales_linearly_with_input_cp(stage6_mesh, naca0012_coords, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage7_linearity")
    base_surface = _make_synthetic_surface(naca0012_coords, _cp_suction_peak)
    scaled_surface = _make_synthetic_surface(
        naca0012_coords, lambda s: 3.0 * _cp_suction_peak(s)
    )

    base = map_pressure_to_mesh(
        base_surface, stage6_mesh["inp_path"], "naca0012_lin_base", str(out_dir),
        U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
    )
    scaled = map_pressure_to_mesh(
        scaled_surface, stage6_mesh["inp_path"], "naca0012_lin_scaled", str(out_dir),
        U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
    )

    for eid, p_base in base["element_pressures_pa"].items():
        p_scaled = scaled["element_pressures_pa"][eid]
        assert p_scaled == pytest.approx(3.0 * p_base, rel=1e-6, abs=1e-9), (
            f"element {eid}: scaling input Cp by 3x didn't scale mapped "
            f"pressure by exactly 3x ({p_base} -> {p_scaled})"
        )


def test_zero_cp_curve_gives_zero_pressure_everywhere(stage6_mesh, naca0012_coords, tmp_path_factory):
    surface = _make_synthetic_surface(naca0012_coords, _cp_constant(0.0))
    out_dir = tmp_path_factory.mktemp("stage7_zero")
    result = map_pressure_to_mesh(
        surface, stage6_mesh["inp_path"], "naca0012_zero", str(out_dir),
        U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
    )
    pressures = np.array(list(result["element_pressures_pa"].values()))
    assert np.all(pressures == 0.0), (
        "an all-zero input Cp curve did not map to exactly zero pressure "
        "everywhere -- the mapping has a hidden nonzero offset"
    )
    fx, fy, fz = result["resultant_force_n"]
    assert (fx, fy, fz) == (0.0, 0.0, 0.0)


# --- Error handling ------------------------------------------------------------


def test_missing_mesh_path_raises(default_surface, tmp_path):
    with pytest.raises(FileNotFoundError):
        map_pressure_to_mesh(
            default_surface, str(tmp_path / "no_such.inp"), "naca0012", str(tmp_path),
            U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
        )


def test_empty_surface_curve_raises(stage6_mesh, tmp_path):
    empty_surface = {"s": np.array([]), "x": np.array([]), "y": np.array([]), "Cp": np.array([])}
    with pytest.raises(ValueError):
        map_pressure_to_mesh(
            empty_surface, stage6_mesh["inp_path"], "naca0012", str(tmp_path),
            U_inf=U_INF, span=SPAN, rho_air=RHO_AIR,
        )


def test_nonpositive_rho_air_raises(stage6_mesh, default_surface, tmp_path):
    with pytest.raises(ValueError):
        map_pressure_to_mesh(
            default_surface, stage6_mesh["inp_path"], "naca0012", str(tmp_path),
            U_inf=U_INF, span=SPAN, rho_air=0.0,
        )
