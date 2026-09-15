"""
Verification tests for Stage 5 -- Structural geometry generation.

Contract (from .claude/airfoil_pipeline_build_spec.md):
  Input:  Stage 0's normalized 2D airfoil coords (unit-chord,
          Selig-ordered, closed) + span, spar locations, rib spacing
          (all "locked decisions", not re-litigated here):
            span = 3.0 x chord
            spar_locations = (0.2, 0.6) x chord
            rib_spacing = 0.5 x chord
  Output: 3D shell geometry (NOT a mesh -- Stage 6's job) --
            - skin: airfoil boundary extruded along the span
            - spar webs: internal shell surfaces at the two spar chord
              fractions, spanning the full span, bounded vertically by
              the actual airfoil thickness at that chord fraction (the
              cross-section is identical at every span station, per the
              locked "replicate uniformly along span" simplification --
              there is no spanwise variation to bound them differently)
            - rib bulkheads: airfoil-shaped filled surfaces at each rib
              span station (root through tip inclusive)
  This is a different tool/geometry representation than Stage 1's CFD
  domain (a 2D-then-extruded VOLUME mesh for OpenFOAM) -- Stage 5 is pure
  B-rep surface geometry, no mesh, matching the spec's explicit
  Stage5/Stage6 split (geometry, then meshing -- mirroring Stage
  0/Stage 1's split on the CFD side).

Design decisions locked before writing this suite:
  - Tool: gmsh's OCC kernel (same as Stage 1), output as a STEP file --
    interoperable, inspectable, and the natural format for "just
    geometry, no mesh".
  - Rib placement: one rib at every multiple of rib_spacing from z=0
    (root) through z=span (tip), inclusive of both ends. With the
    locked defaults (span=3.0, spacing=0.5) that's exactly 7 ribs
    (0, 0.5, ..., 3.0) -- evenly divisible, so no rounding/remainder
    behavior is exercised by the defaults alone; a non-exact ratio is
    tested explicitly below (rounds to the nearest station count that
    keeps spacing <= the requested value, always including the tip).
  - Spar web extent: full span in z; in y, bounded by the airfoil's own
    upper/lower surface coordinates at that x -- NOT a fixed height.

Verification approach: reload the generated STEP file in a fresh gmsh
session and classify surfaces by their own bounding-box geometry (ribs:
collapsed in z; spar webs: collapsed in x; skin: neither) rather than by
internal physical-group names, since STEP does not preserve gmsh's
physical groups -- this tests the actual on-disk artifact any downstream
consumer (Stage 6) would load, not this module's internal state.
"""

import os

import numpy as np
import pytest
import gmsh

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage5_structural_geometry import generate_structural_geometry

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

SPAN = 3.0
SPAR_LOCATIONS = (0.2, 0.6)
RIB_SPACING = 0.5
BBOX_TOL = 1e-6


def fixture_path(name):
    return os.path.join(FIXTURES, name)


def _load_step_surfaces(step_path):
    """
    Reload a STEP file in an isolated gmsh session and return a bounding
    box per surface (dim=2) entity. Always finalizes gmsh, even on error,
    so a failed assertion doesn't leave a stray session for the next test.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("verify")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(dim=2)
        result = []
        for _, tag in surfaces:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            result.append({
                "tag": tag,
                "xmin": xmin, "xmax": xmax,
                "ymin": ymin, "ymax": ymax,
                "zmin": zmin, "zmax": zmax,
            })
        return result
    finally:
        gmsh.finalize()


def _classify(surfaces, span_):
    """Classify surfaces into ribs (z-collapsed), spars (x-collapsed), skin (neither)."""
    ribs, spars, skin = [], [], []
    for s in surfaces:
        z_collapsed = (s["zmax"] - s["zmin"]) < BBOX_TOL
        x_collapsed = (s["xmax"] - s["xmin"]) < BBOX_TOL
        if z_collapsed:
            ribs.append(s)
        elif x_collapsed:
            spars.append(s)
        else:
            skin.append(s)
    return ribs, spars, skin


@pytest.fixture(scope="module")
def naca0012_coords():
    return load_airfoil(fixture_path("naca0012.dat"))


@pytest.fixture(scope="module")
def default_geometry(naca0012_coords, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage5")
    return generate_structural_geometry(
        naca0012_coords, "naca0012", str(out_dir),
        span=SPAN, spar_locations=SPAR_LOCATIONS, rib_spacing=RIB_SPACING,
    )


@pytest.fixture(scope="module")
def default_surfaces(default_geometry):
    return _load_step_surfaces(default_geometry["step_path"])


# --- 1. Geometry validity ----------------------------------------------------


def test_step_file_exists_and_reloads(default_geometry, default_surfaces):
    assert os.path.exists(default_geometry["step_path"])
    assert os.path.getsize(default_geometry["step_path"]) > 0
    assert len(default_surfaces) > 0


def test_surface_counts_match_expected(default_geometry, default_surfaces, naca0012_coords):
    ribs, spars, skin = _classify(default_surfaces, SPAN)
    expected_n_ribs = len(default_geometry["rib_stations"])
    assert len(ribs) == expected_n_ribs
    assert len(spars) == len(SPAR_LOCATIONS)
    # Skin is one lateral surface per boundary edge of the (closed) input
    # loop -- N points closed into a loop gives N-1 edges (coords[0] ==
    # coords[-1] per Stage 0's contract).
    assert len(skin) == len(naca0012_coords) - 1


def test_no_degenerate_surfaces(default_surfaces):
    for s in default_surfaces:
        # Every surface must have SOME extent in at least two axes --
        # a truly degenerate (zero-area) surface would collapse in all three.
        spans = [s["xmax"] - s["xmin"], s["ymax"] - s["ymin"], s["zmax"] - s["zmin"]]
        assert sum(sp > BBOX_TOL for sp in spans) >= 2, f"degenerate surface: {s}"


# --- 2. Dimensional correctness ----------------------------------------------


def test_skin_spans_full_z_range(default_surfaces, naca0012_coords):
    _, _, skin = _classify(default_surfaces, SPAN)
    z_mins = [s["zmin"] for s in skin]
    z_maxs = [s["zmax"] for s in skin]
    assert min(z_mins) == pytest.approx(0.0, abs=1e-6)
    assert max(z_maxs) == pytest.approx(SPAN, abs=1e-6)
    # Every individual skin strip spans the FULL span (pure translation
    # extrusion of the boundary, not stretched/tapered) -- not just the
    # aggregate min/max across all strips.
    for s in skin:
        assert s["zmin"] == pytest.approx(0.0, abs=1e-6)
        assert s["zmax"] == pytest.approx(SPAN, abs=1e-6)


def test_skin_cross_section_matches_stage0_coords(default_surfaces, naca0012_coords):
    _, _, skin = _classify(default_surfaces, SPAN)
    x_all = [s["xmin"] for s in skin] + [s["xmax"] for s in skin]
    y_all = [s["ymin"] for s in skin] + [s["ymax"] for s in skin]
    assert min(x_all) == pytest.approx(naca0012_coords[:, 0].min(), abs=1e-6)
    assert max(x_all) == pytest.approx(naca0012_coords[:, 0].max(), abs=1e-6)
    assert min(y_all) == pytest.approx(naca0012_coords[:, 1].min(), abs=1e-6)
    assert max(y_all) == pytest.approx(naca0012_coords[:, 1].max(), abs=1e-6)


# --- 3. Spar web placement & bounding ----------------------------------------


def _dedupe_sort_by_x(pts):
    """Sort by x and average y over any near-duplicate x (digitization
    noise near the LE would otherwise violate np.interp's strictly-
    increasing-x requirement and silently produce garbage)."""
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    xs, ys = [pts[0, 0]], [pts[0, 1]]
    for x, y in pts[1:]:
        if x - xs[-1] < 1e-9:
            ys[-1] = (ys[-1] + y) / 2.0
        else:
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)


def _airfoil_y_bounds_at_x(coords, x_frac):
    """Reference upper/lower y at a given x, via linear interpolation on
    Stage 0's own coords (independent of Stage 5's implementation)."""
    le_idx = int(np.argmin(coords[:, 0]))
    upper = coords[: le_idx + 1][::-1]  # LE -> TE, x increasing
    lower = coords[le_idx:]             # LE -> TE, x increasing
    upper_x, upper_y = _dedupe_sort_by_x(upper)
    lower_x, lower_y = _dedupe_sort_by_x(lower)
    y_upper = np.interp(x_frac, upper_x, upper_y)
    y_lower = np.interp(x_frac, lower_x, lower_y)
    return y_lower, y_upper


@pytest.mark.parametrize("spar_x", SPAR_LOCATIONS)
def test_spar_web_placement_and_bounds(default_surfaces, naca0012_coords, spar_x):
    _, spars, _ = _classify(default_surfaces, SPAN)
    matches = [s for s in spars if abs(s["xmin"] - spar_x) < 1e-6]
    assert len(matches) == 1, f"expected exactly one spar web at x={spar_x}"
    spar = matches[0]

    assert spar["zmin"] == pytest.approx(0.0, abs=1e-6)
    assert spar["zmax"] == pytest.approx(SPAN, abs=1e-6)

    y_lower_ref, y_upper_ref = _airfoil_y_bounds_at_x(naca0012_coords, spar_x)
    assert spar["ymin"] == pytest.approx(y_lower_ref, abs=1e-4)
    assert spar["ymax"] == pytest.approx(y_upper_ref, abs=1e-4)


# --- 4. Rib station placement -------------------------------------------------


def test_rib_surfaces_at_expected_z_stations(default_geometry, default_surfaces):
    ribs, _, _ = _classify(default_surfaces, SPAN)
    rib_zs = sorted(s["zmin"] for s in ribs)  # zmin == zmax for a z-collapsed surface
    expected_zs = sorted(default_geometry["rib_stations"])
    assert len(rib_zs) == len(expected_zs)
    for actual, expected in zip(rib_zs, expected_zs):
        assert actual == pytest.approx(expected, abs=1e-6)
    # Root and tip are always included, per the locked convention.
    assert rib_zs[0] == pytest.approx(0.0, abs=1e-6)
    assert rib_zs[-1] == pytest.approx(SPAN, abs=1e-6)


def test_rib_cross_section_matches_stage0_coords(default_surfaces, naca0012_coords):
    ribs, _, _ = _classify(default_surfaces, SPAN)
    for rib in ribs:
        assert rib["xmin"] == pytest.approx(naca0012_coords[:, 0].min(), abs=1e-6)
        assert rib["xmax"] == pytest.approx(naca0012_coords[:, 0].max(), abs=1e-6)
        assert rib["ymin"] == pytest.approx(naca0012_coords[:, 1].min(), abs=1e-6)
        assert rib["ymax"] == pytest.approx(naca0012_coords[:, 1].max(), abs=1e-6)


def test_rib_station_rounding_for_non_exact_span(naca0012_coords, tmp_path):
    """
    span=2.2 with rib_spacing=0.5 is NOT an exact multiple (2.2/0.5=4.4).
    Locked convention: always include the tip, so this must NOT silently
    drop the last partial segment or overshoot past the actual tip.
    """
    result = generate_structural_geometry(
        naca0012_coords, "naca0012_nonexact", str(tmp_path),
        span=2.2, spar_locations=SPAR_LOCATIONS, rib_spacing=0.5,
    )
    stations = sorted(result["rib_stations"])
    assert stations[0] == pytest.approx(0.0, abs=1e-6)
    assert stations[-1] == pytest.approx(2.2, abs=1e-6)
    # Strictly increasing, no duplicate/near-duplicate station from
    # rounding (e.g. a station at 2.0 AND one at 2.2 that's really the
    # same point within tolerance would indicate a rounding bug).
    assert all(b - a > 1e-6 for a, b in zip(stations, stations[1:]))


# --- 5. Topological connectivity (CAD-level, not mesh-level) ----------------


def _load_brep_surfaces_with_boundary(brep_path):
    """
    Reload a .brep file (gmsh-native OCC serialization -- unlike STEP,
    guaranteed to preserve the fragmented B-rep topology exactly as OCC
    held it in memory) and return each surface's bounding box plus the
    set of curve tags on its boundary. This lets the test derive shared
    topology itself (matching curve tags between surfaces) instead of
    trusting a count the implementation reports about itself.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("verify_brep")
        gmsh.model.occ.importShapes(brep_path)
        gmsh.model.occ.synchronize()
        result = []
        for _, tag in gmsh.model.getEntities(dim=2):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            boundary = gmsh.model.getBoundary([(2, tag)], combined=False, oriented=False)
            curve_tags = {t for _, t in boundary}
            result.append({
                "tag": tag,
                "xmin": xmin, "xmax": xmax,
                "ymin": ymin, "ymax": ymax,
                "zmin": zmin, "zmax": zmax,
                "curve_tags": curve_tags,
            })
        return result
    finally:
        gmsh.finalize()


def test_ribs_and_spars_share_edges_with_skin(default_geometry):
    """
    Spar webs and ribs must share real B-rep edges/curves with the skin
    surface where they physically meet -- not just be geometrically
    coincident-but-disjoint surfaces. Verified on the .brep artifact
    (gmsh-native, preserves OCC topology exactly -- unlike STEP, which is
    not guaranteed to preserve shared edges for a loose surface
    collection) by matching curve tags directly, rather than trusting a
    count the implementation reports about itself. Stage 6's shell mesh
    needs this shared topology to produce connected nodes at these
    junctions.
    """
    surfaces = _load_brep_surfaces_with_boundary(default_geometry["brep_path"])
    ribs, spars, skin = _classify(surfaces, SPAN)
    skin_curve_tags = set()
    for s in skin:
        skin_curve_tags |= s["curve_tags"]

    shared = 0
    for s in ribs + spars:
        shared += len(s["curve_tags"] & skin_curve_tags)

    assert shared > 0, (
        "no shared edges found between ribs/spars and skin -- surfaces "
        "are geometrically coincident but topologically disconnected, "
        "which Stage 6 cannot mesh into a connected shell model"
    )


# --- Error handling ------------------------------------------------------------


def test_nonpositive_span_raises(naca0012_coords, tmp_path):
    with pytest.raises(ValueError):
        generate_structural_geometry(
            naca0012_coords, "naca0012", str(tmp_path), span=0.0,
        )


def test_spar_location_outside_chord_raises(naca0012_coords, tmp_path):
    with pytest.raises(ValueError):
        generate_structural_geometry(
            naca0012_coords, "naca0012", str(tmp_path),
            spar_locations=(0.2, 1.5),  # 1.5 is past the TE
        )
