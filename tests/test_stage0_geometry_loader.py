"""
Verification tests for Stage 0 — Geometry loader.

Contract (from .claude/airfoil_pipeline_build_spec.md):
  Input:  UIUC .dat file path
  Output: np.ndarray (N, 2), normalized to unit chord, Selig ordering
          (TE -> upper -> LE -> lower -> TE)
  Must handle: unclosed files (first != last point), inconsistent point
               ordering
  Suggested check: no self-intersections
"""

import os

import numpy as np
import pytest

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage0_geometry_loader import (
    _segments_intersect as segments_intersect,
    _polygon_has_self_intersections as polygon_has_self_intersections,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

SYNTHETIC_AIRFOILS = [
    # Analytic NACA 4-digit shapes (see fixtures/generate_fixtures.py) --
    # for deterministic, offline, adversarial/edge-case coverage. Real
    # UIUC data is covered separately in test_stage0_real_uiuc.py; the two
    # are not redundant, see that file's docstring.
    "naca0006.dat",   # very thin symmetric
    "naca0012.dat",   # thin symmetric baseline
    "naca0021.dat",   # thick symmetric
    "naca2412.dat",   # moderate camber
    "naca4412.dat",   # higher camber
    "naca6412.dat",   # high camber
]


def fixture_path(name):
    return os.path.join(FIXTURES, name)


# segments_intersect / polygon_has_self_intersections are re-exported
# above from pipeline.stage0_geometry_loader (the same check load_airfoil
# now runs internally to reject self-intersecting contours) so this
# suite and the pipeline never drift into checking self-intersection two
# different ways.


# --- 1. Type / shape ---------------------------------------------------------


@pytest.mark.parametrize("fname", SYNTHETIC_AIRFOILS)
def test_output_type_and_shape(fname):
    coords = load_airfoil(fixture_path(fname))
    assert isinstance(coords, np.ndarray)
    assert coords.dtype == np.float64
    assert coords.ndim == 2
    assert coords.shape[1] == 2
    assert coords.shape[0] > 2


# --- 2. Unit-chord normalization ---------------------------------------------


@pytest.mark.parametrize("fname", SYNTHETIC_AIRFOILS)
def test_unit_chord_normalization(fname):
    coords = load_airfoil(fixture_path(fname))
    x = coords[:, 0]
    assert x.min() == pytest.approx(0.0, abs=1e-9)
    assert x.max() == pytest.approx(1.0, abs=1e-9)


# --- 3. Selig ordering --------------------------------------------------------


@pytest.mark.parametrize("fname", SYNTHETIC_AIRFOILS + ["scrambled_order.dat", "open_te.dat"])
def test_selig_ordering(fname):
    coords = load_airfoil(fixture_path(fname))
    le_idx = int(np.argmin(coords[:, 0]))

    # LE must sit strictly inside the array, not at either endpoint.
    assert 0 < le_idx < len(coords) - 1

    upper_x = coords[: le_idx + 1, 0]
    lower_x = coords[le_idx:, 0]

    # Upper surface: TE -> LE, x non-increasing.
    assert np.all(np.diff(upper_x) <= 1e-9)
    # Lower surface: LE -> TE, x non-decreasing.
    assert np.all(np.diff(lower_x) >= -1e-9)


# --- 4. Inconsistent ordering is corrected, not passed through --------------


def test_scrambled_input_is_recovered_to_selig_order():
    scrambled = load_airfoil(fixture_path("scrambled_order.dat"))
    reference = load_airfoil(fixture_path("naca4412.dat"))

    # Both should describe the same shape once canonicalized: same point
    # count and the same x-range/ordering behavior (checked structurally,
    # not by exact float equality, since traversal start differs).
    assert scrambled.shape == reference.shape
    le_idx_scrambled = int(np.argmin(scrambled[:, 0]))
    le_idx_reference = int(np.argmin(reference[:, 0]))
    assert le_idx_scrambled == le_idx_reference


# --- 5. Unclosed trailing edge is closed -------------------------------------


def test_open_trailing_edge_is_closed():
    coords = load_airfoil(fixture_path("open_te.dat"))
    assert np.array_equal(coords[0], coords[-1])


def test_closed_input_is_left_alone():
    # naca0012.dat is generated already closed (upper/lower share the LE
    # point only; TE endpoints coincide by construction of the analytic
    # formula at x=1). Closing must be idempotent / a no-op here.
    coords = load_airfoil(fixture_path("naca0012.dat"))
    assert np.array_equal(coords[0], coords[-1])


# --- 6. No self-intersections -------------------------------------------------


@pytest.mark.parametrize("fname", SYNTHETIC_AIRFOILS + ["scrambled_order.dat", "open_te.dat", "duplicate_points.dat"])
def test_no_self_intersections(fname):
    coords = load_airfoil(fixture_path(fname))
    assert not polygon_has_self_intersections(coords)


# --- 7. Determinism ------------------------------------------------------------


@pytest.mark.parametrize("fname", SYNTHETIC_AIRFOILS)
def test_deterministic(fname):
    a = load_airfoil(fixture_path(fname))
    b = load_airfoil(fixture_path(fname))
    assert np.array_equal(a, b)


# --- 8. Adversarial: consecutive duplicate points (found probing Stage 0/1
#        for troubleshooter-agent scoping, 2026-09-14/15 -- a duplicated
#        coordinate crashed Stage 1's CubicSpline resampling with a raw
#        scipy ValueError, since a non-strictly-increasing arc-length
#        parameterization is unusable for spline fitting regardless of
#        mesh generation) ---------------------------------------------------


def test_consecutive_duplicate_points_are_deduped():
    coords = load_airfoil(fixture_path("duplicate_points.dat"))
    deltas = np.diff(coords, axis=0)
    identical = np.all(deltas == 0.0, axis=1)
    assert not identical.any(), (
        "load_airfoil must drop consecutive duplicate coordinates -- a "
        "repeated point produces a zero-length polygon edge, which is "
        "not just cosmetic: Stage 1's arc-length resampling divides by "
        "segment length and crashes outright on one, per the raw "
        "'x must be strictly increasing sequence' scipy error found "
        "probing this fixture before this fix"
    )


def test_dedup_reduces_point_count():
    # duplicate_points.dat has 3 duplicated points injected (see
    # tests/fixtures/generate_fixtures.py) -- deduping must actually
    # remove them, not just tolerate them downstream.
    raw_dat = fixture_path("duplicate_points.dat")
    with open(raw_dat) as f:
        raw_n = sum(1 for line in f.readlines()[1:] if line.strip())
    coords = load_airfoil(raw_dat)
    assert len(coords) < raw_n


# --- 9. Adversarial: genuinely self-intersecting contour is rejected, not
#        silently passed through to an expensive/unmeshable Stage 1 attempt
#        (found probing Stage 0/1 for troubleshooter-agent scoping,
#        2026-09-14/15) -------------------------------------------------------


def test_self_intersecting_contour_is_rejected():
    with pytest.raises(ValueError, match="(?i)self.intersect"):
        load_airfoil(fixture_path("self_intersecting.dat"))


# --- Error handling ------------------------------------------------------------


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_airfoil(fixture_path("does_not_exist.dat"))
