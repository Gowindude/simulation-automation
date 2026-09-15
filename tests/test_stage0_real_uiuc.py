"""
Stage 0 regression tests against REAL (not synthetic) UIUC airfoils.

`test_stage0_geometry_loader.py` uses analytically-generated NACA 4-digit
fixtures. Those are good for fast, deterministic, adversarial/edge-case
coverage (a scrambled-order file, a deliberately-open-TE file -- inputs we
fully control on purpose), but they are not a substitute for testing
against real digitized data, which has quirks no analytic formula
reproduces: uneven point spacing, small real trailing-edge gaps, and
provenance from many different digitization eras/conventions across the
UIUC database.

Concretely, of the 35 real files in fixtures/real_uiuc/ (downloaded once,
checked in -- no network needed at test time), none of the *synthetic*
fixtures ever exercised a genuine nonzero raw TE gap end-to-end: the
synthetic 'open_te.dat' fixture is hand-constructed. This file closes that
gap (pun intended) by running the same battery of checks from
test_stage0_geometry_loader.py against real data, plus a dedicated test
for TE-gap closing on the real files that actually have one.
"""

import os

import numpy as np
import pytest

from data.airfoil_downloader import load_dat_file
from pipeline.stage0_geometry_loader import load_airfoil
from tests.test_stage0_geometry_loader import polygon_has_self_intersections

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "real_uiuc")

ALL_REAL_AIRFOILS = sorted(
    f[:-4] for f in os.listdir(FIXTURES) if f.endswith(".dat")
)

# Files whose raw (pre-Stage-0) coordinates have a genuine, non-negligible
# gap between the first and last recorded point (checked via
# `data.airfoil_downloader.load_dat_file`, i.e. before Stage 0 touches
# them). `m6` has the largest at ~0.52% chord.
REAL_AIRFOILS_WITH_OPEN_TE = [
    "ag24", "ag35", "clarky", "m6", "naca23012", "rc410", "sc20402",
    "usa35b", "whitcomb",
]


def fixture_path(name):
    return os.path.join(FIXTURES, f"{name}.dat")


def _raw_te_gap(name):
    _, raw_points = load_dat_file(fixture_path(name))
    raw = np.array(raw_points, dtype=np.float64)
    return float(np.linalg.norm(raw[0] - raw[-1]))


# --- Sanity: the "open TE" fixture list is actually correct -----------------


def test_open_te_fixture_list_matches_raw_data():
    """Guards against the curated list above silently drifting from the data."""
    actual_gapped = {name for name in ALL_REAL_AIRFOILS if _raw_te_gap(name) > 1e-4}
    assert actual_gapped == set(REAL_AIRFOILS_WITH_OPEN_TE)


# --- Full Stage 0 battery on every real file ---------------------------------


@pytest.mark.parametrize("name", ALL_REAL_AIRFOILS)
def test_output_type_and_shape(name):
    coords = load_airfoil(fixture_path(name))
    assert isinstance(coords, np.ndarray)
    assert coords.dtype == np.float64
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert coords.shape[0] > 2


@pytest.mark.parametrize("name", ALL_REAL_AIRFOILS)
def test_unit_chord_normalization(name):
    coords = load_airfoil(fixture_path(name))
    x = coords[:, 0]
    assert x.min() == pytest.approx(0.0, abs=1e-9)
    assert x.max() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", ALL_REAL_AIRFOILS)
def test_selig_ordering(name):
    coords = load_airfoil(fixture_path(name))
    le_idx = int(np.argmin(coords[:, 0]))
    assert 0 < le_idx < len(coords) - 1

    upper_x = coords[: le_idx + 1, 0]
    lower_x = coords[le_idx:, 0]
    assert np.all(np.diff(upper_x) <= 1e-9)
    assert np.all(np.diff(lower_x) >= -1e-9)


@pytest.mark.parametrize("name", ALL_REAL_AIRFOILS)
def test_no_self_intersections(name):
    coords = load_airfoil(fixture_path(name))
    assert not polygon_has_self_intersections(coords)


@pytest.mark.parametrize("name", ALL_REAL_AIRFOILS)
def test_deterministic(name):
    a = load_airfoil(fixture_path(name))
    b = load_airfoil(fixture_path(name))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ALL_REAL_AIRFOILS)
def test_te_is_closed(name):
    coords = load_airfoil(fixture_path(name))
    assert np.array_equal(coords[0], coords[-1])


# --- Dedicated check: real nonzero TE gaps are actually closed --------------


@pytest.mark.parametrize("name", REAL_AIRFOILS_WITH_OPEN_TE)
def test_real_open_te_gap_is_closed_to_midpoint(name):
    """
    The closed TE point must be the midpoint of the two RAW endpoints,
    expressed in the final normalized frame -- and per Stage 0's contract
    (close the gap, THEN normalize -- see stage0_geometry_loader.load_airfoil),
    that closed point defines the chord's x=1.0 exactly. It is not
    generally the midpoint of the two raw endpoints *after* independently
    normalizing each one first -- that was the pre-fix (buggy) order,
    which could land the closed TE slightly off of x=1.0 whenever the raw
    upper/lower TE points didn't share the same x already (e.g. naca23012:
    raw TE at x=1.00003 upper vs. x=0.99997 lower).
    """
    raw_gap = _raw_te_gap(name)
    assert raw_gap > 1e-4, "fixture list is stale -- see test_open_te_fixture_list_matches_raw_data"

    _, raw_points = load_dat_file(fixture_path(name))
    raw = np.array(raw_points, dtype=np.float64)
    te_closed_raw = (raw[0] + raw[-1]) / 2.0
    # Closing only touches index 0/-1, so x_min (the LE, an interior point) is unaffected.
    x_min = raw[:, 0].min()
    chord = te_closed_raw[0] - x_min
    expected_te_normalized = np.array(
        [(te_closed_raw[0] - x_min) / chord, te_closed_raw[1] / chord]
    )
    assert expected_te_normalized[0] == pytest.approx(1.0, abs=1e-9)  # sanity on the expectation itself

    coords = load_airfoil(fixture_path(name))
    assert np.array_equal(coords[0], coords[-1])
    assert coords[0] == pytest.approx(expected_te_normalized, abs=1e-9)
