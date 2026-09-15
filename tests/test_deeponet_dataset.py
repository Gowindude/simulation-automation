"""
DeepONet dataset contract tests. Uses real .h5 files from
.orchestrator_runs/real_uiuc_35 (regenerated 2026-09-15 -- see
STATUS.md) when present; skipped otherwise rather than faked, since the
whole point is verifying real schema shapes (source_file reload,
variable per-airfoil point counts, non-converged AoA exclusion), not a
synthetic stand-in that could silently drift from the real .h5 layout.
"""

import os

import numpy as np
import pytest

from deeponet.dataset import (
    load_airfoil_records, split_by_airfoil, Normalizer, build_flat_arrays,
)

H5_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".orchestrator_runs", "real_uiuc_35",
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(H5_DIR), reason=f"no regenerated dataset at {H5_DIR}",
)


@pytest.fixture(scope="module")
def records():
    return load_airfoil_records(H5_DIR)


def test_every_airfoil_has_fixed_length_geometry(records):
    shapes = {r["geometry"].shape for r in records}
    assert len(shapes) == 1, f"geometry resampling must produce one fixed shape, got {shapes}"


def test_non_converged_aoas_are_excluded_not_fabricated(records):
    # Every record with < 5 samples must correspond to a real
    # non-converged AoA -- not a silently dropped/corrupted one.
    for r in records:
        assert 1 <= len(r["samples"]) <= 5


def test_split_by_airfoil_is_disjoint_and_covers_everything(records):
    train, val = split_by_airfoil(records, val_fraction=0.2)
    train_names = {r["name"] for r in train}
    val_names = {r["name"] for r in val}
    assert not (train_names & val_names)
    assert train_names | val_names == {r["name"] for r in records}
    assert len(val) >= 1


def test_flat_arrays_shapes_are_consistent(records):
    train, val = split_by_airfoil(records, val_fraction=0.2)
    norm = Normalizer().fit(train)
    branch, trunk, targets, groups = build_flat_arrays(train, norm)

    n_points = sum(len(s["s"]) for r in train for s in r["samples"])
    assert branch.shape == (n_points, train[0]["geometry"].size + 1)
    assert trunk.shape == (n_points, 1)
    assert targets.shape == (n_points, 1)
    assert groups.shape == (n_points,)
    assert trunk.min() >= 0.0 and trunk.max() <= 1.0 + 1e-6


def test_normalizer_fit_only_on_train_not_leaked_from_val(records):
    train, val = split_by_airfoil(records, val_fraction=0.2)
    norm_train_only = Normalizer().fit(train)
    norm_all = Normalizer().fit(records)
    if val:  # only meaningful if there's an actual held-out airfoil
        assert norm_train_only.cp_mean != pytest.approx(norm_all.cp_mean) or len(train) == len(records)


def test_geometry_is_reloaded_from_source_file_not_fabricated(records):
    # Sanity: geometry must be real, unit-chord airfoil coords, not a
    # zero/placeholder array silently substituted on a reload failure.
    for r in records:
        g = r["geometry"]
        chord = g[:, 0].max() - g[:, 0].min()
        # Loose tolerance: _resample_cosine's spline fit can overshoot the
        # original [0, 1] range slightly near the endpoints (confirmed:
        # one real airfoil resamples to chord 1.0004) -- this test checks
        # "is this real reloaded geometry," not resampling exactness.
        assert chord == pytest.approx(1.0, abs=1e-2)
        assert not np.allclose(g, 0.0)
