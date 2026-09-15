"""
Verification tests for the G2Aero downloader/converter (data/g2aero_downloader.py).

Contract (locked before implementation, per .claude/airfoil_pipeline_build_spec.md's
"propose the verification tests ... then agree on them before you build"):
  - `load_real_shapes` must identify real (non-synthetic) shapes as
    those whose `classes` name value appears exactly once -- a repeated
    name is a CST-perturbation baseline reused across ~1000 synthetic
    shapes, not a real airfoil -- and must raise rather than silently
    guess if the heuristic doesn't separate anything.
  - Dedup: an exact (normalized) name match against an existing UIUC
    airfoil is skipped, and a geometry near-duplicate (same shape,
    different name) is also skipped; a genuinely different shape/name
    is kept.
  - `convert_and_dedup` end-to-end: every .dat file it writes must parse
    cleanly through Stage 0's load_airfoil (the same bar the 100-airfoil
    UIUC pull was held to, per STATUS.md).

No real network access — the npz download and the live openei.org URL
are exercised manually (see module docstring), not in this suite.
"""

import numpy as np
import pytest

from data.g2aero_downloader import (
    load_real_shapes, is_near_duplicate, _resample_for_compare, _normalize_name,
    convert_and_dedup,
)


def _naca0012_like(n=41):
    """A closed symmetric airfoil-ish contour, landmark-style (x, y)."""
    x_upper = np.linspace(1, 0, n // 2 + 1)
    x_lower = np.linspace(0, 1, n - len(x_upper))
    t = 0.12
    y_upper = 5 * t * (0.2969 * np.sqrt(x_upper) - 0.1260 * x_upper - 0.3516 * x_upper**2
                       + 0.2843 * x_upper**3 - 0.1015 * x_upper**4)
    y_lower = -5 * t * (0.2969 * np.sqrt(x_lower) - 0.1260 * x_lower - 0.3516 * x_lower**2
                        + 0.2843 * x_lower**3 - 0.1015 * x_lower**4)
    x = np.concatenate([x_upper, x_lower])
    y = np.concatenate([y_upper, y_lower])
    return np.stack([x, y], axis=1)


class TestLoadRealShapes:
    def test_singleton_names_are_real_repeated_names_are_synthetic(self, tmp_path):
        n_landmarks = 21
        # 3 real airfoils (unique names) + 1 synthetic baseline repeated 5x.
        real_names = ["foo", "bar", "baz"]
        synth_shapes = np.stack([_naca0012_like(n_landmarks) for _ in range(5)])
        synth_names = ["baseline1"] * 5
        real_shapes_in = np.stack([_naca0012_like(n_landmarks) for _ in range(3)])

        shapes = np.concatenate([real_shapes_in, synth_shapes])
        classes = np.array(real_names + synth_names)
        npz_path = tmp_path / "curated_airfoils.npz"
        np.savez(npz_path, shapes=shapes, classes=classes)

        real_out, names_out = load_real_shapes(str(npz_path))
        assert len(real_out) == 3
        assert set(names_out.tolist()) == set(real_names)

    def test_raises_if_all_names_unique(self, tmp_path):
        n_landmarks = 21
        shapes = np.stack([_naca0012_like(n_landmarks) for _ in range(4)])
        classes = np.array(["a", "b", "c", "d"])  # all singleton -- heuristic separates nothing
        npz_path = tmp_path / "curated_airfoils.npz"
        np.savez(npz_path, shapes=shapes, classes=classes)

        with pytest.raises(ValueError, match="didn't separate"):
            load_real_shapes(str(npz_path))

    def test_raises_if_no_names_unique(self, tmp_path):
        n_landmarks = 21
        shapes = np.stack([_naca0012_like(n_landmarks) for _ in range(4)])
        classes = np.array(["a", "a", "b", "b"])  # nothing singleton
        npz_path = tmp_path / "curated_airfoils.npz"
        np.savez(npz_path, shapes=shapes, classes=classes)

        with pytest.raises(ValueError, match="didn't separate"):
            load_real_shapes(str(npz_path))


class TestNormalizeName:
    def test_case_and_punctuation_insensitive(self):
        assert _normalize_name("NACA-0012") == _normalize_name("naca0012")
        assert _normalize_name("E 387") == _normalize_name("e387")


class TestGeometryDedup:
    def test_identical_shape_is_a_duplicate(self):
        shape = _naca0012_like()
        resampled = _resample_for_compare(shape)
        assert is_near_duplicate(resampled, [resampled.copy()]) is True

    def test_different_shape_is_not_a_duplicate(self):
        naca0012 = _resample_for_compare(_naca0012_like())
        thick = _naca0012_like().copy()
        thick[:, 1] *= 3.0  # much thicker section -- clearly different, not digitization noise
        thick_resampled = _resample_for_compare(thick)
        assert is_near_duplicate(thick_resampled, [naca0012]) is False

    def test_empty_existing_list_never_flags_duplicate(self):
        resampled = _resample_for_compare(_naca0012_like())
        assert is_near_duplicate(resampled, []) is False


class TestConvertAndDedupEndToEnd:
    def _make_npz(self, tmp_path, real_shapes, real_names, n_synth_baselines=2, n_synth_each=3, n_landmarks=21):
        synth_shapes = []
        synth_names = []
        for i in range(n_synth_baselines):
            base = _naca0012_like(n_landmarks)
            for _ in range(n_synth_each):
                synth_shapes.append(base)
                synth_names.append(f"synthbase{i}")
        all_shapes = np.concatenate([np.stack(real_shapes), np.stack(synth_shapes)])
        all_names = np.array(list(real_names) + synth_names)
        npz_path = tmp_path / "curated_airfoils.npz"
        np.savez(npz_path, shapes=all_shapes, classes=all_names)
        return str(npz_path)

    def test_written_files_all_parse_through_stage0(self, tmp_path):
        """The locked test target: every kept .dat must survive load_airfoil."""
        n_landmarks = 41
        shapes = [_naca0012_like(n_landmarks) for _ in range(5)]
        for i, s in enumerate(shapes):
            s[:, 1] *= 1.0 + 0.6 * i  # well past the 1%-chord dedup threshold between any pair
        names = [f"realfoil{i}" for i in range(5)]
        npz_path = self._make_npz(tmp_path, shapes, names, n_landmarks=n_landmarks)

        uiuc_dir = tmp_path / "uiuc_empty"
        uiuc_dir.mkdir()
        out_dir = tmp_path / "g2aero_out"

        result = convert_and_dedup(str(npz_path), str(uiuc_dir), str(out_dir))

        assert result["n_real"] == 5
        assert result["n_stage0_rejected"] == 0
        written = list(out_dir.glob("*.dat"))
        assert len(written) == result["n_written"] == 5

    def test_dedups_by_name_against_existing_uiuc_corpus(self, tmp_path):
        uiuc_dir = tmp_path / "uiuc"
        uiuc_dir.mkdir()
        with open(uiuc_dir / "naca0012.dat", "w") as f:
            f.write("naca0012\n")
            for x, y in _naca0012_like(41):
                f.write(f"{x:.6f} {y:.6f}\n")

        # A different shape, but the SAME name (case/punctuation-varied) --
        # must be skipped on name alone, not geometry.
        different_shape = _naca0012_like(41)
        different_shape[:, 1] *= 5.0
        npz_path = self._make_npz(tmp_path, [different_shape], ["NACA-0012"], n_landmarks=41)

        out_dir = tmp_path / "g2aero_out"
        result = convert_and_dedup(str(npz_path), str(uiuc_dir), str(out_dir))

        assert result["n_duplicate_name"] == 1
        assert result["n_duplicate_geometry"] == 0
        assert result["n_written"] == 0

    def test_dedups_by_geometry_when_name_differs(self, tmp_path):
        uiuc_dir = tmp_path / "uiuc"
        uiuc_dir.mkdir()
        shape = _naca0012_like(41)
        with open(uiuc_dir / "naca0012.dat", "w") as f:
            f.write("naca0012\n")
            for x, y in shape:
                f.write(f"{x:.6f} {y:.6f}\n")

        # Same geometry, different name -- must be caught by the geometry check.
        npz_path = self._make_npz(tmp_path, [shape.copy()], ["totally_different_name"], n_landmarks=41)

        out_dir = tmp_path / "g2aero_out"
        result = convert_and_dedup(str(npz_path), str(uiuc_dir), str(out_dir))

        assert result["n_duplicate_name"] == 0
        assert result["n_duplicate_geometry"] == 1
        assert result["n_written"] == 0

    def test_limit_caps_processed_shapes(self, tmp_path):
        n_landmarks = 21
        shapes = [_naca0012_like(n_landmarks) for _ in range(5)]
        for i, s in enumerate(shapes):
            s[:, 1] *= 1.0 + 0.05 * i
        names = [f"realfoil{i}" for i in range(5)]
        npz_path = self._make_npz(tmp_path, shapes, names, n_landmarks=n_landmarks)

        uiuc_dir = tmp_path / "uiuc_empty"
        uiuc_dir.mkdir()
        out_dir = tmp_path / "g2aero_out"

        result = convert_and_dedup(str(npz_path), str(uiuc_dir), str(out_dir), limit=2)

        assert result["n_real"] == 2
