"""
DeepONet dataset builder for the ADE pipeline's Cp(s) surface-pressure
output.

Scope decision (STATUS.md, 2026-09-15): the spec's Final Output Schema
also has a per-AoA `stress_field`, but it's keyed by CalculiX's internal
shell-expansion node ids with no (x, y, z) cross-reference stored
anywhere in the pipeline -- a DeepONet trunk needs real query
*locations*, so a stress-field model is untrainable from the current
`.h5` schema without a Stage 9 schema change (node id -> coords
mapping). Scoped to `pressure_vs_arc_length` only: the trunk query `s`
(arc-length position) is already stored per point, exactly what a
DeepONet trunk wants.

Branch input: each airfoil's Stage 0 geometry (reloaded from the .h5's
own `source_file` path -- the .h5 doesn't store raw coords, only the
CFD/FEA outputs) resampled to a fixed point count so every airfoil
produces a same-length branch vector, concatenated with the AoA for
that sample. Trunk input: arc-length `s`. Target: `Cp` at that `s`.

Split is by WHOLE AIRFOIL, not by point or by (airfoil, AoA) sample --
holding out individual points from an airfoil the model has already
seen elsewhere on its own surface tells you almost nothing about
generalization to an unseen shape, which is the only question that
matters for this dataset's actual purpose.
"""

import glob
import os

import h5py
import numpy as np

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage1_mesh import _resample_cosine

GEOMETRY_N_POINTS_PER_SURFACE = 32  # -> 2*32-1 = 63 points per airfoil


def _resample_geometry(coords: np.ndarray, n_per_surface: int = GEOMETRY_N_POINTS_PER_SURFACE) -> np.ndarray:
    """Fixed-length arc-length resampling, reusing Stage 1's own cosine
    resampler so branch-input geometry encoding matches what the CFD
    mesh itself saw, not a separately-invented parameterization."""
    return _resample_cosine(coords, n_per_surface=n_per_surface)


def load_airfoil_records(h5_dir: str) -> list[dict]:
    """
    Read every `airfoil_*.h5` under `h5_dir` (recursively) into a flat
    list of per-airfoil records:
        {
            "name": str,
            "geometry": (M, 2) ndarray (fixed length, resampled),
            "samples": [
                {"aoa_deg": float, "s": (K,) ndarray, "cp": (K,) ndarray},
                ...  # one entry per converged AoA
            ],
        }
    Non-converged AoAs are skipped (no pressure curve exists for them --
    per the spec, they're excluded from training data, not fabricated).
    """
    paths = sorted(glob.glob(os.path.join(h5_dir, "**", "airfoil_*.h5"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No airfoil_*.h5 files found under {h5_dir}")

    records = []
    for path in paths:
        with h5py.File(path, "r") as f:
            name = f["metadata"].attrs["airfoil_name"]
            source_file = f["metadata"].attrs["source_file"]
            aoa_sweep = f["metadata/aoa_sweep_deg"][:]

            geometry = _resample_geometry(load_airfoil(source_file))

            samples = []
            for i, aoa_deg in enumerate(aoa_sweep):
                cfd = f[f"aoa_{i:02d}/cfd"]
                if cfd.attrs["status"] != "converged":
                    continue
                arr = cfd["pressure_vs_arc_length"][:]
                samples.append({
                    "aoa_deg": float(aoa_deg),
                    "s": arr[:, 0].astype(np.float32),
                    "cp": arr[:, 1].astype(np.float32),
                })

            if samples:
                records.append({"name": str(name), "geometry": geometry.astype(np.float32), "samples": samples})

    return records


def split_by_airfoil(records: list[dict], val_fraction: float = 0.2, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Holdout split by whole airfoil (see module docstring for why)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_val = max(1, int(round(len(records) * val_fraction)))
    val_idx = set(order[:n_val].tolist())
    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    return train, val


def split_train_val_test(
    records: list[dict], val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 0,
    fixed_test_names: set[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Three-way holdout split by whole airfoil.

    Val drives checkpoint selection during training (same role as
    split_by_airfoil's val set). Test is never touched during training
    or checkpoint selection -- it exists only to report a real
    generalization number once, at the end. Reusing val for both jobs
    (the previous single-split setup) lets the "best" checkpoint be
    implicitly chosen to look good on the same set used to report
    "how good is the model," which is optimistic, not a held-out result.

    fixed_test_names, if given, pins the test set to those exact airfoil
    names (present in `records`) instead of drawing test randomly by
    `rng.permutation(len(records))`. Without this, re-splitting a
    differently-sized/differently-composed corpus each retrain draws a
    *different* random set of test airfoils every time (seed=0 seeds the
    permutation, not the airfoil identities) -- real symptom hit
    2026-09-16: test_rmse_cp swung 0.449 -> 0.524 -> 0.566 across three
    same-night retrains even as val_loss improved monotonically, purely
    from which airfoils happened to land in each cycle's random test
    draw. Passing the same fixed_test_names across corpora of any size
    makes test_rmse_cp directly comparable retrain to retrain.
    """
    if fixed_test_names is not None:
        test = [r for r in records if r["name"] in fixed_test_names]
        remainder = [r for r in records if r["name"] not in fixed_test_names]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(remainder))
        n_val = max(1, int(round(len(remainder) * val_fraction / (1 - test_fraction))))
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        train = [remainder[i] for i in train_idx]
        val = [remainder[i] for i in val_idx]
        return train, val, test

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n = len(records)
    n_val = max(1, int(round(n * val_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    if n_val + n_test >= n:
        raise ValueError(
            f"val_fraction + test_fraction leaves no training airfoils "
            f"({n_val} val + {n_test} test >= {n} total records)"
        )
    val_idx = order[:n_val]
    test_idx = order[n_val:n_val + n_test]
    train_idx = order[n_val + n_test:]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]
    return train, val, test


class Normalizer:
    """Fit on train records only, applied to both splits -- fitting on
    val data would leak information about the held-out airfoils into
    the normalization statistics."""

    def __init__(self):
        self.aoa_scale = 14.0  # spec's locked sweep max magnitude
        # s (arc length) is normalized per-sample by that curve's own max
        # in build_flat_arrays, not a single global scale -- see its
        # docstring for why (perimeter varies by geometry).
        self.cp_mean = 0.0
        self.cp_std = 1.0

    def fit(self, records: list[dict]) -> "Normalizer":
        all_cp = np.concatenate([s["cp"] for r in records for s in r["samples"]])
        self.cp_mean = float(all_cp.mean())
        self.cp_std = float(all_cp.std() + 1e-8)
        return self

    def transform_cp(self, cp: np.ndarray) -> np.ndarray:
        return (cp - self.cp_mean) / self.cp_std

    def inverse_transform_cp(self, cp_norm: np.ndarray) -> np.ndarray:
        return cp_norm * self.cp_std + self.cp_mean


def build_flat_arrays(records: list[dict], normalizer: Normalizer):
    """
    Flatten (airfoil, AoA, point) into parallel arrays ready for a
    DataLoader:
        branch_inputs: (N, 2*M+1) float32 -- flattened resampled
            geometry + normalized AoA, repeated per query point
        trunk_inputs:  (N, 1) float32 -- s normalized to [0, 1] by each
            sample's own curve max (perimeter varies by geometry, so a
            single global scale would put different airfoils' arc-length
            on inconsistent footing)
        targets:       (N, 1) float32 -- normalized Cp
        sample_group:  (N,) int -- which (airfoil, AoA) sample each row
            belongs to, for diagnostics/plotting, not used in training
    """
    # Preallocated in one pass rather than building a Python list of
    # per-sample arrays and np.concatenate-ing at the end -- the list+
    # concat approach transiently holds both the list AND the final
    # array in memory at once (~2x peak RSS), which is what pushed a
    # 352-airfoil corpus into an OOM kill on a loaded machine. A single
    # preallocated buffer, filled in place, has no such transient copy.
    branch_dim = r_geom_dim = None
    total_n = 0
    for r in records:
        for sample in r["samples"]:
            total_n += len(sample["s"])
    if records and records[0]["samples"]:
        branch_dim = records[0]["geometry"].reshape(-1).shape[0] + 1

    branch_inputs = np.empty((total_n, branch_dim or 1), dtype=np.float32)
    trunk_inputs = np.empty((total_n, 1), dtype=np.float32)
    targets = np.empty((total_n, 1), dtype=np.float32)
    sample_group = np.empty((total_n,), dtype=np.int64)

    offset = 0
    group_id = 0
    for r in records:
        geom_flat = r["geometry"].reshape(-1)
        for sample in r["samples"]:
            aoa_norm = sample["aoa_deg"] / normalizer.aoa_scale
            branch_vec = np.concatenate([geom_flat, [aoa_norm]]).astype(np.float32)
            s = sample["s"]
            s_max = float(s.max()) if s.max() > 0 else 1.0
            s_norm = (s / s_max).astype(np.float32)
            cp_norm = normalizer.transform_cp(sample["cp"])

            n = len(s_norm)
            branch_inputs[offset:offset + n] = branch_vec
            trunk_inputs[offset:offset + n, 0] = s_norm
            targets[offset:offset + n, 0] = cp_norm.astype(np.float32)
            sample_group[offset:offset + n] = group_id
            offset += n
            group_id += 1

    return branch_inputs, trunk_inputs, targets, sample_group
