"""
G2Aero (NREL "curated_airfoils") database downloader & converter.

Supplements the UIUC corpus (data/airfoil_downloader.py) with real,
non-synthetic airfoil geometry consolidated from UIUC + JavaFoil +
NACA-TR-824 (the "BigFoil" compilation), published by NREL:
    https://data.openei.org/submissions/6198  (CC BY 4.0)
    curated_airfoils.npz: 19,164 shapes total.

The npz's `classes` (N,) array was assumed from NREL's own docs to be a
real/synthetic 0/1 label; inspecting the actual downloaded file showed
that's wrong -- `classes` holds each shape's airfoil NAME string. 14
names repeat ~1000x each (the CST-perturbation baselines, 13,012 shapes
total); the remaining 6,152 names each appear exactly once -- those are
the real BigFoil-derived airfoils, identified here by that name
appearing exactly once, not by matching a hardcoded expected count
(NREL's published "6,164" doesn't exactly match what's in the file --
count-based identification is self-describing from the data, a fixed
constant would silently drift if a future npz version changes it).

Because the real subset has actual names, dedup against the existing
UIUC corpus (data/raw/airfoils/) is primarily NAME-based (normalized:
lowercased, non-alphanumeric stripped) -- exact and cheap. A secondary
geometry-based check (max pointwise distance after resampling) catches
the same airfoil under a different name/spelling. Per the user: this is
a best-effort filter, not a guarantee -- an occasional repeat slipping
through is acceptable.
"""

import argparse
import glob
import os
import re

import numpy as np
import requests

from pipeline.stage0_geometry_loader import load_airfoil

CURATED_AIRFOILS_URL = "https://data.openei.org/files/6198/curated_airfoils.npz"


def download_curated_airfoils(save_path: str = "data/raw/g2aero/curated_airfoils.npz") -> str:
    """Stream-download the ~310MB npz if not already present. Idempotent."""
    if os.path.exists(save_path):
        return save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_path = save_path + ".part"
    with requests.get(CURATED_AIRFOILS_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    os.replace(tmp_path, save_path)
    return save_path


def load_real_shapes(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (shapes, names) for only the real (non-synthetic) subset.

    Real = a `classes` name value that appears exactly once in the file.
    A synthetic-perturbation baseline's name is reused for every
    perturbation of it (~1000x each), so any name repeated more than
    once is synthetic by construction, not by a guessed threshold.
    Raises if every name is unique (or none is) -- either would mean the
    repeat-count heuristic isn't distinguishing anything, and silently
    returning "all of it" or "none of it" would be worse than failing
    loud.
    """
    data = np.load(npz_path, allow_pickle=True)
    shapes = data["shapes"]
    classes = data["classes"]
    values, counts = np.unique(classes, return_counts=True)
    singleton_values = set(values[counts == 1].tolist())
    if not singleton_values or len(singleton_values) == len(values):
        raise ValueError(
            f"Repeat-count heuristic didn't separate real from synthetic names "
            f"({len(singleton_values)} singleton names out of {len(values)} total) -- "
            f"refusing to guess."
        )
    is_real = np.array([c in singleton_values for c in classes])
    return shapes[is_real], classes[is_real]


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resample_for_compare(coords: np.ndarray, n: int = 100) -> np.ndarray:
    """Coarse fixed-length resample (arc-length parameterized) used only
    for the geometry dedup distance metric -- not the actual output."""
    d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(d)])
    if t[-1] == 0:
        return np.tile(coords[0], (n, 1))
    t /= t[-1]
    t_query = np.linspace(0, 1, n)
    x = np.interp(t_query, t, coords[:, 0])
    y = np.interp(t_query, t, coords[:, 1])
    return np.stack([x, y], axis=1)


def _load_existing_uiuc(uiuc_dat_dir: str, n_compare: int = 100) -> tuple[set, list]:
    """Names (normalized) and resampled shapes of every parseable
    existing UIUC .dat -- used as the dedup reference set."""
    names = set()
    shapes = []
    for path in sorted(glob.glob(os.path.join(uiuc_dat_dir, "*.dat"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        names.add(_normalize_name(stem))
        try:
            shapes.append(_resample_for_compare(load_airfoil(path), n_compare))
        except Exception:
            continue  # a UIUC file Stage 0 already rejects isn't a real dedup target
    return names, shapes


def is_near_duplicate(candidate_resampled: np.ndarray, existing_resampled: list,
                       threshold: float = 0.01) -> bool:
    """Max pointwise distance (chord-normalized) below `threshold` against
    any existing shape counts as a duplicate. threshold=0.01 (1% chord) --
    loose enough to catch the same airfoil digitized twice, tight enough
    not to conflate two genuinely different thin/symmetric sections."""
    for other in existing_resampled:
        if np.max(np.linalg.norm(candidate_resampled - other, axis=1)) < threshold:
            return True
    return False


def convert_and_dedup(
    npz_path: str,
    uiuc_dat_dir: str = "data/raw/airfoils",
    out_dir: str = "data/airfoils_g2aero",
    limit: int | None = None,
) -> dict:
    """
    Convert real G2Aero shapes to Selig .dat files, skipping anything
    that's a name-match or geometry near-duplicate of an existing
    `uiuc_dat_dir` airfoil, then verify every kept file parses cleanly
    through Stage 0 (the locked test target for this module -- see
    module docstring / STATUS.md).

    Returns {"n_real": ..., "n_duplicate_name": ..., "n_duplicate_geometry": ...,
             "n_written": ..., "n_stage0_rejected": ..., "rejected": [(name, error), ...]}.
    """
    os.makedirs(out_dir, exist_ok=True)
    real_shapes, real_names = load_real_shapes(npz_path)
    if limit is not None:
        real_shapes, real_names = real_shapes[:limit], real_names[:limit]

    existing_names, existing_shapes = _load_existing_uiuc(uiuc_dat_dir)

    n_duplicate_name = 0
    n_duplicate_geometry = 0
    written_paths = []
    seen_this_batch = set()
    for shape, name in zip(real_shapes, real_names):
        norm_name = _normalize_name(str(name))
        if norm_name in existing_names or norm_name in seen_this_batch:
            n_duplicate_name += 1
            continue
        candidate = _resample_for_compare(shape)
        if is_near_duplicate(candidate, existing_shapes):
            n_duplicate_geometry += 1
            continue

        safe_name = re.sub(r"[^a-z0-9_-]", "_", str(name).lower()) or "g2aero_unnamed"
        dat_path = os.path.join(out_dir, f"{safe_name}.dat")
        with open(dat_path, "w") as f:
            f.write(f"{name}\n")
            for x, y in shape:
                f.write(f"{x:.6f} {y:.6f}\n")
        written_paths.append(dat_path)
        # Guard against duplicates within this same G2Aero batch too.
        seen_this_batch.add(norm_name)
        existing_shapes.append(candidate)

    n_stage0_rejected = 0
    rejected = []
    for path in written_paths:
        try:
            load_airfoil(path)
        except Exception as exc:
            n_stage0_rejected += 1
            rejected.append((os.path.basename(path), f"{type(exc).__name__}: {exc}"))
            os.remove(path)  # don't leave a file in the corpus that Stage 0 itself rejects

    return {
        "n_real": len(real_shapes),
        "n_duplicate_name": n_duplicate_name,
        "n_duplicate_geometry": n_duplicate_geometry,
        "n_written": len(written_paths) - n_stage0_rejected,
        "n_stage0_rejected": n_stage0_rejected,
        "rejected": rejected,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert the G2Aero real-airfoil subset.")
    parser.add_argument("--npz-path", default="data/raw/g2aero/curated_airfoils.npz")
    parser.add_argument("--uiuc-dat-dir", default="data/raw/airfoils")
    parser.add_argument("--out-dir", default="data/airfoils_g2aero")
    parser.add_argument("--limit", type=int, default=None, help="Cap on number of real shapes to process (testing).")
    args = parser.parse_args()

    path = download_curated_airfoils(args.npz_path)
    result = convert_and_dedup(path, args.uiuc_dat_dir, args.out_dir, limit=args.limit)
    print(
        f"{result['n_real']} real G2Aero shapes -- "
        f"{result['n_duplicate_name']} skipped as name-duplicates, "
        f"{result['n_duplicate_geometry']} skipped as geometry near-duplicates of existing UIUC airfoils, "
        f"{result['n_written']} written to {args.out_dir}, "
        f"{result['n_stage0_rejected']} rejected by Stage 0's own checks"
    )
    if result["rejected"]:
        print("Rejected:")
        for name, err in result["rejected"][:20]:
            print(f"  {name}: {err}")
