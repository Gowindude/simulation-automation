"""
Extract a compact JSON summary from the orchestrator's .h5 outputs +
batch manifest, for the demo dashboard (an Artifact -- static HTML/JS,
can't read local files at runtime, so this is a one-shot export).

Usage:
    python scripts/build_dashboard_data.py .orchestrator_runs/real_uiuc_35 \
        --out scripts/dashboard_data.json
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np


def _round(x, n=4):
    return round(float(x), n)


def extract(h5_dir: str) -> dict:
    manifest_path = os.path.join(h5_dir, "batch_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    airfoils = []
    paths = sorted(glob.glob(os.path.join(h5_dir, "*", "airfoil_*.h5")))
    for path in paths:
        with h5py.File(path, "r") as f:
            name = str(f["metadata"].attrs["airfoil_name"])
            aoa_sweep = f["metadata/aoa_sweep_deg"][:].tolist()

            aoas = []
            for i, aoa in enumerate(aoa_sweep):
                g = f[f"aoa_{i:02d}"]
                cfd = g["cfd"]
                entry = {
                    "aoa_deg": _round(aoa, 1),
                    "status": str(cfd.attrs["status"]),
                    "Cl": _round(cfd.attrs["Cl"]) if not np.isnan(cfd.attrs["Cl"]) else None,
                    "Cd": _round(cfd.attrs["Cd"]) if not np.isnan(cfd.attrs["Cd"]) else None,
                    "Cl_xfoil": _round(cfd.attrs["Cl_xfoil"]) if not np.isnan(cfd.attrs["Cl_xfoil"]) else None,
                    "Cd_xfoil": _round(cfd.attrs["Cd_xfoil"]) if not np.isnan(cfd.attrs["Cd_xfoil"]) else None,
                }
                if "pressure_vs_arc_length" in cfd:
                    arr = cfd["pressure_vs_arc_length"][:]
                    # subsample to ~120 points for a manageable JSON/plot size
                    step = max(1, len(arr) // 120)
                    entry["cp_curve"] = [[_round(s), _round(cp)] for s, cp in arr[::step]]
                if "fea" in g:
                    fea = g["fea"]
                    entry["max_von_mises_kpa"] = _round(fea.attrs["max_von_mises"] / 1000.0, 1)
                    entry["reaction_force_residual"] = _round(fea.attrs["reaction_force_residual"], 3)
                aoas.append(entry)

            # airfoil outline for a small shape preview, downsampled
            source_file = str(f["metadata"].attrs["source_file"])
            geometry = None
            if os.path.exists(source_file):
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from pipeline.stage0_geometry_loader import load_airfoil
                coords = load_airfoil(source_file)
                step = max(1, len(coords) // 80)
                geometry = [[_round(x), _round(y)] for x, y in coords[::step]]

            airfoils.append({
                "name": name,
                "aoa_sweep": [_round(a, 1) for a in aoa_sweep],
                "aoas": aoas,
                "geometry": geometry,
                "manifest_status": manifest.get(name, {}).get("status"),
            })

    n_total_aoa = sum(len(a["aoas"]) for a in airfoils)
    n_converged_aoa = sum(1 for a in airfoils for x in a["aoas"] if x["status"] == "converged")

    # Pipeline timing, from the manifest's own per-airfoil wall_clock_seconds
    # (orchestrator.py) -- only present for airfoils run after that field was
    # added, so this quietly reports "0 timed" on an older manifest rather
    # than fabricating a number.
    timed = [
        v["wall_clock_seconds"] for v in manifest.values()
        if v.get("status") == "success" and v.get("wall_clock_seconds") is not None
    ]
    pipeline_timing = {
        "n_airfoils_timed": len(timed),
        "mean_seconds_per_airfoil": _round(np.mean(timed), 1) if timed else None,
        "total_seconds": _round(np.sum(timed), 1) if timed else None,
    }

    return {
        "generated_from": h5_dir,
        "n_airfoils": len(airfoils),
        "n_airfoils_success": sum(1 for v in manifest.values() if v.get("status") == "success"),
        "n_total_aoa": n_total_aoa,
        "n_converged_aoa": n_converged_aoa,
        "airfoils": airfoils,
        "pipeline_timing": pipeline_timing,
    }


def extract_deeponet_metrics(checkpoint_dir: str, pipeline_timing: dict) -> dict | None:
    """
    Pull the held-out-test metrics + inference timing written by
    deeponet/train.py's normalizer.json (see that module: test set is
    never used for training or checkpoint selection, so test_rmse_cp is
    a real generalization number, not a training-time proxy). Returns
    None if no trained checkpoint exists yet -- the dashboard's own
    build must not depend on training having been run.
    """
    normalizer_path = os.path.join(checkpoint_dir, "normalizer.json")
    if not os.path.exists(normalizer_path):
        return None
    with open(normalizer_path) as f:
        n = json.load(f)
    if "test_rmse_cp" not in n:
        # Older checkpoint, trained before the train/val/test split existed.
        return None

    n_test_airfoils = len(n.get("test_airfoils", [])) or 1
    points_per_airfoil = n["test_n_points"] / n_test_airfoils
    seconds_per_airfoil_inference = points_per_airfoil * n["test_inference_seconds_per_point"]

    speedup_x = None
    if pipeline_timing.get("mean_seconds_per_airfoil") and seconds_per_airfoil_inference > 0:
        speedup_x = _round(pipeline_timing["mean_seconds_per_airfoil"] / seconds_per_airfoil_inference, 0)

    return {
        "n_train_airfoils": len(n.get("train_airfoils", [])),
        "n_val_airfoils": len(n.get("val_airfoils", [])),
        "n_test_airfoils": n_test_airfoils,
        "best_epoch": n.get("best_epoch"),
        "best_val_loss": _round(n["best_val_loss"], 5) if n.get("best_val_loss") is not None else None,
        "test_rmse_cp": _round(n["test_rmse_cp"], 4),
        "test_n_points": n["test_n_points"],
        "inference_seconds_per_point": n["test_inference_seconds_per_point"],
        # Estimated full-airfoil-sweep inference time: per-point inference
        # time x average query points per test airfoil (one AoA's Cp(s)
        # curve worth of points) -- an estimate, not a measured single-call
        # timing, so it's labeled as such rather than presented as identical
        # in kind to the measured pipeline wall-clock number.
        "estimated_seconds_per_airfoil": seconds_per_airfoil_inference,
        "pipeline_vs_deeponet_speedup_x": speedup_x,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_dir")
    parser.add_argument("--out", default="scripts/dashboard_data.json")
    parser.add_argument("--deeponet-checkpoint-dir", default="deeponet/checkpoints")
    args = parser.parse_args()

    data = extract(args.h5_dir)
    data["deeponet"] = extract_deeponet_metrics(args.deeponet_checkpoint_dir, data["pipeline_timing"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out} ({size_kb:.0f} KB): {data['n_airfoils']} airfoils, "
          f"{data['n_converged_aoa']}/{data['n_total_aoa']} AoA converged")
