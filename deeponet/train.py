"""
Train the Cp(s) DeepONet on the pipeline's .h5 output.

Usage:
    python -m deeponet.train --h5-dir .orchestrator_runs/real_uiuc_35 --epochs 300

Small model, short run, CPU-only (confirmed: torch reports no CUDA on
this machine) -- per the same instinct as everything else built
tonight, a working small result beats an ambitious one that doesn't
finish. Held-out-airfoil validation loss is the number that actually
means something here (see dataset.py's split_by_airfoil docstring);
train loss alone would look good on 175 samples regardless of whether
the model generalizes to an unseen shape.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deeponet.dataset import (
    load_airfoil_records, split_train_val_test, Normalizer, build_flat_arrays,
)
from deeponet.model import DeepONet


def _to_loader(branch, trunk, targets, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(branch), torch.from_numpy(trunk), torch.from_numpy(targets),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train(
    h5_dir, epochs=300, batch_size=256, lr=1e-3,
    val_fraction=0.15, test_fraction=0.15, seed=0, out_dir="deeponet/checkpoints",
):
    os.makedirs(out_dir, exist_ok=True)
    records = load_airfoil_records(h5_dir)
    train_records, val_records, test_records = split_train_val_test(
        records, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed,
    )
    print(
        f"{len(records)} airfoils total -- {len(train_records)} train / "
        f"{len(val_records)} val (checkpoint selection) / {len(test_records)} test (held out, reported once)"
    )

    normalizer = Normalizer().fit(train_records)
    branch_tr, trunk_tr, y_tr, _ = build_flat_arrays(train_records, normalizer)
    branch_va, trunk_va, y_va, group_va = build_flat_arrays(val_records, normalizer)
    print(f"train points: {len(y_tr)}, val points: {len(y_va)}")

    train_loader = _to_loader(branch_tr, trunk_tr, y_tr, batch_size, shuffle=True)

    model = DeepONet(branch_in_dim=branch_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    branch_va_t = torch.from_numpy(branch_va)
    trunk_va_t = torch.from_numpy(trunk_va)
    y_va_t = torch.from_numpy(y_va)

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for bx, tx, y in train_loader:
            opt.zero_grad()
            pred = model(bx, tx)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(y)
            n += len(y)
        train_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():
            val_pred = model(branch_va_t, trunk_va_t)
            val_loss = loss_fn(val_pred, y_va_t).item()
            # Physical-units RMSE (undo Cp normalization) -- the
            # normalized MSE alone doesn't say anything interpretable.
            val_rmse_cp = float(np.sqrt(
                np.mean((normalizer.inverse_transform_cp(val_pred.numpy())
                         - normalizer.inverse_transform_cp(y_va_t.numpy())) ** 2)
            ))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_rmse_cp": val_rmse_cp})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Deep-copy off the GPU/graph, not a reference -- the model
            # keeps training after this point.
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  val_rmse_Cp={val_rmse_cp:.4f}")

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s -- best val_loss={best_val_loss:.5f} at epoch {best_epoch} "
          f"(final epoch {epochs} val_loss={history[-1]['val_loss']:.5f})")

    # Held-out test evaluation -- the BEST-val checkpoint, evaluated once
    # against airfoils that influenced neither training nor checkpoint
    # selection. This is the only number that should be reported as
    # "how well does this generalize"; val_loss/val_rmse_cp above answer
    # a different question (which checkpoint to keep).
    model.load_state_dict(best_state)
    model.eval()
    branch_te, trunk_te, y_te, group_te = build_flat_arrays(test_records, normalizer)
    branch_te_t = torch.from_numpy(branch_te)
    trunk_te_t = torch.from_numpy(trunk_te)
    y_te_t = torch.from_numpy(y_te)

    # Inference timing on the same held-out points, for the "DeepONet
    # prediction speed vs. full pipeline" dashboard comparison --
    # measured here (not in a separate script) so it uses the real
    # trained model and real query volume, not a synthetic stand-in.
    with torch.no_grad():
        # One warmup pass -- excluded from the timing, matches the usual
        # convention of not counting first-call lazy-init overhead.
        model(branch_te_t, trunk_te_t)
        infer_t0 = time.time()
        test_pred = model(branch_te_t, trunk_te_t)
        infer_elapsed = time.time() - infer_t0
        test_loss = loss_fn(test_pred, y_te_t).item()
        test_rmse_cp = float(np.sqrt(
            np.mean((normalizer.inverse_transform_cp(test_pred.numpy())
                     - normalizer.inverse_transform_cp(y_te_t.numpy())) ** 2)
        ))
    n_test_points = len(y_te)
    print(
        f"held-out test ({len(test_records)} airfoils, {n_test_points} points): "
        f"test_loss={test_loss:.5f}  test_rmse_Cp={test_rmse_cp:.4f}  "
        f"inference={infer_elapsed*1000:.2f}ms total "
        f"({infer_elapsed/n_test_points*1e6:.2f}us/point)"
    )

    # Save the BEST-val checkpoint, not the last epoch's -- the two can
    # differ a lot once overfitting sets in (confirmed for real,
    # STATUS.md 2026-09-15: a run on 41 airfoils bottomed out at epoch
    # ~102 then rose for the remaining ~700 epochs). The final-epoch
    # state is also kept, explicitly named, so a caller can still get it
    # if that's genuinely what they want.
    torch.save(best_state, os.path.join(out_dir, "deeponet_cp.pt"))
    torch.save(model.state_dict(), os.path.join(out_dir, "deeponet_cp_final_epoch.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "normalizer.json"), "w") as f:
        json.dump({
            "aoa_scale": normalizer.aoa_scale, "cp_mean": normalizer.cp_mean, "cp_std": normalizer.cp_std,
            "train_airfoils": [r["name"] for r in train_records],
            "val_airfoils": [r["name"] for r in val_records],
            "test_airfoils": [r["name"] for r in test_records],
            "branch_in_dim": int(branch_tr.shape[1]),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_epoch": epochs,
            "test_loss": test_loss,
            "test_rmse_cp": test_rmse_cp,
            "test_n_points": n_test_points,
            "test_inference_seconds_total": infer_elapsed,
            "test_inference_seconds_per_point": infer_elapsed / n_test_points,
        }, f, indent=2)

    return model, normalizer, history, train_records, val_records, test_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", default=".orchestrator_runs/real_uiuc_35")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    train(
        args.h5_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction,
    )
