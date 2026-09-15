"""
Stage 9 -- Aggregation.

Input:  Stage 4 (per-AoA `build_cfd_record()`) + Stage 8 (per-AoA
        `run_calculix_analysis()`, only for AoAs where CFD converged)
        outputs across an airfoil's AoA sweep.
Output: a single `.h5` file per airfoil, matching the spec's Final
        Output Schema (metadata + per-AoA cfd/fea groups) -- the
        eventual training-data format for the Phase 2 neural operator.

`per_aoa_results` is a list of `{"cfd": <dict>, "fea": <dict> | None}`,
one per value in `aoa_sweep_deg`, in the same order. `fea` must be None
whenever `cfd["status"] != "converged"` (Stages 5-8 never ran for that
AoA) -- enforced, not assumed, since a converged-looking fea record next
to a failed cfd status would silently corrupt the training data.

Failed cases are never dropped: every AoA gets its own `aoa_NN/cfd`
group with its real status, whether or not it converged; `aoa_NN/fea`
is simply absent for AoAs that didn't produce structural results.
"""

import os

import h5py
import numpy as np


def aggregate_airfoil_record(
    name, source_file, reynolds, span, spar_locations, rib_spacing,
    aoa_sweep_deg, per_aoa_results, output_dir,
):
    if len(aoa_sweep_deg) != len(per_aoa_results):
        raise ValueError(
            f"aoa_sweep_deg has {len(aoa_sweep_deg)} values but "
            f"per_aoa_results has {len(per_aoa_results)} -- must match 1:1"
        )
    for aoa, record in zip(aoa_sweep_deg, per_aoa_results):
        if record["cfd"]["status"] != "converged" and record["fea"] is not None:
            raise ValueError(
                f"AoA {aoa}: fea result present but cfd status is "
                f"'{record['cfd']['status']}' (not converged) -- Stage 8 "
                "should never have run for this AoA"
            )

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"airfoil_{name}.h5")

    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["airfoil_name"] = name
        meta.attrs["source_file"] = source_file
        meta.attrs["reynolds"] = reynolds
        meta.attrs["span"] = span
        meta.attrs["spar_locations"] = np.array(spar_locations, dtype=float)
        meta.attrs["rib_spacing"] = rib_spacing
        meta.create_dataset("aoa_sweep_deg", data=np.array(aoa_sweep_deg, dtype=float))

        for i, (aoa, record) in enumerate(zip(aoa_sweep_deg, per_aoa_results)):
            aoa_group = f.create_group(f"aoa_{i:02d}")
            aoa_group.attrs["aoa_deg"] = aoa

            cfd_in = record["cfd"]
            cfd = aoa_group.create_group("cfd")
            cfd.attrs["status"] = cfd_in["status"]
            for key in ("Cl", "Cd", "Cl_xfoil", "Cd_xfoil"):
                cfd.attrs[key] = cfd_in[key] if cfd_in[key] is not None else np.nan
            if cfd_in["pressure_vs_arc_length"] is not None:
                cfd.create_dataset(
                    "pressure_vs_arc_length",
                    data=np.array(cfd_in["pressure_vs_arc_length"], dtype=float),
                )

            fea_in = record["fea"]
            if fea_in is not None:
                fea = aoa_group.create_group("fea")
                fea.attrs["max_von_mises"] = fea_in["max_von_mises_pa"]
                fea.attrs["max_stress_location"] = fea_in["max_stress_node"]
                fea.attrs["reaction_force_residual"] = fea_in["reaction_force_residual_n"]
                node_ids = np.array(list(fea_in["stress_field"].keys()), dtype=np.int64)
                von_mises = np.array(list(fea_in["stress_field"].values()), dtype=float)
                fea.create_dataset("stress_field_node_ids", data=node_ids)
                fea.create_dataset("stress_field_von_mises", data=von_mises)

    return {"h5_path": h5_path}
