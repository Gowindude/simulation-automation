"""
Verification tests for Stage 9 -- Aggregation.

Contract (from .claude/airfoil_pipeline_build_spec.md, Stage 9 +
"Final Output Schema", verbatim):
    Input: Stage 4 + Stage 8 outputs across the AoA sweep for one airfoil
    Output: single structured record (JSON metadata + array data -- HDF5
            or .npz) per airfoil: geometry, AoA sweep, pressure fields,
            stress fields

    airfoil_<name>.h5
    |-- metadata
    |   |-- airfoil_name: str
    |   |-- source_file: str
    |   |-- reynolds: float
    |   |-- span, spar_locations, rib_spacing: as locked above
    |   `-- aoa_sweep_deg: [list of 3-5 values]
    |
    |-- per AoA (repeated for each value in aoa_sweep_deg):
    |   |-- cfd/
    |   |   |-- status: "converged" | "non_converged" | "diverged" | "crashed"
    |   |   |-- Cl, Cd: float
    |   |   |-- Cl_xfoil, Cd_xfoil: float
    |   |   `-- pressure_vs_arc_length: array of (s, Cp) pairs
    |   `-- fea/
    |       |-- max_von_mises: float
    |       |-- max_stress_location: node id or (x, y, z)
    |       |-- reaction_force_residual: float
    |       `-- stress_field: array over shell nodes/elements
    |
    `-- failed cases recorded explicitly (never silently dropped)

Design decisions locked before writing this suite:
  - Format: HDF5 via h5py (confirmed installed, v3.16.0) -- matches the
    spec's own illustrated schema literally, one file per airfoil.
  - Input shape: a list of per-AoA records, each `{"cfd": <Stage 4's
    build_cfd_record() dict>, "fea": <Stage 8's run_calculix_analysis()
    dict> | None}`, in the same order as `aoa_sweep_deg`. `fea` is None
    whenever `cfd["status"] != "converged"` (Stages 5-8 never ran for
    that AoA) -- this is what "failed cases recorded explicitly, never
    silently dropped" means concretely: the AoA's `cfd/` group is always
    written (with its real status), and `fea/` is written as an explicit
    empty/absent group rather than the whole AoA being omitted from the
    file.
  - `max_stress_location` stored as the raw CalculiX shell-expansion
    node id Stage 8 returns (`max_stress_node`) -- per the schema's own
    "node id or (x, y, z)" allowance, node id is simplest and exactly
    what Stage 8 already computes.
  - `stress_field` stored as two parallel arrays (node_ids, von_mises)
    rather than a dict, since HDF5 datasets are typed arrays, not maps.

Verification approach: write a real `.h5` file (not fixtures/mocks) with
both a converged and a non-converged synthetic AoA record, then read it
back with h5py directly and check every value round-trips exactly --
this is the on-disk artifact any actual neural-operator dataloader would
read, so it needs to be verified as the real artifact, not the
in-memory dict Stage 9 built it from.
"""

import os

import h5py
import numpy as np
import pytest

from pipeline.stage9_aggregation import aggregate_airfoil_record

SPAN = 3.0
SPAR_LOCATIONS = (0.2, 0.6)
RIB_SPACING = 0.5
AOA_SWEEP = [-2.0, 2.0, 6.0, 10.0, 14.0]


def _converged_cfd_record(aoa):
    return {
        "status": "converged",
        "Cl": 0.1 + 0.01 * aoa,
        "Cd": 0.02 + 0.001 * aoa,
        "Cl_xfoil": 0.11 + 0.01 * aoa,
        "Cd_xfoil": 0.015 + 0.001 * aoa,
        "xfoil_converged": True,
        "pressure_vs_arc_length": [[0.0, 1.0], [0.5, -0.8], [1.0, 0.1]],
    }


def _fea_record():
    return {
        "max_von_mises_pa": 4.0e5,
        "max_stress_node": 42,
        "reaction_force_n": (1.0, 2.0, 0.0),
        "reaction_force_residual_n": 0.05,
        "stress_field": {1: 1.0e5, 2: 2.0e5, 42: 4.0e5, 100: 3.5e5},
    }


def _failed_cfd_record(status):
    return {
        "status": status,
        "Cl": None, "Cd": None, "Cl_xfoil": None, "Cd_xfoil": None,
        "xfoil_converged": None, "pressure_vs_arc_length": None,
    }


@pytest.fixture(scope="module")
def mixed_per_aoa_results():
    """4 converged + 1 non_converged (to exercise the "never silently
    dropped" requirement), matching AOA_SWEEP's 5 values."""
    results = [
        {"cfd": _converged_cfd_record(AOA_SWEEP[0]), "fea": _fea_record()},
        {"cfd": _converged_cfd_record(AOA_SWEEP[1]), "fea": _fea_record()},
        {"cfd": _failed_cfd_record("non_converged"), "fea": None},
        {"cfd": _converged_cfd_record(AOA_SWEEP[3]), "fea": _fea_record()},
        {"cfd": _converged_cfd_record(AOA_SWEEP[4]), "fea": _fea_record()},
    ]
    return results


@pytest.fixture(scope="module")
def default_result(mixed_per_aoa_results, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage9")
    return aggregate_airfoil_record(
        name="naca0012", source_file="tests/fixtures/naca0012.dat",
        reynolds=5e5, span=SPAN, spar_locations=SPAR_LOCATIONS,
        rib_spacing=RIB_SPACING, aoa_sweep_deg=AOA_SWEEP,
        per_aoa_results=mixed_per_aoa_results, output_dir=str(out_dir),
    )


@pytest.fixture(scope="module")
def h5_file(default_result):
    with h5py.File(default_result["h5_path"], "r") as f:
        yield f


# --- 1. Output artifact validity ---------------------------------------------


def test_h5_file_exists_and_opens(default_result):
    assert os.path.exists(default_result["h5_path"])
    assert os.path.getsize(default_result["h5_path"]) > 0
    with h5py.File(default_result["h5_path"], "r") as f:
        assert "metadata" in f


# --- 2. Metadata matches the spec's schema exactly ----------------------------


def test_metadata_fields(h5_file):
    meta = h5_file["metadata"]
    assert meta.attrs["airfoil_name"] == "naca0012"
    assert meta.attrs["source_file"] == "tests/fixtures/naca0012.dat"
    assert meta.attrs["reynolds"] == pytest.approx(5e5)
    assert meta.attrs["span"] == pytest.approx(SPAN)
    assert list(meta.attrs["spar_locations"]) == pytest.approx(list(SPAR_LOCATIONS))
    assert meta.attrs["rib_spacing"] == pytest.approx(RIB_SPACING)
    assert list(meta["aoa_sweep_deg"][:]) == pytest.approx(AOA_SWEEP)


# --- 3. Per-AoA cfd/ group -----------------------------------------------------


def test_cfd_group_present_for_every_aoa(h5_file):
    for i in range(len(AOA_SWEEP)):
        assert f"aoa_{i:02d}/cfd" in h5_file, f"missing cfd/ group for AoA index {i}"


def test_converged_cfd_values_round_trip(h5_file, mixed_per_aoa_results):
    for i, record in enumerate(mixed_per_aoa_results):
        if record["cfd"]["status"] != "converged":
            continue
        cfd = h5_file[f"aoa_{i:02d}/cfd"]
        assert cfd.attrs["status"] == "converged"
        assert cfd.attrs["Cl"] == pytest.approx(record["cfd"]["Cl"])
        assert cfd.attrs["Cd"] == pytest.approx(record["cfd"]["Cd"])
        assert cfd.attrs["Cl_xfoil"] == pytest.approx(record["cfd"]["Cl_xfoil"])
        assert cfd.attrs["Cd_xfoil"] == pytest.approx(record["cfd"]["Cd_xfoil"])
        pval = np.array(record["cfd"]["pressure_vs_arc_length"])
        assert cfd["pressure_vs_arc_length"][:] == pytest.approx(pval)


# --- 4. "Failed cases recorded explicitly, never silently dropped" -----------


def test_non_converged_aoa_is_present_not_omitted(h5_file, mixed_per_aoa_results):
    failed_idx = next(
        i for i, r in enumerate(mixed_per_aoa_results) if r["cfd"]["status"] != "converged"
    )
    assert f"aoa_{failed_idx:02d}" in h5_file, (
        "the non-converged AoA is missing from the file entirely -- "
        "failed cases must be recorded explicitly, not silently dropped"
    )
    cfd = h5_file[f"aoa_{failed_idx:02d}/cfd"]
    assert cfd.attrs["status"] == "non_converged"


def test_failed_aoa_has_no_fea_group(h5_file, mixed_per_aoa_results):
    failed_idx = next(
        i for i, r in enumerate(mixed_per_aoa_results) if r["cfd"]["status"] != "converged"
    )
    assert f"aoa_{failed_idx:02d}/fea" not in h5_file


def test_aoa_sweep_length_matches_group_count(h5_file):
    aoa_groups = [k for k in h5_file.keys() if k.startswith("aoa_")]
    assert len(aoa_groups) == len(AOA_SWEEP)


# --- 5. Per-AoA fea/ group (converged cases only) -----------------------------


def test_fea_group_values_round_trip(h5_file, mixed_per_aoa_results):
    for i, record in enumerate(mixed_per_aoa_results):
        if record["fea"] is None:
            continue
        fea = h5_file[f"aoa_{i:02d}/fea"]
        assert fea.attrs["max_von_mises"] == pytest.approx(record["fea"]["max_von_mises_pa"])
        assert fea.attrs["max_stress_location"] == record["fea"]["max_stress_node"]
        assert fea.attrs["reaction_force_residual"] == pytest.approx(
            record["fea"]["reaction_force_residual_n"]
        )
        node_ids = fea["stress_field_node_ids"][:]
        von_mises = fea["stress_field_von_mises"][:]
        expected = record["fea"]["stress_field"]
        got = dict(zip(node_ids.tolist(), von_mises.tolist()))
        assert got == pytest.approx(expected)


# --- Error handling ------------------------------------------------------------


def test_mismatched_sweep_and_results_length_raises(tmp_path):
    with pytest.raises(ValueError):
        aggregate_airfoil_record(
            name="naca0012", source_file="x.dat", reynolds=5e5,
            span=SPAN, spar_locations=SPAR_LOCATIONS, rib_spacing=RIB_SPACING,
            aoa_sweep_deg=[0.0, 4.0], per_aoa_results=[{"cfd": _converged_cfd_record(0.0), "fea": _fea_record()}],
            output_dir=str(tmp_path),
        )


def test_fea_present_for_non_converged_raises(tmp_path):
    """A converged-looking fea record paired with a non-converged cfd
    status is a contract violation the module should catch, not persist
    silently as inconsistent data."""
    with pytest.raises(ValueError):
        aggregate_airfoil_record(
            name="naca0012", source_file="x.dat", reynolds=5e5,
            span=SPAN, spar_locations=SPAR_LOCATIONS, rib_spacing=RIB_SPACING,
            aoa_sweep_deg=[0.0],
            per_aoa_results=[{"cfd": _failed_cfd_record("crashed"), "fea": _fea_record()}],
            output_dir=str(tmp_path),
        )
