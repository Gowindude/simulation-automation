"""
End-to-End CFD Integration Test

Runs the full pipeline:
  1. Download a raw .dat airfoil (NACA 0012-34)
  2. Geometry Agent: clean .dat, apply cosine spacing, export DXF & CSV
  3. Launch the Fluent solver
  4. Generate a 2D mesh using the DXF via PyFluent meshing & load it
  5. Set boundary conditions (50 m/s inlet, k-omega SST)
  6. Run the simulation
  7. Export pressure_dist.csv
  8. Validate the output

Prerequisites:
    - Licensed Ansys Fluent installation with AWP_ROOT<ver> set
    - pip install ansys-fluent-core numpy scipy requests
"""

import os
import csv
import sys
import requests

# Add project root to path so we can import our modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.airfoil_downloader import load_dat_file
from agents.geometry_agent import GeometryAgent
from physics_cores.ansys_fluent.cfd_tool import FluidistAgent


def validate_pressure_csv(csv_path: str) -> bool:
    """
    Check that the pressure CSV exists, has the right columns,
    and contains at least one row of numeric data.
    """
    if not os.path.exists(csv_path):
        print(f"  FAIL: {csv_path} does not exist.")
        return False

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        expected = {"x_m", "y_m", "pressure_Pa"}
        if not expected.issubset(set(reader.fieldnames or [])):
            print(f"  FAIL: Expected columns {expected}, got {reader.fieldnames}")
            return False

        rows = list(reader)
        if len(rows) == 0:
            print("  FAIL: CSV has headers but no data rows.")
            return False

        try:
            float(rows[0]["x_m"])
            float(rows[0]["y_m"])
            float(rows[0]["pressure_Pa"])
        except (ValueError, KeyError) as e:
            print(f"  FAIL: First row is not numeric: {e}")
            return False

    print(f"  PASS: {csv_path} has {len(rows)} data points with correct columns.")
    return True


def download_test_airfoil(output_dir: str) -> str:
    """Download a raw .dat file from UIUC for testing."""
    os.makedirs(output_dir, exist_ok=True)
    url = "https://m-selig.ae.illinois.edu/ads/coord_seligFmt/naca001234.dat"
    path = os.path.join(output_dir, "naca001234.dat")
    if not os.path.exists(path):
        r = requests.get(url, timeout=15)
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def main():
    """Run the full CFD pipeline and report pass/fail for each step."""
    output_dir = "data/raw"
    output_csv = os.path.join(output_dir, "pressure_dist.csv")
    mesh_path = "airfoil_2d.msh.h5"

    print("=" * 60)
    print("  CFD Integration Test — Full Pipeline")
    print("=" * 60)

    # ── Step 1: Download & Preprocess (Geometry Agent) ─────────────────────
    print("\n[1/6] Downloading & preprocessing airfoil geometry...")
    dat_path = download_test_airfoil(output_dir)
    geo_agent = GeometryAgent(dat_path=dat_path, chord_m=1.0, output_dir=output_dir)
    geo_result = geo_agent.process(n_points=150)
    print(f"  PASS: Geometry Agent generated DXF at {geo_result['dxf_path']}")

    # Load the cleaned coords from the CSV to pass to the FluidistAgent
    # (used for reference, e.g., structural mapping later)
    _, cleaned_coords = load_dat_file(dat_path)  # Just dummy loading here to test the API

    # ── Step 2: Launch solver ──────────────────────────────────────────────
    print("\n[2/6] Launching Fluent solver...")
    fluidist = FluidistAgent(show_gui=False)
    print("  PASS: Fluent solver launched.")

    # ── Step 3: Mesh & Load ────────────────────────────────────────────────
    print("\n[3/6] Generating/loading mesh from DXF...")
    fluidist.generate_or_load_mesh(
        coords=cleaned_coords,
        dxf_file=geo_result["dxf_path"],
        mesh_path=mesh_path,
    )
    print("  PASS: Mesh loaded into solver.")

    # ── Step 4: Set boundary conditions ────────────────────────────────────
    print("\n[4/6] Setting boundary conditions (50 m/s, k-omega SST)...")
    fluidist.set_boundary_conditions(inlet_velocity=50.0)
    print("  PASS: Boundary conditions set.")

    # ── Step 5: Run the simulation ─────────────────────────────────────────
    print("\n[5/6] Running simulation (300 iterations)...")
    fluidist.run_simulation(iterations=300)
    print("  PASS: Simulation complete.")

    # ── Step 6: Export pressure CSV ────────────────────────────────────────
    print("\n[6/6] Exporting pressure distribution...")
    fluidist.export_pressure_csv(output_path=output_csv)

    # ── Clean up ───────────────────────────────────────────────────────────
    fluidist.close()

    # ── Validation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Validation")
    print("=" * 60)
    passed = validate_pressure_csv(output_csv)

    if passed:
        print("  ✅ ALL TESTS PASSED")
    else:
        print("  ❌ TESTS FAILED")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

