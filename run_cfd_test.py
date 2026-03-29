"""
End-to-End CFD Integration Test

Runs the full pipeline:
  1. Generate a 2D mesh (NACA0012 via PyFluent meshing)
  2. Launch the Fluent solver
  3. Set boundary conditions (50 m/s inlet, k-omega SST)
  4. Run the simulation
  5. Export pressure_dist.csv
  6. Validate the output

Prerequisites:
    - Licensed Ansys Fluent installation with AWP_ROOT<ver> set
    - pip install ansys-fluent-core
"""

import os
import csv
import sys

# Add project root to path so we can import our modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

        # Check header columns.
        expected = {"x_m", "y_m", "pressure_Pa"}
        if not expected.issubset(set(reader.fieldnames or [])):
            print(f"  FAIL: Expected columns {expected}, got {reader.fieldnames}")
            return False

        # Check at least one data row with numeric values.
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


def main():
    """Run the full CFD pipeline and report pass/fail for each step."""
    output_csv = "data/raw/pressure_dist.csv"
    mesh_path = "airfoil_2d.msh.h5"

    print("=" * 60)
    print("  CFD Integration Test — Full Pipeline")
    print("=" * 60)

    # ── Step 1: Launch solver + generate/load mesh ─────────────────────────
    # FluidistAgent launches Fluent in solver mode. generate_or_load_mesh
    # will detect that no .msh.h5 exists and trigger mesh_airfoil.py to
    # generate one using the NACA0012 example geometry from PyFluent.
    print("\n[1/5] Launching Fluent solver...")
    agent = FluidistAgent(show_gui=False)
    print("  PASS: Fluent solver launched.")

    print("\n[2/5] Generating/loading mesh...")
    # Pass dummy coords — in the current workflow the actual geometry comes
    # from the .fmd file, not from these coordinates. These are stored for
    # reference only (e.g. for later structural mapping).
    sample_coords = [(0.0, 0.0), (0.5, 0.06), (1.0, 0.0)]
    agent.generate_or_load_mesh(sample_coords, mesh_path=mesh_path)
    print("  PASS: Mesh loaded into solver.")

    # ── Step 2: Set boundary conditions ────────────────────────────────────
    print("\n[3/5] Setting boundary conditions (50 m/s, k-omega SST)...")
    agent.set_boundary_conditions(inlet_velocity=50.0)
    print("  PASS: Boundary conditions set.")

    # ── Step 3: Run the simulation ─────────────────────────────────────────
    print("\n[4/5] Running simulation (300 iterations)...")
    agent.run_simulation(iterations=300)
    print("  PASS: Simulation complete.")

    # ── Step 4: Export pressure CSV ────────────────────────────────────────
    print("\n[5/5] Exporting pressure distribution...")
    agent.export_pressure_csv(output_path=output_csv)

    # ── Step 5: Clean up ───────────────────────────────────────────────────
    agent.close()

    # ── Validation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Validation")
    print("=" * 60)
    passed = validate_pressure_csv(output_csv)

    print("\n" + "=" * 60)
    if passed:
        print("  ✅ ALL TESTS PASSED")
    else:
        print("  ❌ TESTS FAILED")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
