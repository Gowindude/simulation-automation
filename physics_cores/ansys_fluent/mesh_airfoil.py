"""
2D Airfoil Meshing Tool for the Aero-Structural Optimisation System

Generates a 2D triangular mesh around an airfoil profile using
PyFluent's meshing mode and the dedicated 2D meshing workflow.

The workflow is:
    1. Launch Fluent in MESHING mode (double precision).
    2. Load a .fmd geometry file containing the airfoil + far-field domain.
    3. Configure sizing controls (global, body-of-influence, edge, curvature).
    4. Add boundary layers on the airfoil wall for BL resolution.
    5. Generate the 2D surface mesh.
    6. Export as .msh.h5 — ready for the solver in cfd_tool.py.

Key constraint:
    PyFluent's 2D meshing mode CANNOT switch to solver mode directly.
    The mesh must be exported as .msh.h5, then a separate solver session
    must be launched to read it. This is a PyFluent API limitation.

Prerequisites:
    - Licensed copy of Ansys Fluent (same as cfd_tool.py)
    - pip install ansys-fluent-core
"""

import os
import ansys.fluent.core as pyfluent
from ansys.fluent.core import examples


def generate_airfoil_mesh(
    geometry_file: str | None = None,
    output_mesh: str = "airfoil_2d.msh.h5",
    length_unit: str = "m",
) -> str:
    """
    Generate a 2D mesh around an airfoil using PyFluent's 2D meshing workflow.

    Args:
        geometry_file: Path to a .fmd geometry file. If None, downloads the
                       NACA0012 example from PyFluent's built-in examples.
        output_mesh:   Output path for the exported .msh.h5 mesh file.
        length_unit:   Unit of the geometry file dimensions ("m", "mm", etc.).

    Returns:
        Absolute path to the generated .msh.h5 file.
    """
    # ── Step 1: Get or download the geometry file ──────────────────────────
    # The 2D meshing workflow needs a .fmd (Fluent Meshing Data) file, which
    # contains the airfoil shape + surrounding far-field domain as CAD geometry.
    # If the user hasn't provided one, we fall back to PyFluent's built-in
    # NACA0012 example — this lets us test the full pipeline immediately.
    if geometry_file is None:
        geometry_file = examples.download_file("NACA0012.fmd", "pyfluent/airfoils")
        print(f"Using built-in NACA0012 geometry: {geometry_file}")
        # The PyFluent example NACA0012.fmd uses millimetres internally.
        length_unit = "mm"
    else:
        print(f"Using provided geometry: {geometry_file}")

    # ── Step 2: Launch Fluent in meshing mode ──────────────────────────────
    # We use double precision because the mesh quality metrics (skewness,
    # aspect ratio) are computed more accurately at 64-bit. Meshing is not
    # as sensitive as the solver, but it avoids edge cases in thin BL cells.
    print("Launching Fluent in meshing mode...")
    meshing = pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.MESHING,
        precision=pyfluent.Precision.DOUBLE,
        processor_count=4,
    )

    # ── Step 3: Initialise the 2D meshing workflow ─────────────────────────
    # two_dimensional_meshing() returns a workflow object with task-
    # specific attributes like load_cad_geometry, define_global_sizing, etc.
    # This is the PyFluent-native way to do 2D meshing (as opposed to the
    # classic TaskObject["..."] journal-style API).
    two_d = meshing.two_dimensional_meshing()

    # Load the .fmd geometry into the mesher.
    load_cad = two_d.load_cad_geometry
    load_cad.file_name = geometry_file
    load_cad.length_unit = length_unit
    # Refaceting re-triangulates the imported CAD surface. We disable it
    # because the .fmd already has the right tessellation from SpaceClaim.
    load_cad.refaceting.refacet = False
    load_cad()
    print("Geometry loaded successfully.")

