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

    # ── Step 4: Update boundaries ──────────────────────────────────────────
    # Auto-detect boundary zones from the imported geometry. Using "zone"
    # selection type tells the mesher to name boundaries based on the zone
    # labels defined in the .fmd file (e.g. "airfoil", "inlet", "outlet").
    update_bnd = two_d.update_boundaries
    update_bnd.selection_type = "zone"
    update_bnd()
    print("Boundaries updated from geometry zones.")

    # ── Step 5: Define global sizing ───────────────────────────────────────
    # These controls set the coarsest / finest cell sizes for the whole domain.
    # curvature_normal_angle: max angle (degrees) between adjacent face normals.
    #   Lower = more cells on curved edges. 20° is a good balance for airfoils.
    # max_size: largest cell anywhere in the far field (keeps the mesh small).
    # min_size: smallest cell allowed globally (prevents over-refinement).
    # size_functions: "Curvature" adapts cell size to match surface curvature —
    #   crucial for the leading edge where the radius of curvature is tiny.
    global_sz = two_d.define_global_sizing
    global_sz.curvature_normal_angle = 20
    global_sz.max_size = 2000.0
    global_sz.min_size = 5.0
    global_sz.size_functions = "Curvature"
    global_sz()
    print("Global sizing defined (curvature-based, min=5, max=2000).")

    # ── Step 6: Add local sizing controls ──────────────────────────────────
    # We add three refinement zones to capture the physics accurately:
    local_sz = two_d.add_local_sizing_wtm

    # 6a) Body of Influence (BOI) — a region around the airfoil where we
    #     force smaller cells. This captures the wake and pressure gradients
    #     that extend downstream of the trailing edge.
    local_sz.add_child = "yes"
    local_sz.boi_control_name = "boi_1"
    local_sz.boi_execution = "Body Of Influence"
    local_sz.boi_face_label_list = ["boi"]
    local_sz.boi_size = 50.0
    local_sz.boi_zoneor_label = "label"
    local_sz.draw_size_control = True
    local_sz.add_child_and_update(defer_update=False)
    print("  Local sizing: Body of Influence added.")

    # 6b) Edge sizing at the trailing edge — the TE has a sharp geometric
    #     discontinuity where flow separation occurs, so we need very fine
    #     cells there to capture the pressure drop accurately.
    local_sz.add_child = "yes"
    local_sz.boi_control_name = "edgesize_1"
    local_sz.boi_execution = "Edge Size"
    local_sz.boi_size = 5.0
    local_sz.boi_zoneor_label = "label"
    local_sz.draw_size_control = True
    local_sz.edge_label_list = ["airfoil-te"]
    local_sz.add_child_and_update(defer_update=False)
    print("  Local sizing: Trailing edge refinement added.")

    # 6c) Curvature-based sizing on the airfoil surface — the leading edge
    #     has a very small radius of curvature, so we refine heavily there
    #     to resolve the stagnation point and suction peak.
    local_sz.add_child = "yes"
    local_sz.boi_control_name = "curvature_1"
    local_sz.boi_curvature_normal_angle = 10
    local_sz.boi_execution = "Curvature"
    local_sz.boi_max_size = 2
    local_sz.boi_min_size = 1.5
    local_sz.boi_scope_to = "edges"
    local_sz.boi_zoneor_label = "label"
    local_sz.draw_size_control = True
    local_sz.edge_label_list = ["airfoil"]
    local_sz.add_child_and_update(defer_update=False)
    print("  Local sizing: Airfoil curvature refinement added.")

    # ── Step 7: Add boundary layers ────────────────────────────────────────
    # Boundary layers are thin, structured rows of cells stacked on the
    # airfoil wall. They are essential for RANS simulations because the
    # turbulence model (k-omega SST) needs to resolve the velocity gradient
    # inside the boundary layer — if the first cell is too thick, the wall
    # shear stress and pressure predictions will be wildly wrong.
    # We use the aspect-ratio method with 4 layers as a starting point;
    # for production runs you'd increase to 10-20 layers with a growth ratio.
    add_bl = two_d.add_2d_boundary_layers
    add_bl.add_child = "yes"
    add_bl.control_name = "aspect-ratio_1"
    add_bl.number_of_layers = 4
    add_bl.offset_method_type = "aspect-ratio"
    add_bl.add_child_and_update(defer_update=False)
    print("Boundary layers added (4 layers, aspect-ratio method).")

    # ── Step 8: Generate the 2D surface mesh ───────────────────────────────
    # show_advanced_options must be True to access the zone-merging options.
    # We disable zone merging so each boundary (airfoil, inlet, outlet, etc.)
    # keeps its own named zone — the solver needs these names to apply BCs.
    gen_mesh = two_d.generate_initial_surface_mesh
    prefs = gen_mesh.surface_2d_preferences
    prefs.show_advanced_options = True
    prefs.merge_edge_zones_based_on_labels = "no"
    prefs.merge_face_zones_based_on_labels = "no"
    gen_mesh()
    print("2D surface mesh generated.")

    # ── Step 9: Export the mesh as .msh.h5 ─────────────────────────────────
    # We can't switch to solver mode from 2D meshing (PyFluent limitation),
    # so we export the mesh to disk. The solver session in cfd_tool.py will
    # read this file via settings.file.read_case().
    output_mesh = os.path.abspath(output_mesh)
    meshing.tui.file.export.fluent_2d_mesh(output_mesh)
    print(f"Mesh exported to: {output_mesh}")

    # ── Step 10: Clean up ──────────────────────────────────────────────────
    meshing.exit()
    print("Meshing session closed.")

    return output_mesh


# ─── CLI entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a 2D airfoil mesh using PyFluent meshing."
    )
    parser.add_argument(
        "--geometry",
        default=None,
        help="Path to .fmd geometry file. Default: uses built-in NACA0012.",
    )
    parser.add_argument(
        "--output",
        default="airfoil_2d.msh.h5",
        help="Output .msh.h5 file path (default: airfoil_2d.msh.h5).",
    )
    parser.add_argument(
        "--unit",
        default="m",
        help="Length unit in the geometry file (default: m).",
    )
    args = parser.parse_args()

    mesh_path = generate_airfoil_mesh(
        geometry_file=args.geometry,
        output_mesh=args.output,
        length_unit=args.unit,
    )
    print(f"\nDone! Mesh ready at: {mesh_path}")
