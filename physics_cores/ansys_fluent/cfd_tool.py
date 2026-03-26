"""
Fluidist Tool (CFD) for Aero-Structural Optimization System
"""
import os
import csv
import ansys.fluent.core as pyfluent

class FluidistAgent:
    """Agent responsible for running 2D aerodynamic analysis via Ansys Fluent."""

    def __init__(self, show_gui: bool = False):
        """
        Launch the Ansys Fluent solver and store the session as self.solver.

        Args:
            show_gui (bool): Set True to open the Fluent GUI for visual debugging.
                             False (default) runs in batch mode — faster for automation.
        """
        print("Launching Ansys Fluent...")

        # We use 2D double-precision ("2ddp") because airfoil boundary layers have
        # very thin, high-gradient regions where single-precision floating point
        # would accumulate error and give us garbage pressure readings.
        self.solver = pyfluent.launch_fluent(
            precision="double",     # double precision float for accuracy in thin BL regions
            processor_count=4,      # parallelise across 4 CPU cores to speed up the solve
            mode="solver",          # we want the solver, not the mesher
            version="2d",           # 2D planar mode — our airfoil cross-section lives in a plane
            show_gui=show_gui,
        )

        print("Fluent launched successfully.")

    def generate_or_load_mesh(self, coords: list, mesh_path: str = "airfoil_2d.msh"):
        """
        Check if a mesh file already exists and load it; otherwise write geometry
        coordinates to a temp file for the mesher to use later.

        Args:
            coords (list of tuple): [(x1, y1), ...] in Meters (SI).
            mesh_path (str): Path to an existing .msh case file.
        """
        # Store coords for reference — these are the raw (x,y) defining the airfoil profile.
        # All values are in Meters as per SI convention.
        self.coords = coords

        if os.path.exists(mesh_path):
            # If a mesh was already generated (e.g. from a previous run), skip re-meshing.
            # Re-meshing is expensive, so we reuse it when nothing has changed geometrically.
            print(f"Existing mesh found at '{mesh_path}'. Loading...")
            self.solver.file.read(file_type="case", file_name=mesh_path)
        else:
            # No mesh on disk — write the coordinates out to a CSV so the mesher
            # can pick them up. In a full workflow this feeds into Fluent Meshing
            # or SpaceClaim via a journal file.
            coords_path = os.path.splitext(mesh_path)[0] + "_coords.csv"
            with open(coords_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x_m", "y_m"])   # header — units are Meters
                writer.writerows(coords)
            print(f"No mesh found. Coordinates written to '{coords_path}' for mesher.")

