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

    def set_boundary_conditions(self, inlet_velocity: float = 50.0):
        """
        Configure the physics: set the turbulence model, working fluid, and inlet velocity.

        Args:
            inlet_velocity (float): Freestream velocity at the inlet in m/s (SI). Default 50 m/s.
        """
        # ---- Turbulence Model ----
        # k-omega SST (Shear Stress Transport) blends k-omega near the wall and k-epsilon
        # in the freestream. This makes it much better at capturing the flow separation
        # that happens on the airfoil's suction side and near the trailing edge — exactly
        # the regions that drive surface pressure errors.
        self.solver.setup.models.viscous.model = "k-omega-sst"
        print("Turbulence model set to k-omega SST.")

        # ---- Working Fluid ----
        # Air at sea-level ISA conditions. Density and viscosity are already set
        # by default in Fluent's material library, so we just need to reference it.
        self.solver.setup.materials.fluid["air"]  # ensures air is the active fluid

        # ---- Inlet Boundary Condition ----
        # "velocity-inlet" named zone. The zone name "inlet" must match what is
        # defined in the mesh — if you rename it in the mesher, update here too.
        inlet = self.solver.setup.boundary_conditions.velocity_inlet["inlet"]
        # Set the magnitude — value comes in as m/s (SI), Fluent expects m/s by default.
        inlet.momentum.velocity_magnitude.value = inlet_velocity
        print(f"Velocity inlet set to {inlet_velocity} m/s (SI).")

