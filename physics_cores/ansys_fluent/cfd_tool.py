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
