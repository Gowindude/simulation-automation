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
