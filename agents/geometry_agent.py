"""
Geometry Agent — Autonomous Airfoil Preprocessor for CFD

This agent takes a raw .dat airfoil coordinate file and makes autonomous
engineering decisions to prepare it for high-fidelity CFD simulation.

Responsibilities:
    1. Trailing Edge Analysis — detect blunt vs sharp TE, choose treatment
    2. Cosine Spacing — re-sample for LE/TE gradient clustering
    3. DXF Export — produce geometry that Fluent Meshing can import
    4. Far-Field Domain — generate C-shaped domain boundaries
    5. Processing Log — explain every decision the agent made and why

The agent reasons about each airfoil independently and logs its
engineering judgments so a human reviewer can audit the preprocessing.

Prerequisites:
    pip install numpy scipy
"""

import os
import math
import logging
import numpy as np
from scipy.interpolate import CubicSpline
