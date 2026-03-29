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


class GeometryAgent:
    """
    Autonomous agent that preprocesses raw .dat airfoil coordinates for CFD.

    The agent makes its own engineering decisions about:
      - Trailing edge treatment (sharpen blunt TE vs snap sharp TE)
      - Point distribution (cosine spacing for gradient resolution)

    Every decision is logged with reasoning so a human can audit it.
    """

    # Trailing edge gap threshold as a fraction of chord length.
    # If the TE gap exceeds this, the TE is "blunt" and needs sharpening.
    # 0.5% is the standard cutoff in aerospace — thinner than that is
    # effectively sharp at the mesh resolution we're using.
    TE_GAP_THRESHOLD = 0.005

    def __init__(self, dat_path: str, chord_m: float = 1.0, output_dir: str = "data/raw"):
        """
        Initialise the agent with a raw .dat file and chord length.

        Args:
            dat_path:   Path to the raw UIUC .dat airfoil coordinate file.
            chord_m:    Physical chord length in metres. The .dat file has
                        normalised (x/c, y/c) coords; we multiply by this
                        to get SI metres for the CFD solver.
            output_dir: Directory to write cleaned outputs into.
        """
        self.dat_path = dat_path
        self.chord_m = chord_m
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Derive the airfoil name from the filename (e.g. "naca0012" from "naca0012.dat").
        self.name = os.path.splitext(os.path.basename(dat_path))[0]

        # Set up a dedicated logger that writes to both console and a log file.
        # This is the agent's "reasoning journal" — every decision gets recorded
        # so a human can review WHY a particular treatment was chosen.
        self.log_path = os.path.join(output_dir, f"{self.name}_processing.log")
        self.logger = logging.getLogger(f"GeometryAgent.{self.name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # avoid duplicate handlers on re-init

        # File handler — permanent record of the agent's reasoning.
        fh = logging.FileHandler(self.log_path, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)

        # Console handler — so the user sees progress in real time.
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[GeometryAgent] %(message)s"))
        self.logger.addHandler(ch)

        self.logger.info(f"Agent initialised for '{self.name}'")
        self.logger.info(f"  Source file : {dat_path}")
        self.logger.info(f"  Chord length: {chord_m} m")

        # Load the raw coordinates immediately — they're small (< 1 KB).
        self.raw_coords = self._load_dat(dat_path)
        self.logger.info(f"  Raw points  : {len(self.raw_coords)}")

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load_dat(self, path: str) -> np.ndarray:
        """
        Parse a UIUC .dat file into an Nx2 numpy array of (x/c, y/c).

        Skips the first line (airfoil name) and any lines that don't
        contain exactly two floats. Returns normalised coordinates.
        """
        coords = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip the name header
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        return np.array(coords)

    def _analyze_trailing_edge(self) -> dict:
        """
        Inspect the trailing edge and decide on treatment.

        The TE gap is the Euclidean distance between the first and last
        coordinate points. In Selig format, the first point is the upper
        surface at the TE and the last point is the lower surface at the TE.

        Returns a dict with the analysis results and chosen treatment.
        """
        first = self.raw_coords[0]
        last = self.raw_coords[-1]

        # Euclidean distance between the two TE points, normalised by chord.
        gap = np.linalg.norm(first - last)
        gap_pct = gap * 100  # as percentage of chord

        self.logger.info("─── Trailing Edge Analysis ───")
        self.logger.info(f"  Upper TE point : ({first[0]:.6f}, {first[1]:.6f})")
        self.logger.info(f"  Lower TE point : ({last[0]:.6f}, {last[1]:.6f})")
        self.logger.info(f"  TE gap         : {gap:.6f} c  ({gap_pct:.3f}% chord)")
        self.logger.info(f"  Threshold      : {self.TE_GAP_THRESHOLD * 100:.1f}% chord")

        if gap > self.TE_GAP_THRESHOLD:
            treatment = "sharpen"
            reason = (
                f"TE gap ({gap_pct:.3f}%) exceeds {self.TE_GAP_THRESHOLD * 100:.1f}% threshold. "
                f"This is a blunt trailing edge — common on thick utility airfoils. "
                f"A blunt TE creates excess pressure drag due to the base-pressure deficit "
                f"behind the flat TE face. Sharpening via cubic spline brings the upper "
                f"and lower surfaces together smoothly, eliminating the drag penalty."
            )
        else:
            treatment = "snap"
            reason = (
                f"TE gap ({gap_pct:.3f}%) is within {self.TE_GAP_THRESHOLD * 100:.1f}% threshold. "
                f"This is effectively sharp — the gap is smaller than a typical BL cell. "
                f"Snapping both TE points to their midpoint guarantees a watertight "
                f"geometry for the mesher without altering the aerodynamic shape."
            )

        self.logger.info(f"  Decision       : {treatment.upper()}")
        self.logger.info(f"  Reasoning      : {reason}")

        return {"gap": gap, "gap_pct": gap_pct, "treatment": treatment, "reason": reason}

