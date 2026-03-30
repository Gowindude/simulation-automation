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

    def _split_surfaces(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the coordinate array into upper and lower surfaces.

        Selig format goes from TE (upper) → LE → TE (lower). The leading
        edge is the point with the smallest x-coordinate. We find that
        index and split there.

        Returns:
            (upper, lower) — each is an Mx2 array, both starting from LE.
        """
        # The leading edge is the point closest to x=0.
        le_idx = np.argmin(coords[:, 0])

        # Upper surface: from TE down to LE. Reverse so it goes LE → TE.
        upper = coords[:le_idx + 1][::-1]
        # Lower surface: from LE down to TE. Already in LE → TE order.
        lower = coords[le_idx:]

        self.logger.info(f"  Split at LE index {le_idx}: "
                         f"upper={len(upper)} pts, lower={len(lower)} pts")
        return upper, lower

    def _treat_trailing_edge(
        self, upper: np.ndarray, lower: np.ndarray, treatment: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the chosen trailing edge treatment.

        'sharpen': use a cubic spline to smoothly bring the last ~10% of
                   each surface toward a shared TE point. This eliminates
                   the blunt-TE base pressure drag without adding a sharp
                   corner (which would cause mesh quality issues).

        'snap':    move both TE endpoints to their midpoint. Simple and
                   safe when the gap is already negligibly small.

        Returns updated (upper, lower) arrays.
        """
        if treatment == "snap":
            # Average the two TE points to get a single closed point.
            te_mid = (upper[-1] + lower[-1]) / 2.0
            upper[-1] = te_mid
            lower[-1] = te_mid
            self.logger.info(f"  Snapped TE to midpoint: ({te_mid[0]:.6f}, {te_mid[1]:.6f})")

        elif treatment == "sharpen":
            # Cubic-spline the last 10% of each surface toward a shared TE point.
            # We compute a blending target at x=1.0, y=0 (symmetric TE),
            # then smoothly transition the last few points toward it.
            te_target = np.array([(upper[-1, 0] + lower[-1, 0]) / 2, 0.0])

            for surface, label in [(upper, "upper"), (lower, "lower")]:
                n = len(surface)
                # Blend the last 10% of points toward the TE target.
                n_blend = max(3, int(0.1 * n))
                for i in range(n_blend):
                    # Linear blend weight: 0 at the start of the blend region, 1 at TE.
                    w = (i + 1) / n_blend
                    idx = n - n_blend + i
                    surface[idx] = (1 - w) * surface[idx] + w * te_target

                self.logger.info(f"  Sharpened {label} TE: blended last {n_blend} points "
                                 f"toward ({te_target[0]:.6f}, {te_target[1]:.6f})")

        return upper, lower

    def _apply_cosine_spacing(
        self, surface: np.ndarray, n_points: int = 150
    ) -> np.ndarray:
        """
        Re-sample a surface using cosine-distributed x-coordinates.

        Why cosine spacing? Uniform point distribution puts the same number
        of points on the flat mid-chord as on the highly-curved leading edge.
        That's terrible for CFD — the LE and TE need 5-10x more resolution
        because that's where the pressure gradients (and therefore the forces
        we're trying to predict) are steepest.

        Cosine spacing uses x = 0.5 * (1 - cos(θ)) for θ ∈ [0, π], which
        naturally clusters points at x=0 (LE) and x=1 (TE).

        Args:
            surface:  Mx2 array of (x, y) from LE to TE.
            n_points: Number of re-sampled points (default 150).

        Returns:
            n_points × 2 array with cosine-spaced coordinates.
        """
        # Build a cubic spline of y(x) along the surface.
        # We need to parameterise by arc length first, not x, because
        # the surface can be multi-valued in y near the LE.
        dx = np.diff(surface[:, 0])
        dy = np.diff(surface[:, 1])
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate(([0], np.cumsum(ds)))  # arc-length parameter
        s /= s[-1]  # normalise to [0, 1]

        # Spline x(s) and y(s) separately.
        spline_x = CubicSpline(s, surface[:, 0])
        spline_y = CubicSpline(s, surface[:, 1])

        # Cosine-distributed parameter values. The cos transform clusters
        # values near s=0 (LE) and s=1 (TE) where the curvature is highest.
        theta = np.linspace(0, np.pi, n_points)
        s_new = 0.5 * (1 - np.cos(theta))

        x_new = spline_x(s_new)
        y_new = spline_y(s_new)

        return np.column_stack([x_new, y_new])



    def _write_csv(self, coords: np.ndarray, path: str):
        """
        Write cleaned coordinates to a CSV file (x_m, y_m in SI metres).

        This is the handoff format for downstream tools that don't read DXF
        (e.g. the Structuralist agent needs surface coordinates for load mapping).
        """
        import csv as csv_mod
        with open(path, "w", newline="") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["x_m", "y_m"])
            for pt in coords:
                writer.writerow([f"{pt[0]:.8f}", f"{pt[1]:.8f}"])
        self.logger.info(f"  CSV written to: {path} ({len(coords)} points)")

    # ── Public interface ────────────────────────────────────────────────────

    def process(self, n_points: int = 150) -> dict:
        """
        Run the full autonomous preprocessing pipeline.

        This is the agent's main entry point. It chains all decisions together:
        analyse → split → treat TE → cosine resample → build domain → export.

        Every decision is logged to the processing log file so a human can
        review the agent's reasoning after the fact.

        Args:
            n_points: Number of cosine-spaced points per surface (default 150).

        Returns:
            Dict with paths to all generated output files.
        """
        self.logger.info("=" * 60)
        self.logger.info("  GEOMETRY AGENT — Starting autonomous processing")
        self.logger.info("=" * 60)

        # ── Phase 1: Trailing edge analysis ────────────────────────────────
        te_result = self._analyze_trailing_edge()

        # ── Phase 2: Split into upper/lower surfaces ──────────────────────
        self.logger.info("─── Surface Splitting ───")
        upper, lower = self._split_surfaces(self.raw_coords.copy())

        # ── Phase 3: Apply TE treatment ───────────────────────────────────
        self.logger.info("─── Trailing Edge Treatment ───")
        upper, lower = self._treat_trailing_edge(upper, lower, te_result["treatment"])

        # ── Phase 4: Cosine re-spacing ────────────────────────────────────
        self.logger.info("─── Cosine Re-spacing ───")
        self.logger.info(f"  Re-sampling each surface to {n_points} cosine-spaced points...")
        self.logger.info(f"  Before: upper={len(upper)} pts, lower={len(lower)} pts")

        upper_cs = self._apply_cosine_spacing(upper, n_points)
        lower_cs = self._apply_cosine_spacing(lower, n_points)
        self.logger.info(f"  After : upper={len(upper_cs)} pts, lower={len(lower_cs)} pts")

        # Combine back into a single closed loop:
        # upper_cs goes LE → TE. So upper_cs[::-1] goes TE → LE.
        # lower_cs goes LE → TE. 
        # So we stack TE(up) → LE ... LE → TE(low) to make a continuous clockwise loop.
        # Skip the duplicate LE point at the junction (lower_cs[1:]).
        airfoil_combined = np.vstack([upper_cs[::-1], lower_cs[1:]]) * self.chord_m
        self.logger.info(f"  Combined airfoil: {len(airfoil_combined)} points "
                         f"(scaled to {self.chord_m} m chord)")

        # ── Phase 5: Export clean coordinates ─────────────────────────────
        self.logger.info("─── Export CSV ───")
        csv_path = os.path.join(self.output_dir, f"{self.name}_cleaned.csv")
        self._write_csv(airfoil_combined, csv_path)

        # ── Phase 6: Generate CAD Domain (.step) ──────────────────────────
        self.logger.info("─── CAD Generation ───")
        from agents.cad_builder import CADBuilderAgent
        cad_agent = CADBuilderAgent(output_dir=self.output_dir)
        step_path = cad_agent.generate_domain_step(
            airfoil_coords=airfoil_combined,
            chord_m=self.chord_m,
            name=self.name
        )

        # ── Summary ───────────────────────────────────────────────────────
        self.logger.info("=" * 60)
        self.logger.info("  GEOMETRY AGENT — Processing complete")
        self.logger.info(f"  Airfoil      : {self.name}")
        self.logger.info(f"  TE treatment : {te_result['treatment']} "
                         f"(gap: {te_result['gap_pct']:.3f}%)")
        self.logger.info(f"  Points/surface: {n_points} (cosine-spaced)")
        self.logger.info(f"  Outputs      :")
        self.logger.info(f"    CSV  : {csv_path}")
        self.logger.info(f"    STEP : {step_path}")
        self.logger.info(f"    Log  : {self.log_path}")
        self.logger.info("=" * 60)

        return {
            "csv_path": csv_path,
            "step_path": step_path,
            "log_path": self.log_path,
            "te_analysis": te_result,
            "n_points_per_surface": n_points,
        }

# ─── CLI entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Geometry Agent: preprocess .dat airfoils for CFD."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the raw .dat airfoil coordinate file.",
    )
    parser.add_argument(
        "--chord", type=float, default=1.0,
        help="Chord length in metres (default: 1.0).",
    )
    parser.add_argument(
        "--output-dir", default="data/raw",
        help="Output directory for cleaned files (default: data/raw).",
    )
    parser.add_argument(
        "--points", type=int, default=150,
        help="Points per surface after cosine re-spacing (default: 150).",
    )
    args = parser.parse_args()

    agent = GeometryAgent(
        dat_path=args.input,
        chord_m=args.chord,
        output_dir=args.output_dir,
    )
    result = agent.process(n_points=args.points)

    print(f"\nDone! Files generated:")
    for key in ("csv_path", "dxf_path", "log_path"):
        print(f"  {key}: {result[key]}")
