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

    def _build_c_domain(self, chord: float) -> dict[str, np.ndarray]:
        """
        Generate a C-shaped far-field domain around the airfoil.

        The C-domain is the standard choice for external aero because:
          - The semicircular inlet captures flow from all upstream angles.
          - The straight downstream section lets the wake develop naturally.
          - It's much more mesh-efficient than a rectangular domain.

        Distances are in the same units as the airfoil (chord * self.chord_m).
        Standard CFD practice: 20c upstream, 40c downstream, 20c top/bottom.
        """
        # Domain extents (multiples of chord length).
        r_inlet = 20.0 * chord   # semicircle radius for the upstream inlet
        x_outlet = 40.0 * chord  # downstream extent for the outlet

        # Semicircular inlet arc from top to bottom (180° arc centered at LE).
        n_arc = 80
        angles = np.linspace(np.pi / 2, -np.pi / 2, n_arc)
        inlet = np.column_stack([r_inlet * np.cos(angles), r_inlet * np.sin(angles)])

        # Straight top edge from inlet arc end to outlet.
        top = np.array([inlet[0], [x_outlet, r_inlet]])
        # Straight bottom edge from outlet to inlet arc end.
        bottom = np.array([[x_outlet, -r_inlet], inlet[-1]])
        # Outlet: vertical line at x_outlet.
        outlet = np.array([[x_outlet, r_inlet], [x_outlet, -r_inlet]])

        self.logger.info(f"  C-domain built: radius={r_inlet:.1f}, outlet_x={x_outlet:.1f}")
        return {"inlet": inlet, "top": top, "bottom": bottom, "outlet": outlet}

    def _write_dxf(
        self, airfoil: np.ndarray, domain: dict[str, np.ndarray], path: str
    ):
        """
        Write a minimal DXF file containing the airfoil and domain boundaries.

        DXF is a text-based CAD exchange format that Fluent Meshing can import
        directly. We write it by hand (no external libs) because we only need
        LWPOLYLINE entities — the simplest DXF entity type.

        Each boundary (airfoil, inlet, top, bottom, outlet) is written as a
        separate polyline on its own layer so the mesher can assign boundary
        types by layer name.
        """
        def write_polyline(f, points: np.ndarray, layer: str, closed: bool = False):
            """Write a single LWPOLYLINE entity to the DXF file."""
            f.write("0\nLWPOLYLINE\n")
            f.write(f"8\n{layer}\n")          # layer name → becomes zone name
            f.write(f"90\n{len(points)}\n")   # number of vertices
            f.write(f"70\n{1 if closed else 0}\n")  # 1 = closed polyline
            for pt in points:
                f.write(f"10\n{pt[0]:.8f}\n")  # X coordinate
                f.write(f"20\n{pt[1]:.8f}\n")  # Y coordinate

        with open(path, "w") as f:
            # DXF header — minimal required sections.
            f.write("0\nSECTION\n2\nENTITIES\n")

            # Airfoil polyline — closed loop.
            write_polyline(f, airfoil, layer="airfoil", closed=True)

            # Domain boundaries — each on its own named layer.
            for name, pts in domain.items():
                write_polyline(f, pts, layer=name, closed=False)

            # End of entities section and file.
            f.write("0\nENDSEC\n0\nEOF\n")

        self.logger.info(f"  DXF written to: {path}")

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

