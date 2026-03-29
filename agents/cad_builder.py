"""
CAD Builder Agent for CFD Domains

Uses the `build123d` OpenCASCADE kernel to generate robust, watertight
3D/2D CAD faces from airfoil coordinate points.
Exports strict .step files that Ansys Fluent Meshing imports flawlessly.
"""

import os
import logging
import build123d as bd
import numpy as np

class CADBuilderAgent:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        self.logger = logging.getLogger("CADBuilderAgent")
        
        # Ensure output dir exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up console logging if not configured
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.INFO)

    def generate_domain_step(
        self, 
        airfoil_coords: np.ndarray, 
        chord_m: float = 1.0, 
        name: str = "airfoil"
    ) -> str:
        """
        Builds a 2D Face of a C-shaped fluid domain with the airfoil 
        subtracted as a hole. Exports it as a .step file.

        Args:
            airfoil_coords: (N, 2) numpy array of closed-loop X,Y coordinates.
            chord_m: Chord length used to scale the domain size.
            name: Base name for the output file.

        Returns:
            Absolute path to the generated .step file.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"  CAD BUILDER AGENT — Generating STEP file for {name}")
        self.logger.info("=" * 60)
        
        # Ensure the airfoil curve is fully closed
        pts = [(float(pt[0]), float(pt[1])) for pt in airfoil_coords]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
            self.logger.info("  Closed the airfoil coordinate loop manually.")

        # Domain extents
        r_inlet = 20.0 * chord_m
        x_outlet = 40.0 * chord_m
        
        self.logger.info(f"  Building C-domain (Inlet R={r_inlet:.1f}, Outlet={x_outlet:.1f})")

        with bd.BuildSketch() as sketch:
            # 1. Build the Fluid Domain Boundary (Outer perimeter)
            with bd.BuildLine() as domain_outline:
                p1 = (0, r_inlet)
                p2 = (x_outlet, r_inlet)
                p3 = (x_outlet, -r_inlet)
                p4 = (0, -r_inlet)
                # Downstream straight boundary elements
                bd.Line(p1, p2)
                bd.Line(p2, p3)
                bd.Line(p3, p4)
                # Upstream semicircular inlet
                bd.ThreePointArc(p4, (-r_inlet, 0), p1)
            # Fill the outline to create a solid 2D face
            bd.make_face()
            self.logger.info("  Domain outer boundary face created.")

            # 2. Build the Airfoil Boundary (Inner hole)
            with bd.BuildLine() as airfoil_outline:
                # Use a Polyline to rigidly respect the exact cosine spacing
                # provided by the Geometry Agent (spline might overshoot).
                bd.Polyline(*pts)
            # Subtract the airfoil outline from the domain face to create the hole
            bd.make_face(mode=bd.Mode.SUBTRACT)
            self.logger.info("  Airfoil hole subtracted from domain.")

        # Export using global exporters method
        step_filename = f"{name}_domain.step"
        step_path = os.path.abspath(os.path.join(self.output_dir, step_filename))
        
        # In build123d, sketch.sketch contains the actual TopoDS_Face that can be exported.
        bd.export_step(sketch.sketch, step_path)
        
        self.logger.info(f"  STEP export successful: {step_path}")
        self.logger.info("=" * 60)
        
        return step_path

if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Generate CAD from cleaned CSV")
    parser.add_argument("--csv", required=True, help="Path to cleaned airfoil CSV")
    parser.add_argument("--name", default="test_airfoil", help="Name prefix for export")
    parser.add_argument("--chord", type=float, default=1.0, help="Chord length")
    args = parser.parse_args()

    # Read CSV
    points = []
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((float(row["x_m"]), float(row["y_m"])))
    coords = np.array(points)

    agent = CADBuilderAgent(output_dir="data/raw")
    out_path = agent.generate_domain_step(coords, chord_m=args.chord, name=args.name)
    print(f"Done! {out_path}")
