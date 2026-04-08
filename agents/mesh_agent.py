"""
Mesh Agent - gmsh-based 2D CFD Mesh Generation

Writes a standalone Python script that uses gmsh (open-source, no license
required) to mesh the C-domain STEP file produced by CADBuilderAgent.

After gmsh succeeds, _convert_to_fluent_msh() rewrites the output as a
native Fluent ASCII MSH with:
  - (2 2) dimension declaration (gmsh MSH has no such header)
  - 2D node coordinates (gmsh always writes x y z even for 2D)
  - Explicit typed cell zone (fluid)
  - All boundary face zones with BC type codes
  - Interior face zone with full cell adjacency

This is necessary because Fluent 2D raises "Null Domain Pointer" when
reading gmsh MSH format directly.
"""

import os
import sys
import subprocess
import logging
from collections import defaultdict


class MeshAgent:
    def __init__(self, output_dir: str = "data/mesh"):
        self.output_dir = output_dir
        self.logger = logging.getLogger("MeshAgent")
        os.makedirs(output_dir, exist_ok=True)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.INFO)

    def generate_mesh(self, step_path: str, name: str = "airfoil") -> str:
        """
        Generates the mesh via gmsh subprocess, then converts to native
        Fluent ASCII MSH format. Returns absolute path to the Fluent MSH.
        """
        gmsh_path   = os.path.abspath(os.path.join(self.output_dir, f"{name}_2d.msh"))
        fluent_path = os.path.abspath(os.path.join(self.output_dir, f"{name}_2d_fluent.msh"))
        script_path = os.path.abspath(os.path.join(self.output_dir, f"mesh_gen_{name}.py"))
        step_path_abs = os.path.abspath(step_path)

        params = {
            "bl_layers": 15,
            "bl_ratio": 1.2,
            "bl_size": 1e-4,   # first layer height (m) — ~y+=1 at 50 m/s
            "min_size": 0.005,
            "max_size": 2.0,
        }

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            self.logger.info("-" * 60)
            self.logger.info(f"  MESH AGENT - Attempt {attempt}/{max_retries} for {name}")
            self.logger.info(f"  BL Settings: {params['bl_layers']} layers, "
                             f"ratio {params['bl_ratio']}, size {params['bl_size']:.2e}")

            self._write_script(script_path, step_path_abs, gmsh_path, params)

            self.logger.info("  Launching gmsh subprocess...")
            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                output = result.stdout + result.stderr

                if "MESH_SUCCESS" in output and result.returncode == 0:
                    self.logger.info(f"  gmsh succeeded. Converting to Fluent MSH format...")
                    self._convert_to_fluent_msh(gmsh_path, fluent_path)
                    self.logger.info(f"  Fluent MSH ready: {fluent_path}")
                    return fluent_path
                else:
                    self.logger.warning(f"  gmsh failed. Exit code: {result.returncode}")
                    self.logger.warning(f"  Subprocess Output:\n{output[-3000:]}")
                    if attempt < max_retries:
                        self.logger.info("  Relaxing BL parameters for next attempt...")
                        params["bl_layers"] = max(5, int(params["bl_layers"] * 0.7))
                        params["bl_ratio"]  = min(1.5, params["bl_ratio"] + 0.1)
                        params["bl_size"]   = params["bl_size"] * 2.0

            except subprocess.TimeoutExpired:
                self.logger.error("  gmsh subprocess timed out (5 mins).")
                if attempt == max_retries:
                    raise RuntimeError("Failed to mesh: gmsh timed out on all attempts.")

        raise RuntimeError(f"Failed to mesh airfoil {name} after {max_retries} attempts.")

    # ------------------------------------------------------------------
    # Fluent ASCII MSH converter
    # ------------------------------------------------------------------

    def _convert_to_fluent_msh(self, gmsh_path: str, fluent_path: str) -> None:
        """
        Convert a gmsh MSH v2 file to Fluent-native ASCII MSH.

        Fluent 2D raises "Null Domain Pointer" when reading gmsh MSH because:
          - gmsh MSH has no dimension header; Fluent can't tell it's 2D
          - gmsh always writes 3D coordinates (x y z) even for planar meshes
          - Fluent needs explicit cell zone, face zones, and BC type codes

        This writes the proper Fluent section-based format so Fluent's
        read_mesh TUI command builds the domain without errors.
        """
        import meshio

        mesh = meshio.read(gmsh_path)

        # Strip z-coordinate — Fluent 2D expects (x, y) only
        nodes = mesh.points[:, :2]
        N = len(nodes)

        # tag-number → zone name (from gmsh $PhysicalNames)
        tag_to_name = {int(v[0]): k for k, v in mesh.field_data.items()}
        phys = mesh.cell_data.get("gmsh:physical", [])

        # Separate fluid cells from boundary edges
        cells        = []        # list of int arrays, 0-indexed node IDs
        cell_ftypes  = []        # 1 = tri, 3 = quad  (Fluent element type codes)
        bnd_edges    = defaultdict(list)  # zone_name -> list of [n0, n1]

        for i, block in enumerate(mesh.cells):
            tags_arr = phys[i] if i < len(phys) else []
            for j, elem in enumerate(block.data):
                tag  = int(tags_arr[j]) if j < len(tags_arr) else 0
                name = tag_to_name.get(tag, f"zone_{tag}")
                if block.type == "triangle":
                    cells.append(elem)
                    cell_ftypes.append(1)
                elif block.type == "quad":
                    cells.append(elem)
                    cell_ftypes.append(3)
                elif block.type == "line":
                    bnd_edges[name].append(elem)

        Nc = len(cells)

        # Build undirected edge -> [(cell_id_0based, directed_n0, directed_n1)] map
        edge_cells = defaultdict(list)
        for cid, cell in enumerate(cells):
            n = len(cell)
            for i in range(n):
                n0, n1 = int(cell[i]), int(cell[(i + 1) % n])
                key = (min(n0, n1), max(n0, n1))
                edge_cells[key].append((cid, n0, n1))

        # Interior faces — edges shared by exactly 2 cells
        interior = []
        for key, adj in edge_cells.items():
            if len(adj) == 2:
                (c0, n0, n1), (c1, _, _) = adj
                interior.append((n0 + 1, n1 + 1, c0 + 1, c1 + 1))  # 1-based

        # Boundary faces — match each boundary edge to its adjacent cell
        # Convention: cr = fluid cell, cl = 0 (exterior)
        bnd_faces = {}
        for zone_name, edges in bnd_edges.items():
            faces = []
            for e in edges:
                n0, n1 = int(e[0]), int(e[1])
                key = (min(n0, n1), max(n0, n1))
                adj = edge_cells.get(key, [])
                if adj:
                    cid, cn0, cn1 = adj[0]
                    # If the cell's directed edge goes n0->n1, the cell is to
                    # the LEFT of that direction — reverse so cell is on right.
                    if cn0 == n0:
                        faces.append((n1 + 1, n0 + 1, cid + 1, 0))
                    else:
                        faces.append((n0 + 1, n1 + 1, cid + 1, 0))
                else:
                    faces.append((n0 + 1, n1 + 1, 0, 0))
            bnd_faces[zone_name] = faces

        total_faces = len(interior) + sum(len(v) for v in bnd_faces.values())

        # Fluent BC type codes (hex) and zone type strings
        BC_CODE = {
            "inlet":            0x14,   # velocity-inlet
            "outlet":           0x9,    # pressure-outlet
            "symmetry_top":     0x7,    # symmetry
            "symmetry_bottom":  0x7,
            "airfoil":          0x3,    # wall
        }
        FLUENT_TYPE = {
            "fluid":            "fluid",
            "inlet":            "velocity-inlet",
            "outlet":           "pressure-outlet",
            "symmetry_top":     "symmetry",
            "symmetry_bottom":  "symmetry",
            "airfoil":          "wall",
            "interior":         "interior",
        }

        # Assign zone IDs: fluid=2, boundaries from 3 upward, interior = last
        bnd_names = list(bnd_faces.keys())
        zone_id = {"fluid": 2}
        for i, zn in enumerate(bnd_names):
            zone_id[zn] = 3 + i
        zone_id["interior"] = 3 + len(bnd_names)

        with open(fluent_path, "w") as out:
            out.write('(0 "Converted from gmsh by MeshAgent")\n\n')
            out.write("(2 2)\n\n")   # 2D declaration

            # --- Nodes ---
            out.write(f"(10 (0 1 {N} 0 2))\n")
            out.write(f"(10 (1 1 {N} 1 2)\n(\n")
            for x, y in nodes:
                out.write(f"{x:.10e} {y:.10e}\n")
            out.write("))\n\n")

            # --- Cells ---
            out.write(f"(12 (0 1 {Nc} 0))\n")
            unique_etypes = set(cell_ftypes)
            if len(unique_etypes) == 1:
                out.write(f"(12 ({zone_id['fluid']} 1 {Nc} 1 {cell_ftypes[0]}))\n\n")
            else:
                # Mixed mesh (tri + quad from BL) — list element types explicitly
                out.write(f"(12 ({zone_id['fluid']} 1 {Nc} 1 0)\n(\n")
                for ct in cell_ftypes:
                    out.write(f"{ct}\n")
                out.write("))\n\n")

            # --- Faces header ---
            out.write(f"(13 (0 1 {total_faces} 0 0))\n")

            # Boundary face zones
            fc = 1
            for zn, faces in bnd_faces.items():
                if not faces:
                    continue
                bc  = BC_CODE.get(zn, 0x3)
                zid = zone_id[zn]
                first, last = fc, fc + len(faces) - 1
                out.write(f"(13 ({zid} {first} {last} {bc:x} 2)\n(\n")
                for n0, n1, cr, cl in faces:
                    out.write(f"{n0} {n1} {cr} {cl}\n")
                out.write("))\n\n")
                fc = last + 1

            # Interior face zone
            zid_int = zone_id["interior"]
            first, last = fc, fc + len(interior) - 1
            out.write(f"(13 ({zid_int} {first} {last} 2 2)\n(\n")
            for n0, n1, cr, cl in interior:
                out.write(f"{n0} {n1} {cr} {cl}\n")
            out.write("))\n\n")

            # --- Zone info (45 sections) ---
            out.write(f"(45 ({zone_id['fluid']} fluid fluid)())\n")
            for zn in bnd_names:
                ft = FLUENT_TYPE.get(zn, "wall")
                out.write(f"(45 ({zone_id[zn]} {ft} {zn})())\n")
            out.write(f"(45 ({zone_id['interior']} interior interior)())\n")

        self.logger.info(
            f"  Converter: {Nc} cells ({cell_ftypes.count(1)} tri + "
            f"{cell_ftypes.count(3)} quad), {total_faces} faces, {N} nodes"
        )

    # ------------------------------------------------------------------
    # gmsh script writer
    # ------------------------------------------------------------------

    def _write_script(self, script_path: str, step_path: str, mesh_path: str, params: dict):
        """Writes the gmsh Python script that the subprocess will execute."""
        bl_layers = params["bl_layers"]
        bl_ratio  = params["bl_ratio"]
        bl_size   = params["bl_size"]
        min_size  = params["min_size"]
        max_size  = params["max_size"]

        step_fwd = step_path.replace("\\", "/")
        mesh_fwd = mesh_path.replace("\\", "/")

        script_content = f"""import sys
import os

def main():
    try:
        import gmsh
    except ImportError:
        print("MESH_ERROR: gmsh not installed. Run: python -m pip install gmsh")
        sys.exit(1)

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        # --- 1. Load STEP via OpenCASCADE kernel (preserves B-rep topology) ---
        gmsh.model.occ.importShapes("{step_fwd}")
        gmsh.model.occ.synchronize()

        surfaces = gmsh.model.getEntities(2)
        curves   = gmsh.model.getEntities(1)

        if not surfaces:
            raise RuntimeError("No surfaces found after importing STEP.")

        print(f"INFO: {{len(surfaces)}} surface(s), {{len(curves)}} curve(s) loaded.")

        # --- 2. Classify curves into boundary zones by bounding box ---
        inlet_tags, outlet_tags, top_tags, bot_tags, airfoil_curve_tags = [], [], [], [], []
        for dim, tag in curves:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            if xmin < -1.0:
                inlet_tags.append(tag)
            elif xmin > 39.0:
                outlet_tags.append(tag)
            elif ymin > 19.0:
                top_tags.append(tag)
            elif ymax < -19.0:
                bot_tags.append(tag)
            else:
                airfoil_curve_tags.append(tag)

        print(f"INFO: inlet={{len(inlet_tags)}} outlet={{len(outlet_tags)}} "
              f"top={{len(top_tags)}} bot={{len(bot_tags)}} airfoil={{len(airfoil_curve_tags)}}")

        gmsh.model.addPhysicalGroup(1, inlet_tags,         name="inlet")
        gmsh.model.addPhysicalGroup(1, outlet_tags,        name="outlet")
        gmsh.model.addPhysicalGroup(1, top_tags,           name="symmetry_top")
        gmsh.model.addPhysicalGroup(1, bot_tags,           name="symmetry_bottom")
        gmsh.model.addPhysicalGroup(1, airfoil_curve_tags, name="airfoil")
        gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], name="fluid")

        # --- 3. Global mesh sizing ---
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", {min_size})
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", {max_size})
        gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.RecombineAll", 0)

        # --- 4. Boundary layer on airfoil (non-fatal) ---
        if airfoil_curve_tags:
            try:
                f = gmsh.model.mesh.field
                bl = f.add("BoundaryLayer")
                f.setNumbers(bl, "CurvesList", airfoil_curve_tags)
                f.setNumber(bl, "Size",     {bl_size})
                f.setNumber(bl, "Ratio",    {bl_ratio})
                f.setNumber(bl, "NbLayers", {bl_layers})
                f.setNumber(bl, "Quads",    1)
                f.setAsBoundaryLayer(bl)
                print(f"INFO: BL set — {{len(airfoil_curve_tags)}} curves, "
                      f"{bl_layers} layers, ratio {bl_ratio}, size {bl_size:.2e}")
            except Exception as bl_err:
                print(f"WARNING: BL failed ({{bl_err}}). Proceeding without inflation layers.")

        # --- 5. Generate 2D mesh ---
        gmsh.model.mesh.generate(2)

        # --- 6. Export gmsh MSH (intermediate — converter produces Fluent MSH) ---
        os.makedirs(os.path.dirname("{mesh_fwd}") or ".", exist_ok=True)
        gmsh.write("{mesh_fwd}")
        gmsh.finalize()

        if not os.path.exists("{mesh_fwd}"):
            raise RuntimeError("gmsh.write() completed but file not found: {mesh_fwd}")

        print("MESH_SUCCESS")

    except Exception as e:
        print(f"MESH_ERROR: {{e}}")
        try:
            gmsh.finalize()
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True)
    parser.add_argument("--name", default="test")
    args = parser.parse_args()

    agent = MeshAgent()
    agent.generate_mesh(args.step, name=args.name)
