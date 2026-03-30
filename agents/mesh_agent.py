"""
Mesh Agent for Dynamic PyFluent Script Generation

Writes a standalone Python script to drive PyFluent Meshing.
Monitors the execution in a subprocess and autonomously adjusts meshing
parameters (like Boundary Layer ratios) if the mesher struggles, allowing
graceful retries without propagating crashes to the main orchestrator.
"""

import os
import sys
import subprocess
import logging

class MeshAgent:
    def __init__(self, output_dir: str = "data/raw"):
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
        Generates the mesh by writing and invoking a PyFluent script.
        Implements an autonomous retry loop if the mesher fails.
        """
        mesh_path = os.path.abspath(os.path.join(self.output_dir, f"{name}_2d.msh.h5"))
        script_path = os.path.abspath(os.path.join(self.output_dir, f"mesh_gen_{name}.py"))
        step_path_abs = os.path.abspath(step_path)
        
        # Initial Boundary Layer parameters
        params = {
            "bl_layers": 15,
            "bl_ratio": 1.2
        }
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            self.logger.info("=" * 60)
            self.logger.info(f"  MESH AGENT — Attempt {attempt}/{max_retries} for {name}")
            self.logger.info(f"  BL Settings: {params['bl_layers']} layers, ratio {params['bl_ratio']}")
            
            # Step 1: Write the dynamic script
            self._write_script(script_path, step_path_abs, mesh_path, params)
            
            # Step 2: Execute the script in an isolated subprocess
            self.logger.info("  Launching PyFluent subprocess...")
            try:
                # We use creationflags to prevent Fluent GUI from popping up cmd windows
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=300 # 5 minutes max
                )
                
                output = result.stdout + result.stderr
                
                if "MESH_SUCCESS" in output and result.returncode == 0:
                    self.logger.info(f"  Meshing Successful! Saved to {mesh_path}")
                    return mesh_path
                else:
                    self.logger.warning(f"  Meshing failed in subprocess. Exit code: {result.returncode}")
                    self.logger.debug(f"  Subprocess Output:\n{output[-1000:]}") # Show last 1000 chars of error
                    
                    # Step 3: Autonomously adjust parameters for next retry
                    if attempt < max_retries:
                        self.logger.info("  Adjusting inflation layers to improve mesh robustness...")
                        params["bl_layers"] = max(5, int(params["bl_layers"] * 0.7)) # Reduce layers
                        params["bl_ratio"] = 1.1 # Safer transition ratio
            
            except subprocess.TimeoutExpired:
                self.logger.error("  Meshing subprocess timed out (5 mins).")
                if attempt == max_retries:
                    raise RuntimeError("Failed to mesh airfoil: PyFluent timed out on all attempts.")
                    
        raise RuntimeError(f"Failed to mesh airfoil {name} after {max_retries} attempts.")


    def _write_script(self, script_path: str, step_path: str, mesh_path: str, params: dict):
        """
        Writes the exact python script that the subprocess will execute.
        This provides perfect isolation and bypasses the Jupyter/IPython notebook 
        quirks that PyFluent sometimes encounters out-of-core.
        """
        # We must escape backslashes in Windows paths for the Python string
        step_str = step_path.replace("\\", "\\\\")
        mesh_str = mesh_path.replace("\\", "\\\\")
        
        script_content = f"""import os
import sys
import ansys.fluent.core as pyfluent

def main():
    try:
        from ansys.fluent.core import launch_fluent, FluentMode, Precision, Dimension
        meshing = launch_fluent(
            mode=FluentMode.MESHING,
            dimension=Dimension.TWO,
            precision=Precision.DOUBLE,
            processor_count=4,
            ui_mode="gui"
        )
        two_d = meshing.two_dimensional_meshing()

        print("Loading CAD...")
        load_cad = two_d.load_cad_geometry_2d
        load_cad.file_name = r"{step_path}"
        load_cad.length_unit = "m"
        load_cad.refaceting.refacet = False
        load_cad()

        print("Updating Boundaries...")
        update_bnd = two_d.update_boundaries
        update_bnd.selection_type = "zone"
        update_bnd()

        print("Adding Boundary Layers...")
        # Note: Depending on CAD import, the edge zone might be named differently
        # (e.g. edge-1). We'll try wildcard *airfoil* or just the default edge.
        add_bl = two_d.add_boundary_layers
        add_bl.add_controls = True
        add_bl.bl_control_name = "airfoil_bl"
        add_bl.local_regions = "*"
        add_bl.offset_method_type = "smooth-transition"
        add_bl.number_of_layers = {params['bl_layers']}
        add_bl.transition_ratio = {params['bl_ratio']}
        add_bl()

        print("Generating Mesh...")
        two_d.generate_mesh()

        print(f"Exporting to {mesh_str}...")
        meshing.meshing.File.WriteMesh(FileName=r"{mesh_str}")
        
        meshing.exit()
        print("MESH_SUCCESS")
        
    except Exception as e:
        print(f"MESH_ERROR: {{str(e)}}")
        try:
            meshing.exit()
        except:
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
