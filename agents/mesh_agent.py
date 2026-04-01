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
        
        # Scheme eval requires forward slashes
        step_fixed = step_path.replace("\\", "/")
        mesh_fixed = mesh_path.replace("\\", "/")
        
        # Define local vars for the f-string interpolation
        bl_layers = params["bl_layers"]
        bl_ratio = params["bl_ratio"]
        
        # Use local paths to avoid space-related TUI parsing errors
        step_local = os.path.basename(step_path)
        mesh_local = os.path.basename(mesh_path)
        work_dir = os.path.dirname(os.path.abspath(script_path))
        
        script_content = f"""import os
import sys
import ansys.fluent.core as pyfluent

def main():
    try:
        from ansys.fluent.core import launch_fluent, FluentMode, Precision, Dimension
        import os
        os.environ["ANSYS_NO_WINDOWS_USER_AUTH"] = "1"
        os.environ["PYFLUENT_START_INSTANCE"] = "1"

        # 1. Set the Python process directory first
        os.chdir(r"{work_dir}") 

        # 2. Remove 'working_directory' from the launch call
        meshing = launch_fluent(
            mode=FluentMode.MESHING,
            dimension=Dimension.THREE,
            precision=Precision.DOUBLE,
            processor_count=4
        )

        # 3. Use the session-level chdir to ensure Fluent is in the right spot
        # If this fails, use: meshing.tui.file.chdir(r"{work_dir}")
        try:
            meshing.chdir(r"{work_dir}")
        except:
            pass

        # ── HYBRID SCHEME INJECTION (UNIVERSAL FLUENT API) ─────────────
        def send_tui(cmd_str):
            # No changes needed here, your TUI logic is solid
            escaped_cmd = cmd_str.replace('"', '\\\\"')
            meshing.scheme.eval(f'(ti-menu-load-string "{{escaped_cmd}}")')
        
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
