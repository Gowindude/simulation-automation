import os
import sys
import ansys.fluent.core as pyfluent

def main():
    try:
        from ansys.fluent.core import launch_fluent, FluentMode, Precision, Dimension
        import os
        os.environ["ANSYS_NO_WINDOWS_USER_AUTH"] = "1"
        os.environ["PYFLUENT_START_INSTANCE"] = "1"

        # 1. Set the Python process directory first
        os.chdir(r"C:\Users\quack\Documents\Projects\Sim Automation\simulation-automation\data\raw") 

        # 2. Remove 'working_directory' from the launch call
        meshing = launch_fluent(
            mode=FluentMode.MESHING,
            dimension=Dimension.THREE,
            precision=Precision.DOUBLE,
            processor_count=4
        )

        # 3. Use the session-level chdir to ensure Fluent is in the right spot
        # If this fails, use: meshing.tui.file.chdir(r"C:\Users\quack\Documents\Projects\Sim Automation\simulation-automation\data\raw")
        try:
            meshing.chdir(r"C:\Users\quack\Documents\Projects\Sim Automation\simulation-automation\data\raw")
        except:
            pass

        # ── HYBRID SCHEME INJECTION (UNIVERSAL FLUENT API) ─────────────
        def send_tui(cmd_str):
            # No changes needed here, your TUI logic is solid
            escaped_cmd = cmd_str.replace('"', '\\"')
            meshing.scheme.eval(f'(ti-menu-load-string "{escaped_cmd}")')
        
    except Exception as e:
        print(f"MESH_ERROR: {str(e)}")
        try:
            meshing.exit()
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
