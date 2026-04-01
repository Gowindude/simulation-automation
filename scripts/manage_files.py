"""
Cleanup Manager for Simulation Automation Pipeline
Handles project directory organization, log redirection, and periodic pruning
of transient simulation artifacts while preserving final results.
"""

import os
import shutil
import glob
from pathlib import Path

class CleanupManager:
    """Manages project file organization and autonomous cleanup of simulation artifacts."""
    
    # New directory structure
    DIRS = {
        "inputs": "data/airfoils",
        "geometry": "data/geometry",
        "mesh": "data/mesh",
        "results": "data/results",
        "logs": "data/fluent_logs"
    }

    def __init__(self, root_dir: str = "."):
        self.root = Path(root_dir).resolve()
        self.ensure_dirs()

    def ensure_dirs(self):
        """Creates the standardized project directory structure if it doesn't exist."""
        for d in self.DIRS.values():
            os.makedirs(self.root / d, exist_ok=True)

    def organize_loose_files(self):
        """Moves orphaned Fluent transcripts, cleanup scripts, and test results from root to logs."""
        # Generic Fluent files
        patterns = [
            "fluent-*.trn",
            "cleanup-fluent-*.bat",
            "fluent-*-error.log",
            "test_out*.json",
            "test_out*.txt",
            "e2e_test_out.*",
            "mesh_gen_*.py" # Move generated mesh scripts if found in root
        ]
        
        target_log_dir = self.root / self.DIRS["logs"]
        target_mesh_dir = self.root / self.DIRS["mesh"]

        for pattern in patterns:
            for file_path in glob.glob(str(self.root / pattern)):
                file_name = os.path.basename(file_path)
                
                # Special cases: move mesh generation scripts to data/mesh
                if file_name.startswith("mesh_gen_"):
                    dest = target_mesh_dir / file_name
                else:
                    dest = target_log_dir / file_name
                
                try:
                    # Use shutil.move to handle cross-device moves
                    shutil.move(file_path, dest)
                except Exception as e:
                    print(f"Error moving {file_name}: {e}")

    def prune_transient_files(self, keep_count: int = 5):
        """
        Deletes old transient files in logs and mesh directories.
        Keeps the n-most recently modified files.
        """
        # We prune logs and mesh scripts, but NEVER results
        prune_targets = [
            self.root / self.DIRS["logs"],
            self.root / self.DIRS["mesh"]
        ]

        for target_dir in prune_targets:
            if not target_dir.exists():
                continue
                
            # Get list of all files in the directory sorted by modification time (newest first)
            files = sorted(
                glob.glob(str(target_dir / "*")),
                key=os.path.getmtime,
                reverse=True
            )
            
            # Identify files to delete (those beyond the keep_count)
            # IMPORTANT: We only prune .trn, .bat, and .py in mesh/scripts
            prunable_extensions = {".trn", ".bat", ".log", ".py", ".json", ".txt"}
            
            to_delete = files[keep_count:]
            for f in to_delete:
                path = Path(f)
                if path.suffix in prunable_extensions:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"Error pruning {f}: {e}")

    @staticmethod
    def bootstrap():
        """Class method to quickly run a full organization and pruning cycle."""
        mgr = CleanupManager()
        print("Organizing loose simulation files...")
        mgr.organize_loose_files()
        print("Pruning old transient logs...")
        mgr.prune_transient_files(keep_count=5)
        print("Project hierarchy clean.")

if __name__ == "__main__":
    CleanupManager.bootstrap()
