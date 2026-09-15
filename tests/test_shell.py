"""
Verification tests for pipeline/_shell.py -- the platform-dispatching
shell helper that replaces the 4 near-identical `_run_wsl`/`_to_wsl_path`
pairs previously duplicated in stage1_mesh.py, stage3_run.py,
stage4_postprocess.py, and stage8_calculix_run.py.

Contract (locked before implementation, per .claude/airfoil_pipeline_build_spec.md's
"propose the verification tests ... then agree on them before you build",
and the GitHub Actions goal this exists for -- a Linux runner has no WSL,
so the dispatch must be real, not a Windows-only shim):
  - On Windows: run_shell wraps the command in `wsl.exe -- bash -lc`,
    exactly matching the previous 4 implementations' behavior (so real
    WSL/OpenFOAM/CalculiX runs on this machine are unaffected).
  - On Linux: run_shell runs `bash -lc` directly, no wsl.exe wrapper
    (there is none on a native Linux runner).
  - source_openfoam=True prepends the OpenFOAM env-sourcing line on
    both platforms; source_openfoam=False (stage8/CalculiX) doesn't.
  - to_linux_path: Windows paths translate to /mnt/<drive>/... form
    (unchanged behavior); a path on Linux is returned as an absolute
    native path, not translated.

subprocess.run itself is mocked throughout -- no real WSL/bash call is
exercised here, only the argv/command construction.
"""

from unittest.mock import patch, MagicMock

import pytest

from pipeline import _shell


class TestToLinuxPath:
    @patch("pipeline._shell.platform.system", return_value="Windows")
    def test_windows_translates_to_wsl_mnt_form(self, _mock_system):
        result = _shell.to_linux_path(r"C:\Users\quack\project\case")
        assert result == "/mnt/c/Users/quack/project/case"

    @patch("pipeline._shell.platform.system", return_value="Linux")
    def test_linux_returns_absolute_native_path_unchanged(self, _mock_system):
        result = _shell.to_linux_path("/home/runner/project/case")
        assert result == "/home/runner/project/case"


class TestRunShell:
    @patch("pipeline._shell.subprocess.run")
    @patch("pipeline._shell.platform.system", return_value="Windows")
    def test_windows_wraps_in_wsl_exe(self, _mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        _shell.run_shell("checkMesh", timeout=60)
        argv = mock_run.call_args[0][0]
        assert argv[:3] == ["wsl.exe", "--", "bash"]
        assert argv[-1] == "checkMesh"

    @patch("pipeline._shell.subprocess.run")
    @patch("pipeline._shell.platform.system", return_value="Linux")
    def test_linux_runs_bash_directly_no_wsl(self, _mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        _shell.run_shell("checkMesh", timeout=60)
        argv = mock_run.call_args[0][0]
        assert argv == ["bash", "-lc", "checkMesh"]
        assert "wsl.exe" not in argv

    @patch("pipeline._shell.subprocess.run")
    @patch("pipeline._shell.platform.system", return_value="Windows")
    def test_source_openfoam_prepends_env_sourcing(self, _mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        _shell.run_shell("checkMesh", timeout=60, source_openfoam=True)
        argv = mock_run.call_args[0][0]
        assert "source /opt/openfoam12/etc/bashrc" in argv[-1]
        assert "checkMesh" in argv[-1]

    @patch("pipeline._shell.subprocess.run")
    @patch("pipeline._shell.platform.system", return_value="Windows")
    def test_no_source_openfoam_by_default(self, _mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        _shell.run_shell("ccx myjob")
        argv = mock_run.call_args[0][0]
        assert argv[-1] == "ccx myjob"

    @patch("pipeline._shell.subprocess.run")
    @patch("pipeline._shell.platform.system", return_value="Linux")
    def test_linux_source_openfoam_prepends_env_sourcing_without_wsl(self, _mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        _shell.run_shell("foamRun", timeout=60, source_openfoam=True)
        argv = mock_run.call_args[0][0]
        assert argv[0] == "bash"
        assert "source /opt/openfoam12/etc/bashrc" in argv[-1]

    @patch("pipeline._shell.subprocess.run")
    @patch("pipeline._shell.platform.system", return_value="Windows")
    def test_timeout_and_capture_passed_through(self, _mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        _shell.run_shell("checkMesh", timeout=42)
        kwargs = mock_run.call_args[1]
        assert kwargs["timeout"] == 42
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True

    @patch("pipeline._shell.subprocess.run")
    @patch("pipeline._shell.platform.system", return_value="Windows")
    def test_input_piped_to_stdin(self, _mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        _shell.run_shell("xfoil", input="LOAD foo.dat\n")
        kwargs = mock_run.call_args[1]
        assert kwargs["input"] == "LOAD foo.dat\n"
