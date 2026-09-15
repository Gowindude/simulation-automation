"""
Platform-dispatching shell helper for Stage 1/3/4/8's external tool
calls (OpenFOAM, CalculiX, xfoil).

Consolidates 4 near-identical `_run_wsl`/`_to_wsl_path` pairs that
previously lived separately in stage1_mesh.py, stage3_run.py,
stage4_postprocess.py, and stage8_calculix_run.py -- one Windows-only
implementation each, wrapping every call in `wsl.exe -- bash -lc`
because OpenFOAM/CalculiX/xfoil have no native Windows build and this
project has only ever run on Windows+WSL until now.

A GitHub Actions Linux runner has no WSL layer at all -- OpenFOAM and
CalculiX would run natively there, so wrapping in `wsl.exe` would be
wrong on that platform, not just redundant. This module dispatches on
`platform.system()`: `wsl.exe -- bash -lc <cmd>` on Windows (unchanged
behavior, real WSL runs on this machine are unaffected), plain
`bash -lc <cmd>` on Linux.
"""

import os
import platform
import posixpath
import subprocess


def to_linux_path(path: str) -> str:
    """
    On Windows, translate an absolute Windows path to its WSL
    `/mnt/<drive>/...` form (unchanged behavior from the previous
    per-stage `_to_wsl_path` implementations).

    On Linux, the path is already native -- returned as an absolute
    path, not translated. Uses posixpath explicitly (not os.path) so
    this is testable/correct regardless of the host OS this code
    happens to be developed/tested on -- os.path is Windows semantics
    on a Windows dev machine even when simulating the Linux branch.
    """
    if platform.system() != "Windows":
        return posixpath.abspath(path)
    win_path = os.path.abspath(path)
    drive, rest = os.path.splitdrive(win_path)
    drive_letter = drive.rstrip(":").lower()
    rest = rest.replace("\\", "/")
    return f"/mnt/{drive_letter}{rest}"


def run_shell(
    bash_cmd: str, timeout: int = 300, source_openfoam: bool = False, input: str | None = None,
) -> subprocess.CompletedProcess:
    """
    Run `bash_cmd` against OpenFOAM/CalculiX/xfoil.

    Windows: via `wsl.exe -- bash -lc` -- the only way to reach a
    Linux-only solver from a Windows host.
    Linux (e.g. a GitHub Actions runner): via `bash -lc` directly --
    already the native environment, no WSL exists there to wrap it in.

    source_openfoam=True prepends OpenFOAM's env-sourcing line (Stage
    1/3/4's own convention). Stage 8's CalculiX call doesn't need it,
    so it stays opt-in, not automatic.

    input, if given, is piped to the command's stdin (e.g. Stage 4's
    XFOIL script, fed to the interactive `xfoil` prompt).
    """
    full_cmd = f"source /opt/openfoam12/etc/bashrc && {bash_cmd}" if source_openfoam else bash_cmd
    if platform.system() == "Windows":
        argv = ["wsl.exe", "--", "bash", "-lc", full_cmd]
    else:
        argv = ["bash", "-lc", full_cmd]
    return subprocess.run(argv, input=input, capture_output=True, text=True, timeout=timeout)
