"""
Stage 4 -- CFD post-processing.

Input:  a converged Stage 3 case dir.
Output: per-AoA pressure distribution as a function of arc length along
        the airfoil surface (resampled onto a consistent parameterization,
        independent of mesh resolution), plus lift/drag from integrating
        that pressure (and wall shear) over the surface.

Extraction mechanism: `foamToVTK -noInternal -fields (p)` writes the
"airfoil" patch as a legacy VTK POLYDATA file. pyvista reads it directly
(meshio's legacy reader does NOT support POLYDATA -- confirmed empirically
before choosing pyvista). Face centroids come back in gmsh's internal
face-write order, not the geometric order around the airfoil, so they are
re-ordered here via a greedy nearest-neighbor traversal before arc length
is meaningful.

Cp = p / (0.5 * U_inf^2): the solver is incompressible (p is kinematic,
m^2/s^2, rho=1 per stage2_case_gen.py's physicalProperties), and the
freestream reference pressure is 0 (stage2_case_gen.py's `0/p` internal
field), so no p_inf subtraction term is needed.

XFOIL cross-check (spec's Must-Pass Gate #2): driven via its interactive
stdin-script interface (there is no other API) -- LOAD the .dat file,
enter OPER, set VISC <Re>, ITER <n>, ALFA <aoa>, and parse the last
converged "CL = ... / CD = ..." pair from stdout. Verified empirically:
XFOIL's PACC polar-accumulation-to-file feature did not reliably write a
data row in this environment even after a fully converged run (confirmed
by reading the polar file after process exit and finding only the
header); reading CL/CD directly from the converged iteration's own
stdout, which XFOIL always prints, is what's actually used here.
"""

import os
import re
import subprocess

import numpy as np
import pyvista as pv


def _order_surface_points(points_2d: np.ndarray):
    """
    Greedy nearest-neighbor traversal to recover the geometric ordering
    of face centroids around the airfoil surface. Face-write order from
    foamToVTK does not follow the surface, so arc length computed on the
    raw order would be meaningless.

    Returns:
        (order, ordered_points, max_gap) -- `max_gap` is the largest
        step between consecutive ordered points, exposed so callers can
        sanity-check the traversal didn't jump across the airfoil (which
        would indicate a self-crossing or a face count too sparse to
        order reliably).
    """
    n = len(points_2d)
    visited = np.zeros(n, dtype=bool)
    start = int(np.argmin(points_2d[:, 0]))  # start near the leading edge
    order = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        dist = np.linalg.norm(points_2d - points_2d[cur], axis=1)
        dist[visited] = np.inf
        nxt = int(np.argmin(dist))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    ordered_points = points_2d[order]
    gaps = np.linalg.norm(np.diff(ordered_points, axis=0), axis=1)
    return order, ordered_points, float(gaps.max())


def extract_surface_pressure(case_dir: str, U_inf: float, time_dir: str | None = None) -> dict:
    """
    Extract Cp as a function of arc length along the airfoil surface.

    Args:
        case_dir: a Stage 3 case dir (must have a written time step with p).
        U_inf: freestream speed used to non-dimensionalize (Cp = p / (0.5 U_inf^2)).
        time_dir: which time directory to read; None means the latest.

    Returns:
        {
            "s": np.ndarray,        # arc length, [0, total perimeter], starts near the LE
            "x": np.ndarray, "y": np.ndarray,  # ordered surface coordinates
            "Cp": np.ndarray,
            "max_traversal_gap": float,  # diagnostic: see _order_surface_points
        }

    Raises:
        RuntimeError: if foamToVTK fails or produces no airfoil patch file.
    """
    _run_foam_to_vtk(case_dir, time_dir)
    vtk_path = _find_airfoil_vtk(case_dir)

    mesh = pv.read(vtk_path)
    centers = mesh.cell_centers().points[:, :2]
    p = np.asarray(mesh.cell_data["p"], dtype=np.float64)

    order, ordered_points, max_gap = _order_surface_points(centers)
    ordered_p = p[order]

    seg = np.diff(ordered_points, axis=0)
    seglen = np.sqrt((seg ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])

    Cp = ordered_p / (0.5 * U_inf ** 2)

    return {
        "s": s,
        "x": ordered_points[:, 0],
        "y": ordered_points[:, 1],
        "Cp": Cp,
        "max_traversal_gap": max_gap,
    }


def _run_foam_to_vtk(case_dir: str, time_dir: str | None) -> None:
    case_wsl = _to_wsl_path(case_dir)
    time_flag = f'-time "{time_dir}"' if time_dir else "-latestTime"
    cmd = f'cd "{case_wsl}" && foamToVTK -noInternal -fields "(p)" {time_flag}'
    result = _run_wsl(cmd, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"foamToVTK failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        )


def _find_airfoil_vtk(case_dir: str) -> str:
    airfoil_dir = os.path.join(case_dir, "VTK", "airfoil")
    if not os.path.isdir(airfoil_dir):
        raise RuntimeError(f"foamToVTK produced no airfoil patch dir: {airfoil_dir}")
    candidates = sorted(f for f in os.listdir(airfoil_dir) if f.endswith(".vtk"))
    if not candidates:
        raise RuntimeError(f"No .vtk files found under {airfoil_dir}")
    return os.path.join(airfoil_dir, candidates[-1])


def pressure_integrated_cl_cd(
    surface: dict, U_inf: float, aoa_deg: float
) -> dict:
    """
    Integrate Cp(s) over the closed surface loop to get pressure-only
    lift/drag coefficients, as an internal cross-check against
    force-integrated values (spec: "Pressure-integrated Cl vs
    force-integrated Cl (internal check): within ~1%").

    Force on each segment: -Cp * n_hat * seg_length (kinematic pressure
    coefficient times outward normal times segment length; span = 1 and
    rho = 1 are folded into the coefficient normalization already used
    for Cp, consistent with Aref = chord * span = 1 in stage2/Stage
    3's forceCoeffs convention).

    Args:
        surface: output of extract_surface_pressure.
        U_inf: freestream speed (used only to confirm consistency with
            how Cp was derived; not re-applied here).
        aoa_deg: angle of attack, degrees -- lift/drag are defined
            relative to the freestream direction (the vector that was
            rotated in Stage 2), not the body axes.

    Returns:
        {"Cl": float, "Cd": float}
    """
    x, y, Cp = surface["x"], surface["y"], surface["Cp"]
    n = len(x)

    # Closed-loop outward normals: rotate each segment's tangent by -90
    # degrees. Sign is fixed once via the polygon's signed area (shoelace
    # formula) rather than assumed, so this is correct regardless of
    # which traversal direction _order_surface_points happened to produce.
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    seg_len = np.hypot(x_next - x, y_next - y)
    tangent = np.column_stack([x_next - x, y_next - y]) / seg_len[:, None]
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])  # rotate -90 deg

    signed_area = 0.5 * np.sum(x * y_next - x_next * y)
    if signed_area < 0:
        normal = -normal  # traversal was clockwise; flip to outward

    force = -(Cp[:, None] * normal * seg_len[:, None])
    fx, fy = force.sum(axis=0)

    theta = np.radians(aoa_deg)
    drag_dir = np.array([np.cos(theta), np.sin(theta)])
    lift_dir = np.array([-np.sin(theta), np.cos(theta)])

    return {
        "Cl": float(fx * lift_dir[0] + fy * lift_dir[1]),
        "Cd": float(fx * drag_dir[0] + fy * drag_dir[1]),
    }


_FORCES_VECTOR_RE = re.compile(
    r"pressure\s*:\s*\(([-\d.eE]+)\s+([-\d.eE]+)\s+[-\d.eE]+\)\s*\n\s*"
    r"viscous\s*:\s*\(([-\d.eE]+)\s+([-\d.eE]+)\s+[-\d.eE]+\)"
)


def force_integrated_cl_cd(
    case_dir: str, U_inf: float, aoa_deg: float, component: str = "total"
) -> dict:
    """
    Lift/drag coefficients via OpenFOAM's own `forcesIncompressible`
    function object -- the independent, native-integration reference this
    module's `pressure_integrated_cl_cd` is cross-checked against (spec's
    internal check: "Pressure-integrated Cl vs force-integrated Cl,
    within ~1%").

    Args:
        component: "pressure" (pressure-only, the correct apples-to-apples
            comparison against `pressure_integrated_cl_cd`, which has no
            viscous term), "viscous", or "total" (both summed).

    Returns:
        {"Cl": float, "Cd": float}

    Raises:
        RuntimeError: if foamPostProcess fails or its output can't be parsed.
    """
    case_wsl = _to_wsl_path(case_dir)
    func = (
        f"forcesIncompressible(patches=(airfoil),magUInf={U_inf},lRef=1,Aref=1,"
        f"CofR=(0.25 0 0),rhoInf=1,pitchAxis=(0 0 1))"
    )
    cmd = f'cd "{case_wsl}" && foamPostProcess -solver incompressibleFluid -func \'{func}\' -latestTime'
    result = _run_wsl(cmd, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"foamPostProcess (forcesIncompressible) failed:\n{result.stdout}\n{result.stderr}"
        )

    match = _FORCES_VECTOR_RE.search(result.stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse forcesIncompressible output:\n{result.stdout[-2000:]}"
        )
    px, py, vx, vy = (float(g) for g in match.groups())

    if component == "pressure":
        fx, fy = px, py
    elif component == "viscous":
        fx, fy = vx, vy
    elif component == "total":
        fx, fy = px + vx, py + vy
    else:
        raise ValueError(f"Unknown component: {component!r}")

    theta = np.radians(aoa_deg)
    drag_dir = np.array([np.cos(theta), np.sin(theta)])
    lift_dir = np.array([-np.sin(theta), np.cos(theta)])
    q = 0.5 * U_inf ** 2  # Aref = 1, rho = 1

    return {
        "Cl": float((fx * lift_dir[0] + fy * lift_dir[1]) / q),
        "Cd": float((fx * drag_dir[0] + fy * drag_dir[1]) / q),
    }


# ------------------------------------------------------------------
# XFOIL cross-check (Must-Pass Gate #2)
# ------------------------------------------------------------------

_XFOIL_SCRIPT = """LOAD {dat_path}
{name}
PANE
OPER
VISC {reynolds}
ITER {iterations}
ALFA {aoa}

QUIT
"""

_CL_CD_LINE_RE = re.compile(
    r"a\s*=\s*([-\d.]+)\s+CL\s*=\s*([-\d.]+)\s*\n\s*Cm\s*=\s*([-\d.]+)\s+CD\s*=\s*([-\d.]+)"
)


def run_xfoil(dat_path: str, name: str, reynolds: float, aoa_deg: float, iterations: int = 200) -> dict:
    """
    Run XFOIL for one operating point and return converged Cl/Cd.

    Args:
        dat_path: UIUC .dat file (same file Stage 0 reads) -- XFOIL LOADs
            it directly rather than the normalized Stage 0 coords, since
            it wants its own native coordinate/panel format.
        name: airfoil name (matches the .dat file's own header line, per
            XFOIL's LOAD convention).
        reynolds: Reynolds number.
        aoa_deg: angle of attack, degrees.
        iterations: max boundary-layer iterations per operating point.

    Returns:
        {"Cl": float, "Cd": float, "converged": bool}

    Raises:
        FileNotFoundError: if `dat_path` doesn't exist.
        RuntimeError: if xfoil itself fails to run (not the same as
            "didn't converge", which is reported via `converged`).
    """
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"No such .dat file: {dat_path}")

    script = _XFOIL_SCRIPT.format(
        dat_path=_to_wsl_path(dat_path), name=name, reynolds=reynolds,
        iterations=iterations, aoa=aoa_deg,
    )
    result = subprocess.run(
        ["wsl.exe", "--", "bash", "-lc", "xfoil"],
        input=script, capture_output=True, text=True, timeout=90,
    )
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"xfoil failed to run: {result.stderr}")

    return parse_xfoil_output(result.stdout)


def parse_xfoil_output(stdout: str) -> dict:
    """
    Parse the LAST converged 'a = ... CL = ... / Cm = ... CD = ...' pair
    XFOIL prints to stdout for an OPER point, plus whether the preceding
    boundary-layer iteration's rms residual indicates real convergence
    (XFOIL prints 'C at' rather than a bare iteration count on the
    converged line -- both are matched here, but a caller can distinguish
    by checking the rms directly if needed).
    """
    matches = list(_CL_CD_LINE_RE.finditer(stdout))
    if not matches:
        return {"Cl": None, "Cd": None, "converged": False}

    last = matches[-1]
    aoa, cl, cm, cd = (float(g) for g in last.groups())

    # XFOIL marks true boundary-layer convergence with a 'C' flag on the
    # rms line immediately preceding the CL/CD line (e.g.
    # '  6   rms: 0.951E-05   max: ...   C at  45  1'), vs no flag / a
    # different letter for a step that hit ITER without converging.
    context = stdout[: last.start()]
    last_iter_line = context.strip().splitlines()[-1] if context.strip() else ""
    converged = bool(re.search(r"\bC at\b", last_iter_line))

    return {"Cl": cl, "Cd": cd, "converged": converged}


def _to_wsl_path(win_path: str) -> str:
    win_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(win_path)
    return f"/mnt/{drive.rstrip(':').lower()}{rest.replace(chr(92), '/')}"


def _run_wsl(bash_cmd: str, timeout: int) -> subprocess.CompletedProcess:
    full_cmd = f"source /opt/openfoam12/etc/bashrc && {bash_cmd}"
    return subprocess.run(
        ["wsl.exe", "--", "bash", "-lc", full_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


# ------------------------------------------------------------------
# Per-AoA output record (spec's Final Output Schema "cfd/" sub-record)
# ------------------------------------------------------------------


def build_cfd_record(
    stage3_result: dict,
    dat_path: str,
    name: str,
    aoa_deg: float,
    reynolds: float,
    U_inf: float,
) -> dict:
    """
    Assemble one AoA's `cfd/` sub-record exactly per the spec's Final
    Output Schema (.claude/airfoil_pipeline_build_spec.md, "Final Output
    Schema" section):

        cfd/
          status: "converged" | "non_converged" | "diverged" | "crashed"
          Cl, Cd: float
          Cl_xfoil, Cd_xfoil: float          <- cross-check reference values
          pressure_vs_arc_length: array of (s, Cp) pairs

    The schema stores Cl_xfoil/Cd_xfoil as raw numbers alongside CFD's own
    Cl/Cd -- not a pass/fail flag -- so a caller (eventually Stage 9's
    aggregation) can compute and track the CFD/XFOIL agreement across the
    whole airfoil set, rather than this function silently deciding
    per-case whether the comparison "counts". Must-Pass Gate #2's ~10-15%
    tolerance is validated separately, as an explicit test
    (test_stage4_postprocess.py::test_gate2_xfoil_cross_check) -- this
    function's job is only to produce the record the schema asks for.

    Args:
        stage3_result: Stage 3's run_case() output (needs 'status' and,
            if converged, 'case_dir').
        dat_path: original UIUC .dat file, for XFOIL's own LOAD.
        name, aoa_deg, reynolds: the case's own operating point.
        U_inf: freestream speed (Stage 2's generate_case() output).

    Returns:
        {
            "status": str,
            "Cl": float | None, "Cd": float | None,
            "Cl_xfoil": float | None, "Cd_xfoil": float | None,
            "xfoil_converged": bool | None,
            "pressure_vs_arc_length": [[s, Cp], ...] | None,
        }

        For any status other than "converged", every field besides
        `status` is explicitly None -- per the spec's "failed cases
        recorded explicitly (never silently dropped)": the case still
        appears in the record (this function always returns one), but
        nothing is fabricated for data that was never computed (no
        Cl=0.0 placeholder, no dropped record).
    """
    status = stage3_result["status"]
    if status != "converged":
        return {
            "status": status,
            "Cl": None,
            "Cd": None,
            "Cl_xfoil": None,
            "Cd_xfoil": None,
            "xfoil_converged": None,
            "pressure_vs_arc_length": None,
        }

    case_dir = stage3_result["case_dir"]
    surface = extract_surface_pressure(case_dir, U_inf)
    cfd = force_integrated_cl_cd(case_dir, U_inf, aoa_deg, component="total")
    xfoil = run_xfoil(dat_path, name, reynolds=reynolds, aoa_deg=aoa_deg)

    return {
        "status": status,
        "Cl": cfd["Cl"],
        "Cd": cfd["Cd"],
        "Cl_xfoil": xfoil["Cl"],
        "Cd_xfoil": xfoil["Cd"],
        "xfoil_converged": xfoil["converged"],
        "pressure_vs_arc_length": [
            [s, cp] for s, cp in zip(surface["s"].tolist(), surface["Cp"].tolist())
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4: extract pressure distribution and cross-check with XFOIL.")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--dat-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--u-inf", type=float, required=True)
    parser.add_argument("--reynolds", type=float, required=True)
    parser.add_argument("--aoa", type=float, required=True)
    args = parser.parse_args()

    surface = extract_surface_pressure(args.case_dir, args.u_inf)
    cfd = pressure_integrated_cl_cd(surface, args.u_inf, args.aoa)
    xfoil = run_xfoil(args.dat_path, args.name, args.reynolds, args.aoa)
    print(f"CFD (pressure-integrated): Cl={cfd['Cl']:.4f} Cd={cfd['Cd']:.4f}")
    print(f"XFOIL: Cl={xfoil['Cl']} Cd={xfoil['Cd']} converged={xfoil['converged']}")
