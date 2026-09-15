"""
Stage 1 — CFD meshing (gmsh -> OpenFOAM).

Input:  Stage 0 output (normalized, Selig-ordered, unit-chord airfoil
        coords) + domain sizing params.
Output: an OpenFOAM polyMesh (written by `gmshToFoam`) plus a `checkMesh`
        pass/fail verdict with the parsed quality metrics.

Domain: C-grid, far-field radius 15c, wake extent 20c downstream of the
trailing edge (locked decisions in the build spec). The geometry is built
directly with gmsh's OCC kernel -- no STEP/build123d hop -- since the
spec's Stage 1 contract takes the coords array straight to a .msh file and
this project's Stage 1 owns the domain shape as well as the mesh.

A single 'farfield' patch covers the whole outer boundary
(freestreamVelocity/freestreamPressure) rather than separate
inlet/outlet/symmetry patches, so the sweep's higher-AoA cases don't end
up with actual outflow hitting a patch fixed as an inlet. OpenFOAM has no
true 2D solver, so the 2D domain face is extruded one chord length in Z
with the front/back faces marked 'empty'.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
from scipy.interpolate import CubicSpline

from pipeline.troubleshooter import diagnose_mesh_failure, log_troubleshooter_call
from pipeline._shell import run_shell as _run_wsl_raw
from pipeline._shell import to_linux_path as _to_wsl_path


# ------------------------------------------------------------------
# Boundary resampling for meshing
# ------------------------------------------------------------------

def _resample_cosine(coords: np.ndarray, n_per_surface: int = 150) -> np.ndarray:
    """
    Re-sample the closed airfoil loop to cosine-spaced points per surface,
    for meshing purposes only -- this does not alter Stage 0's contract or
    its returned array, which stays the exact-digitized geometry.

    Some real UIUC .dat files have as few as ~30 raw points with roughly
    even spacing (not clustered at the LE, where curvature is highest).
    Feeding that directly to the mesher as a straight-line polygon produces
    locally skewed cells where the coarse polygon crosses a high-curvature
    region -- confirmed empirically on `goe398.dat` (33 points): raw
    points gave checkMesh a max skewness of 13.7 (hard fail); cosine
    resampling to 150 points/surface via cubic-spline arc-length
    interpolation brought it to 2.1 (clean pass), with no other change.
    """
    le_idx = int(np.argmin(coords[:, 0]))
    upper = coords[: le_idx + 1]  # TE -> LE
    lower = coords[le_idx:]       # LE -> TE

    def _resample_arm(arm):
        d = np.diff(arm, axis=0)
        ds = np.sqrt((d ** 2).sum(axis=1))
        s = np.concatenate([[0], np.cumsum(ds)])
        s /= s[-1]
        spline_x = CubicSpline(s, arm[:, 0])
        spline_y = CubicSpline(s, arm[:, 1])
        theta = np.linspace(0, np.pi, n_per_surface)
        s_new = 0.5 * (1 - np.cos(theta))
        return np.column_stack([spline_x(s_new), spline_y(s_new)])

    upper_rs = _resample_arm(upper)
    lower_rs = _resample_arm(lower)
    return np.vstack([upper_rs, lower_rs[1:]])  # skip duplicate LE point


# ------------------------------------------------------------------
# Windows <-> WSL plumbing
# ------------------------------------------------------------------

def _run_wsl(bash_cmd: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a command against OpenFOAM (WSL on Windows, native on Linux --
    see pipeline/_shell.py) with the OpenFOAM environment sourced."""
    return _run_wsl_raw(bash_cmd, timeout=timeout, source_openfoam=True)


# ------------------------------------------------------------------
# gmsh geometry + mesh generation (runs on the Windows side)
# ------------------------------------------------------------------

_GMSH_SCRIPT_TEMPLATE = '''
import sys
import json

def main():
    try:
        import gmsh
    except ImportError:
        print("MESH_ERROR: gmsh not installed. Run: python -m pip install gmsh")
        sys.exit(1)

    coords = json.loads(r"""{coords_json}""")
    far_field_radius = {far_field_radius}
    wake_extent = {wake_extent}
    extrusion_thickness = {extrusion_thickness}
    bl_layers = {bl_layers}
    bl_ratio = {bl_ratio}
    bl_size = {bl_size}
    min_size = {min_size}
    max_size = {max_size}
    mesh_path = r"{mesh_path}"

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.add("airfoil_domain")
        occ = gmsh.model.occ

        # --- 1. Outer C-grid boundary -----------------------------------
        # LE is at (0, 0); TE at (1, 0) (unit-chord, per Stage 0 contract).
        r = far_field_radius
        x_out = 1.0 + wake_extent  # wake extent measured downstream of TE

        p_top = occ.addPoint(0, r, 0)
        p_top_out = occ.addPoint(x_out, r, 0)
        p_bot_out = occ.addPoint(x_out, -r, 0)
        p_bot = occ.addPoint(0, -r, 0)
        p_left = occ.addPoint(-r, 0, 0)
        p_center = occ.addPoint(0, 0, 0)

        l_top = occ.addLine(p_top, p_top_out)
        l_out = occ.addLine(p_top_out, p_bot_out)
        l_bot = occ.addLine(p_bot_out, p_bot)
        a_bot_left = occ.addCircleArc(p_bot, p_center, p_left)
        a_left_top = occ.addCircleArc(p_left, p_center, p_top)

        outer_loop = occ.addCurveLoop([l_top, l_out, l_bot, a_bot_left, a_left_top])
        outer_face = occ.addPlaneSurface([outer_loop])

        # --- 2. Airfoil hole ----------------------------------------------
        # Straight-line polygon through the exact Stage 0 points (a spline
        # could overshoot between points and self-intersect near the LE/TE).
        pt_tags = [occ.addPoint(x, y, 0) for x, y in coords[:-1]]  # last==first
        n = len(pt_tags)
        airfoil_lines = [occ.addLine(pt_tags[i], pt_tags[(i + 1) % n]) for i in range(n)]
        airfoil_loop = occ.addCurveLoop(airfoil_lines)
        airfoil_face = occ.addPlaneSurface([airfoil_loop])

        occ.synchronize()
        cut_result, _ = occ.cut([(2, outer_face)], [(2, airfoil_face)])
        occ.synchronize()
        domain_face = cut_result[0][1]

        # --- 3. Sizing + boundary layer on the airfoil wall ----------------
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

        try:
            f = gmsh.model.mesh.field
            bl = f.add("BoundaryLayer")
            f.setNumbers(bl, "CurvesList", airfoil_lines)
            f.setNumber(bl, "Size", bl_size)
            f.setNumber(bl, "Ratio", bl_ratio)
            f.setNumber(bl, "NbLayers", bl_layers)
            f.setNumber(bl, "Quads", 1)
            f.setAsBoundaryLayer(bl)
            print(f"INFO: BL set - {{len(airfoil_lines)}} curves, {{bl_layers}} layers")
        except Exception as bl_err:
            print(f"WARNING: BL failed ({{bl_err}}). Proceeding without inflation layers.")

        # --- 4. Extrude one layer in Z (pseudo-2D for OpenFOAM) ------------
        out = occ.extrude(
            [(2, domain_face)], 0, 0, extrusion_thickness,
            numElements=[1], recombine=True,
        )
        occ.synchronize()

        # out[0] = far face (2D, at z=extrusion_thickness) -> "front"
        # out[1] = volume (3D) -> "fluid"
        # out[2:] = lateral faces, one per boundary curve of domain_face, in
        #           the same order as getBoundary() below.
        front_face = out[0][1]
        volume = out[1][1]
        lateral_faces = [e[1] for e in out[2:]]

        boundary_curves = gmsh.model.getBoundary(
            [(2, domain_face)], combined=False, oriented=False
        )
        boundary_curve_tags = [c[1] for c in boundary_curves]

        assert len(boundary_curve_tags) == len(lateral_faces), (
            f"Curve/lateral-face count mismatch: "
            f"{{len(boundary_curve_tags)}} vs {{len(lateral_faces)}}"
        )

        airfoil_curve_set = set(airfoil_lines)
        airfoil_lateral_faces = []
        farfield_lateral_faces = []
        for curve_tag, lateral_tag in zip(boundary_curve_tags, lateral_faces):
            if curve_tag in airfoil_curve_set:
                airfoil_lateral_faces.append(lateral_tag)
            else:
                farfield_lateral_faces.append(lateral_tag)

        if not airfoil_lateral_faces:
            raise RuntimeError("No lateral faces matched to the airfoil boundary curves.")
        if not farfield_lateral_faces:
            raise RuntimeError("No lateral faces matched to the farfield boundary curves.")

        # --- 5. Physical groups (these become OpenFOAM boundary patches) ---
        gmsh.model.addPhysicalGroup(2, [domain_face], name="back")
        gmsh.model.addPhysicalGroup(2, [front_face], name="front")
        gmsh.model.addPhysicalGroup(2, airfoil_lateral_faces, name="airfoil")
        gmsh.model.addPhysicalGroup(2, farfield_lateral_faces, name="farfield")
        gmsh.model.addPhysicalGroup(3, [volume], name="fluid")

        # --- 6. Generate 3D mesh (extrudes the 2D mesh into one layer) -----
        gmsh.model.mesh.generate(3)

        import os as _os
        _os.makedirs(_os.path.dirname(mesh_path) or ".", exist_ok=True)
        gmsh.write(mesh_path)
        gmsh.finalize()

        if not _os.path.exists(mesh_path):
            raise RuntimeError(f"gmsh.write() completed but file not found: {{mesh_path}}")

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
'''


def _is_bl_self_intersection_failure(gmsh_output: str) -> bool:
    """
    True if a gmsh failure's own output matches the diagnosed boundary-
    layer self-overlap signature (a thick/high-curvature geometry's BL
    extrusion crossing itself before a mesh is even produced), rather
    than some other, unrecognized gmsh failure.

    Checks only the terminal "Edge not recovered" error, not the
    "intersections in the 1D mesh" warning that can precede it -- found
    the hard way (STATUS.md, 2026-09-15): that warning is a generic gmsh
    edge-splitting-and-retry notice that also appears ahead of a
    genuinely different terminal failure ("Could not find extruded
    node..."), so checking for it alone silently swallowed an
    unrecognized failure into this (wrong) deterministic fix instead of
    routing it to the troubleshooter.
    """
    return "Edge not recovered" in gmsh_output


def generate_mesh(
    coords,
    name: str,
    output_dir: str,
    far_field_radius: float = 15.0,
    wake_extent: float = 20.0,
    extrusion_thickness: float = 1.0,
    bl_layers: int = 10,
    bl_ratio: float = 1.2,
    bl_size: float = 1e-3,
    min_size: float = 0.01,
    max_size: float = 5.0,
    mesh_points_per_surface: int = 150,
    max_retries: int = 3,
    enable_troubleshooter: bool = False,
    troubleshooter_log_path: str | None = None,
) -> str:
    """
    Build the C-grid domain around `coords` directly in gmsh and mesh it.

    Args:
        coords: (N, 2) Stage 0 output -- unit-chord, Selig-ordered, closed.
        name: Base name for output files.
        output_dir: Directory for the generated .msh and gmsh script.
        mesh_points_per_surface: the airfoil boundary is cosine-resampled
            to this many points per surface before meshing (see
            `_resample_cosine`) -- independent of how many raw points the
            source .dat file had.
        bl_size: first boundary-layer cell height at the wall. Default
            1e-3 is chosen for universal `checkMesh` Gate #1 robustness
            across the full UIUC geometry set, per the spec's own
            rationale for wall functions in the first place
            ("Robustness across ~hundreds of UIUC geometries matters
            more than boundary-layer accuracy for this demo").
            A larger value (7e-3, confirmed via `foamPostProcess -func
            yPlus` on naca0012/AoA=4/Re=5e5/nu=1.5e-5) puts y+ in the
            spec's locked wall-function range of 30-300 (14.4 -> 89.8
            average) -- but pushes max skewness past OpenFOAM's pass
            threshold on at least 3 of the 35 real UIUC fixtures
            (naca0021, ah79100c, naca633418; see STATUS.md's
            troubleshooter-candidates section for the measured
            skewness values at 1e-3/3e-3/7e-3). Callers that need
            physical accuracy (e.g. an XFOIL cross-check) on a specific
            geometry should pass a larger `bl_size` explicitly rather
            than relying on this default; this is a genuine per-geometry
            calibration trade-off, not something one scalar default can
            satisfy across the whole geometry zoo.
        enable_troubleshooter: if True, a gmsh failure that doesn't match
            any known deterministic signature (see
            _is_bl_self_intersection_failure) is diagnosed by
            `pipeline.troubleshooter.diagnose_mesh_failure` (an LLM
            judgment call via the local `claude` CLI) instead of the
            generic increase-bl_size ladder. Off by default -- opt-in
            per the spec's own agent-scope caution, and because it costs
            real Claude usage and ~10s of wall time per call. If the
            troubleshooter itself fails (CLI unavailable, bad output),
            falls back to the generic ladder rather than raising.
        troubleshooter_log_path: JSONL log of every troubleshooter
            invocation (inputs, proposed params, reasoning, outcome).
            Ignored if enable_troubleshooter is False.

    Returns:
        Absolute path to the generated gmsh MSH v2.2 file.

    Raises:
        RuntimeError: if gmsh fails on every retry.
    """
    os.makedirs(output_dir, exist_ok=True)
    mesh_path = os.path.abspath(os.path.join(output_dir, f"{name}_3d.msh"))
    script_path = os.path.abspath(os.path.join(output_dir, f"mesh_gen_{name}.py"))
    coords = np.asarray(coords, dtype=np.float64)
    mesh_coords = _resample_cosine(coords, n_per_surface=mesh_points_per_surface)
    coords_json = json.dumps(mesh_coords.tolist())
    geometry_stats = {
        "n_points": int(len(coords)),
        "max_thickness_estimate": float(coords[:, 1].max() - coords[:, 1].min()),
        "chord": 1.0,
    }

    params = dict(
        far_field_radius=far_field_radius,
        wake_extent=wake_extent,
        extrusion_thickness=extrusion_thickness,
        bl_layers=bl_layers,
        bl_ratio=bl_ratio,
        bl_size=bl_size,
        min_size=min_size,
        max_size=max_size,
    )

    last_output = ""
    pending_log_record = None
    for attempt in range(1, max_retries + 1):
        script_content = _GMSH_SCRIPT_TEMPLATE.format(
            coords_json=coords_json,
            mesh_path=mesh_path.replace("\\", "/"),
            **params,
        )
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        result = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True, timeout=300
        )
        last_output = result.stdout + result.stderr
        succeeded = "MESH_SUCCESS" in last_output and result.returncode == 0

        if pending_log_record is not None:
            pending_log_record["outcome"] = "succeeded" if succeeded else "failed"
            if troubleshooter_log_path:
                log_troubleshooter_call(troubleshooter_log_path, pending_log_record)
            pending_log_record = None

        if succeeded:
            return mesh_path

        if attempt < max_retries:
            if _is_bl_self_intersection_failure(last_output):
                # Diagnosed failure (found probing a 40%-thick synthetic
                # airfoil, 2026-09-14/15): the boundary-layer offset
                # self-overlaps near tight leading-edge curvature before
                # gmsh even produces a mesh ("Edge not recovered" /
                # "intersections in the 1D mesh"). The generic ladder
                # below *increases* bl_size on every failure, which makes
                # this specific failure worse, not better -- confirmed
                # empirically that *reducing* bl_size fixes it (and the
                # resulting mesh still passes checkMesh's Gate #1).
                params["bl_size"] = max(1e-5, params["bl_size"] * 0.3)
            elif enable_troubleshooter:
                current_params = dict(params)
                try:
                    decision = diagnose_mesh_failure(last_output, current_params, geometry_stats)
                    params["bl_size"] = decision["bl_size"]
                    params["bl_layers"] = decision["bl_layers"]
                    params["bl_ratio"] = decision["bl_ratio"]
                    pending_log_record = {
                        "name": name,
                        "gmsh_output": last_output[-2000:],
                        "current_params": current_params,
                        "proposed_params": decision,
                        "reasoning": decision["reasoning"],
                    }
                except Exception as e:
                    # The agent's own failure (CLI unavailable, bad
                    # output) must never crash the pipeline -- fall back
                    # to the generic ladder for this attempt instead.
                    if troubleshooter_log_path:
                        log_troubleshooter_call(troubleshooter_log_path, {
                            "name": name, "gmsh_output": last_output[-2000:],
                            "current_params": current_params,
                            "outcome": "agent_error", "error": f"{type(e).__name__}: {e}",
                        })
                    params["bl_layers"] = max(3, int(params["bl_layers"] * 0.7))
                    params["bl_ratio"] = min(1.5, params["bl_ratio"] + 0.1)
                    params["bl_size"] = params["bl_size"] * 2.0
            else:
                params["bl_layers"] = max(3, int(params["bl_layers"] * 0.7))
                params["bl_ratio"] = min(1.5, params["bl_ratio"] + 0.1)
                params["bl_size"] = params["bl_size"] * 2.0

    raise RuntimeError(
        f"Failed to mesh '{name}' after {max_retries} attempts.\nLast output:\n{last_output[-3000:]}"
    )


# ------------------------------------------------------------------
# gmshToFoam conversion + checkMesh (run inside WSL)
# ------------------------------------------------------------------

_MINIMAL_CONTROL_DICT = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application     foamRun;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
"""


def convert_to_openfoam(msh_path: str, case_dir: str) -> None:
    """
    Run `gmshToFoam` inside WSL to convert a gmsh MSH into an OpenFOAM
    polyMesh under `<case_dir>/constant/polyMesh/`.

    Raises:
        RuntimeError: if gmshToFoam exits non-zero.
    """
    os.makedirs(os.path.join(case_dir, "system"), exist_ok=True)
    control_dict_path = os.path.join(case_dir, "system", "controlDict")
    with open(control_dict_path, "w") as f:
        f.write(_MINIMAL_CONTROL_DICT)

    case_wsl = _to_wsl_path(case_dir)
    msh_wsl = _to_wsl_path(msh_path)

    result = _run_wsl(f'cd "{case_wsl}" && gmshToFoam "{msh_wsl}"')
    if result.returncode != 0:
        raise RuntimeError(
            f"gmshToFoam failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    poly_mesh_dir = os.path.join(case_dir, "constant", "polyMesh")
    required = ["points", "faces", "owner", "neighbour", "boundary"]
    missing = [f for f in required if not os.path.exists(os.path.join(poly_mesh_dir, f))]
    if missing:
        raise RuntimeError(
            f"gmshToFoam reported success but polyMesh is incomplete, missing: {missing}"
        )

    _set_patch_types(
        os.path.join(poly_mesh_dir, "boundary"),
        {
            # gmshToFoam always types every patch it creates as generic
            # `patch`, including the extrusion end-caps. For a
            # single-cell-thick pseudo-2D mesh those must be typed `empty`
            # -- otherwise OpenFOAM treats the mesh as genuinely 3D, which
            # corrupts unrelated quality metrics wholesale (verified
            # empirically: checkMesh's "cell determinant" and
            # aspect-ratio checks fail on effectively every cell in the
            # mesh, not just the thin boundary-layer ones, until this is
            # fixed -- confirmed by replicating OpenFOAM's own bundled
            # `nacaAirfoil` tutorial, which does this exact same
            # patch-type fixup for the same reason).
            "front": "empty",
            "back": "empty",
            # `wall` (not generic `patch`) is required for wall-function
            # turbulence BCs and near-wall post-processing on the airfoil
            # surface. `farfield` is left as `patch`: OpenFOAM's
            # freestreamVelocity/freestreamPressure BCs apply on a
            # standard `patch`-typed boundary.
            "airfoil": "wall",
        },
    )


def _set_patch_types(boundary_path: str, patch_types: dict) -> None:
    """Rewrite the `type`/`physicalType` entries for named patches in an OpenFOAM `boundary` file."""
    with open(boundary_path, "r") as f:
        text = f.read()

    for name, patch_type in patch_types.items():
        pattern = re.compile(
            rf"(^\s*{re.escape(name)}\s*\{{)(.*?)(\}})",
            re.DOTALL | re.MULTILINE,
        )

        def _replace(match, patch_type=patch_type):
            body = match.group(2)
            body = re.sub(r"type\s+\w+;", f"type            {patch_type};", body)
            body = re.sub(r"physicalType\s+\w+;", f"physicalType    {patch_type};", body)
            return match.group(1) + body + match.group(3)

        text, n = pattern.subn(_replace, text)
        if n == 0:
            raise RuntimeError(f"Could not find patch '{name}' in {boundary_path} to retype.")

    with open(boundary_path, "w") as f:
        f.write(text)


def check_mesh(case_dir: str) -> dict:
    """
    Run `checkMesh` inside WSL and parse its output for the spec's Must-Pass
    Gate #1: zero negative-volume cells, and non-orthogonality/skewness
    passing OpenFOAM's *own* pass/fail verdict for those two checks (its
    "... check OK." lines), not a threshold re-implemented here.

    Deliberately NOT part of the gate: aspect ratio and "cell determinant"
    warnings. Verified empirically against OpenFOAM 12's own bundled,
    validated `nacaAirfoil` tutorial mesh -- it also fails the determinant
    check (for a subset of its boundary-layer cells), so requiring a clean
    determinant result would reject a mesh OpenFOAM's own reference case
    doesn't meet either. This matches the spec's Gate #1 wording, which
    lists only negative volumes and non-orthogonality/skewness.

    Deliberately runs plain `checkMesh`, not `-allTopology -allGeometry`:
    those extra checks aren't part of the gate, and on a real (bad-quality)
    UIUC case they hung for 5+ minutes on a mesh where plain `checkMesh`
    completed in under a second and reported the same real defect
    (skewness) that made the mesh fail anyway.

    Returns:
        {
            "passed": bool,                       # the spec's Gate #1 only
            "negative_volume_cells": int | None,   # None = couldn't parse
            "non_orthogonality_ok": bool,
            "max_non_orthogonality_deg": float | None,   # informational
            "skewness_ok": bool,
            "max_skewness": float | None,                # informational
            "mesh_ok": bool,           # checkMesh's own blanket "Mesh OK." (informational only)
            "raw_output": str,
        }
    """
    case_wsl = _to_wsl_path(case_dir)
    result = _run_wsl(f'cd "{case_wsl}" && checkMesh', timeout=60)
    output = result.stdout + result.stderr

    # Exact wording from OpenFOAM's own source
    # (src/meshCheck/primitiveMeshCheck/primitiveMeshCheck.C):
    #   fail: "***Zero or negative cell volume detected.  Minimum negative
    #          volume: X, Number of negative volume cells: N"
    #   pass: "Min volume = X. Max volume = Y.  Total volume = Z.  Cell volumes OK."
    neg_vol_fail = re.search(
        r"Zero or negative cell volume detected\.\s*Minimum negative volume:\s*"
        r"[-\d.eE]+,\s*Number of negative volume cells:\s*(\d+)",
        output,
    )
    if neg_vol_fail:
        negative_volume_cells = int(neg_vol_fail.group(1))
    elif "Cell volumes OK." in output:
        negative_volume_cells = 0
    else:
        negative_volume_cells = None  # couldn't parse -- must not silently pass

    # src/meshCheck/polyMeshCheck/polyMeshCheck.C: "Mesh non-orthogonality Max: X average: Y"
    # followed by either "Non-orthogonality check OK." or "***Number of non-orthogonality errors: N."
    non_ortho_match = re.search(
        r"Mesh non-orthogonality Max:\s*([\d.]+)\s*average:\s*([\d.]+)", output
    )
    max_non_ortho = float(non_ortho_match.group(1)) if non_ortho_match else None
    non_orthogonality_ok = "Non-orthogonality check OK." in output

    # "Max skewness = X OK." (pass) or "***Max skewness = X" (fail)
    skew_match = re.search(r"Max skewness = ([\d.]+)", output)
    max_skew = float(skew_match.group(1)) if skew_match else None
    skewness_ok = bool(re.search(r"Max skewness = [\d.]+ OK\.", output))

    mesh_ok = "Mesh OK" in output

    passed = (
        result.returncode == 0
        and negative_volume_cells == 0
        and non_orthogonality_ok
        and skewness_ok
    )

    return {
        "passed": passed,
        "negative_volume_cells": negative_volume_cells,
        "non_orthogonality_ok": non_orthogonality_ok,
        "max_non_orthogonality_deg": max_non_ortho,
        "skewness_ok": skewness_ok,
        "max_skewness": max_skew,
        "mesh_ok": mesh_ok,
        "raw_output": output,
    }


def run_stage1(coords, name: str, output_dir: str, **mesh_kwargs) -> dict:
    """
    Full Stage 1 pipeline: build+mesh domain -> gmshToFoam -> checkMesh.

    Returns:
        {
            "msh_path": str,
            "case_dir": str,
            "check": <check_mesh() dict>,
        }

    Raises:
        RuntimeError: if meshing or gmshToFoam fails outright (checkMesh
                      failing the quality gate is reported, not raised).
    """
    msh_path = generate_mesh(coords, name, output_dir, **mesh_kwargs)
    case_dir = os.path.join(output_dir, f"{name}_case")
    convert_to_openfoam(msh_path, case_dir)
    check = check_mesh(case_dir)
    return {"msh_path": msh_path, "case_dir": case_dir, "check": check}


if __name__ == "__main__":
    import argparse
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.stage0_geometry_loader import load_airfoil

    parser = argparse.ArgumentParser(description="Stage 1: mesh an airfoil domain for OpenFOAM.")
    parser.add_argument("--input", required=True, help="Path to the .dat file.")
    parser.add_argument("--name", default="airfoil")
    parser.add_argument("--output-dir", default="data/mesh")
    args = parser.parse_args()

    coords = load_airfoil(args.input)
    result = run_stage1(coords, args.name, args.output_dir)
    print(f"Mesh: {result['msh_path']}")
    print(f"Case: {result['case_dir']}")
    print(f"checkMesh passed: {result['check']['passed']}")
