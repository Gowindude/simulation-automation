"""
Stage 8 -- CalculiX execution.

Input:  Stage 6's shell mesh (`.inp`) + Stage 7's per-element pressure
        loads + material properties (generic aluminum placeholder,
        configurable).
Output: a `.frd` result file -> parsed von Mises stress field, plus a
        `.dat` file with root reaction forces for Gate #4.

Assembles a complete, executable CalculiX deck around Stage 6's mesh via
*INCLUDE (no mesh content is duplicated): a root NSET (every node at
z=0), *SHELL SECTION for each of skin/spar/rib (all sharing the same
placeholder material/thickness -- the spec doesn't distinguish by
region), a cantilever *BOUNDARY (root fully fixed, tip free), and a
*DLOAD line per skin element.

CalculiX's shell pressure label "P" is positive ALONG the element's own
connectivity (node-order) normal -- confirmed empirically via a single-
element probe deck's `.frd` displacement output (D3 came out positive,
i.e. deflection in the same direction as the RH-rule connectivity
normal, for a positive P). Since gmsh's per-triangle node ordering is
arbitrary, each element's raw connectivity normal may not match Stage
7's established "true outward" convention, and Stage 7's own force
convention is force = -pressure * true_outward_normal * area (positive
pressure pushes INWARD). Equating CalculiX's physical force
(+P * raw_normal * area) to Stage 7's intended force
(-pressure_pa * true_outward_normal * area) gives
P = -pressure_pa when raw_normal already matches true_outward, and
P = +pressure_pa when it's flipped -- i.e. `P = -pressure_pa * sign`,
where `sign` is Stage 7's own outward-vs-raw-normal check, recomputed
here (`_outward_sign`).
"""

import os
import re
import subprocess

import numpy as np

from pipeline.stage6_structural_mesh import parse_inp


def _to_wsl_path(win_path: str) -> str:
    win_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(win_path)
    drive_letter = drive.rstrip(":").lower()
    rest = rest.replace("\\", "/")
    return f"/mnt/{drive_letter}{rest}"


def _run_wsl(bash_cmd: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl.exe", "--", "bash", "-lc", bash_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _raw_triangle_normal(nodes, tri_node_ids):
    p0, p1, p2 = (np.array(nodes[n]) for n in tri_node_ids[:3])
    raw = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(raw)
    centroid = (p0 + p1 + p2) / 3.0
    return centroid, area, raw / (2.0 * area)


def _outward_sign(centroid, raw_normal, poly_cx, poly_cy):
    dot = (centroid[0] - poly_cx) * raw_normal[0] + (centroid[1] - poly_cy) * raw_normal[1]
    return 1.0 if dot > 0 else -1.0


def _write_nset(lines, name, node_ids, per_line=8):
    lines.append(f"*NSET, NSET={name}")
    ids = sorted(node_ids)
    for i in range(0, len(ids), per_line):
        lines.append(", ".join(str(n) for n in ids[i:i + per_line]))


def _parse_frd_max_von_mises(frd_path):
    with open(frd_path) as f:
        text = f.read()

    m = re.search(r"-4\s+STRESS.*?\n((?: -5.*\n)+)((?: -1.*\n)+)", text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"no STRESS block found in {frd_path}")

    # Fixed-width columns, NOT space-delimited: a 4+ digit node id
    # directly abutting a negative value leaves no separating space
    # (confirmed empirically: "      3808-2.67249E+04..."). Format is a
    # 3-char key, a 10-char node id field, then six 12-char value fields.
    max_vm = 0.0
    max_node = None
    stress_field = {}  # node_id -> max von Mises seen for that node (top/bottom points)
    for line in m.group(2).splitlines():
        if len(line) < 85:
            continue
        node_id = int(line[3:13])
        sxx, syy, szz, sxy, syz, szx = (
            float(line[13 + 12 * i: 25 + 12 * i]) for i in range(6)
        )
        vm = (0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
              + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)) ** 0.5
        stress_field[node_id] = max(vm, stress_field.get(node_id, 0.0))
        if vm > max_vm:
            max_vm = vm
            max_node = node_id

    return max_vm, max_node, stress_field


def _parse_dat_total_force(dat_path):
    with open(dat_path) as f:
        text = f.read()
    m = re.search(
        r"total force \(fx,fy,fz\) for set \w+ and time.*\n\n\s*"
        r"([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)",
        text,
    )
    if not m:
        raise RuntimeError(f"no 'total force' line found in {dat_path}")
    return tuple(float(v) for v in m.groups())


def run_calculix_analysis(
    inp_path, loads_result, name, output_dir,
    E=70e9, nu=0.33, rho=2700.0, thickness=0.002,
):
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Stage 6 mesh not found: {inp_path}")
    if E <= 0.0:
        raise ValueError(f"E must be positive, got {E}")
    if not (0.0 < nu < 0.5):
        raise ValueError(f"nu must be in (0, 0.5), got {nu}")
    if rho <= 0.0:
        raise ValueError(f"rho must be positive, got {rho}")
    if thickness <= 0.0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    os.makedirs(output_dir, exist_ok=True)

    parsed = parse_inp(inp_path)
    nodes = parsed["nodes"]
    elements_by_id = {el["id"]: el for el in parsed["elements"]}
    skin_ids = parsed["elsets"]["skin"]

    root_ids = [nid for nid, (x, y, z) in nodes.items() if abs(z) < 1e-6]
    if not root_ids:
        raise RuntimeError("no root (z=0) nodes found in the mesh")

    skin_x = [nodes[nid][0] for eid in skin_ids for nid in elements_by_id[eid]["nodes"]]
    skin_y = [nodes[nid][1] for eid in skin_ids for nid in elements_by_id[eid]["nodes"]]
    poly_cx, poly_cy = float(np.mean(skin_x)), float(np.mean(skin_y))

    lines = [f"*INCLUDE, INPUT={os.path.basename(inp_path)}"]
    _write_nset(lines, "ROOT", root_ids)

    lines += [
        "*MATERIAL, NAME=PLACEHOLDER_ALU",
        "*ELASTIC",
        f"{E}, {nu}",
        "*DENSITY",
        f"{rho}",
    ]
    for region in ("skin", "spar", "rib"):
        if parsed["elsets"].get(region):
            lines += [
                f"*SHELL SECTION, ELSET={region}, MATERIAL=PLACEHOLDER_ALU",
                f"{thickness}",
            ]

    lines += [
        "*BOUNDARY",
        "ROOT, 1, 6",
        "*STEP",
        "*STATIC",
        "*DLOAD",
    ]
    for eid in skin_ids:
        el = elements_by_id[eid]
        centroid, area, raw_normal = _raw_triangle_normal(nodes, el["nodes"])
        sign = _outward_sign(centroid, raw_normal, poly_cx, poly_cy)
        p_pa = loads_result["element_pressures_pa"].get(eid, 0.0)
        lines.append(f"{eid}, P, {-p_pa * sign}")

    lines += [
        "*EL FILE",
        "S",
        "*NODE PRINT, NSET=ROOT, TOTALS=YES",
        "RF",
        "*END STEP",
    ]

    job_name = f"{name}_stage8"
    assembled_inp = os.path.join(output_dir, f"{job_name}.inp")
    with open(assembled_inp, "w") as f:
        f.write("\n".join(lines) + "\n")

    # inp_path (the *INCLUDE target) must sit next to the assembled deck.
    inp_copy = os.path.join(output_dir, os.path.basename(inp_path))
    if os.path.abspath(inp_copy) != os.path.abspath(inp_path):
        with open(inp_path) as src, open(inp_copy, "w") as dst:
            dst.write(src.read())

    wsl_dir = _to_wsl_path(output_dir)
    result = _run_wsl(f'cd "{wsl_dir}" && ccx {job_name}', timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ccx failed (rc={result.returncode}): {result.stdout}\n{result.stderr}")

    frd_path = os.path.join(output_dir, f"{job_name}.frd")
    dat_path = os.path.join(output_dir, f"{job_name}.dat")
    if not os.path.exists(frd_path):
        raise RuntimeError(f"ccx did not produce {frd_path}: {result.stdout}")

    max_vm, max_node, stress_field = _parse_frd_max_von_mises(frd_path)
    reaction_force_n = _parse_dat_total_force(dat_path)

    applied = np.array(loads_result["resultant_force_n"])
    reaction = np.array(reaction_force_n)
    residual = float(np.linalg.norm(reaction + applied))

    return {
        "inp_path": assembled_inp,
        "frd_path": frd_path,
        "dat_path": dat_path,
        "max_von_mises_pa": max_vm,
        "max_stress_node": max_node,
        # stress_field is keyed by CalculiX's internal shell-expansion
        # node ids (top/bottom companion nodes), not Stage 6/7's mesh
        # node ids -- there's no meaningful cross-reference needed here,
        # Stage 9 stores this as its own independent (id, von_mises) array.
        "stress_field": stress_field,
        "reaction_force_n": reaction_force_n,
        "reaction_force_residual_n": residual,
    }
