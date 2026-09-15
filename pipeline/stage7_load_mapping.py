"""
Stage 7 -- Load mapping (fluid -> structure).

Input:  Stage 4's live `extract_surface_pressure()` result (s, x, y, Cp)
        for one AoA + Stage 6's shell mesh (`.inp`).
Output: a physical pressure (Pa) per skin element, written to a JSON
        artifact, plus the mapped resultant force for Gate #3 checking.

The CFD solver runs at rho=1 (Cp is dimensionless), but a real,
meter-scale shell structure needs a physical pressure in Pa:
`p_Pa = Cp * 0.5 * rho_air * U_inf^2`, with `rho_air` a new parameter
this stage introduces (default 1.225 kg/m^3, ISA sea level).

Only "skin" elements (Stage 6's ELSET) receive direct aerodynamic
pressure -- spar/rib elements are internal members. For each skin
element, its centroid's (x, y) (z ignored -- load is uniform along span,
the locked decision) is projected onto Stage 4's own ordered surface
polyline via nearest-segment projection to find its arc-length position,
then Cp is interpolated there via `scipy.interpolate.interp1d`. The
polyline is treated as a closed loop (wrapping the last point back to
the first) since the airfoil surface it represents is closed.

Sign convention mirrors Stage 4's own force-integration code
(`pipeline.stage4_postprocess.pressure_integrated_cl_cd`): force on an
element = -pressure * area * outward_normal, i.e. a positive Cp
(higher than freestream) pushes INWARD.
"""

import json
import os

import numpy as np
from scipy.interpolate import interp1d

from pipeline.stage6_structural_mesh import parse_inp


def _triangle_geometry(nodes, tri_node_ids):
    p0, p1, p2 = (np.array(nodes[n]) for n in tri_node_ids[:3])
    raw_normal = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(raw_normal)
    unit_normal = raw_normal / (2.0 * area)
    centroid = (p0 + p1 + p2) / 3.0
    return centroid, area, unit_normal


def _closed_loop_arrays(surface):
    x = np.asarray(surface["x"], dtype=float)
    y = np.asarray(surface["y"], dtype=float)
    s = np.asarray(surface["s"], dtype=float)
    Cp = np.asarray(surface["Cp"], dtype=float)

    x_ext = np.concatenate([x, [x[0]]])
    y_ext = np.concatenate([y, [y[0]]])
    seglens = np.hypot(np.diff(x_ext), np.diff(y_ext))
    s_ext = np.concatenate([s, [s[-1] + seglens[-1]]])
    Cp_ext = np.concatenate([Cp, [Cp[0]]])
    return x_ext, y_ext, s_ext, Cp_ext, seglens


def _project_to_arc_length(qx, qy, x_ext, y_ext, s_ext, seglens):
    """Nearest-segment projection of a query point onto the closed
    polyline; returns the corresponding arc-length position."""
    p0 = np.column_stack([x_ext[:-1], y_ext[:-1]])
    p1 = np.column_stack([x_ext[1:], y_ext[1:]])
    d = p1 - p0
    q = np.array([qx, qy])
    t = np.sum((q - p0) * d, axis=1) / np.sum(d * d, axis=1)
    t = np.clip(t, 0.0, 1.0)
    proj = p0 + t[:, None] * d
    dist = np.hypot(*(q - proj).T)
    i = int(np.argmin(dist))
    return s_ext[i] + t[i] * seglens[i]


def map_pressure_to_mesh(surface, inp_path, name, output_dir, U_inf, span, rho_air=1.225):
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Stage 6 mesh not found: {inp_path}")
    if len(surface["s"]) == 0:
        raise ValueError("surface pressure curve is empty")
    if rho_air <= 0.0:
        raise ValueError(f"rho_air must be positive, got {rho_air}")

    os.makedirs(output_dir, exist_ok=True)

    parsed = parse_inp(inp_path)
    nodes = parsed["nodes"]
    elements_by_id = {el["id"]: el for el in parsed["elements"]}
    skin_ids = parsed["elsets"]["skin"]

    x_ext, y_ext, s_ext, Cp_ext, seglens = _closed_loop_arrays(surface)
    cp_interp = interp1d(
        s_ext, Cp_ext, kind="linear", bounds_error=False,
        fill_value=(Cp_ext[0], Cp_ext[-1]),
    )

    q_dyn = 0.5 * rho_air * U_inf ** 2
    poly_cx = float(np.mean(surface["x"]))
    poly_cy = float(np.mean(surface["y"]))

    element_pressures_pa = {}
    force_dot_normal = []
    total_force = np.zeros(3)

    for eid in skin_ids:
        el = elements_by_id[eid]
        centroid, area, normal = _triangle_geometry(nodes, el["nodes"])
        if (centroid[0] - poly_cx) * normal[0] + (centroid[1] - poly_cy) * normal[1] < 0:
            normal = -normal

        s_q = _project_to_arc_length(centroid[0], centroid[1], x_ext, y_ext, s_ext, seglens)
        cp_q = float(cp_interp(s_q))
        p_pa = cp_q * q_dyn
        element_pressures_pa[eid] = p_pa

        force_vec = -p_pa * area * normal
        total_force += force_vec
        force_dot_normal.append(float(np.dot(force_vec, normal)))

    result = {
        "loads_path": os.path.join(output_dir, f"{name}_loads.json"),
        "rho_air": rho_air,
        "U_inf": U_inf,
        "span": span,
        "element_pressures_pa": element_pressures_pa,
        "resultant_force_n": tuple(total_force.tolist()),
        "per_element_force_dot_normal": force_dot_normal,
    }

    with open(result["loads_path"], "w") as f:
        json.dump({
            "rho_air": rho_air,
            "U_inf": U_inf,
            "span": span,
            "element_pressures_pa": {str(k): v for k, v in element_pressures_pa.items()},
            "resultant_force_n": list(total_force),
        }, f)

    return result
