"""
Stage 5 -- Structural geometry generation.

Input:  Stage 0's normalized 2D airfoil coords (unit-chord, Selig-ordered,
        closed) + span, spar locations, rib spacing.
Output: 3D shell B-rep geometry (NOT a mesh -- Stage 6's job) exported as
        both a STEP file (interoperable, inspectable) and a .brep file
        (gmsh/OCC-native serialization) -- the skin (airfoil boundary
        extruded along the span), two spar webs (internal planar surfaces
        at the given chord fractions, bounded in y by the airfoil's own
        upper/lower surface at that x), and rib bulkheads (airfoil-shaped
        filled surfaces at each rib span station, root through tip
        inclusive).

Return schema (all keys always present):
    step_path   -- str, path to the exported STEP file
    brep_path   -- str, path to the exported .brep file
    rib_stations -- list[float], z-coordinates of every rib, sorted ascending
    n_skin_panels -- int, number of skin surfaces (== len(coords) - 1)
    n_spar_webs  -- int, number of spar web surfaces (== len(spar_locations))

Root (z=0) and tip (z=span) ribs are built by reusing the exact same gmsh
curve entities as the skin panels' top/bottom boundary edges, rather than
via a boolean fragment/glue step -- fragmenting the whole assembly would
also split every skin panel at each *interior* rib station (the rib's
polygon boundary at 0 < z < span is geometrically embedded in the middle
of a skin panel, not on its boundary), which would blow up the skin
surface count far past len(coords) - 1. Sharing curve tags at construction
time gives real shared B-rep topology at root/tip without that side
effect; interior ribs and spar webs are independent surfaces, coincident
with the skin but not topologically fused to it -- conformal meshing
across those junctions is Stage 6's concern.
"""

import os

import gmsh
import numpy as np

from pipeline._gmsh_isolation import run_isolated


def _dedupe_sort_by_x(pts):
    """Sort by x and average y over any near-duplicate x, so np.interp's
    strictly-increasing-x requirement isn't silently violated by
    digitization noise near the leading edge."""
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    xs, ys = [pts[0, 0]], [pts[0, 1]]
    for x, y in pts[1:]:
        if x - xs[-1] < 1e-9:
            ys[-1] = (ys[-1] + y) / 2.0
        else:
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)


def _airfoil_y_bounds_at_x(coords, x_frac):
    """Upper/lower y at a given chordwise fraction, via linear
    interpolation on Stage 0's own coords."""
    le_idx = int(np.argmin(coords[:, 0]))
    upper = coords[: le_idx + 1][::-1]  # LE -> TE, x increasing
    lower = coords[le_idx:]             # LE -> TE, x increasing
    upper_x, upper_y = _dedupe_sort_by_x(upper)
    lower_x, lower_y = _dedupe_sort_by_x(lower)
    y_upper = float(np.interp(x_frac, upper_x, upper_y))
    y_lower = float(np.interp(x_frac, lower_x, lower_y))
    return y_lower, y_upper


def _rib_stations(span, rib_spacing):
    """One rib at every multiple of rib_spacing from z=0 through z=span,
    inclusive of both ends -- always including the tip even when span
    isn't an exact multiple of rib_spacing."""
    n_full = int(np.floor(span / rib_spacing + 1e-9))
    stations = [i * rib_spacing for i in range(n_full + 1)]
    if stations[-1] < span - 1e-6:
        stations.append(span)
    return stations


def generate_structural_geometry(
    coords,
    name,
    output_dir,
    span=3.0,
    spar_locations=(0.2, 0.6),
    rib_spacing=0.5,
):
    """
    Validates inputs, then runs the actual gmsh work in an isolated
    subprocess (see pipeline/_gmsh_isolation.py) -- calling gmsh.finalize()
    in-process corrupts later WSL calls (Stages 1/3/4/8), and Stage 8
    must run after this stage's gmsh-built geometry, so isolation (not
    call reordering) is the only fix that holds for a single airfoil's
    full 0-9 run.
    """
    if span <= 0.0:
        raise ValueError(f"span must be positive, got {span}")
    for x_frac in spar_locations:
        if not (0.0 <= x_frac <= 1.0):
            raise ValueError(
                f"spar location {x_frac} is outside the chord (must be in [0, 1])"
            )

    os.makedirs(output_dir, exist_ok=True)
    coords = np.asarray(coords, dtype=float)

    return run_isolated(
        _generate_structural_geometry_worker,
        coords, name, output_dir, span, spar_locations, rib_spacing,
    )


def _generate_structural_geometry_worker(
    coords, name, output_dir, span, spar_locations, rib_spacing,
):
    n_pts = len(coords) - 1  # coords is closed: coords[0] == coords[-1]
    rib_stations = _rib_stations(span, rib_spacing)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(f"{name}_stage5")
        occ = gmsh.model.occ

        pts0 = [occ.addPoint(x, y, 0.0) for x, y in coords[:-1]]
        pts_span = [occ.addPoint(x, y, span) for x, y in coords[:-1]]
        lines0 = [occ.addLine(pts0[i], pts0[(i + 1) % n_pts]) for i in range(n_pts)]
        lines_span = [
            occ.addLine(pts_span[i], pts_span[(i + 1) % n_pts]) for i in range(n_pts)
        ]
        vlines = [occ.addLine(pts0[i], pts_span[i]) for i in range(n_pts)]

        for i in range(n_pts):
            j = (i + 1) % n_pts
            loop = occ.addCurveLoop([lines0[i], vlines[j], -lines_span[i], -vlines[i]])
            occ.addPlaneSurface([loop])

        root_loop = occ.addCurveLoop(lines0)
        occ.addPlaneSurface([root_loop])
        tip_loop = occ.addCurveLoop(lines_span)
        occ.addPlaneSurface([tip_loop])

        for z0 in rib_stations[1:-1]:
            pts_z = [occ.addPoint(x, y, z0) for x, y in coords[:-1]]
            lines_z = [
                occ.addLine(pts_z[k], pts_z[(k + 1) % n_pts]) for k in range(n_pts)
            ]
            loop_z = occ.addCurveLoop(lines_z)
            occ.addPlaneSurface([loop_z])

        for x_frac in spar_locations:
            y_lo, y_hi = _airfoil_y_bounds_at_x(coords, x_frac)
            p1 = occ.addPoint(x_frac, y_lo, 0.0)
            p2 = occ.addPoint(x_frac, y_hi, 0.0)
            p3 = occ.addPoint(x_frac, y_hi, span)
            p4 = occ.addPoint(x_frac, y_lo, span)
            l1 = occ.addLine(p1, p2)
            l2 = occ.addLine(p2, p3)
            l3 = occ.addLine(p3, p4)
            l4 = occ.addLine(p4, p1)
            spar_loop = occ.addCurveLoop([l1, l2, l3, l4])
            occ.addPlaneSurface([spar_loop])

        occ.synchronize()

        step_path = os.path.join(output_dir, f"{name}_structural.step")
        brep_path = os.path.join(output_dir, f"{name}_structural.brep")
        gmsh.write(step_path)
        gmsh.write(brep_path)
    finally:
        gmsh.finalize()

    return {
        "step_path": step_path,
        "brep_path": brep_path,
        "rib_stations": rib_stations,
        "n_skin_panels": n_pts,
        "n_spar_webs": len(spar_locations),
    }
