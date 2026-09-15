"""
Stage 6 -- Structural meshing.

Input:  Stage 5's structural geometry (`.brep`, gmsh/OCC-native, chosen
        over the STEP file specifically because it preserves the shared
        B-rep edges Stage 5 built at the root/tip ribs).
Output: a shell element mesh as a CalculiX-compatible Abaqus-dialect
        `.inp` file.

Ribs and spars are geometrically coincident with, but not CAD-fused to,
the skin at interior span stations (a deliberate Stage 5 decision --
fusing them there would have split Stage 5's own full-span skin panels).
This stage resolves that: the whole imported assembly is self-fragmented
(`occ.fragment(surfaces, surfaces)`) before meshing, which conformally
splits overlapping regions at their true intersection and gives ribs,
spars, and skin genuinely shared mesh nodes at every junction -- the
property a spar/rib shell model needs for CalculiX to transfer load
through it correctly.

gmsh's own `.inp` writer has no shell-element option -- it types every
2D surface element as CPS3/CPS4 (2D continuum), never S3/S4 (shell).
`_rewrite_as_shell_elements` post-processes the written file: converts
CPS3/CPS4 element headers to S3/S4, and drops the boundary curve
elements (T3D2) gmsh also writes, which aren't part of a shell mesh.

Return schema:
    inp_path   -- str, path to the CalculiX-ready .inp file
    mesh_size  -- float, the characteristic element size used (echoed
                  back since it's a tunable parameter, not a locked spec
                  value -- see the module docstring in
                  tests/test_stage6_structural_mesh.py)
    n_skin_surfaces, n_spar_surfaces, n_rib_surfaces -- int, post-
                  fragmentation surface counts per region
"""

import os
import re

import gmsh

from pipeline._gmsh_isolation import run_isolated


def parse_inp(inp_path):
    """
    Parse gmsh's Abaqus-dialect .inp output into nodes, elements, and
    named element sets. Only understands the subset gmsh actually
    writes (confirmed by direct inspection): `*NODE`, `*ELEMENT,
    type=X, ELSET=Y`, `*ELSET,ELSET=Y`.

    Returns:
        {
            "nodes": {int: (x, y, z)},
            "elements": [{"id": int, "type": str, "nodes": [int, ...]}],
            "elsets": {str: set of element ids},
        }
    """
    with open(inp_path) as f:
        lines = [l.rstrip("\n") for l in f]

    nodes = {}
    elements = []
    elsets = {}

    i = 0
    section = None
    current_type = None
    current_elset_name = None
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("*NODE"):
            section = "NODE"
            i += 1
            continue
        if line.startswith("*ELEMENT"):
            section = "ELEMENT"
            m_type = re.search(r"type=(\w+)", line)
            m_elset = re.search(r"ELSET=(\w+)", line)
            current_type = m_type.group(1) if m_type else None
            current_elset_name = m_elset.group(1) if m_elset else None
            i += 1
            continue
        if line.startswith("*ELSET"):
            section = "ELSET"
            m_elset = re.search(r"ELSET=(\w+)", line)
            current_elset_name = m_elset.group(1)
            elsets.setdefault(current_elset_name, set())
            i += 1
            continue
        if line.startswith("*"):
            section = None
            i += 1
            continue

        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if section == "NODE":
            nid = int(parts[0])
            x, y, z = (float(v) for v in parts[1:4])
            nodes[nid] = (x, y, z)
        elif section == "ELEMENT":
            eid = int(parts[0])
            node_ids = [int(v) for v in parts[1:]]
            elements.append({"id": eid, "type": current_type, "nodes": node_ids})
            elsets.setdefault(current_elset_name, set()).add(eid)
        elif section == "ELSET":
            elsets[current_elset_name].update(int(v) for v in parts)
        i += 1

    return {"nodes": nodes, "elements": elements, "elsets": elsets}


def _classify_surfaces(surfaces, tol=1e-6):
    skin, spar, rib = [], [], []
    for _, tag in surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        z_collapsed = (zmax - zmin) < tol
        x_collapsed = (xmax - xmin) < tol
        if z_collapsed:
            rib.append(tag)
        elif x_collapsed:
            spar.append(tag)
        else:
            skin.append(tag)
    return skin, spar, rib


def _rewrite_as_shell_elements(inp_path):
    """
    gmsh's Abaqus writer has no shell-element option: rewrite CPS3/CPS4
    (2D continuum) element headers to S3/S4 (shell), and drop the T3D2
    boundary-curve element blocks entirely -- they aren't part of a
    shell mesh and CalculiX has no use for them here.
    """
    with open(inp_path) as f:
        lines = f.readlines()

    out = []
    skip_block = False
    for line in lines:
        if line.startswith("*ELEMENT"):
            if "type=T3D2" in line:
                skip_block = True
                continue
            skip_block = False
            line = line.replace("type=CPS4", "type=S4").replace("type=CPS3", "type=S3")
            out.append(line)
            continue
        if line.startswith("*") and not line.startswith("*ELEMENT"):
            skip_block = False
        if skip_block:
            continue
        out.append(line)

    with open(inp_path, "w") as f:
        f.writelines(out)


def generate_structural_mesh(brep_path, name, output_dir, mesh_size=0.05):
    """
    Validates inputs, then runs the actual gmsh work in an isolated
    subprocess (see pipeline/_gmsh_isolation.py) -- see Stage 5's
    docstring for why isolation, not call reordering, is required here.
    """
    if not os.path.exists(brep_path):
        raise FileNotFoundError(f"Stage 5 geometry not found: {brep_path}")
    if mesh_size <= 0.0:
        raise ValueError(f"mesh_size must be positive, got {mesh_size}")

    os.makedirs(output_dir, exist_ok=True)

    result = run_isolated(
        _generate_structural_mesh_worker, brep_path, name, output_dir, mesh_size,
    )
    _rewrite_as_shell_elements(result["inp_path"])
    return result


def _generate_structural_mesh_worker(brep_path, name, output_dir, mesh_size):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(f"{name}_stage6")
        occ = gmsh.model.occ
        occ.importShapes(brep_path)
        occ.synchronize()

        surfaces = gmsh.model.getEntities(dim=2)
        occ.fragment(surfaces, surfaces)
        occ.synchronize()
        surfaces = gmsh.model.getEntities(dim=2)

        skin_tags, spar_tags, rib_tags = _classify_surfaces(surfaces)
        if skin_tags:
            gmsh.model.addPhysicalGroup(2, skin_tags, name="skin")
        if spar_tags:
            gmsh.model.addPhysicalGroup(2, spar_tags, name="spar")
        if rib_tags:
            gmsh.model.addPhysicalGroup(2, rib_tags, name="rib")

        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.2)
        gmsh.model.mesh.generate(2)

        inp_path = os.path.join(output_dir, f"{name}_structural_mesh.inp")
        gmsh.write(inp_path)
    finally:
        gmsh.finalize()

    return {
        "inp_path": inp_path,
        "mesh_size": mesh_size,
        "n_skin_surfaces": len(skin_tags),
        "n_spar_surfaces": len(spar_tags),
        "n_rib_surfaces": len(rib_tags),
    }
