"""
Verification tests for Stage 6 -- Structural meshing.

Contract (from .claude/airfoil_pipeline_build_spec.md, Stage 6, verbatim):
    Input: Stage 5 geometry
    Output: shell element mesh (gmsh or CalculiX's `cgx`) suitable for
            CalculiX input deck

The spec leaves the details open -- Stages 5-7 are flagged in the spec's
own "Build order" section as "new territory, build and inspect the mesh
visually before running CalculiX blind" -- so the concrete engineering
decisions below fill that gap, each one a direct consequence of the two
words that ARE locked: "shell element" and "CalculiX input deck".

Design decisions locked before writing this suite:
  - Tool: gmsh (same as Stages 1 and 5), output as an Abaqus-dialect
    `.inp` file -- gmsh's native CalculiX-compatible mesh export format.
  - "suitable for CalculiX input deck" requires true shell elements
    (CalculiX types S3/S4), not gmsh's raw 2D output. Confirmed
    empirically: gmsh's own `.inp` writer types 2D surface elements as
    `CPS3`/`CPS4` (2D continuum plane-stress), never a shell type -- gmsh
    has no option to request shell typing directly, since it doesn't know
    the target analysis. Stage 6 must rewrite these element-type
    declarations after gmsh writes the file, or the output is not
    actually usable by CalculiX for a shell analysis (a *SHELL SECTION
    card in CalculiX requires elements literally typed S3/S4).
  - Boundary curve elements (`T3D2`, one per meshed edge) that gmsh also
    writes are dropped from the final `.inp` -- they aren't part of a
    shell element mesh and would corrupt any "all elements are shell
    type" check if left in.
  - Conformal fragmentation before meshing (`occ.fragment(surfaces,
    surfaces)`, i.e. self-fragment the whole imported assembly): Stage 5
    deliberately left interior ribs/spars geometrically coincident with,
    but NOT topologically fused to, the skin (fusing them there would
    have split Stage 5's own full-span skin panels, breaking ITS
    contract). Stage 6 is explicitly where that gets resolved -- a shell
    mesh of a spar/rib structure is only physically meaningful for FEA
    if ribs, spars, and skin share actual mesh NODES at every junction
    (root, tip, AND every interior rib/spar crossing), not just
    coincident coordinates, since CalculiX transfers load through shared
    nodes. Verified empirically on a standalone rib+spar test case before
    writing this suite: self-fragmenting a compound of coincident
    surfaces DOES split them at their true intersection and produces
    genuinely shared topology at the cut.
  - Physical groups ("skin", "spar", "rib") assigned by post-fragment
    bounding-box classification (same z-collapsed / x-collapsed / neither
    rule Stage 5's own test suite uses) -- confirmed empirically that
    gmsh's Abaqus writer emits a `*ELSET,ELSET=<physical-group-name>`
    block per named physical group, giving CalculiX addressable regions
    for later material/section assignment (Stage 8's job, not this
    stage's).
  - Mesh size is a TUNABLE PARAMETER, not a locked decision like Stage
    5's span/spar/rib defaults -- there is no single "correct" element
    size the spec dictates, and efficiency-vs-accuracy is an open
    calibration question for later stages (analogous to Stage 1's
    bl_size, which stayed a per-run override rather than one true
    default). Tests below verify the MECHANISM (finer size -> more
    elements; a sane bound at the starting default) rather than pinning
    one exact element count.

Verification approach: parse the actual `.inp` file gmsh writes (a
lightweight regex-based parser below, matching gmsh's exact Abaqus
dialect confirmed empirically) rather than trust any count the
implementation reports about itself -- this tests the on-disk artifact
Stage 8 will actually load, the same principle Stage 5's suite used for
its STEP/.brep verification.
"""

import os

import numpy as np
import pytest

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage5_structural_geometry import generate_structural_geometry
from pipeline.stage6_structural_mesh import generate_structural_mesh, parse_inp

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

SPAN = 3.0
SPAR_LOCATIONS = (0.2, 0.6)
RIB_SPACING = 0.5
DEFAULT_MESH_SIZE = 0.05
COORD_TOL = 1e-6


def fixture_path(name):
    return os.path.join(FIXTURES, name)


def _triangle_area(nodes, tri_node_ids):
    p = [np.array(nodes[n]) for n in tri_node_ids[:3]]
    return 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))


# --- Fixtures ------------------------------------------------------------


@pytest.fixture(scope="module")
def naca0012_coords():
    return load_airfoil(fixture_path("naca0012.dat"))


@pytest.fixture(scope="module")
def stage5_geometry(naca0012_coords, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage6_stage5geom")
    return generate_structural_geometry(
        naca0012_coords, "naca0012", str(out_dir),
        span=SPAN, spar_locations=SPAR_LOCATIONS, rib_spacing=RIB_SPACING,
    )


@pytest.fixture(scope="module")
def default_mesh(stage5_geometry, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage6")
    return generate_structural_mesh(
        stage5_geometry["brep_path"], "naca0012", str(out_dir),
        mesh_size=DEFAULT_MESH_SIZE,
    )


@pytest.fixture(scope="module")
def default_parsed(default_mesh):
    return parse_inp(default_mesh["inp_path"])


# --- 1. Output artifact validity ------------------------------------------


def test_inp_file_exists_and_parses(default_mesh, default_parsed):
    assert os.path.exists(default_mesh["inp_path"])
    assert os.path.getsize(default_mesh["inp_path"]) > 0
    assert len(default_parsed["nodes"]) > 0
    assert len(default_parsed["elements"]) > 0


# --- 2. "shell element mesh" -- true shell typing, not gmsh's raw output --


def test_all_elements_are_shell_type(default_parsed):
    types_seen = {el["type"] for el in default_parsed["elements"]}
    assert types_seen <= {"S3", "S4"}, (
        f"found non-shell element types {types_seen - {'S3', 'S4'}} -- "
        "gmsh's raw .inp writer types 2D surfaces as CPS3/CPS4 (continuum), "
        "which CalculiX cannot use for a *SHELL SECTION; Stage 6 must "
        "rewrite these to S3/S4"
    )


def test_no_boundary_line_elements(default_parsed):
    types_seen = {el["type"] for el in default_parsed["elements"]}
    assert "T3D2" not in types_seen, (
        "boundary curve elements (T3D2) leaked into the final mesh -- "
        "these aren't part of a shell element mesh and should be dropped"
    )


# --- 3. Mesh validity -------------------------------------------------------


def test_no_degenerate_elements(default_parsed):
    for el in default_parsed["elements"]:
        area = _triangle_area(default_parsed["nodes"], el["nodes"])
        assert area > 1e-10, f"degenerate element {el['id']}: area={area}"


# --- 4. The core "did fragmentation actually work" check -------------------


def test_no_duplicate_coincident_nodes(default_parsed):
    """
    A conformal mesh (ribs/spars genuinely fused to the skin via
    occ.fragment before meshing) has exactly one node per physical
    location -- never two distinct node IDs at the same 3D point. This
    is the direct, general proof that the fragment step worked: it
    doesn't assume which surfaces touch which, it just checks that
    nowhere in the whole mesh did two coincident points fail to weld
    into one shared node. If fragmentation had been skipped (or failed),
    every rib/spar-to-skin junction would show up here as a duplicate.
    """
    seen = {}
    duplicates = []
    for nid, (x, y, z) in default_parsed["nodes"].items():
        key = (round(x / COORD_TOL), round(y / COORD_TOL), round(z / COORD_TOL))
        if key in seen:
            duplicates.append((seen[key], nid))
        else:
            seen[key] = nid
    assert not duplicates, (
        f"found {len(duplicates)} pairs of distinct node IDs at coincident "
        f"coordinates (e.g. {duplicates[:5]}) -- fragmentation did not "
        "conformally weld ribs/spars to the skin, so CalculiX would see "
        "these regions as structurally disconnected"
    )


# --- 5. Region coverage for CalculiX section/material assignment -----------


def test_physical_regions_present(default_parsed):
    elsets = default_parsed["elsets"]
    for region in ("skin", "spar", "rib"):
        assert region in elsets, f"missing ELSET '{region}' in .inp output"
        assert len(elsets[region]) > 0, f"ELSET '{region}' has no elements"


def test_physical_regions_partition_all_elements(default_parsed):
    all_ids = {el["id"] for el in default_parsed["elements"]}
    covered = set()
    for region in ("skin", "spar", "rib"):
        covered |= default_parsed["elsets"][region]
    assert covered == all_ids, (
        "skin/spar/rib ELSETs don't cover every element -- some elements "
        "belong to no named region and couldn't get a section assignment"
    )


# --- 6. Mesh size as a tunable knob, not a locked value ---------------------


def test_finer_mesh_size_yields_more_elements(stage5_geometry, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage6_sizes")
    coarse = generate_structural_mesh(
        stage5_geometry["brep_path"], "naca0012_coarse", str(out_dir), mesh_size=0.15,
    )
    fine = generate_structural_mesh(
        stage5_geometry["brep_path"], "naca0012_fine", str(out_dir), mesh_size=0.03,
    )
    n_coarse = len(parse_inp(coarse["inp_path"])["elements"])
    n_fine = len(parse_inp(fine["inp_path"])["elements"])
    assert n_fine > n_coarse, (
        f"finer mesh_size (0.03) produced {n_fine} elements, not more than "
        f"coarse (0.15)'s {n_coarse} -- mesh_size isn't actually controlling "
        "resolution"
    )


def test_default_mesh_size_produces_reasonable_element_count(default_parsed):
    n = len(default_parsed["elements"])
    assert 50 < n < 200_000, (
        f"{n} elements at the default mesh_size ({DEFAULT_MESH_SIZE}) is "
        "outside a sane range -- either grossly under- or over-resolved"
    )


# --- Error handling ----------------------------------------------------------


def test_missing_brep_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_structural_mesh(
            str(tmp_path / "does_not_exist.brep"), "naca0012", str(tmp_path),
        )


def test_nonpositive_mesh_size_raises(stage5_geometry, tmp_path):
    with pytest.raises(ValueError):
        generate_structural_mesh(
            stage5_geometry["brep_path"], "naca0012", str(tmp_path), mesh_size=0.0,
        )
