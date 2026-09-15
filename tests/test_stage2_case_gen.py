"""
Verification tests for Stage 2 -- CFD case generation.

Contract (from .claude/airfoil_pipeline_build_spec.md, plus the follow-up
decision on the spec's "Reynolds number (or fixed freestream speed)"
ambiguity -- see below):

  Input:  Stage 1 mesh case dir (constant/polyMesh already converted) +
          one AoA (deg) + Reynolds number
  Output: a new OpenFOAM case dir containing:
            - 0/{U,p,nuTilda,nut}   freestream fields rotated per AoA
                                    (rotate the vector, not the geometry --
                                    one mesh serves the whole AoA sweep).
                                    All four, not just U -- Stage 3's own
                                    convergence logic (spec line 99) tracks
                                    initial residuals for Ux, Uy, p AND
                                    nuTilda, and Spalart-Allmaras needs
                                    nut too, so a case missing any of
                                    these is not a partially-correct
                                    Stage-2 output, it's one Stage 3 can't
                                    run at all.
            - constant/polyMesh     REUSED from the Stage 1 mesh case, not
                                     regenerated
            - constant/physicalProperties   (nu) -- OpenFOAM 12's
                                     Foundation-line name; NOT
                                     `transportProperties`, verified
                                     against the bundled
                                     tutorials/incompressibleFluid/airFoil2D
                                     case, which is the spec's own named
                                     baseline ("Matches tutorial baseline
                                     already validated (airFoil2D)")
            - constant/momentumTransport   Spalart-Allmaras (locked
                                     decision, see spec's "Locked
                                     decisions" table) -- also lives under
                                     constant/, not system/, in OF12; same
                                     verification source as above
            - system/controlDict

  Reynolds vs. fixed speed: the spec leaves this open ("Reynolds number
  (or fixed freestream speed)"). Locked here as Reynolds number, because
  (a) Stage 4's XFOIL cross-check (Must-Pass Gate #2) is Re-driven, and
  (b) a fixed speed would put every airfoil in the eventual multi-airfoil
  sweep at a different, uncontrolled Re, breaking cross-airfoil
  comparability. Since Stage 0 normalizes every airfoil to unit chord,
  U_inf = Reynolds * nu is a direct closed-form conversion -- nothing is
  lost by taking Re as the input and deriving speed internally.

These tests parse the *actual* generated OpenFOAM dictionaries via
`foamDictionary` (inside WSL) rather than grepping raw text, the same
approach test_stage1_mesh.py uses for checkMesh output -- so a
dictionary that's syntactically wrong (which foamDictionary would refuse
to parse) fails loudly instead of passing a naive string-match test.

All dictionary paths and `foamDictionary` invocation syntax below (the
`/`-separated `-entry` scope, e.g. `boundaryField/farfield/type`, NOT
`.`-separated) were verified empirically against the real bundled
`airFoil2D` tutorial before writing these tests, not assumed from a
generic/older OpenFOAM layout -- the `.`-separated form and the classic
`constant/transportProperties` + `system/turbulenceProperties` paths both
fail outright on this OpenFOAM 12 install.
"""

import hashlib
import os
import re

import numpy as np
import pytest

from pipeline.stage0_geometry_loader import load_airfoil
from pipeline.stage1_mesh import run_stage1
from pipeline.stage2_case_gen import generate_case

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# Full spec-locked sweep plus explicit negative-AoA and zero-AoA edge
# cases -- the spec's own sweep (-2, 2, 6, 10, 14) already includes one
# negative value, but we also probe a larger-magnitude negative AoA
# (-10) to stress the atan2 sign handling further from zero.
AOA_SWEEP_DEG = [-10.0, -2.0, 0.0, 2.0, 6.0, 10.0, 14.0]

REYNOLDS_VALUES = [2e5, 5e5, 1e6]

NU_DEFAULT = 1.5e-5


def _to_wsl_path(win_path: str) -> str:
    win_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(win_path)
    return f"/mnt/{drive.rstrip(':').lower()}{rest.replace(chr(92), '/')}"


def _run_wsl(bash_cmd: str, timeout: int = 60):
    import subprocess

    full_cmd = f"source /opt/openfoam12/etc/bashrc && {bash_cmd}"
    return subprocess.run(
        ["wsl.exe", "--", "bash", "-lc", full_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _foam_dict_value(case_dir: str, rel_file: str, entry: str) -> str:
    """Read one entry from an OpenFOAM dictionary via foamDictionary.

    `entry` uses '/' as the scope separator, e.g.
    'boundaryField/farfield/freestreamValue' -- OpenFOAM 12's
    foamDictionary rejects '.'-separated scope paths outright (verified:
    it raises "Cannot find entry boundaryField.inlet.type" against the
    real airFoil2D tutorial).

    `-writePrecision 15` is required: without it, `-value` echoes numbers
    at OpenFOAM's default 6-significant-figure stream precision
    regardless of how many digits are actually in the file (verified with
    a literal 16-sig-fig value: default read back as '7.38606', only
    `-writePrecision 15` recovered '7.38605814759156'). Omitting this
    doesn't affect Stage 2's output correctness at all -- it's purely a
    read-back artifact of this test's verification tool -- but it made
    the direction/magnitude checks below fail on rounding noise that has
    nothing to do with the pipeline.
    """
    case_wsl = _to_wsl_path(case_dir)
    result = _run_wsl(
        f'foamDictionary -writePrecision 15 -case "{case_wsl}" -entry "{entry}" -value "{rel_file}"'
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"foamDictionary failed reading {entry} from {rel_file}:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def _parse_foam_vector(raw: str) -> np.ndarray:
    """Parse OpenFOAM's '(x y z)' vector literal into a numpy array."""
    match = re.search(
        r"\(\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*\)", raw
    )
    assert match, f"could not parse vector from: {raw!r}"
    return np.array([float(g) for g in match.groups()])


def _parse_dimensioned_scalar(raw: str) -> float:
    """Parse OpenFOAM's '[dims] value' or bare 'value' scalar form.

    physicalProperties' nu is written as a dimensioned quantity, e.g.
    '[ 0 2 -1 0 0 0 0 ] 1e-05' (verified against the airFoil2D tutorial)
    -- the numeric value is whatever trails the closing ']', not the
    first number in the string (which would just be a dimension
    exponent).
    """
    after_bracket = raw.split("]")[-1] if "]" in raw else raw
    match = re.search(r"[-\d.eE]+", after_bracket)
    assert match, f"could not parse scalar from: {raw!r}"
    return float(match.group())


def _list_boundary_patches(case_dir: str) -> list:
    """Extract patch names from constant/polyMesh/boundary (same
    top-level-brace-name parsing style as test_stage1_mesh.py's
    `_patch_type`, kept independent of foamDictionary since `boundary` is
    a bare list, not a keyed dictionary -- `-keywords` doesn't expose the
    patch names for it, confirmed empirically)."""
    with open(os.path.join(case_dir, "constant", "polyMesh", "boundary")) as f:
        text = f.read()
    # Require leading indentation on both the name and its brace line --
    # excludes the FoamFile header block, whose 'FoamFile'/'{' sit at
    # column 0 with no indentation (verified against a real generated
    # boundary file: patch entries are 4-space indented, the header isn't).
    return re.findall(r"^[ \t]+(\w+)\n[ \t]+\{", text, re.MULTILINE)


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture(scope="module")
def mesh_case_dir(tmp_path_factory):
    """One Stage 1 mesh, reused as the input to every Stage 2 case below."""
    out_dir = tmp_path_factory.mktemp("stage2_mesh")
    coords = load_airfoil(os.path.join(FIXTURES, "naca0012.dat"))
    result = run_stage1(coords, "naca0012", str(out_dir))
    assert result["check"]["passed"]
    return result["case_dir"]


@pytest.fixture(scope="module")
def sweep_cases(mesh_case_dir, tmp_path_factory):
    """Generate one Stage 2 case per AoA in the sweep, from the same mesh."""
    out_dir = tmp_path_factory.mktemp("stage2_cases")
    reynolds = 5e5
    cases = {}
    for aoa in AOA_SWEEP_DEG:
        cases[aoa] = generate_case(
            mesh_case_dir=mesh_case_dir,
            name="naca0012",
            aoa_deg=aoa,
            reynolds=reynolds,
            output_dir=str(out_dir),
            nu=NU_DEFAULT,
        )
    return cases


# --- 1. Freestream direction matches AoA (sign + magnitude) -----------------


@pytest.mark.parametrize("aoa", AOA_SWEEP_DEG)
def test_freestream_direction_matches_aoa(sweep_cases, aoa):
    case_dir = sweep_cases[aoa]["case_dir"]
    U_inf = sweep_cases[aoa]["U_inf"]

    raw = _foam_dict_value(case_dir, "0/U", "boundaryField/farfield/freestreamValue")
    vec = _parse_foam_vector(raw)

    theta = np.radians(aoa)
    expected = np.array([U_inf * np.cos(theta), U_inf * np.sin(theta), 0.0])

    assert vec == pytest.approx(expected, abs=1e-6)
    # Direction check independent of the magnitude, so a bug that gets the
    # magnitude right but the sign/rotation wrong can't hide behind it.
    assert np.arctan2(vec[1], vec[0]) == pytest.approx(theta, abs=1e-9)
    assert np.linalg.norm(vec) == pytest.approx(U_inf, abs=1e-9)


def test_zero_aoa_is_purely_axial(sweep_cases):
    raw = _foam_dict_value(
        sweep_cases[0.0]["case_dir"], "0/U", "boundaryField/farfield/freestreamValue"
    )
    vec = _parse_foam_vector(raw)
    assert vec[1] == pytest.approx(0.0, abs=1e-9)
    assert vec[0] > 0


# --- 1b. Every patch has a boundaryField entry in EVERY field Stage 3 needs -


# Spalart-Allmaras needs nuTilda + nut; the solver needs p. A case missing
# any of these can't run at all -- distinct from (and cheaper to catch
# than) a case that runs but converges wrong.
REQUIRED_FIELD_FILES = ["U", "p", "nuTilda", "nut"]


@pytest.mark.parametrize("aoa", AOA_SWEEP_DEG)
def test_every_boundary_patch_has_entry_in_every_required_field(sweep_cases, aoa):
    case_dir = sweep_cases[aoa]["case_dir"]
    patches = _list_boundary_patches(case_dir)
    assert set(patches) == {"farfield", "airfoil", "front", "back"}, (
        "test's own patch-name assumption is stale -- Stage 1's mesh "
        f"boundary patches changed: {patches}"
    )

    for field in REQUIRED_FIELD_FILES:
        field_path = f"0/{field}"
        assert os.path.exists(os.path.join(case_dir, field_path)), field_path
        for patch in patches:
            # Raises RuntimeError (via _foam_dict_value) if the patch has
            # no boundaryField entry at all -- foamDictionary itself
            # reports "Cannot find entry" for a missing key, so this is
            # the field/patch pair actually going unfound, not a
            # false-pass on absence.
            patch_type = _foam_dict_value(case_dir, field_path, f"boundaryField/{patch}/type")
            assert patch_type, f"{field_path}: patch '{patch}' has an empty type"


# --- 2. Mesh is reused, never regenerated for a new AoA ----------------------


def test_mesh_reused_not_regenerated(mesh_case_dir, sweep_cases):
    source_points = os.path.join(mesh_case_dir, "constant", "polyMesh", "points")
    source_hash = _sha256_file(source_points)

    hashes = set()
    for aoa, case in sweep_cases.items():
        case_points = os.path.join(case["case_dir"], "constant", "polyMesh", "points")
        assert os.path.exists(case_points)
        hashes.add(_sha256_file(case_points))

    # Every AoA case's mesh is byte-identical to the source AND to each
    # other -- if Stage 2 re-meshed per AoA (even correctly), this would
    # fail, since re-triangulation is not guaranteed to be bit-identical
    # even for the same geometry.
    assert hashes == {source_hash}


# --- 3. Turbulence model is Spalart-Allmaras, per the locked decision -------


@pytest.mark.parametrize("aoa", AOA_SWEEP_DEG)
def test_momentum_transport_is_spalart_allmaras(sweep_cases, aoa):
    case_dir = sweep_cases[aoa]["case_dir"]
    sim_type = _foam_dict_value(case_dir, "constant/momentumTransport", "simulationType")
    assert sim_type == "RAS"

    ras_model = _foam_dict_value(case_dir, "constant/momentumTransport", "RAS/model")
    assert ras_model == "SpalartAllmaras"


# --- 3b. controlDict is actually runnable, not just present ----------------


@pytest.mark.parametrize("aoa", AOA_SWEEP_DEG)
def test_control_dict_specifies_solver(sweep_cases, aoa):
    """
    Regression guard: an earlier version of this module wrote a
    syntactically valid controlDict (foamDictionary parses it fine, and
    'a file exists' would pass) that was missing the `solver` entry
    foamRun requires -- discovered only by actually running `foamRun`
    against a real generated case, which failed with "FOAM FATAL ERROR:
    solver not specified in the controlDict". Exactly the "clean,
    non-crashing, plausible-looking output that is actually wrong" case
    the spec's Must-Pass-Gates section is about, just one level up (a
    config file, not a physics result) -- so it gets a permanent test
    rather than being left as a one-off manual fix.
    """
    case_dir = sweep_cases[aoa]["case_dir"]
    application = _foam_dict_value(case_dir, "system/controlDict", "application")
    assert application == "foamRun"
    solver = _foam_dict_value(case_dir, "system/controlDict", "solver")
    assert solver == "incompressibleFluid"


# --- 4. Freestream speed is derived correctly from Reynolds number ---------


@pytest.mark.parametrize("reynolds", REYNOLDS_VALUES)
def test_freestream_speed_matches_reynolds(mesh_case_dir, tmp_path, reynolds):
    case = generate_case(
        mesh_case_dir=mesh_case_dir,
        name="naca0012",
        aoa_deg=4.0,
        reynolds=reynolds,
        output_dir=str(tmp_path),
        nu=NU_DEFAULT,
    )

    # Read nu back from the actual generated dictionary rather than
    # asserting against the NU_DEFAULT literal directly, so this test
    # checks internal self-consistency (U_inf matches whatever nu Stage 2
    # actually wrote) rather than merely echoing the input.
    nu_raw = _foam_dict_value(case["case_dir"], "constant/physicalProperties", "nu")
    nu_written = _parse_dimensioned_scalar(nu_raw)

    # Unit-chord geometry (Stage 0 contract) => Re = U_inf * c / nu = U_inf / nu.
    expected_U_inf = reynolds * nu_written
    assert case["U_inf"] == pytest.approx(expected_U_inf, rel=1e-9)

    raw = _foam_dict_value(
        case["case_dir"], "0/U", "boundaryField/farfield/freestreamValue"
    )
    vec = _parse_foam_vector(raw)
    assert np.linalg.norm(vec) == pytest.approx(expected_U_inf, rel=1e-6)


# --- Error handling ----------------------------------------------------------


def test_missing_mesh_case_dir_raises(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        generate_case(
            mesh_case_dir=str(tmp_path / "does_not_exist"),
            name="naca0012",
            aoa_deg=2.0,
            reynolds=5e5,
            output_dir=str(tmp_path / "out"),
        )


def test_nonpositive_reynolds_raises(mesh_case_dir, tmp_path):
    with pytest.raises(ValueError):
        generate_case(
            mesh_case_dir=mesh_case_dir,
            name="naca0012",
            aoa_deg=2.0,
            reynolds=0.0,
            output_dir=str(tmp_path),
        )
