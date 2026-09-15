"""
Stage 2 -- CFD case generation.

Input:  a Stage 1 mesh case dir (contains `constant/polyMesh`, already
        converted via gmshToFoam) + one AoA (degrees) + Reynolds number.
Output: a new OpenFOAM case dir containing the Stage 1 mesh (reused, not
        regenerated) plus `0/{U,p,nuTilda,nut}`, `constant/physicalProperties`,
        `constant/momentumTransport`, and `system/controlDict`.

Reynolds number (not a fixed freestream speed) is the canonical input:
Stage 4's XFOIL cross-check is Re-driven, and a fixed speed would put
every airfoil in a multi-airfoil sweep at a different, uncontrolled Re.
Since Stage 0 normalizes every airfoil to unit chord, U_inf = Re * nu is
a direct closed-form conversion.

The freestream vector is rotated via atan2, not the geometry -- one mesh
serves the whole AoA sweep (rotating the geometry per AoA would require
re-meshing every case).

Dictionary layout (OpenFOAM 12, Foundation line) verified against the
bundled `tutorials/incompressibleFluid/airFoil2D` tutorial, the spec's own
named baseline: `momentumTransport` and `physicalProperties` live under
`constant/`, not `system/`/`transportProperties` (the classic OpenFOAM.com
layout).
"""

import os
import shutil

import numpy as np

_BOUNDARY_PATCHES = ("farfield", "airfoil", "front", "back")


def _foam_header(obj_class: str, obj_name: str, location: str) -> str:
    return f"""FoamFile
{{
    format      ascii;
    class       {obj_class};
    location    "{location}";
    object      {obj_name};
}}
"""


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _vector_field_file(name: str, internal, farfield_value, airfoil_bc: str) -> str:
    ix, iy, iz = internal
    fx, fy, fz = farfield_value
    return _foam_header("volVectorField", name, "0") + f"""
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({ix:.10g} {iy:.10g} {iz:.10g});

boundaryField
{{
    farfield
    {{
        type            freestreamVelocity;
        freestreamValue uniform ({fx:.10g} {fy:.10g} {fz:.10g});
    }}

    airfoil
    {{
        type            {airfoil_bc};
    }}

    front
    {{
        type            empty;
    }}

    back
    {{
        type            empty;
    }}
}}
"""


def _scalar_field_file(
    name: str,
    dimensions: str,
    internal: float,
    farfield_type: str,
    farfield_value,
    airfoil_type: str,
    airfoil_value=None,
) -> str:
    airfoil_block = f"        type            {airfoil_type};\n"
    if airfoil_value is not None:
        airfoil_block += f"        value           uniform {airfoil_value:.10g};\n"

    return _foam_header("volScalarField", name, "0") + f"""
dimensions      {dimensions};

internalField   uniform {internal:.10g};

boundaryField
{{
    farfield
    {{
        type            {farfield_type};
        freestreamValue uniform {farfield_value:.10g};
    }}

    airfoil
    {{
{airfoil_block}    }}

    front
    {{
        type            empty;
    }}

    back
    {{
        type            empty;
    }}
}}
"""


def _write_field_files(case_dir: str, U_vec, nu: float) -> None:
    Ux, Uy = U_vec

    _write(
        os.path.join(case_dir, "0", "U"),
        _vector_field_file("U", (Ux, Uy, 0.0), (Ux, Uy, 0.0), "noSlip"),
    )
    _write(
        os.path.join(case_dir, "0", "p"),
        _scalar_field_file(
            "p", "[0 2 -2 0 0 0 0]", 0.0, "freestreamPressure", 0.0, "zeroGradient"
        ),
    )

    # Spalart-Allmaras freestream nuTilda: matches the ratio used by the
    # spec's own validated baseline (tutorials/incompressibleFluid/airFoil2D:
    # nu=1e-5, nuTilda=nut=0.14 -> ratio ~14000x nu), not an arbitrary
    # small multiple -- an under-scaled freestream turbulence value was
    # confirmed (empirically, via a real foamRun) to leave nuTilda
    # oscillating in an exact limit cycle and never converging.
    nu_tilda_inf = 14000.0 * nu
    _write(
        os.path.join(case_dir, "0", "nuTilda"),
        _scalar_field_file(
            "nuTilda", "[0 2 -1 0 0 0 0]", nu_tilda_inf, "freestream", nu_tilda_inf,
            "fixedValue", airfoil_value=0.0,
        ),
    )
    _write(
        os.path.join(case_dir, "0", "nut"),
        _scalar_field_file(
            "nut", "[0 2 -1 0 0 0 0]", nu_tilda_inf, "freestream", nu_tilda_inf,
            "nutUSpaldingWallFunction", airfoil_value=0.0,
        ),
    )


def _write_physical_properties(case_dir: str, nu: float) -> None:
    content = _foam_header("dictionary", "physicalProperties", "constant") + f"""
viscosityModel  constant;

rho             [1 -3 0 0 0 0 0] 1;

nu              [0 2 -1 0 0 0 0] {nu:.10g};
"""
    _write(os.path.join(case_dir, "constant", "physicalProperties"), content)


def _write_momentum_transport(case_dir: str) -> None:
    content = _foam_header("dictionary", "momentumTransport", "constant") + """
simulationType RAS;

RAS
{
    model           SpalartAllmaras;

    turbulence      on;

    printCoeffs     on;
}
"""
    _write(os.path.join(case_dir, "constant", "momentumTransport"), content)


def _write_control_dict(case_dir: str) -> None:
    content = _foam_header("dictionary", "controlDict", "system") + """
application     foamRun;

solver          incompressibleFluid;

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         1000;

deltaT          1;

writeControl    timeStep;
writeInterval   100;

purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;
"""
    _write(os.path.join(case_dir, "system", "controlDict"), content)


def _copy_mesh(mesh_case_dir: str, case_dir: str) -> None:
    src = os.path.join(mesh_case_dir, "constant", "polyMesh")
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"No constant/polyMesh found under mesh case dir: {mesh_case_dir}"
        )
    dst = os.path.join(case_dir, "constant", "polyMesh")
    shutil.copytree(src, dst)


def generate_case(
    mesh_case_dir: str,
    name: str,
    aoa_deg: float,
    reynolds: float,
    output_dir: str,
    nu: float = 1.5e-5,
) -> dict:
    """
    Build one AoA case from a Stage 1 mesh: reuse the mesh, write the
    rotated freestream fields, transport properties, turbulence model,
    and control dict.

    Args:
        mesh_case_dir: Stage 1 output case dir (has constant/polyMesh).
        name: Base name for the case directory.
        aoa_deg: Angle of attack in degrees. The freestream vector is
            rotated by this angle via atan2; the geometry/mesh is not
            touched.
        reynolds: Reynolds number (canonical input -- see module
            docstring). Must be > 0.
        output_dir: Directory under which the new case dir is created.
        nu: Kinematic viscosity used to derive U_inf = reynolds * nu
            (chord = 1, per Stage 0's unit-chord contract).

    Returns:
        {"case_dir": str, "U_inf": float}

    Raises:
        FileNotFoundError: if `mesh_case_dir` has no constant/polyMesh.
        ValueError: if `reynolds` is not strictly positive.
    """
    if not os.path.isdir(mesh_case_dir):
        raise FileNotFoundError(f"No such mesh case dir: {mesh_case_dir}")
    if reynolds <= 0:
        raise ValueError(f"Reynolds number must be > 0, got {reynolds}")

    U_inf = reynolds * nu
    theta = np.radians(aoa_deg)
    U_vec = (U_inf * np.cos(theta), U_inf * np.sin(theta))

    case_dir = os.path.abspath(os.path.join(output_dir, f"{name}_aoa{aoa_deg:+.1f}"))
    os.makedirs(case_dir, exist_ok=True)

    _copy_mesh(mesh_case_dir, case_dir)
    _write_field_files(case_dir, U_vec, nu)
    _write_physical_properties(case_dir, nu)
    _write_momentum_transport(case_dir)
    _write_control_dict(case_dir)

    return {"case_dir": case_dir, "U_inf": U_inf}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: generate an OpenFOAM case for one AoA.")
    parser.add_argument("--mesh-case", required=True, help="Stage 1 mesh case dir.")
    parser.add_argument("--name", default="airfoil")
    parser.add_argument("--aoa", type=float, required=True, help="Angle of attack, degrees.")
    parser.add_argument("--reynolds", type=float, required=True)
    parser.add_argument("--output-dir", default="data/cases")
    parser.add_argument("--nu", type=float, default=1.5e-5)
    args = parser.parse_args()

    result = generate_case(
        args.mesh_case, args.name, args.aoa, args.reynolds, args.output_dir, nu=args.nu
    )
    print(f"Case: {result['case_dir']}")
    print(f"U_inf: {result['U_inf']:.6g}")
