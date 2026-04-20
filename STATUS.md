# ADE Pipeline — Status

Last updated: 2026-04-10

## Agent Pipeline

| Stage | Agent | Status | Notes |
|-------|-------|--------|-------|
| 1 | Librarian (Geometry) | Working | `GeometryAgent` + `CADBuilderAgent` produce `naca001234_domain.step` |
| 2 | Mesh Agent | Debugging | gmsh works; 3 converter bugs fixed (hex, inline `(`, BC codes); cr/cl orientation under test |
| 3 | Fluidist (CFD) | Implemented; unverified | `run_from_mesh()` complete; full BC set; PyFluent 0.20+ field API; divergence check; blocked on mesh load |
| 4 | Structuralist (FEA) | Not built | `physics_cores/ansys_mech/` placeholder only |
| 5 | Surrogate (PINN) | Not built | DeepXDE/PyTorch; depends on CFD + FEA output |
| 6 | Troubleshooter | Not built | Log monitor + LLM error interpreter |
| 7 | Lead (Orchestrator) | Not built | LangGraph state machine |

## MSH Converter Bug History

Three bugs fixed in `MeshAgent._convert_to_fluent_msh()`:

1. **Decimal integers** — all counts, indices, face data `(n0 n1 cr cl)` must be hex. Caused "unable to read coordinates of node N" parse overflow.
2. **Data block `(` on wrong line** — must be `(section_header)(\n` not `(section_header)\n(\n`. Caused "Build Grid: Aborted" + SIGSEGV.
3. **Wrong BC type codes** — `0x9` is pressure-far-field (3D only, caused SIGSEGV in 2D); `0x14` is mass-flow-inlet. Correct codes: `0xa`=velocity-inlet, `0x5`=pressure-outlet.

## Current Blocker

Mesh loads (zones recognised, no crash) but Fluent GUI reported "11528 cells with non-positive volume" with cr=c1 orientation.

Mesh has been regenerated with cr=c0 (original orientation). Load in Fluent GUI (2D DP) to verify volumes are positive. If volumes are positive, run `python run_cfd_test.py`.

## Output Files (Confirmed on Disk)

- `data/geometry/naca001234_domain.step` — C-domain STEP, confirmed valid
- `data/mesh/naca001234_2d.msh` — gmsh intermediate MSH
- `data/mesh/naca001234_2d_fluent.msh` — Fluent ASCII MSH, regenerated 2026-04-10 with cr=c0

## Next Actions

1. Load `data/mesh/naca001234_2d_fluent.msh` in Fluent GUI (2D Double Precision, File > Read > Mesh)
2. Confirm zones appear and volumes are positive (no "non-positive volume" warning)
3. If clean: `python run_cfd_test.py`
4. If still non-positive: the cr/cl convention needs further investigation — consider trying pure tri mesh (no BL quads) to isolate the issue
