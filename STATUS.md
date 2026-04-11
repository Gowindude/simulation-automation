# ADE Pipeline — Status

Last updated: 2026-04-09

## Agent Pipeline

| Stage | Agent | Status | Notes |
|-------|-------|--------|-------|
| 1 | Librarian (Geometry) | Working | `GeometryAgent` + `CADBuilderAgent` produce `naca001234_domain.step` |
| 2 | Mesh Agent | Debugging | gmsh meshing works; two converter bugs fixed (hex indices, inline `(`); awaiting confirmed Fluent load |
| 3 | Fluidist (CFD) | Implemented; unverified | `FluidistAgent.run_from_mesh()` complete; full BC set (inlet/outlet/symmetry); PyFluent 0.20+ field data API; post-solve divergence check; blocked on Fluent MSH load |
| 4 | Structuralist (FEA) | Not built | `physics_cores/ansys_mech/` placeholder only |
| 5 | Surrogate (PINN) | Not built | DeepXDE/PyTorch; depends on CFD + FEA output data |
| 6 | Troubleshooter | Not built | Log monitor + LLM error interpreter |
| 7 | Lead (Orchestrator) | Not built | LangGraph state machine; drives the design loop |

## Known Blockers

- Fluent load of `naca001234_2d_fluent.msh` not yet manually verified in GUI. Regenerate the file after the inline-`(` fix before testing.

## MSH Converter Bug History

Two bugs were fixed in `MeshAgent._convert_to_fluent_msh()`:

1. **Decimal integers** — all counts, indices, and face data values (n0, n1, cr, cl) must be hex. Decimal caused "unable to read coordinates of node N" parse overflow.
2. **Data block `(` on wrong line** — Fluent's parser requires `(section_header)(\n`, not `(section_header)\n(\n`. The bare `(` on its own line caused "Build Grid: Aborted due to critical error" + SIGSEGV.

## Output Files (Confirmed on Disk)

- `data/geometry/naca001234_domain.step` — C-domain STEP, confirmed valid (gmsh loads it)
- `data/mesh/naca001234_2d.msh` — gmsh intermediate MSH
- `data/mesh/naca001234_2d_fluent.msh` — Fluent ASCII MSH; regenerate after latest fix before testing

## Next Actions

1. Delete stale `data/mesh/naca001234_2d_fluent.msh`, regenerate with `python agents/mesh_agent.py --step data/geometry/naca001234_domain.step --name naca001234`
2. Open Fluent GUI in **2D Double Precision** mode, load the new MSH via File > Read > Mesh, confirm zones appear (inlet, outlet, airfoil, symmetry_top, symmetry_bottom)
3. If MSH loads cleanly: run `python run_cfd_test.py`, confirm `data/results/pressure_dist.csv` is produced
4. If MSH load fails: debug `MeshAgent._convert_to_fluent_msh()` — check zone type codes and face-cell adjacency orientation
