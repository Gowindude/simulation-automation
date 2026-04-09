# ADE Pipeline — Status

Last updated: 2026-04-09

## Agent Pipeline

| Stage | Agent | Status | Notes |
|-------|-------|--------|-------|
| 1 | Librarian (Geometry) | Working | `GeometryAgent` + `CADBuilderAgent` produce `naca001234_domain.step` |
| 2 | Mesh Agent | Debugging | gmsh meshing works; Fluent ASCII MSH converter written; hex formatting fix applied; awaiting confirmed Fluent load |
| 3 | Fluidist (CFD) | Not started | `FluidistAgent` class exists in `cfd_tool.py`; `run_from_mesh()` method not yet implemented |
| 4 | Structuralist (FEA) | Not built | `physics_cores/ansys_mech/` placeholder only |
| 5 | Surrogate (PINN) | Not built | DeepXDE/PyTorch; depends on CFD + FEA output data |
| 6 | Troubleshooter | Not built | Log monitor + LLM error interpreter |
| 7 | Lead (Orchestrator) | Not built | LangGraph state machine; drives the design loop |

## Known Blockers

- `run_cfd_test.py:130` — `tui.file.read_mesh` call is broken (method does not exist); needs to be `settings.file.read_case(file_name=...)`. Not yet committed.
- Fluent load of `naca001234_2d_fluent.msh` not yet manually verified in GUI. Until confirmed, the MSH converter output may still have format issues.

## Output Files (Confirmed on Disk)

- `data/geometry/naca001234_domain.step` — C-domain STEP, confirmed valid (gmsh loads it)
- `data/mesh/naca001234_2d.msh` — gmsh intermediate MSH
- `data/mesh/naca001234_2d_fluent.msh` — Fluent ASCII MSH (729KB); hex indices applied; not yet Fluent-verified

## Next Actions

1. Open Fluent GUI manually, load `data/mesh/naca001234_2d_fluent.msh`, confirm zones appear
2. Fix `run_cfd_test.py:130` and commit
3. Implement `FluidistAgent.run_from_mesh()` in `cfd_tool.py`
4. Run full pipeline, confirm `data/results/pressure_dist.csv` is produced
