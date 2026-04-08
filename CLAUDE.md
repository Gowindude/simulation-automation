# ADE Project — CLAUDE.md

## Project Overview

Autonomous Design Evaluator (ADE): a 6-agent pipeline that runs high-fidelity CFD and FEA simulations on NACA airfoil shapes, collects the results as training data, and uses them to train a Physics-Informed Neural Network (PINN). The PINN eventually acts as a fast surrogate model — predicting aerodynamic and structural performance for any airfoil without running a full simulation each time. The lead agent uses the PINN to search the design space and only calls Ansys to verify the best candidates.

The pipeline processes a single airfoil as: `.dat` → cleaned geometry → CAD STEP → gmsh mesh → Fluent ASCII MSH → CFD solve → pressure CSV → (FEA) → PINN training data.

Current status: Agents 1–2 are implemented. Agents 3–6 are not yet built.

---

## 6-Agent Architecture

| # | Name | Role | Status | Entry Point |
|---|------|------|--------|-------------|
| 1 | Librarian | Downloads/generates NACA .dat files from UIUC or naca_gen.py | Working | `data/airfoil_downloader.py` |
| 2 | Fluidist (CFD Lead) | Geometry → mesh → Fluent CFD solve → pressure CSV | In progress | `agents/geometry_agent.py`, `agents/mesh_agent.py`, `physics_cores/ansys_fluent/cfd_tool.py` |
| 3 | Structuralist (FEA Lead) | Maps pressure from Fluidist onto structural model via PyMechanical | Not built | `physics_cores/ansys_mech/` |
| 4 | Surrogate (PINN Trainer) | Trains DeepXDE/PyTorch PINN on CFD+FEA outputs | Not built | TBD |
| 5 | Troubleshooter | Monitors logs, interprets divergence, instructs Fluidist to retry | Not built | TBD |
| 6 | Lead (Orchestrator) | Uses PINN predictions to decide which shapes to verify in Ansys | Not built | TBD |

Agents pass a **Design Dictionary** between each other. Minimum keys in that dict:
```python
{
    "name": "naca001234",
    "dat_path": "data/airfoils/naca001234.dat",
    "step_path": "data/geometry/naca001234_domain.step",
    "mesh_path": "data/mesh/naca001234_2d_fluent.msh",
    "pressure_csv": "data/results/pressure_dist.csv",
}
```

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.13 (MS Store) | Runtime |
| ansys-fluent-core (PyFluent) | latest | Fluent solver automation |
| gmsh | 4.15.2 | Open-source meshing (replaces Fluent Meshing) |
| meshio | 5.3.5 | Read gmsh MSH files in the converter |
| build123d | latest | OpenCASCADE-based CAD for domain STEP generation |
| numpy / scipy | latest | Geometry preprocessing |
| DeepXDE / PyTorch | TBD | PINN training (Agent 4, not yet built) |
| Ansys Fluent | 2025 R2 Student | Solver only (meshing mode blocked by Student license) |
| Ansys Mechanical | Student | FEA (Agent 3, not yet built) |

Python venv is at `.venv/` — always activate before running anything:
```bash
.venv/Scripts/activate
python run_cfd_test.py
```

---

## File Structure

```
simulation-automation/
├── CLAUDE.md                       # this file
├── run_cfd_test.py                 # end-to-end integration test (the base-case pipeline)
│
├── agents/
│   ├── geometry_agent.py           # Agent 1+2a: cleans .dat, applies cosine spacing, calls CAD builder
│   ├── cad_builder.py              # Agent 2b: build123d C-domain STEP with airfoil hole
│   └── mesh_agent.py               # Agent 2c: gmsh meshing + Fluent ASCII MSH converter
│
├── physics_cores/
│   ├── ansys_fluent/
│   │   ├── cfd_tool.py             # FluidistAgent: Fluent launch, BCs, solve, CSV export
│   │   └── mesh_airfoil.py         # Legacy: Fluent meshing workflow (blocked by Student license, kept for reference)
│   └── ansys_mech/                 # Agent 3 placeholder (empty)
│
├── data/
│   ├── airfoils/                   # Input: raw .dat files from UIUC
│   ├── geometry/                   # Agent 1 output: cleaned CSV, STEP
│   ├── mesh/                       # Agent 2 output: gmsh .msh, Fluent .msh, generated scripts
│   ├── fluent_logs/                # Fluent transcript (.trn) and error logs
│   ├── results/                    # CFD output: pressure_dist.csv
│   └── raw/                        # Deprecated staging area (managed by CleanupManager)
│
└── scripts/
    └── manage_files.py             # CleanupManager: bootstraps dirs, organises loose files
```

Generated files that are NOT committed (see .gitignore): `data/mesh/*.msh`, `data/results/`, `data/fluent_logs/*.trn`, `*.trn` in root.

---

## Rules & Conventions

**Naming:**
- Variables use `snake_case`. Agent class names use `PascalCase` (e.g. `MeshAgent`, `FluidistAgent`).
- Airfoil names are lowercase with no spaces: `naca001234`, `naca0012`.
- All intermediate files are named `{airfoil_name}_{type}.{ext}` (e.g. `naca001234_domain.step`, `naca001234_2d_fluent.msh`).
- Fluent zone names are fixed: `inlet`, `outlet`, `airfoil`, `symmetry_top`, `symmetry_bottom`, `fluid`, `interior`. Do not rename these — they are hardcoded in both mesh generation and `cfd_tool.py` BC setup.

**Comments:**
- Plain text only. No markdown, no emoji, no `#---` decorative lines in inline comments.
- Comments explain the *why*, not the *what*. Obvious code gets no comment.
- Docstrings on all public methods. One-line summary, then Args/Returns if non-trivial.

**Units:**
- Everything is SI throughout: metres, m/s, Pascals. Normalised airfoil coords (x/c, y/c) are explicitly labelled and converted to metres before any solver call.

**Error handling:**
- Agents raise `RuntimeError` on unrecoverable failures. No silent `except: pass` blocks.
- Subprocess calls (gmsh) check `returncode` and scan stdout for `MESH_SUCCESS` / `MESH_ERROR` sentinel strings.

---

## Common Commands

```bash
# Full pipeline (geometry → mesh → CFD solve → CSV)
python run_cfd_test.py

# Mesh agent only (skip geometry, use existing STEP)
python agents/mesh_agent.py --step data/geometry/naca001234_domain.step --name naca001234

# CAD only
python agents/cad_builder.py --csv data/geometry/naca001234_cleaned.csv --name naca001234

# Fluent solver only (load existing mesh, run, export)
python physics_cores/ansys_fluent/cfd_tool.py --mesh data/mesh/naca001234_2d_fluent.msh

# Check a Fluent transcript for errors
grep -i "error\|warning\|null domain" data/fluent_logs/*.trn
```

---

## Debugging Gotchas

These are issues that have already burned time. Do not repeat them.

### Ansys Student License Constraints
- **Meshing mode is completely blocked.** `FluentMode.MESHING` exits immediately with "Mesher mode is not supported - Starting Solver mode." The solution is gmsh for all meshing. `mesh_airfoil.py` is kept for reference only.
- **Only 1 solver core allowed.** `processor_count=4` causes a `BAD TERMINATION / SIGSEGV` in `Auto_Partition` because the MPI partitioner fails on gmsh-origin meshes. Always use `processor_count=1`.
- **License is tied to a specific Windows user account.** Run all scripts from the account that installed Ansys Student. Running elevated (admin UAC) or from a different user causes silent license checkout failures.

### WNUA (Windows Network User Authentication)
PyFluent launches Fluent as a subprocess; WNUA blocks gRPC communication between them when the process user differs from the Windows session user. Three bypasses are required together — if any one is missing, Fluent may launch but PyFluent can't connect:
1. `os.environ["ANSYS_NO_WINDOWS_USER_AUTH"] = "1"` — set in `cfd_tool.py` before `import ansys.fluent.core`
2. `os.environ["ANSYSLI_NO_USER_CHECK"] = "1"` — same location
3. `additional_arguments="-nwnua"` — passed to `pyfluent.launch_fluent()`

The "ACCESS DENIED" messages that appear at Fluent shutdown (`ANSYS Product Improvement Program`) are benign telemetry failures, not solve errors.

### gmsh STEP Import
- Use `gmsh.model.occ.importShapes(path)` + `occ.synchronize()` to load a STEP file. This preserves B-rep topology.
- Do NOT use `gmsh.merge()` + `classifySurfaces()`. Those only work on triangulated/tessellated surfaces, not analytical STEP B-rep. They silently load 0 curves and produce an unmeshed domain.
- `gmsh.model.mesh.checkMesh()` does not exist in gmsh 4.15 — remove any call to it.

### Fluent 2D and the gmsh MSH Format
Fluent 2D cannot read gmsh MSH files directly. Two reasons:
1. gmsh MSH has no dimension declaration. Fluent 2D needs `(2 2)` in the header or it raises "Null Domain Pointer."
2. gmsh always writes 3D node coordinates `(x y z)` even for flat 2D meshes. Fluent 2D rejects this.

The fix is `MeshAgent._convert_to_fluent_msh()`, which reads the gmsh MSH with meshio and writes a native Fluent ASCII MSH with the correct header, 2D node coords, typed mixed cell zone, face-cell adjacency, and BC-typed boundary zones.

Fluent ASCII MSH section structure (in order):
```
(0 "comment")
(2 2)                          <- dimension: 2D
(10 (0 1 N 0 2))               <- node header (N = total node count)
(10 (zone_id first last 1 2)(  <- node data
  x y
  ...
))
(12 (0 1 Nc 0))                <- cell header
(12 (zone_id 1 Nc 1 0)(        <- mixed cell zone (type 0 = mixed)
  1                            <- 1 = triangle
  3                            <- 3 = quad
  ...
))
(13 (0 1 Nf 0 0))              <- face header
(13 (zone_id first last bc 2)( <- boundary face zone; bc is hex e.g. 14 = velocity-inlet
  n0 n1 cr cl                  <- 1-based node indices; cr=fluid cell, cl=0 for boundary
  ...
))
(13 (zone_id first last 2 2)(  <- interior faces; bc=2
  ...
))
(45 (zone_id type name)())     <- zone descriptors
```

Fluent BC type codes: `0x14`=velocity-inlet, `0x9`=pressure-outlet, `0x7`=symmetry, `0x3`=wall, `0x2`=interior.

### PyFluent TUI — Reading a Mesh File
`solver.tui.file.read_mesh()` does not exist in PyFluent. Use either:
```python
solver.tui.file.read_case(mesh_path)      # TUI wrapper (accepts .msh extension)
solver.settings.file.read_case(file_name=mesh_path)  # settings API
```

### Windows Paths in Generated Python Scripts
When `mesh_agent.py` writes a gmsh script to disk and embeds file paths as string literals, backslashes cause `SyntaxError: (unicode error) 'unicodeescape'`. Fix: convert all paths to forward slashes before embedding:
```python
step_fwd = step_path.replace("\\", "/")
```

### Face-Cell Adjacency in the MSH Converter
Fluent requires every face to list `(n0 n1 cr cl)` where `cr` = the fluid cell to the right of the directed edge `n0→n1`, and `cl` = the cell to the left (0 for exterior). Getting the orientation wrong is silent but produces incorrect pressure gradients. The rule: if the cell's stored half-edge goes `n0→n1`, the cell is already to the *left* — reverse the edge so the cell ends up on the *right*.

---

## Current Focus: Base Case Pipeline

The immediate goal is to prove the full pipeline runs end-to-end for one hardcoded airfoil (NACA 001234) and produces `data/results/pressure_dist.csv`. No agent generalisation, no parameter sweeps — just one working run.

Test with: `python run_cfd_test.py`

Once that passes, the next step is parameterising `MeshAgent` and `FluidistAgent` so the Lead Agent can hand them arbitrary airfoil names and get results back.
