# ADE Project — CLAUDE.md

## Project Overview

Autonomous Design Evaluator (ADE): a 6-agent pipeline that runs high-fidelity CFD and FEA simulations on NACA airfoil shapes, collects the results as training data, and uses them to train a Physics-Informed Neural Network (PINN). The PINN acts as a fast surrogate model — predicting aerodynamic and structural performance for any airfoil without running a full simulation. The lead agent uses the PINN to search the design space and only calls Ansys to verify the best candidates.

Pipeline: `.dat` → cleaned geometry → CAD STEP → gmsh mesh → Fluent ASCII MSH → CFD solve → pressure CSV → FEA → PINN training data.

Read `STATUS.md` for current per-agent status and known blockers.

---

## 6-Agent Architecture

| # | Name | Role | Status | Entry Point |
|---|------|------|--------|-------------|
| 1 | Librarian | Downloads/generates NACA .dat files | Working | `data/airfoil_downloader.py` |
| 2 | Fluidist (CFD Lead) | Geometry → mesh → Fluent CFD → pressure CSV | In progress | `agents/geometry_agent.py`, `agents/mesh_agent.py`, `physics_cores/ansys_fluent/cfd_tool.py` |
| 3 | Structuralist (FEA Lead) | Maps pressure onto structural model via PyMechanical | Not built | `physics_cores/ansys_mech/` |
| 4 | Surrogate (PINN Trainer) | Trains DeepXDE/PyTorch PINN on CFD+FEA outputs | Not built | TBD |
| 5 | Troubleshooter | Monitors logs, interprets divergence, retries | Not built | TBD |
| 6 | Lead (Orchestrator) | Uses PINN to decide which shapes to verify | Not built | TBD |

Agents pass a **Design Dictionary** between each other:
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
| ansys-fluent-core (PyFluent) | 0.20+ | Fluent solver automation |
| gmsh | 4.15.2 | Open-source meshing (replaces Fluent Meshing) |
| meshio | 5.3.5 | Read gmsh MSH files in the converter |
| build123d | latest | OpenCASCADE-based CAD for domain STEP generation |
| numpy / scipy | latest | Geometry preprocessing |
| DeepXDE | latest | PINN training (Agent 4) |
| PyTorch | 2.0+ | DeepXDE backend (preferred over TF) |
| LangGraph | 0.1+ | Agent orchestration state machine (Agent 6) |
| Ansys Fluent | 2025 R2 Student | Solver only (meshing mode blocked) |
| Ansys Mechanical | Student | FEA (Agent 3, not yet built) |

```bash
# Always activate venv first
.venv/Scripts/activate
python run_cfd_test.py
```

---

## File Structure

```
simulation-automation/
├── CLAUDE.md                       # this file — read at start of every session
├── STATUS.md                       # per-agent pipeline status — update at end of session
├── run_cfd_test.py                 # end-to-end integration test
│
├── agents/
│   ├── geometry_agent.py           # cleans .dat, cosine spacing, calls CAD builder
│   ├── cad_builder.py              # build123d C-domain STEP with airfoil hole
│   └── mesh_agent.py               # gmsh meshing + Fluent ASCII MSH converter
│
├── physics_cores/
│   ├── ansys_fluent/
│   │   ├── cfd_tool.py             # FluidistAgent: launch, BCs, solve, CSV export
│   │   └── mesh_airfoil.py         # Legacy Fluent meshing (Student license blocked — reference only)
│   └── ansys_mech/                 # Agent 3 placeholder
│
├── data/
│   ├── airfoils/                   # Input: raw .dat files
│   ├── geometry/                   # Agent 1 output: cleaned CSV, STEP
│   ├── mesh/                       # Agent 2 output: gmsh .msh, Fluent .msh, scripts
│   ├── fluent_logs/                # Fluent transcripts (.trn)
│   ├── results/                    # CFD output: pressure_dist.csv
│   └── raw/                        # Deprecated staging (managed by CleanupManager)
│
└── scripts/
    └── manage_files.py             # CleanupManager: bootstraps dirs, organises files
```

Generated files not committed: `data/mesh/*.msh`, `data/results/`, `data/fluent_logs/*.trn`.

---

## Rules & Conventions

**Naming:** `snake_case` variables, `PascalCase` agent classes. Airfoil names lowercase: `naca0012`. Files: `{name}_{type}.{ext}`. Fluent zone names are fixed (hardcoded in mesh generation and `cfd_tool.py`): `inlet`, `outlet`, `airfoil`, `symmetry_top`, `symmetry_bottom`, `fluid`, `interior`.

**Comments:** Plain text only. Explain the *why*, not the *what*. Docstrings on all public methods.

**Units:** SI throughout — metres, m/s, Pascals. Normalised airfoil coords (x/c, y/c) must be converted to metres before any solver call.

**Errors:** Agents raise `RuntimeError` on unrecoverable failures. No silent `except: pass`. Subprocess calls check `returncode` and scan stdout for `MESH_SUCCESS` / `MESH_ERROR`.

---

## Common Commands

```bash
# Full pipeline
python run_cfd_test.py

# Mesh agent only (skip geometry, use existing STEP)
python agents/mesh_agent.py --step data/geometry/naca001234_domain.step --name naca001234

# CAD only
python agents/cad_builder.py --csv data/geometry/naca001234_cleaned.csv --name naca001234

# Fluent solver only
python physics_cores/ansys_fluent/cfd_tool.py --mesh data/mesh/naca001234_2d_fluent.msh

# Scan Fluent transcripts for errors
grep -i "error\|warning\|null domain" data/fluent_logs/*.trn
```

---

## Debugging Gotchas

### Ansys Student License Constraints
- **Meshing mode is completely blocked.** `FluentMode.MESHING` exits immediately. Use gmsh for all meshing. `mesh_airfoil.py` is reference only.
- **Only 1 solver core.** `processor_count=4` causes `BAD TERMINATION / SIGSEGV` in `Auto_Partition`. Always `processor_count=1`.
- **License is tied to a specific Windows user.** Running elevated (admin UAC) or from a different user causes silent license failures.

### WNUA (Windows Network User Authentication)
All three bypasses are required together. Missing any one causes Fluent to launch but PyFluent can't connect:
1. `os.environ["ANSYS_NO_WINDOWS_USER_AUTH"] = "1"` — in `cfd_tool.py` before the import
2. `os.environ["ANSYSLI_NO_USER_CHECK"] = "1"` — same location
3. `additional_arguments="-nwnua"` — in `launch_fluent()`

Shutdown "ACCESS DENIED" messages (`ANSYS Product Improvement Program`) are benign telemetry, not solver errors.

### gmsh STEP Import
- Use `gmsh.model.occ.importShapes(path)` + `occ.synchronize()`. This preserves B-rep topology.
- Do NOT use `gmsh.merge()` + `classifySurfaces()` — those only work on tessellated surfaces, silently load 0 curves.
- `gmsh.model.mesh.checkMesh()` does not exist in gmsh 4.15 — remove any call to it.

### Fluent ASCII MSH Format (Critical)
Fluent 2D cannot read gmsh MSH directly — no `(2 2)` header, 3D coordinates. `MeshAgent._convert_to_fluent_msh()` writes the native format. Three rules that have already caused crashes:

**1. All integers must be hexadecimal.** This includes counts, first/last indices, zone IDs, face data (`n0 n1 cr cl`), and element type codes. Decimal values cause "unable to read coordinates of node N" parse overflow.

**2. The data block `(` must open on the same line as the section header.** Write `(10 (1 1 N 1 2)(\n` not `(10 (1 1 N 1 2)\n(\n`. A bare `(` on its own line causes "Build Grid: Aborted due to critical error" + SIGSEGV.

**3. Face-cell adjacency orientation.** `cr` = fluid cell to the right of directed edge `n0→n1`, `cl` = 0 for boundary. If the cell's stored half-edge goes `n0→n1`, the cell is on the left — reverse the edge.

Section order: `(2 2)` → `(10 ...)` nodes → `(12 ...)` cells → `(13 ...)` faces → `(45 ...)` zone descriptors.

BC type hex codes: `0x14`=velocity-inlet, `0x9`=pressure-outlet, `0x7`=symmetry, `0x3`=wall, `0x2`=interior.

### PyFluent API
**Loading a mesh:** `solver.tui.file.read_mesh()` does not exist. Use:
```python
solver.file.read_mesh(file_name=mesh_path)    # Settings API — preferred
solver.tui.file.read_case(mesh_path)           # TUI — also works
# Note: solver.settings.file.read_case() does NOT exist
```

**Field data:** `SurfaceFieldDataRequest` and `ScalarFieldDataRequest` moved to a submodule — import from there, not from `ansys.fluent.core` directly (that raises `ImportError`):
```python
from ansys.fluent.core.field_data_interfaces import (
    SurfaceFieldDataRequest, ScalarFieldDataRequest, SurfaceDataType
)

field_data = solver.fields.field_data
batch = field_data.new_batch()
batch.add_requests(
    SurfaceFieldDataRequest(data_types=[SurfaceDataType.FacesCentroid], surfaces=["airfoil"]),
    ScalarFieldDataRequest(field_name="pressure", surfaces=["airfoil"])
)
response = batch.get_response()
centroids = response.get_field_data(surface_request)["airfoil"].face_centroids
pressures = response.get_field_data(pressure_request)["airfoil"]
```

Use `add_requests()` — `add_surfaces_request()` is deprecated. The Fluent field name for gauge static pressure is `"pressure"`, not `"static-pressure"`.

**Boundary conditions:**
```python
setup = solver.settings.setup
setup.models.viscous.model.set_state("k-omega")
setup.models.viscous.k_omega_model.set_state("sst")
inlet = setup.boundary_conditions.velocity_inlet["inlet"]
inlet.momentum.velocity_magnitude.set_state(50.0)
```

### Windows Paths in Generated Scripts
Backslashes in string literals cause `SyntaxError: unicodeescape`. Always convert before embedding in generated scripts:
```python
step_fwd = step_path.replace("\\", "/")
```

---

## Agent 4 — PINN Surrogate (DeepXDE)

Uses PyTorch backend. Set before importing DeepXDE: `os.environ['DDE_BACKEND'] = 'pytorch'`.

**Data-driven training on CFD CSV:**
```python
import deepxde as dde, numpy as np, pandas as pd

df = pd.read_csv("data/results/pressure_dist.csv")
X = df[["x_m", "y_m"]].values.astype(np.float32)
y = df["pressure_Pa"].values.reshape(-1, 1).astype(np.float32)

bc = dde.icbc.PointSetBC(X, y)
geom = dde.geometry.Rectangle([-0.5, -1.0], [1.5, 1.0])
data = dde.data.PDE(geom, pde=None, bcs=[bc], num_domain=0, num_boundary=0)

net = dde.nn.FNN([2, 128, 128, 64, 1], activation="tanh")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
model.train(epochs=15000, display_every=500)
```

Checkpoint during training:
```python
ckpt = dde.callbacks.ModelCheckpoint("models/airfoil_pinn", save_better_only=True, period=500)
model.train(epochs=15000, callbacks=[ckpt])
```

Inference:
```python
import torch
net = dde.nn.FNN([2, 128, 128, 64, 1], activation="tanh")
ckpt = torch.load("models/airfoil_pinn-best_loss.pt")
net.load_state_dict(ckpt["model_state_dict"])
net.eval()
```

Normalise inputs to `[-1, 1]` before training. Store the scaler for inference.

---

## Agent 6 — Orchestrator (LangGraph)

LangGraph does NOT require LangChain — `pip install langgraph` only. Nodes are plain Python functions; no LLM required.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

class DesignState(TypedDict):
    design_dict: dict
    cfd_converged: bool
    fea_results: dict

def cfd_node(state: DesignState) -> dict:
    results = FluidistAgent().run_from_mesh(state["design_dict"]["mesh_path"], ...)
    return {"cfd_converged": results["converged"], "design_dict": state["design_dict"]}

def route_after_cfd(state: DesignState) -> str:
    return "fea" if state["cfd_converged"] else "troubleshooter"

graph = StateGraph(DesignState)
graph.add_node("cfd", cfd_node, retry_policy=RetryPolicy(max_attempts=2))
graph.add_node("troubleshooter", troubleshooter_node)
graph.add_node("fea", fea_node)
graph.add_edge(START, "cfd")
graph.add_conditional_edges("cfd", route_after_cfd, {"fea": "fea", "troubleshooter": "troubleshooter"})
graph.add_edge("fea", END)

app = graph.compile()
result = app.invoke({"design_dict": {...}, "cfd_converged": False, "fea_results": {}})
```

Router functions return a string (node name). State updates are merged automatically — return only changed keys.

---

## Maintaining CLAUDE.md and STATUS.md

At the end of every session:

1. Update `STATUS.md` — change the relevant agent's status row if anything changed.
2. Update `CLAUDE.md` only if the session produced a new gotcha, confirmed a stage working end-to-end, or changed a convention/API. Do not add in-progress debugging or "we tried X" notes.
3. Fix `CLAUDE.md` immediately mid-session if you find it contradicts the actual code.
4. Keep `CLAUDE.md` under ~250 lines. Compress sections if they grow.

---

## Current Focus

Prove the pipeline works end-to-end for one hardcoded airfoil (NACA 001234) and produces `data/results/pressure_dist.csv`. Test with: `python run_cfd_test.py`
