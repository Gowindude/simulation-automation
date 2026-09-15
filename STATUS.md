# ADE Pipeline — Status

Last updated: 2026-09-15 (train/val/test split, dashboard DeepONet metrics, G2Aero corpus, native-Linux shell dispatch for GH Actions)

### DeepONet: real train/val/test split exposes a generalization gap (2026-09-15)

The previous train/val split (`deeponet/dataset.py::split_by_airfoil`) used
the same held-out airfoils both to pick the best checkpoint AND as the
number reported for "how good is this" -- optimistic, not a true
held-out result. Added `split_train_val_test` (3-way, by whole airfoil)
and wired `deeponet/train.py` to report a test-set metric computed once,
after training, on airfoils that influenced neither training nor
checkpoint selection.

Re-ran training on the same 41-airfoil dataset (29 train / 6 val / 6
test, `--epochs 800`): best val_loss=0.406 (epoch 53) vs. **held-out
test_loss=1.894, test_rmse_Cp=1.15** -- the real generalization number is
~4.6x worse than what the val-only setup was reporting. Confirms the
41-airfoil corpus is still too small/narrow for the model to generalize
well, not just an academic distinction -- worth keeping in mind before
trusting any dashboard "accuracy" number as production-quality.

Also measured real DeepONet inference speed on this same test set:
0.25us/point (CPU), i.e. a full one-airfoil Cp(s) sweep estimated at
~0.3ms -- vs. a real measured single-airfoil full-pipeline wall-clock
of **131.8s** (naca0012, 5/5 AoA, WSL/gmsh/OpenFOAM/CalculiX, measured
fresh this session, not the older ~116s/airfoil batch-average number).
Both numbers now surface on the dashboard (`scripts/dashboard.html`'s
new "DeepONet surrogate: speed & accuracy" panel, fed by
`scripts/build_dashboard_data.py`'s new `pipeline_timing`/`deeponet`
sections) -- degrades gracefully (panel hidden) if a manifest predates
per-airfoil timing or a checkpoint predates the 3-way split.

### G2Aero corpus added: 358 new real airfoils (2026-09-15)

Per user request, pulled NREL's G2Aero `curated_airfoils.npz`
(data.openei.org/submissions/6198, CC BY 4.0) as a second real-airfoil
source alongside UIUC. **Correction to the dataset's own published
docs**: NREL's page describes `classes` as distinguishing real vs.
6,164 synthetic shapes; the actual downloaded file's `classes` array
holds each shape's NAME string, not a 0/1 label -- 14 names repeat
~1,000x each (the CST-perturbation baselines, 13,012 shapes), the
remaining 6,152 appear exactly once (the real BigFoil-derived
airfoils). `data/g2aero_downloader.py::load_real_shapes` identifies
"real" by that repeat-count heuristic, not a hardcoded expected number
(which would silently drift if a future file version changes it).

Real, non-name-based dedup also added (`is_near_duplicate`, resampled
max-pointwise-distance, 1% chord threshold) since G2Aero's real subset
substantially overlaps UIUC (BigFoil itself incorporates UIUC). Real
run against the full 1,666-file UIUC corpus (`data/raw/airfoils/`):
6,152 real G2Aero shapes -> 1,028 name-duplicates + 4,674
geometry-duplicates + 92 rejected by Stage 0's own self-intersection
check (mostly exotic theoretical shapes -- Joukowsky sections, extreme
aspect-ratio GA airfoils) -> **358 net new airfoils** written to
`data/airfoils_g2aero/`, all verified parsing cleanly through Stage 0
(the same bar the original 100-airfoil UIUC pull was held to). Not yet
run through the CFD/FEA pipeline itself -- that's the GitHub Actions
work below.

### Native-Linux shell dispatch: unblocks running the pipeline on GitHub Actions (2026-09-15)

`wsl.exe` was hardcoded via 4 separately-duplicated `_run_wsl`/
`_to_wsl_path` helper pairs (stage1_mesh.py, stage3_run.py,
stage4_postprocess.py, stage8_calculix_run.py) -- fine for a
Windows+WSL-only project, but a GitHub Actions Linux runner has no WSL
layer at all, so `wsl.exe` would just fail there, not silently work.
Consolidated into `pipeline/_shell.py::run_shell`/`to_linux_path`,
platform-dispatched via `platform.system()`: `wsl.exe -- bash -lc` on
Windows (byte-for-byte the same invocation the old duplicated helpers
made -- confirmed via the full real-hardware Stage 1/3/4/8 test suite,
76 tests + 1 xfail, all still passing after the refactor), plain
`bash -lc` on Linux (no wsl.exe wrapper, since none exists there).

Added `.github/workflows/airfoil_smoke_test.yml` (`workflow_dispatch`
only, not on every push): installs OpenFOAM 12 + CalculiX + xfoil +
gmsh natively via apt/pip, runs one real airfoil (naca0012) through the
full Stage 0-9 chain, asserts a real `.h5` with >=1 converged AoA. This
is the locked target before any matrix fan-out over more airfoils, per
the build spec's own "propose the verification target, then build to
it" convention. **Not yet run for real** -- apt package names
(`openfoam12`, `calculix-ccx`, `xfoil`) and the OpenFOAM Foundation apt
repo URL are correct per their own published install docs but unverified
against an actual GitHub Actions runner; expect at least one real
iteration to shake out install issues before this passes cleanly. Not
yet pushed to the remote -- pending confirmation, since it's the first
thing in this project that touches GitHub Actions minutes / CI.

Researched (not yet acted on): further free compute beyond local
`max_workers` -- GitHub Actions' free-tier matrix jobs (up to 20
concurrent runners, unlimited minutes on a public repo) are the
strongest option once the smoke test above is proven; Oracle Cloud's
Always Free ARM tier is a fallback for any single airfoil needing more
than a 6-hour job cap, but has known provisioning/ARM-build friction.

### First real GitHub Actions run: apt/pip install verified, real bug found and fixed (2026-09-15)

Ran the smoke-test workflow for real (`gh workflow run`, `feature/mesh-agent`,
after pushing just the workflow file to `main` too -- GitHub only allows
`workflow_dispatch` to be dispatched once the workflow file exists on the
default branch, even to run against a different ref). **Every install
step succeeded**: `openfoam12`, `calculix-ccx`, `xfoil` all installed
cleanly from the exact apt source/package names cross-checked against
this project's own working WSL install -- that verification approach
held up for real, not just in theory.

The run itself **looked like a 28-minute hang** (job-level
`timeout-minutes: 30` eventually force-killed it) but the real failure
happened in the first 10 seconds: the workflow ran its Python snippet
via `python - <<HEREDOC` (stdin), which sets `__main__.__file__` to
`<stdin>` -- not a real path. `pipeline/_gmsh_isolation.py`'s spawned
multiprocessing child (Stage 5/6's gmsh isolation) crashed during
interpreter bootstrap trying to re-import that non-existent path,
*before* it ever reached `_worker_entry` to report anything back. The
parent's `queue.get()` had no timeout, so it waited forever -- a real
10-second crash was indistinguishable from a genuine still-running solve
until GitHub's own job timeout intervened.

**Two fixes, not one** (STATUS's own past pattern -- a hardcoded-path
gotcha needs both the direct fix and a defensive bound, or the same
failure mode just recurs somewhere else):
1. `scripts/ci_smoke_test.py` -- a real file with `if __name__ ==
   "__main__":`, called via `python scripts/ci_smoke_test.py`, not
   piped through stdin. Root-cause fix for *this* incident.
2. `pipeline/_gmsh_isolation.py::run_isolated` now takes a bounded
   `timeout` (default 600s, generous vs. real ~132s single-airfoil
   solves) and polls rather than blocking indefinitely -- a child that
   dies without ever reporting now raises a RuntimeError within one
   poll interval (~0.2s), and a child that's still running past budget
   raises TimeoutError, distinct failure modes rather than one
   indistinguishable hang. Covered by `tests/test_gmsh_isolation.py` (5
   tests, including a real subprocess death simulated via `os._exit`).
   This means any *future* bootstrap-crash-shaped bug (not just this
   exact stdin gotcha) fails fast instead of silently eating a job's
   entire timeout budget.

Not yet re-run for real after these fixes -- next step is triggering
the workflow again to confirm the actual solve now completes (or fails
with a real, fast, legible error if something else is still wrong).

### Known flaky/load-sensitive real-hardware test (observed, not fixed, 2026-09-15)

`test_orchestrator_real_multi_airfoil.py::test_real_parallel_batch_converges_with_no_corruption_and_beats_serial`
failed on its wall-clock threshold (621s actual vs. <348s required) while
this session had heavy concurrent load (a 310MB download, DeepONet
training, and back-to-back pytest runs all sharing the same WSL VM/16
cores). All of that run's actual correctness assertions passed (4/4
airfoils, 5/5 AoA converged each, no corruption) -- only the timing
assertion failed, and 621s is even slower than the test's own naive
*serial* estimate (464s), which points at contention, not a real
regression (STATUS.md's own prior measurement of this exact test was
237.9s under quieter conditions). Not re-verified under quiet
conditions this session due to the ~10min-per-run cost -- worth a clean
rerun before trusting it either way.

## Current architecture: deterministic pipeline (`.claude/airfoil_pipeline_build_spec.md`)

The project pivoted from the agent-based Ansys Fluent/Mechanical pipeline
(see "Superseded" section below) to a deterministic, stage-by-stage
pipeline built on gmsh + OpenFOAM (WSL) + CalculiX, per the build spec.
No agents yet — agent layer is explicitly out of scope until stages 0-9
work end-to-end for one airfoil. Code lives in `pipeline/`, tests in
`tests/`.

| Stage | Module | Status | Tests |
|-------|--------|--------|-------|
| 0 — Geometry loader | `pipeline/stage0_geometry_loader.py` | Done | 258 passed (`test_stage0_geometry_loader.py`, `test_stage0_real_uiuc.py`) |
| 1 — CFD meshing (gmsh -> OpenFOAM) | `pipeline/stage1_mesh.py` | Done | 96 passed (`test_stage1_mesh.py`, `test_stage1_real_uiuc.py`) |
| 2 — CFD case generation | `pipeline/stage2_case_gen.py` | Done | 35 passed (`test_stage2_case_gen.py`) |
| — Pipeline integration (0->1->2 handoffs) | — | Done | 12 passed (`test_pipeline_integration.py`) |
| 3 — CFD execution + convergence check | `pipeline/stage3_run.py` | Done | 13 passed (`test_stage3_run.py`) |
| 4 — CFD post-processing | `pipeline/stage4_postprocess.py` | Done (mechanism + schema record); Gate #2 fails for real, see below | 16 passed + 1 xfail (`test_stage4_postprocess.py`) |
| 5 — Structural geometry generation | `pipeline/stage5_structural_geometry.py` | Done | 13 passed (`test_stage5_structural_geometry.py`) |
| 6 — Structural meshing | `pipeline/stage6_structural_mesh.py` | Done | 11 passed (`test_stage6_structural_mesh.py`) |
| 7 — Load mapping (fluid -> structure) | `pipeline/stage7_load_mapping.py` | Done, incl. Must-Pass Gate #3 | 11 passed (`test_stage7_load_mapping.py`) |
| 8 — CalculiX execution | `pipeline/stage8_calculix_run.py` | Done, incl. Must-Pass Gate #4 | 10 passed (`test_stage8_calculix_run.py`) |
| 9 — Aggregation | `pipeline/stage9_aggregation.py` | Done | 10 passed (`test_stage9_aggregation.py`) |

**All 10 stages (0-9) are now built.** Per the spec's own build order
step 8, `pipeline/orchestrator.py` now loops the full chain over an
airfoil list, with progress/failure tracking and an opt-in parallel
mode. Built test-first (`tests/test_orchestrator.py`, 15 tests written
before the implementation): the batch-loop layer (`run_batch` —
manifest bookkeeping, resume, failure isolation, progress callbacks,
sequential-vs-parallel dispatch) is tested against an injected fake
`run_one`, so it runs in seconds with no WSL/gmsh dependency; the
single-airfoil chain layer (`run_single_airfoil`) has separate tests
monkeypatching the real stage functions to check failure
categorization (Stage 0/1 = `"geometry_mesh"` per the spec's
agent-scope split, everything else = `"other"`) and that a checkMesh
Gate #1 failure or a per-AoA solver crash halts/logs correctly without
raising. One non-mocked test confirms the real nesting shape a parallel
batch depends on (`ProcessPoolExecutor` worker → `run_isolated()`'s own
spawned child for Stage 5/6's gmsh work — not daemonic-forbidden the
way `multiprocessing.Pool` would be). `run_single_airfoil` was then
run for real (naca0012, one AoA, no mocks) as the actual composition
check: Cl=0.371/Cd=0.0486, matching the hand-verified single-airfoil
run above to 3 decimal places, `.h5` written, FEA stage ran — confirms
Stage 0->1->2->3->4->5->6->7->8->9 wired correctly end-to-end through
the orchestrator, not just through each stage's own test suite.

Design notes / open items:
- `max_workers=1` (sequential loop, no process pool) is the default.
  `max_workers>1` uses `ProcessPoolExecutor`; no concurrent-WSL
  production run has been validated yet (only the isolation-nesting
  shape was checked directly), so treat >1 as opt-in until that's done.
- The manifest (`<output_dir>/batch_manifest.json`) intentionally stores
  only a per-AoA status summary, not the full per-AoA result — the
  `.h5` already holds the real arrays (pressure curves, ~23k-entry
  stress fields per converged AoA per STATUS's own numbers); keeping
  those in the manifest too would mean re-serializing several MB of
  JSON after every single airfoil in a batch.
- Resume treats the manifest as the completion authority only in
  combination with the filesystem: a `"success"` entry whose `h5_path`
  no longer exists is rerun, not trusted blindly.
- Each airfoil gets its own `<output_dir>/<name>/` subdirectory --
  Stage 8 copies its `.inp` next to the assembled CalculiX deck, which
  would otherwise race across airfoils sharing one directory.
- Not yet done: a real multi-airfoil (sequential or parallel) run
  through the orchestrator itself -- only a single airfoil, single AoA,
  through `run_single_airfoil` directly has been confirmed for real so
  far. The full 3-airfoil confirmation earlier in this file predates
  the orchestrator module and used ad hoc (uncommitted) runner scripts.

### Reframing the goal: pipeline is supporting infra, dataset is the point

Per the build spec's own words -- "the pipeline's purpose is generating
training data for a neural operator" -- automating the CFD/FEA chain is
the supporting goal; the primary goal is producing a (geometry, AoA) ->
(pressure field, stress field) dataset large and diverse enough to train
a DeepONet (spec's own recommendation over FNO, since output isn't on a
fixed grid). Scaling the orchestrator and building the Stage 0/1
troubleshooter agent (next up) both serve that directly. Corpus size:
the full UIUC database (~1600 airfoils) is available but likely
insufficient alone -- organically collected, so it clusters (many
near-duplicate general-aviation/glider shapes) rather than evenly
covering thickness/camber space. Plan is to supplement with synthetic
parametric geometry (NACA 4/5-digit equations first -- simple, closed-
form, easy to validate against Stage 0's own self-intersection/closure
checks; CST parameterization later if broader coverage than the NACA
family is needed) once the real corpus's own failure modes are known
(see below) -- generating synthetic shapes before knowing which shapes
break the pipeline would mean doing it twice.

### Real multi-airfoil orchestrator runs (2026-09-14)

`tests/test_orchestrator_real_multi_airfoil.py` (real WSL/gmsh/ccx, no
mocks; skip with `ADE_SKIP_REAL_TESTS=1`) confirms `max_workers=2`: 4
real airfoils (naca0006, naca4412, clarky, e387; not the 3 used in the
first real orchestrator confirmation) each converged 5/5 AoA in 237.9s
total vs. the ~464s serial estimate (4 x 116s/airfoil from the
3-airfoil run) -- a real ~1.96x speedup, no WSL/gmsh cross-process
corruption. Machine: 16 physical cores, 31 GB RAM (`Get-CimInstance
Win32_Processor`/`wsl.exe -- nproc` both report 16) -- well above what
`max_workers=4` needs.

Then ran the **full 35-airfoil `tests/fixtures/real_uiuc/` corpus** at
`max_workers=4` (all 35 parse cleanly at Stage 0 first, confirmed
before the batch) -- **35/35 airfoils completed (aggregated .h5
written), 0 airfoil-level failures**, ~968 MB total output (~28
MB/airfoil incl. intermediates, not just the ~2 MB `.h5` -- the
intermediate case dirs are the real disk cost at scale, not the final
data). At AoA level: 162/175 (92.6%) converged. Also **deliberately
interrupted the batch mid-run** (killed at 8/35 done) and restarted
with `resume=True` (the default) -- confirmed for real, not just via
the mocked test suite, that it picked up exactly where it left off with
no re-run of completed airfoils.

**8 airfoils had a non-5/5 AoA outcome** -- the first real failure
taxonomy data, exactly what STATUS's "Deferred to a future
Troubleshooter/agent layer" section said was missing before that work
could start:
- ah79100c, naca633418, naca633618, s1223: one or two AoAs
  `non_converged` (Stage 3's own legitimate classification, not an
  exception) -- ah79100c and the bl_size/skewness tension are already
  documented above as a known hard case; naca633418/633618 (6-series,
  thin) and s1223 (a cambered low-Re section) failing to converge at a
  specific AoA rather than every AoA fits the spec's own framing
  ("likely unsteady flow/stall... needs interpretation, not a fixed
  retry") -- real, reproducible candidates for the troubleshooter, not
  noise.
- e423, fx60100, fx60126, fx63137: first-AoA `"crashed"` status (an
  actual exception in the orchestrator's Stage 2/3/4 per-AoA try/except,
  not a Stage 3 convergence classification) -- all four happened on the
  exact first batch of 4 workers dispatched immediately after the
  manual interrupt-and-restart above. **Investigated, not just
  observed:** re-ran these same 4 airfoils under the identical
  max_workers=4 burst condition afterward (no interrupt this time) and
  got zero crashes -- 3 of the 4 fully converged 5/5, the 4th showed
  ordinary `non_converged` (not `"crashed"`) on the AoAs that had
  crashed before. No orphaned `wsl.exe`/`wslhost` processes were found
  afterward either. This points at the deliberate `TaskStop` interrupt
  itself (likely transient contention from the just-killed run's
  in-flight WSL/gmsh subprocesses) as the trigger, not a persistent
  `max_workers=4` concurrency bug -- but it is not fully root-caused
  (the interrupt was via TaskStop on the parent bash task, whose effect
  on already-spawned `ProcessPoolExecutor` workers and their own child
  `wsl.exe`/`run_isolated` processes wasn't traced directly). Treat
  killing an in-flight parallel batch as a real, if narrow, risk until
  this is traced further -- prefer letting a batch finish or checking
  for orphaned processes after an interrupt before immediately
  restarting.

**Manifest bug found and fixed during this investigation:**
`_summarize_for_manifest` (added to fix the earlier manifest-bloat
issue, see below) was trimming per-AoA diagnostic error strings along
with the bulky pressure/stress arrays -- meaning the crashed-AoA
investigation above initially had no persisted error text to look at
and had to be reproduced from scratch. Fixed: manifest entries now keep
a `per_aoa_errors` list (small strings, `None` for non-crashed AoAs)
alongside `per_aoa_status`, covered by
`test_manifest_entry_excludes_bulky_per_aoa_arrays_but_keeps_error_text`.

**Not yet decided:** whether `max_workers=4` is the right production
default (16 cores support it comfortably; the open question is whether
routine batches should push higher, e.g. matched closer to physical
core count, or whether OpenFOAM/CalculiX's own per-solve resource use
caps the useful ceiling below the core count -- not yet measured at
`max_workers>4`). Also flagged, not yet acted on: `foamRun` currently
runs with no `decomposePar`/MPI parallelization within a single case at
all (a stale constraint from the old Ansys Student-license era, not a
real one now) -- worth measuring whether per-solve parallelism or more
concurrent airfoils gives better throughput on this hardware before
committing to one axis.

### Troubleshooter agent investigation (2026-09-15): no agent-shaped work found yet

Before building the spec's Stage 0/1 Troubleshooter agent, deeply
investigated the 8 non-5/5-AoA airfoils the 35-airfoil run surfaced, per
the spec's own rule that agent infrastructure needs real failures to
characterize first. Conclusion, backed by direct experiments rather than
inspection alone: **this specific failure mode is not agent-shaped, and
the real Stage 0/1 failure modes found afterward were closeable with
plain deterministic code, not agent judgment.**

**The AoA non-convergence cluster (ah79100c, naca633418, naca633618,
s1223) is a Stage 2/3 numerics finding, explicitly out of the spec's
agent scope, not a mesh-quality issue:**
- Signature across all affected AoAs: pressure-equation residual
  frozen at 1.1-3.9x the 1e-5 threshold while Ux/Uy/nuTilda are already
  converged to <1e-6 -- a narrow numerical plateau, not divergence, not
  general unsteadiness. Failures cluster at low/moderate AoA (-2 deg to
  +10 deg), not the spec's assumed high-AoA-stall mechanism -- 3 of 4
  tested airfoils converge cleanly at +14 deg.
- **Falsified experimentally, not just suspected:** extending `endTime`
  from 1000 to 3000 iterations on the worst case (ah79100c, AoA=-2 deg,
  p=3.91e-5) moved the residual by 0.006% -- ruling out "still
  decreasing, just needs more time" (the one deterministic Stage 3 fix
  the spec already sanctions). Raising `nNonOrthogonalCorrectors` from
  0 to 2 on the same case made it *worse* (p went to 1.09e-4) -- ruling
  out the obvious mesh/numerics lever too.
- Mesh-quality correlation tested and rejected: non-orthogonality
  ranged 59.3-81.4 deg across the 4 airfoils with no consistent
  relationship to which/how many AoAs failed (naca633618 at the
  *lowest* non-orthogonality, 59.3 deg, still failed at the same AoA as
  naca633418 at 74.5 deg). All 4 meshes pass Gate #1 with real skewness
  margin (2.28-3.59 vs ~4.0 threshold) -- not the documented
  bl_size-vs-skewness conflict.
- Per the spec's own rule ("if a case fails for a reason that isn't
  'geometry/mesh was bad', log and exclude, don't hand to an agent"):
  these 13/175 AoAs (the 8 airfoils' non-converged/crashed points) are
  correctly excluded from the training set as-is. No code change from
  this finding -- the existing non-converged handling is already
  correct.

**Then probed Stage 0/1 directly with adversarial synthetic geometries**
(8 cases explicitly designed to break meshing: 0.5%/1% extreme-thin,
40%-thick, m=9-12% extreme-camber, a cusped-TE/high-camber/thin
combination, a genuinely self-intersecting contour, consecutive
duplicate points) -- **5 passed cleanly** (including the exact thin/
high-camber pathologies the spec named as the agent's reason for
existing), and the 3 that failed were all closeable deterministically:
- **`duplicate_points.dat`** (consecutive duplicate coordinate, a
  plausible digitization artifact): crashed Stage 1's `CubicSpline`
  resampling with a raw scipy `ValueError` ("x must be strictly
  increasing"). Fixed: `load_airfoil` now dedupes consecutive duplicate
  points (`pipeline/stage0_geometry_loader.py::_dedupe_consecutive`).
- **`self_intersecting.dat`** (genuinely self-crossing contour, not
  just scrambled ordering): previously passed through Stage 0 silently
  and burned a full 3-attempt Stage 1 gmsh retry ladder before failing
  with a much less legible gmsh error. Fixed: `load_airfoil` now
  rejects a self-intersecting contour with a clear `ValueError` before
  Stage 1 is ever called (`_polygon_has_self_intersections`, promoted
  from the test suite's own check -- both now share one implementation).
- **`extreme_thick_t40pct`** (40% thickness, not a realistic airfoil but
  a real gmsh boundary-layer-extrusion failure): gmsh's BL offset
  self-overlaps near the LE's tight curvature ("Edge not recovered" /
  "intersections in the 1D mesh"), *before* checkMesh ever runs. The
  existing blind retry ladder (`generate_mesh`) *increases* `bl_size` on
  every failure -- confirmed empirically that this makes this specific
  failure worse. Fixed: retry now checks the gmsh output for this exact
  signature (`_is_bl_self_intersection_failure`) and *reduces* `bl_size`
  instead for that case only, keeping the existing increase-ladder for
  any other/unrecognized failure. Verified for real (not just via the
  mocked unit tests): the same 40%-thick geometry now passes Stage 1
  automatically, and the resulting mesh clears Gate #1 with real margin
  (skewness 1.05, non-orthogonality 56.0 deg).
- All fixed cases re-verified against the full existing Stage 0/1/
  integration suite (342 tests total) plus new tests covering the fixes
  themselves (`test_consecutive_duplicate_points_are_deduped`,
  `test_self_intersecting_contour_is_rejected`,
  `test_bl_self_intersection_signature_triggers_reduced_bl_size_retry`,
  `test_generic_failure_signature_keeps_existing_increase_ladder`) --
  all pass, no real UIUC geometry is newly rejected. New fixtures
  (`duplicate_points.dat`, `self_intersecting.dat`) committed to
  `tests/fixtures/generate_fixtures.py` for reproducibility.

**Conclusion: no agent-shaped work has been found yet.** Every real
Stage 0/1 failure mode surfaced so far -- across a real 35-airfoil batch
and 8 deliberately adversarial synthetic geometries -- was closeable
with a small, targeted, deterministic fix (a dedupe, a rejection check,
an error-signature-keyed retry direction), not judgment under ambiguity.
Per the spec's own conditioning ("nothing for an agent to recover from
until real failures exist to characterize"), building LLM-judgment
agent infrastructure now would be solving a problem that doesn't exist
yet. Next real test of this: a broader UIUC pull (beyond the 35 curated
`tests/fixtures/real_uiuc/` files) or the synthetic-geometry generation
work (NACA-parametric/CST, see above) actually run through the
pipeline -- either could still surface a genuinely ambiguous Stage 0/1
failure the fixes above don't cover, which is what would justify the
agent layer. Investigation artifacts are no longer deleted after use --
`.orchestrator_runs/`, `.investigation_out/`, and similar are now
`.gitignore`d rather than removed, after this investigation lost the
35-airfoil run's case logs to a premature cleanup and had to spend a
real WSL re-run recovering data it already had.

**Correction to the conclusion above, same night (2026-09-15):** the
user pushed back on "no agent-shaped work" -- correctly. What actually
happened tonight *was* diagnose-react-fix, done live, three times; it
only looked like "no agent needed" because each diagnosis got turned
into a hardcoded pattern match immediately, closing the door behind it.
That's real and valuable (the next occurrence of the *same* signature
is now free), but it only defers the question: the next genuinely new
gmsh failure signature (and one *will* show up, scaling past 35
airfoils) has no hardcoded branch waiting for it, and the pipeline goes
back to "blind retry 3 times, then give up." So the actual scope isn't
"no agent needed" -- it's "build the bounded version of what the human
just did, automatically, for whatever shows up next." See "Stage 1
Troubleshooter agent" below for what got built from that reframing.

### Stage 1 Troubleshooter agent (`pipeline/troubleshooter.py`, 2026-09-15)

Built per the correction above: `diagnose_mesh_failure()` shells out to
the local `claude` CLI in print mode (`claude -p --output-format json
--json-schema ...`), giving it the gmsh failure output, current mesh
params, and geometry stats, and gets back a structured
`{reasoning, bl_size, bl_layers, bl_ratio}` decision. Wired into
`generate_mesh`'s retry loop as a third branch: known signatures
(`_is_bl_self_intersection_failure`) still get the deterministic fix
first; an unrecognized signature, on a non-final retry, calls the
troubleshooter instead of the blind ladder -- opt-in
(`enable_troubleshooter=False` by default, matching `run_batch`'s own
opt-in default for `max_workers>1`). Every call is logged to a JSONL
file (`log_troubleshooter_call`) with inputs, proposed params,
reasoning, and outcome, so a recurring pattern is future material for
promoting into a hardcoded rule, the same way tonight's own adversarial
findings became `_is_bl_self_intersection_failure`.

**Billing, confirmed empirically, not assumed:** `claude auth status`
reports `authMethod: "claude.ai"`, `subscriptionType: "max"`, no
`ANTHROPIC_API_KEY` set -- `claude -p` calls bill against Claude
subscription plan usage, not separate metered API charges. Do not add
`--bare` to this call: it explicitly requires an API key and never
reads OAuth/subscription auth.

**Two real Windows-specific bugs found and fixed before this worked at
all** (both would have silently misfired in production, not just failed
loudly):
1. `subprocess.run(["claude", ...])` raised `FileNotFoundError:
   [WinError 2]` -- `claude` resolves to a `.CMD` npm shim
   (`shutil.which` confirms), which `CreateProcess` can't exec directly.
   Fixed with `shell=True` (Python still applies correct Win32 argv
   quoting via `list2cmdline` before handing the joined command to
   `cmd.exe`).
2. Passing the prompt as an argv element (even under `shell=True`)
   silently corrupted/failed real calls once the prompt embedded real
   gmsh output -- multi-line text, `[100%]`-style percent signs, ANSI
   escapes (`\x1b[1m\x1b[31m`). `cmd.exe` re-tokenizes an already
   list2cmdline-quoted string with its own metacharacter rules (`%`,
   `^`, `&`, `|`, `<`, `>`) on top of Win32 escaping -- not safe for
   arbitrary content. Fixed by passing the prompt over stdin
   (`input=prompt`, no prompt in argv) -- only the fixed flags
   (`-p`, `--output-format`, `--json-schema`) ever pass through
   `cmd.exe`'s parser.

**Real generalization test, not a re-run of an already-fixed case**
(`test_real_troubleshooter_diagnoses_novel_spike_failure`, gated behind
`ADE_RUN_LLM_TESTS=1`): found a genuinely novel, unrecognized failure by
deliberately searching for one -- an isolated single point spiked 0.35
units outside a naca0012's normal envelope produces `"Could not find
extruded node ... in surface N"`, distinct from the already-fixed `"Edge
not recovered"` signature (confirmed: this new signature's full gmsh
output does NOT contain `"Edge not recovered"`, so the deterministic
branch correctly skips it and routes to the troubleshooter). Real,
non-mocked runs against this case: the troubleshooter is invoked
correctly every time, gives real per-attempt reasoning grounded in the
actual error text and geometry stats (not templated -- verified this
directly by reading the reasoning text from 3 separate real attempts,
each one different and each one correctly identifying "thick/high-
curvature geometry, BL offset too aggressive for the local radius of
curvature" from the numbers given), and proposes genuinely different
parameters each attempt rather than repeating a guess. It did **not**
fully resolve this particular case within 3 real attempts -- this is a
deliberately extreme, unrealistic synthetic geometry (no real airfoil
looks like a single 0.35-unit spike), chosen specifically to be
unrecognized by the deterministic rules, not chosen to be easy. The test
asserts the honest bar (real invocation, grounded per-attempt reasoning,
genuinely differing proposals) rather than outright success on a
contrived worst case.

**Not yet done:** no promotion pipeline from the troubleshooter's JSONL
log to a new hardcoded `_is_*_failure` rule -- that stays a manual
step (as it was for tonight's 3 findings) until there's enough real
log volume to see what's worth automating.

### DeepONet training pipeline (`deeponet/`, 2026-09-15)

Built and trained end-to-end on the real 34-airfoil dataset (35 airfoils
attempted, `whitcomb` excluded -- Stage 1 meshing failure, see the
36-airfoil regen note below). Scoped to `Cp(s)` only, not the FEA
stress field -- confirmed the spec's `stress_field` output has no
node-id-to-(x,y,z) mapping anywhere in the pipeline (STATUS's own Stage
9 details above: "keyed by CalculiX's internal shell-expansion node
ids... no cross-reference is needed" -- true for the schema's original
purpose, but means a DeepONet trunk has no real query *location* for
stress). `pressure_vs_arc_length`'s `s` is exactly the trunk input a
DeepONet wants, already stored per point.

- `deeponet/dataset.py`: loads every `.h5`'s `pressure_vs_arc_length`
  (non-converged AoAs excluded, not fabricated) plus geometry reloaded
  from the `.h5`'s own `source_file` path (the `.h5` itself doesn't
  store raw coords) resampled to a fixed 63 points via Stage 1's own
  `_resample_cosine` (so branch-input geometry encoding matches what the
  CFD mesh itself saw). Split by **whole airfoil**, not by point or by
  (airfoil, AoA) sample -- holding out points from an airfoil the model
  saw elsewhere on its own surface says nothing about generalizing to an
  unseen shape. `Normalizer` fits only on the train split. Covered by
  `tests/test_deeponet_dataset.py` (6 tests, run against the real
  regenerated dataset, skipped if it's absent -- not faked, since the
  point is verifying real schema shapes).
- `deeponet/model.py`: standard branch/trunk dot-product DeepONet (Lu et
  al. 2021) -- branch encodes (resampled geometry, AoA), trunk encodes
  query arc-length `s`. Chosen over FNO per the build spec's own
  reasoning (pressure curves vary in length/resolution per airfoil; a
  fixed-grid FNO input would need lossy resampling that DeepONet's
  arbitrary-query trunk avoids).
- `deeponet/train.py`: CPU-only (confirmed: `torch.cuda.is_available()`
  is `False` on this machine, `torch==2.11.0+cpu`), held-out-airfoil
  validation loss as the real metric (train loss alone looks good
  regardless of whether the model generalizes to an unseen shape, given
  only ~34 airfoils). Checkpoint + normalizer + train/val airfoil split
  saved to `deeponet/checkpoints/`.

**Two full real training runs**, both `python -m deeponet.train --h5-dir
.orchestrator_runs/real_uiuc_35 --epochs 800` (CPU-only, ~5 min each):

- **Run 1, 34 airfoils** (27 train / 7 held out): train loss
  0.985 -> 0.062; held-out val loss 1.028 -> 0.376 (best 0.322 at epoch
  790) -- val loss tracked train loss down essentially the whole run,
  a clean generalizing result.
- **Run 2, 41 airfoils** (after downloading 100 more real UIUC airfoils
  from the public database and re-running the orchestrator -- see
  below; 33 train / 8 held out: a18, a18sm, a63a108c, goe398, m6, mh60,
  naca23012, rg15): train loss 0.955 -> 0.066 (still monotonic); but
  **held-out val loss bottoms out at epoch ~102 (0.577) then rises and
  oscillates for the remaining ~700 epochs (0.69-1.09 range, ends at
  0.890)** -- confirmed by sampling val_loss every 50 epochs across the
  full run, not a one-point artifact. This is real overfitting past
  ~epoch 100-150, not the "mild signal" the run-1 note speculated about
  -- run 2's larger, more diverse (less curated) corpus made it visible
  where run 1's smaller run either hadn't reached that regime yet or
  got lucky with its particular train/val split.

**Fixed same night:** `train.py` was only saving the *final*-epoch
checkpoint, not the best-val one -- `deeponet_cp.pt` was silently the
epoch-800 (overfit) weights from run 2, not the better-generalizing
epoch-102 ones, with nothing in the filename or checkpoint itself to
warn a future reader. Fixed: `train.py` now tracks `best_val_loss`
across the run, saves that state dict as `deeponet_cp.pt` (the final-
epoch weights are kept too, explicitly renamed
`deeponet_cp_final_epoch.pt`, for a caller who genuinely wants them),
and records `best_epoch`/`best_val_loss`/`final_epoch` in
`normalizer.json` so the gap between them is visible without having to
cross-reference `history.json` by hand. Re-ran training with the fix
(**run 3**, same 41-airfoil dataset, same 33/8 split): confirms the
overfitting pattern is real and consistent across runs, not a one-off
-- best val_loss this time was even earlier, **epoch 29** (val_loss
0.732), vs. run 2's epoch 102. `deeponet_cp.pt` now correctly holds
that epoch-29 checkpoint, not epoch 800's.

**Immediate next steps for training:** (1) more airfoils and/or
regularization (dropout, weight decay, or simply a smaller model) --
41 airfoils is still thin for a DeepONet by ML-dataset standards, and
overfitting emerging this early (epoch ~30-100 of 800, consistent
across two independent runs) says the model has more capacity than the
data supports right now; (2) proper early stopping in `train.py`
(currently always runs the full requested epoch count and relies on
best-checkpoint tracking after the fact -- fine for a first pass, but
wastes ~700 epochs of compute once overfitting has clearly set in);
(3) hyperparameter sweep (branch/trunk width, `p` embedding dim) only
after (1) and (2), so a sweep isn't just measuring which config
overfits fastest.

### Corpus scale-up: 41 airfoils (2026-09-15, continued)

Per explicit user request ("continue running the pipeline to train the
DeepONet"), downloaded 100 additional real airfoils from the public
UIUC database (`data/airfoil_downloader.py`'s existing scraper, source
`m-selig.ae.illinois.edu` -- the same academic source the original 35
fixtures came from; confirmed with the user before running, since a
bulk external download needs explicit sign-off). Verified all 100
parse through Stage 0 before committing to a full pipeline run: 99/100
clean, one (`30p-30n`, a NASA multi-element high-lift configuration --
slat+main+flap as one file, not a single closed contour) correctly
rejected by tonight's own self-intersection guard -- real validation of
that guard against genuinely unexpected real-world data, not just the
adversarial synthetic cases it was built against. Two more files from
the same family (`30p-30n-flap`, `30p-30n-slat`, the isolated flap/slat
elements alone) passed Stage 0 but were correctly excluded later
(`geometry_mesh` failure) -- the pipeline's layered gates catching what
each individual check didn't.

Ran the orchestrator over the combined 134-airfoil `real_uiuc/`
directory (`resume=True`, so the 34 already-done airfoils were skipped,
not re-run) at `max_workers=4`. **The batch was killed partway through
by the OS for low system memory** -- not a code bug: checked afterward,
no orphaned `python`/`wsl.exe` processes remained, and the top memory
consumers at kill time were the browser (several hundred MB each,
multiple tabs) and the WSL VM itself, not a leak in the pipeline. 45
airfoils had completed by then (41 success + the 3 excluded `30p-30n*`
files + 1 other). **Given this session was also well past its allotted
time budget at this point, chose not to retry/relaunch the remaining
~90 airfoils** -- retrained on the 41 available instead (see above) and
stopped there. Resuming the rest of the 100-airfoil download is a
clean, mechanical next step (`resume=True` already means it picks up
exactly where it left off) whenever there's a session with memory
headroom and time for it.

### Demo dashboard (Artifact, 2026-09-15)

Built per the explicit instruction that this "does not depend on
training succeeding" and "must not be last-and-rushed" -- built from the
`.h5` files directly (`scripts/build_dashboard_data.py` extracts a
compact JSON: per-airfoil Cp(s) curves subsampled to ~120 points,
Cl/Cd/Cl_xfoil/Cd_xfoil per AoA, max von Mises + Gate #4 residual per
converged AoA, geometry outline), embedded into a static HTML/JS page
(`scripts/dashboard.html` + `scripts/embed_dashboard_data.py` ->
`scripts/dashboard_publish.html`) since an Artifact can't read local
files at runtime.

IBM Plex Mono/Sans pairing (technical/engineering register, matching the
subject), warm-neutral light palette + a matching dark palette (both
verified by screenshot -- see below), airfoil list with status-pill
filtering (clean/partial/failed), per-airfoil detail: shape outline, AoA
toggle, Cp(s) chart, Cl-vs-AoA (CFD solid + XFOIL dashed) and Cd-vs-AoA
charts (hand-rolled inline SVG, no charting library), per-AoA summary
table with FEA results. All real data, no placeholders.

**Verified visually, not just by code review** -- no Claude-in-Chrome
extension available in this environment, so installed Playwright +
Chromium (`python -m playwright install chromium`) for one-time local
screenshots (light mode, dark mode, both against real data) rather than
skipping the check. Caught and fixed one real bug this way: the chart
y-axis label precision was fixed at 1 decimal, which collapsed a
narrow-range axis (Cd: 0.02-0.19) to two identical "0.1" tick labels --
fixed with a range-aware decimal-count function
(`tickDecimals()` in the chart code).

Published as a Claude Artifact (kept in sync with `url=` across 3
republishes as the dataset grew from 8 -> 12 -> 34 airfoils during the
same regen run) rather than a local-only file, since the explicit goal
is "a demo and show to people" -- a shareable link, not a screenshot.

### 35-airfoil dataset: lost, then regenerated for real (2026-09-15)

Real mistake, caught and fixed, worth recording plainly: the original
35-airfoil `.h5` dataset (the "Real multi-airfoil orchestrator runs"
section above) was deleted by this session's own `rm -rf
.orchestrator_runs` cleanup after the troubleshooter investigation --
that dataset *was* the DeepONet training set, and nothing downstream
(training, dashboard, demo) exists without it. Caught before any
training/dashboard work was built on top of it, and the batch was
re-run for real (`max_workers=4`, ~1321s wall clock, unattended) rather
than faked or skipped. `.orchestrator_runs/` is `.gitignore`d, not
deleted, from here on (see the troubleshooter-investigation section
above, which made the same mistake once already the same night).

This regen also produced one small, real, unexplained data point:
`naca633418` converged 5/5 AoA this time, vs. 4/5 (one `non_converged`
at AoA=+2 deg) in the original run with identical inputs/params --
OpenFOAM's SIMPLE solve is not literally bit-deterministic run-to-run
under real machine/timing conditions apparently including whether a
given borderline case crosses the 1e-5 threshold. Not investigated
further tonight (not on the critical path); worth knowing if a future
session sees a case flip status between two runs and wonders whether
something broke.

Full current suite: 494 passed + 1 xfailed as of the last full run
before tonight's Stage 0/1 fixes and new modules (`pytest tests/`, ~9
min -- most of it is Stage 1/2/3/4 shelling out to WSL for
`gmshToFoam`/`checkMesh`/`foamDictionary`/`foamRun`/`foamToVTK`/`xfoil`;
Stage 8 also shells to WSL for `ccx`). Not re-counted exactly after
tonight's additions (Stage 0/1 dedup + self-intersection fixes,
`pipeline/troubleshooter.py`, `deeponet/`) -- a full `pytest tests/`
run is known to intermittently hit the documented pytest-only WSL
exhaustion issue below at this suite's size, so treat any single full-
suite number as noisy; per-module runs (`test_stage0_geometry_loader.py`,
`test_stage1_mesh.py`, `test_troubleshooter.py`, `test_deeponet_dataset.py`)
all pass cleanly in isolation, which is what was actually verified
tonight.
Breakdown (as of the last exact count): 258 (Stage 0: 38 synthetic + 220 real-UIUC) + 96 (Stage 1: 26 +
70 real-UIUC) + 35 (Stage 2) + 12 (integration) + 13 (Stage 3) + 16
(Stage 4) + 13 (Stage 5) + 11 (Stage 6) + 11 (Stage 7) + 10 (Stage 8) +
10 (Stage 9) = 494 passed + 1 xfailed, plus tonight's new/changed
coverage: 42 (Stage 0, +4 adversarial) + 30 (Stage 1, +2 retry-direction)
+ 15 (orchestrator) + 13 (troubleshooter, +1 real/gated) + 6 (deeponet
dataset) not yet folded into a fresh full-suite total.

### Stage 9 details

HDF5 (h5py) output matching the spec's Final Output Schema literally --
`metadata` group (attrs + `aoa_sweep_deg` dataset) plus one `aoa_NN/`
group per AoA, each with a `cfd/` group (always present) and an `fea/`
group (present only when that AoA's CFD converged). "Failed cases
recorded explicitly, never silently dropped" is enforced, not just
documented: the module raises `ValueError` if an `fea` result is passed
for a non-converged AoA (a contract violation that would otherwise
silently corrupt training data), and a dedicated test confirms a
non-converged AoA's `cfd/` group still exists in the file with its real
status, rather than being omitted. `stress_field` is stored as two
parallel arrays (node ids, von Mises) since HDF5 datasets are typed
arrays, not maps -- keyed by CalculiX's internal shell-expansion node
ids (Stage 8's own ids), not Stage 6/7's mesh node ids; no cross-
reference is needed for the schema's purposes.

### Full single-airfoil pipeline (Stages 0-9) confirmed for real

Ran the complete pipeline, real data throughout (no synthetic/fixture
shortcuts), for naca0012 across the spec's locked full AoA sweep
(-2, 2, 6, 10, 14 degrees, Re=5e5): all 5 converged, Stage 5/6 geometry
+ mesh shared once across the sweep (AoA-independent, per spec), Stage
9 wrote a real 1.93 MB `.h5` file with all 5 AoAs present and a full
23,480-point stress field each. Every physical trend is sane:

| AoA | Cl | Cd | max von Mises | Gate #4 residual |
|---|---|---|---|---|
| -2 deg | -0.186 | 0.0412 | 208 kPa | 1.45% |
| 2 deg | 0.188 | 0.0412 | 208 kPa | 1.44% |
| 6 deg | 0.546 | 0.0610 | 608 kPa | 1.17% |
| 10 deg | 0.860 | 0.0997 | 959 kPa | 1.20% |
| 14 deg | 1.104 | 0.1560 | 1233 kPa | 1.26% |

Cl rises monotonically and roughly linearly with AoA (matches thin-
airfoil theory's ~0.11/deg slope reasonably well over this range,
consistent with the spec's own "quick-reference" sanity table); Cd and
peak stress both rise with AoA too, as expected (more lift -> more
load -> more stress). Gate #4 stayed comfortably under 1.5% at every
AoA, not just the one case checked during Stage 8 development -- this
is the single-airfoil confirmation the multi-airfoil scale-up work was
explicitly gated on.

### Multi-airfoil (3 airfoils) full pipeline confirmed for real

Before starting any multi-airfoil orchestration work, directly tested
the actual thing that mattered: does running more than one airfoil's
full 0-9 chain, in one process, actually work, or does the gmsh/WSL
history in this project mean it silently degrades at the 2nd/3rd
airfoil? Ran 3 geometrically diverse real airfoils (naca0012 thin
symmetric, naca2412 cambered, naca0021 thick symmetric) back-to-back in
one process, each through the full real 5-AoA sweep (0-9, no shortcuts)
-- **15/15 real CFD+FEA solves completed, all 3 `.h5` files written,
zero WSL/gmsh corruption errors anywhere in the run.** Gate #4 stayed
in the same 1.2-1.8% band as the single-airfoil case at every AoA except
one: naca2412 at AoA=-2 deg showed 6.7% -- not a physics failure, just
relative-tolerance math blowing up because `Cl` there is nearly zero
(-0.0009, a near-zero-lift condition for a cambered section at slightly
negative incidence) -- the absolute residual is still tiny, only the
*ratio* to a near-zero applied load is large. Worth an absolute-residual
floor alongside the relative check if Gate #4 needs to run unattended at
scale (not yet added -- noted here, not hidden).

Also stress-tested `run_isolated()`'s isolation mechanism directly (40
consecutive gmsh-isolated calls in a plain script, more than the ~35
that triggered a pytest-only issue): every `wsl.exe` check after every
5 calls succeeded, confirming no real per-call resource leak in
production usage. `pipeline/_gmsh_isolation.py::run_isolated()` was
still hardened defensively (explicit `queue.close()`/`join_thread()` and
`proc.close()` after every call) as cheap insurance for 1000-airfoil
scale, even though nothing pointed at a real leak.

### Stage 8 details

Real, non-trivial investigation was required to get Gate #4 right --
worth recording since it's the kind of finding that would otherwise get
rediscovered painfully later:

1. **CalculiX shell pressure DLOAD investigation.** Built 10 progressively
   simpler standalone probe decks (`ccx` run directly via WSL, outside
   the pipeline) before writing any pipeline code, because getting the
   sign/magnitude convention wrong here is exactly the "plausible-looking
   wrong stress field" Gate #4 exists to catch. Found and ruled out
   several false leads (P vs P1 label -- no difference; suspected NSET
   under-coverage in `*NODE PRINT` totals -- not the cause) before
   isolating the real, confirmed CalculiX behavior: **a load applied
   directly at an already-BOUNDARY-constrained node/DOF does not fully
   appear in that node's printed reaction** (cross-validated against the
   `.frd` file's independent `FORC` record -- not a print-formatting
   bug, a real property of solving a system with a prescribed DOF). This
   was large (up to 100%) in the tiny toy decks used to investigate it,
   because the fixed edge was most of the whole mesh; for a real
   structural mesh (thousands of elements, root being one edge among
   many) it's a small effect -- confirmed directly on the real mesh:
   **2.8% residual**, comfortably under Gate #4's tolerance, not a
   coincidental near-miss.
2. **Two real, separate bugs caught by empirical verification before
   the test suite was declared green:**
   - `.frd` stress parsing: CalculiX's fixed-width columns run together
     for 4+ digit node IDs (`"3808-2.67249E+04"`, no space before a
     negative value) -- naive `.split()` breaks; fixed with explicit
     column-offset slicing (3-char key + 10-char node id + six 12-char
     value fields, confirmed empirically against known-good short lines
     first).
   - `.dat` reaction-force regex: `[-\d.eE]+` doesn't include `+`, so it
     couldn't match the exponent sign in `4.014019E+00`, silently
     truncating the match. Both bugs were caught because Gate #4 kept
     failing/erroring until they were fixed -- exactly the value of
     writing the equilibrium check first and refusing to loosen it.
   - **The actual DLOAD sign convention bug**: a single-element probe's
     `.frd` displacement output was misread verbally on first pass
     (assumed positive P deflects opposite the connectivity normal; the
     printed `D3` was actually positive, i.e. the SAME direction).
     Building Stage 8 on that wrong assumption produced a Gate #4
     failure with ~197% residual -- almost exactly double, the signature
     of a clean sign flip rather than a real physics error (magnitudes
     agreed to ~3%, only the sign was backwards). Corrected to
     `P = -pressure_pa * sign`; re-verified against the real mesh
     afterward (2.8% residual, not just "test passes").
3. **Material/thickness**: generic aluminum (E=70 GPa, nu=0.33,
   rho=2700 kg/m^3) and uniform 2mm shell thickness across skin/spar/rib,
   both explicit placeholders per the spec's own allowance, both wired
   as configurable parameters (confirmed: halving thickness measurably
   increases peak stress, not a dead knob).
4. **BC**: cantilever, full 6-DOF fixity at every node with z=0 (root),
   tip free -- the obvious, standard choice for this structure, not a
   genuinely ambiguous call.

### Stage 7 details

Confirmed for real, full chain Stage 0->1->2->3->4->5->6->7 for naca0012
with real CFD data (Cl=0.371, Cd=0.0486): Stage 7 mapped pressure onto
20,640 skin elements, resultant force (0.83, 38.58, ~0) N. Hand-calc
cross-check: `L = Cl * q * chord * span = 0.371 * 34.45 * 1 * 3 ≈
38.35 N`, matching the mapped lift component within ~0.6%. The mapped
drag component (0.83 N) is much smaller than `Cd * q * chord * span ≈
5.02 N` -- expected, not a bug: Stage 7 only maps *pressure* (per its
contract), while the CFD `Cd` includes viscous shear, which dominates
drag for this thin, attached-flow case.

This is also where the spec's **Must-Pass Gate #3 (load conservation,
~1%, non-negotiable)** lives: `test_gate3_load_conservation` computes an
independent hand-calc reference (a standalone trapezoidal integration of
Cp(s) around the 2D polygon, written fresh in the test file, NOT calling
`pipeline.stage4_postprocess.pressure_integrated_cl_cd` -- checking the
pipeline against itself would defeat the point of an independent check)
and compares it against Stage 7's actual mapped 3D resultant. First
attempt failed at ~2600% off due to a real bug *in the test's hand-calc*
(forgot to multiply by dynamic pressure q -- the raw polygon integral of
a dimensionless Cp is a force *coefficient*, not yet a force). After
fixing that, a residual ~2.9% gap remained -- traced to the hand-calc
using a rectangle-rule (per-vertex) Cp sampling on the original 118-point
digitized polygon, coarser than the actual pipeline's true
piecewise-linear interpolation integrated over thousands of fine mesh
elements. Switching the hand-calc to trapezoidal (segment-midpoint-
averaged) sampling of the *same* underlying curve brought the two
independently-computed numbers to agreement within 3e-12 (floating-point
noise) -- confirming Gate #3 passes for a real, non-trivial reason, not
because the tolerance was loosened to fit.

Two new real decisions this stage introduces:
- `rho_air` (default 1.225 kg/m^3, ISA sea level) -- the CFD solver runs
  at rho=1 (Cp is dimensionless), but mapping onto a real, meter-scale
  shell structure needs physical Pa: `p_Pa = Cp * 0.5 * rho_air *
  U_inf^2`. No real air density existed anywhere in the pipeline before
  this stage.
- Only "skin" elements (Stage 6's ELSET) receive direct aerodynamic
  pressure; spar/rib elements get none directly (verified explicitly:
  `test_non_skin_elements_get_no_direct_load`).

Because the pressure-to-load mapping is linear (interpolate + multiply
by a constant), and Stage 8's CalculiX solve will be linear-elastic
(stress scales proportionally with applied load), the suite verifies
both the "slope" and "intercept" of that relationship directly:
scaling the entire input Cp(s) curve by k scales every mapped element
pressure by exactly k (`test_pressure_scales_linearly_with_input_cp`),
and an all-zero Cp curve maps to exactly zero pressure everywhere, not a
hidden nonzero offset (`test_zero_cp_curve_gives_zero_pressure_everywhere`)
-- the physically ideal no-load case, and the cleanest possible
reference point precisely because of that linearity.

**Planned for Stage 8, not built yet:** an XFOIL-equivalent cheap
independent check for the structural side -- closed-form beam theory
(bending moment from the integrated pressure load, max stress via M*c/I
on the spar cross-section) as a fast, non-FEA cross-check against
CalculiX's stress output, the same role XFOIL plays for Gate #2.

### Stage 6 details

Confirmed for real, full chain Stage 0->1->2->3->4->5->6 for naca0012
(single-process, since Stage 5/6 come after all WSL calls for that one
airfoil -- the gmsh/WSL cross-process issue below only bites the *next*
airfoil's Stage 1, not this one): CFD converged (Cl=0.372, Cd=0.0486),
Stage 6 produced 732 skin + 12 spar + 21 rib mesh surfaces, valid
CalculiX `.inp` written.

Two real engineering decisions Stage 6 owns, both empirically verified
(not assumed) before locking:
- **gmsh's raw `.inp` export has no shell-element option** -- it types
  every 2D surface as `CPS3`/`CPS4` (2D continuum), never `S3`/`S4`
  (shell), which CalculiX's `*SHELL SECTION` requires. Stage 6
  post-processes the written file to fix element types and drops the
  boundary `T3D2` line elements gmsh also writes.
- **Conformal fragmentation** (`occ.fragment(surfaces, surfaces)`,
  self-fragmenting the whole imported assembly) is what actually fuses
  ribs/spars to the skin at every junction, not just root/tip -- this is
  explicitly where Stage 5's deferred "interior ribs/spars are coincident
  but not CAD-fused" gap gets resolved. Verified directly (not assumed
  from the mechanism working on a toy case): the shipped test suite
  parses the real `.inp` output and asserts there are no two distinct
  node IDs at coincident coordinates anywhere in the mesh -- the general,
  artifact-level proof that fragmentation actually welded every junction,
  not a hand-picked spot check.

Mesh size (`mesh_size`, default 0.05 chord units) is left as a tunable
parameter, not a locked spec value -- there's no single correct
efficiency/accuracy tradeoff yet, so the test suite verifies the
mechanism (finer size -> strictly more elements; a sane element-count
range at the default) rather than pinning one true value. Calibrating it
is future work, same status as Stage 1's `bl_size`.

**Stage 5 has no data dependency on Stages 1-4** -- per the spec, it
branches directly off Stage 0's coords (span/spar/rib params only), in
parallel with the CFD branch (Stages 1-4); the two branches only merge at
Stage 7 (load mapping needs both Stage 4's pressure output and Stage 6's
mesh). So there is no Stage-4-to-5 "seam" to integration-test the way
`test_pipeline_integration.py` does for Stages 0-1-2. Verified for real
instead by running `generate_structural_geometry` directly (not just via
the fixture-based unit suite) against 5 diverse real UIUC geometries
(naca0012/2412/6412/0021, ah79100c -- including shapes that stressed
Stage 1's mesh-quality gate) -- all produced valid STEP/`.brep` output
with the expected panel/rib counts.

Also ran the **full merged chain** (Stage 0->1->2->3->4->5, real CFD
solve + real XFOIL cross-check + real Stage 5 geometry, not fixtures) for
those same 5 airfoils, each fully converged:

| Airfoil | Stage3 | CFD Cl / Cd | XFOIL Cl / Cd | XFOIL converged | Stage5 panels/ribs |
|---|---|---|---|---|---|
| naca0012 | converged | 0.372 / 0.0486 | 0.4804 / 0.00899 | yes | 118 / 7 |
| naca2412 | converged | 0.556 / 0.0515 | 0.7063 / 0.00829 | no | 118 / 7 |
| naca6412 | converged | 0.925 / 0.0632 | 1.1147 / 0.01003 | no | 118 / 7 |
| naca0021 | converged | 0.324 / 0.0722 | 0.412 / 0.01021 | yes | 118 / 7 |
| ah79100c | converged | 1.125 / 0.0679 | -- / -- (XFOIL non-conv.) | no | 96 / 7 |

This run is what surfaced the gmsh/WSL cross-process gotcha documented
below -- it only appears when chaining all 6 stages across *multiple*
airfoils in one process, a scenario no existing per-stage or per-branch
test suite exercises.

Implementation note: ribs at the root (z=0) and tip (z=span) reuse the
exact same gmsh curve entities as the skin panels' top/bottom boundary
edges (built once, referenced by both), rather than via a boolean
fragment/glue step after the fact. Fragmenting the whole assembly
together would also split every skin panel at each *interior* rib
station -- an interior rib's polygon boundary (0 < z < span) is
geometrically embedded in the middle of a skin panel, not on its
boundary, so OCC's fragment op would cut the panel there to keep every
edge on a face boundary. That would blow up the skin surface count far
past the expected `len(coords) - 1`. Sharing curve tags at construction
time gives real, verifiable shared B-rep topology at root/tip (checked
in `test_ribs_and_spars_share_edges_with_skin` via `getBoundary` curve-tag
matching on the reloaded `.brep`, not a self-reported count) without that
side effect. Interior ribs and spar webs are independent surfaces,
geometrically coincident with the skin but not topologically fused to
it -- conformal meshing across those junctions is Stage 6's concern.
`.brep` (gmsh/OCC-native) is exported alongside the STEP specifically
because STEP export is not guaranteed to preserve shared-edge topology
for a loose surface collection the way OCC's native format does.

**Must-Pass Gate #2 (XFOIL cross-check) fails for real, and this is now a
thoroughly investigated, evidenced conclusion, not an open bug** —
tracked as a real, non-hidden `xfail(strict=True)` in
`test_stage4_postprocess.py::test_gate2_xfoil_cross_check`. Measured
(naca0012, AoA=4°, Re=5e5): CFD Cl 0.361-0.372 vs XFOIL 0.4804 (~22-25%
off); CFD Cd 0.044-0.049 vs XFOIL 0.00899 (4.9-5.5x off).

Nine hypotheses tested, each with a direct measurement (full detail in
`test_stage4_postprocess.py`'s module docstring):
1. y+ out of the locked wall-function range (14.4 avg, fixed to 89.8) — negligible effect.
2. Bulk mesh resolution — refined 2649→11812 cells (max face area 8.5→1.3): Cl +1.4%, Cd -5%. A real mesh-convergence study (further refinement to ~110k cells failed to even converge in 1000 iterations) shows the solution is already mesh-independent, not under-resolved.
3. Domain size 15c→50c — negligible.
4. Sign/projection convention — recomputed Cl/Cd with the projection angle flipped to -4°: Cl unchanged, Cd went the wrong direction (negative). Rules out a sign bug.
5. Combined `farfield` patch not holding the prescribed incidence — measured the solved far-field velocity directly: 3.91° vs. prescribed 4.0°. Correct.
6. Combined patch vs. the validated tutorial's separate inlet/outlet — built a one-off mesh replicating the tutorial's exact patch split; Cl/Cd came back virtually identical.
7. Fully-turbulent SA vs. XFOIL's free-transition assumption — reran XFOIL forced fully-turbulent: viscous Cd improved (2.6x→1.9x off) but the *dominant* pressure-drag term stayed ~8-10x off, and Cl barely moved.
8. Wall-function y+ (30-300, the spec's locked choice) vs. proper wall-resolved SA (y+~1, what SA is actually designed for per external literature) — regenerated at y+ avg 1.01: statistically identical Cl/Cd to both y+=14.4 and y+=89.8. Tested across three full y+ regimes with zero meaningful effect.
9. Bias consistency across 4 diverse airfoils (naca0006/0012/2412/6412) — Cl ratio (CFD/XFOIL) stayed in a fairly tight 0.77-0.88 band; Cd ratio ranged 3.6x-6.3x and grew with camber/thickness (XFOIL itself didn't converge for the 2 cambered cases, weakening confidence there specifically). **Lift bias is reasonably consistent across shapes; drag bias is not.**

External corroboration (WebSearch): published OpenFOAM-vs-XFOIL comparisons at comparable Reynolds numbers report the same qualitative pattern — "OpenFOAM predicted higher drag coefficients than XFOIL... consistently... across all cases," for every RANS turbulence model tested. This is a documented, expected RANS-vs-panel-method characteristic, not a bug specific to this pipeline.

**Conclusion and resolution:** this is a converged, mesh-independent, correctly-implemented RANS/SA solution that genuinely disagrees with XFOIL for this case — a CFD-methodology finding, not a fixable code defect. Per the spec's own output schema (`Cl_xfoil`/`Cd_xfoil` stored as raw numbers alongside CFD's own `Cl`/`Cd`, not a boolean pass/fail), `pipeline/stage4_postprocess.py::build_cfd_record` now assembles the exact per-AoA `cfd/` record the schema specifies — tracking the CFD/XFOIL relationship as data for downstream (Stage 9 aggregation, and eventually the Lead agent) to consume, rather than the pipeline silently discarding or gating on it. Because the drag bias specifically is shape-dependent (item 9 above), **drag/efficiency-based rankings between candidate airfoils should not be trusted from this data without further work; lift-based rankings are on safer ground.** Because XFOIL didn't converge for 2 of the 4 airfoils tested, getting real (converged) XFOIL references for cambered/thick sections is worth doing before trusting the Cd-ratio trend further.

**The pipeline is genuinely runnable end-to-end as of Stage 3**, confirmed
with a real solve: naca0012, AoA=4°, Re=5e5 converges in ~365-370
iterations (`tests/fixtures/solver_logs/real_ours_naca0012_aoa4_converged.log`).
Getting there required fixing a real bug: `stage2_case_gen.py`'s
Spalart-Allmaras freestream `nuTilda`/`nut` was set to `5 * nu` (a guessed
small multiple); the spec's own validated `airFoil2D` tutorial baseline
uses effectively `14000 * nu`. At the wrong ratio, `nuTilda` got stuck in
an exact two-value oscillation and never converged at any AoA tested (0°
and 4°, both to 1000 iterations) — see `stage2_case_gen.py`'s
`nu_tilda_inf` comment and `real_ours_naca0012_aoa0_oscillating.log`
(kept as a real fixture of the failure mode). Also fixed: Stage 2's
`controlDict` was missing the `solver incompressibleFluid;` entry
`foamRun` requires — syntactically valid, silently unrunnable; now a
permanent regression test (`test_control_dict_specifies_solver`).

Stage 3 also owns writing `system/fvSchemes`/`fvSolution` (copied
verbatim from the `airFoil2D` tutorial — verified to reference no patch
names, so they apply unmodified to this pipeline's farfield/airfoil
patches), since Stage 2's contract doesn't include solver numerics.

## FIXED: gmsh + WSL subprocess in the same long-lived process

**Scope is broader than first characterized.** Originally found chaining
multiple airfoils' full pipelines back-to-back in one process (Stage 5's
`gmsh.finalize()` breaking the *next airfoil's* Stage 1 WSL call). Since
then, reproduced again *within a single airfoil's own chain*: calling
Stage 4's `extract_surface_pressure` (WSL, via `foamToVTK`) *after*
Stages 5/6 (gmsh) in the same process throws the identical
`FileNotFoundError: [WinError 2]` on `subprocess.run(["wsl.exe", ...])`,
even for the SAME airfoil. So the real rule is: **any WSL-dependent call
after any `gmsh.finalize()` in the same process fails**, regardless of
airfoil identity -- not specifically "the next airfoil's Stage 1." Two
confirmed workarounds, both demonstrated for real in this session's
runner scripts: (a) do all WSL-dependent work before any gmsh stage,
within a given airfoil's own chain (this is what unblocked Stage
0->1->2->3->4->5->6->7 running cleanly for naca0012 in one process --
Stage 4's pressure extraction was reordered to happen before Stage 5/6);
(b) isolate each airfoil's full pipeline run in its own subprocess, for
the cross-airfoil case.

Neither `wsl.exe`'s resolution (`where wsl.exe` succeeds) nor
`os.environ["PATH"]` nor `os.getcwd()` change when this triggers (all
checked directly) -- something in gmsh's OCC-backed `finalize()` leaves
the process in a state Windows' `CreateProcess` can no longer use to
locate `wsl.exe`. Exact internal mechanism still not identified, but the
trigger (any gmsh finalize before any subsequent WSL call, same process)
and both workarounds are now confirmed empirically across two different
manifestations, not guessed from one case.

**Fixed for real** (this was not deferrable: Stage 8/CalculiX also runs
via WSL -- confirmed, `ccx` 2.17 is installed only inside WSL, not
natively on Windows -- and Stage 8 structurally must run *after* Stages
5/6's gmsh output, so even a single airfoil's full 0-9 run would have
hit this at Stage 8 with no way to fix it by reordering, unlike Stage 4).
`pipeline/_gmsh_isolation.py::run_isolated()` runs all gmsh work in a
fresh `multiprocessing` **spawn** subprocess (there is no `fork` on
Windows) instead of in-process -- confirmed empirically that this
prevents whatever `gmsh.finalize()` corrupts from ever reaching the
process that later makes WSL calls. `pipeline/stage5_structural_geometry.py`
and `pipeline/stage6_structural_mesh.py` were refactored: each public
function now validates inputs, then calls `run_isolated(_worker_fn, ...)`
where `_worker_fn` holds the actual gmsh logic (unchanged) -- the public
API and all existing tests were unaffected by the refactor (13 + 11
still pass). Verified directly, not just via the test suite passing: ran
`generate_structural_geometry`/`generate_structural_mesh` then
immediately called `subprocess.run(["wsl.exe", ...])` in the same
process -- succeeds now, both for a single airfoil's own chain and
across two different airfoils back-to-back. The full real chain (Stage
0->1->2->3->4->5->6->7) was re-confirmed end-to-end for naca0012 with
this fix in place.

**New constraint this fix introduces:** any top-level script (not a
pytest test module -- those are unaffected by the `__main__` re-exec
issue specifically, since pytest's `__main__` is the pytest runner, not
the test file) that calls a Stage 5/6 function **must** guard its
top-level code with `if __name__ == "__main__":`. Windows'
`multiprocessing` spawn re-executes an unguarded script's top-level code
inside the spawned child (a documented Python gotcha, confirmed the hard
way here: an unguarded ad-hoc runner script recursively re-ran its
entire pipeline chain inside the isolated gmsh subprocess, producing
duplicate output and a directory-collision crash, until guarded). Any
future Stage 6-9 orchestrator or demo script must follow this pattern.

**Known remaining issue, pytest-specific, NOT a production bug:**
`pytest tests/` (the full suite in one process) intermittently fails
Stage 8's tests with the exact same `FileNotFoundError: [WinError 2]`
signature -- reproduced twice, including once with zero concurrent WSL
usage from anything else, ruling out simple resource contention as the
sole explanation. This is very likely `run_isolated()`'s spawn mechanism
accumulating some process-level resource (handle/semaphore) across the
~35 gmsh-isolated calls Stage 5/6/7's combined test suites make in one
pytest session, eventually breaking a later WSL call -- plausible given
production usage (a real pipeline run calls Stage 5/6 once or twice per
airfoil, nowhere near pytest's per-session volume) has been verified
clean multiple times with **zero** such failures, including a full real
5-AoA sweep (see below). Practical mitigation confirmed: run Stage 8's
tests as a separate `pytest` invocation from Stage 5/6/7's (e.g. `pytest
tests/ --ignore=tests/test_stage8_calculix_run.py` then `pytest
tests/test_stage8_calculix_run.py` separately) -- both pass cleanly this
way. Not root-caused further since it doesn't affect the actual
pipeline, only a single-process full-suite pytest run's ordering; worth
revisiting if it starts affecting CI.

## Deferred to a future Troubleshooter/agent layer (do not hard-code a fix now)

Two places identified so far where a fixed rule can't reliably make the
right call — both are judgment-under-ambiguity, which is what the spec's
Agent 5 (Troubleshooter) is for, not more deterministic code:

1. **Stage 1 mesh-quality retries** (`pipeline/stage1_mesh.py::generate_mesh`).
   Currently blind-retries with degraded boundary-layer params (fewer
   layers, larger ratio/size) up to 3 times and gives up with no
   diagnosis of *why* a given geometry failed `checkMesh`. A troubleshooter
   agent could inspect the specific failure (skewness vs. negative volume
   vs. non-orthogonality) and choose a targeted fix instead of a fixed
   degradation schedule.
2. **Stage 3 non-converged/plateaued branch** (spec lines 102-104). The
   spec explicitly says: if residuals plateau rather than still
   decreasing, do NOT just rerun longer — it's likely unsteady flow/stall
   at high AoA or a mesh quality issue, and needs interpretation, not a
   fixed retry count.
3. **Stage 1 boundary-layer sizing vs. y+ vs. skewness, in conflict**
   (`pipeline/stage1_mesh.py::generate_mesh`'s `bl_size`). Two
   requirements that a single scalar parameter cannot satisfy across the
   full UIUC geometry set: y+ needs to land in the locked wall-function
   range (30-300), which needs a larger `bl_size` (7e-3 gets naca0012 to
   y+≈89.8 from a default-1e-3 y+≈14.4) — but `checkMesh`'s skewness gate
   (Must-Pass Gate #1, non-negotiable) then fails on other geometries.
   Measured max skewness (pass threshold ≈4.0):
   | Geometry | bl_size=1e-3 | bl_size=3e-3 | bl_size=7e-3 |
   |---|---|---|---|
   | naca0021 | 2.73 (pass) | 4.41 (fail) | — |
   | ah79100c | 2.28 (pass) | 4.53 (fail) | — |
   | naca633418 | 3.59 (pass) | 5.33 (fail) | — |
   | whitcomb | 3.11 (pass) | 3.17 (pass) | fails at 7e-3 |
   A fixed value can't win on both axes for every geometry — this needs
   either per-geometry calibration (possibly adjusting `bl_layers`/
   `bl_ratio` instead of just `bl_size`, or local refinement near
   high-curvature regions) or accepting a documented per-run override
   (current approach: `stage1_mesh.py` defaults to 1e-3 for universal
   Gate #1 robustness per the spec's own stated rationale for wall
   functions; `test_stage4_postprocess.py` passes `bl_size=7e-3`
   explicitly for its one XFOIL-comparison geometry).

## Superseded: agent-based Ansys Fluent/Mechanical pipeline

The 6-agent architecture in `CLAUDE.md` (Librarian/Fluidist/Structuralist/
Surrogate/Troubleshooter/Lead using Ansys Fluent + Mechanical via PyFluent)
predates the pivot above. `agents/`, `physics_cores/ansys_fluent/`, and
`run_cfd_test.py` reflect that earlier approach; several of those files
were removed from this branch. `CLAUDE.md` has not yet been updated to
match — treat its 6-agent table and Ansys-specific gotchas as historical
context, not current status, until it's revised.
