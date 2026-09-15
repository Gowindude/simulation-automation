"""
Stage 1 Troubleshooter -- LLM-judgment fallback for gmsh mesh-generation
failures whose error signature doesn't match any known deterministic
rule in generate_mesh's retry ladder.

Scope, per the build spec's agent-scope carve-out
(.claude/airfoil_pipeline_build_spec.md lines 7-14): Stage 1 mesh
generation ONLY, and only as a last resort after the deterministic
signature-matching (e.g. _is_bl_self_intersection_failure) fails to
recognize the error. CFD execution (Stage 3), load mapping (Stage 7),
and CalculiX execution (Stage 8) are explicitly out of scope -- nothing
here touches those.

Shells out to the local `claude` CLI in print mode (`claude -p`), not
the Anthropic API directly -- this bills against the caller's existing
Claude subscription plan usage (confirmed empirically: `claude auth
status` reports no ANTHROPIC_API_KEY, authMethod "claude.ai"), not
separate metered API charges. Do NOT pass --bare: it explicitly requires
an API key and never reads OAuth/subscription auth, which would silently
switch billing modes.

Every invocation should be logged via log_troubleshooter_call (inputs,
proposed params, reasoning, and -- once known -- whether it actually
fixed the mesh). A recurring pattern in that log is exactly the material
for graduating a new deterministic rule into generate_mesh, the same way
this project's own adversarial-probing findings (2026-09-14/15) became
hardcoded fixes rather than staying agent-handled forever -- the agent's
necessary scope should shrink over time, not grow.
"""

import json
import os
import subprocess
import time

_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "bl_size": {"type": "number"},
        "bl_layers": {"type": "integer"},
        "bl_ratio": {"type": "number"},
    },
    "required": ["reasoning", "bl_size", "bl_layers", "bl_ratio"],
}

_PROMPT_TEMPLATE = """A gmsh boundary-layer airfoil mesh generation attempt failed. You are diagnosing a Stage 1 CFD meshing failure in a pipeline that meshes UIUC-style airfoil geometries (2D C-grid domain, boundary-layer extrusion near the airfoil wall) for CFD analysis.

Geometry stats:
{geometry_stats}

Current mesh parameters:
{current_params}

gmsh error output (tail):
{gmsh_output}

Propose ONE new parameter set to try next. Constraints: bl_size in [1e-5, 1e-2], bl_layers in [3, 20], bl_ratio in [1.05, 1.5]. Ground your reasoning in the specific error text and geometry stats given -- don't guess blindly or just repeat the current values."""


def diagnose_mesh_failure(
    gmsh_output: str, current_params: dict, geometry_stats: dict, timeout: int = 90,
) -> dict:
    """
    Ask the local `claude` CLI (print mode, schema-validated structured
    output) to diagnose a gmsh mesh-generation failure and propose new
    boundary-layer parameters.

    Returns:
        {"reasoning": str, "bl_size": float, "bl_layers": int, "bl_ratio": float}

    Raises:
        RuntimeError: if the `claude` CLI itself fails, times out, or
            returns output that doesn't validate -- callers must treat
            this as "the agent couldn't help" and fall back to the
            deterministic retry ladder, never let it crash the pipeline.
    """
    prompt = _PROMPT_TEMPLATE.format(
        geometry_stats=json.dumps(geometry_stats, indent=2),
        current_params=json.dumps(current_params, indent=2),
        gmsh_output=gmsh_output[-2000:],
    )
    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--output-format", "json",
                "--json-schema", json.dumps(_SCHEMA),
            ],
            input=prompt,
            capture_output=True, text=True, timeout=timeout,
            # Windows: `claude` resolves to a `.CMD` npm shim, which
            # CreateProcess can't exec directly (only native .exe) --
            # shell=True routes the joined argv through `cmd.exe /c`.
            # That means cmd.exe re-tokenizes the *already* list2cmdline-
            # quoted string with its own metacharacter rules (%, ^, &, |,
            # <, >) on top of Win32 argv escaping -- a prompt is NOT safe
            # to put in that argv list, because it embeds raw gmsh output
            # and json.dumps() blobs that can contain those characters
            # (confirmed the hard way: the real spike-failure troubleshooter
            # call silently failed/corrupted this way -- see STATUS.md).
            # Passing the prompt over stdin instead means only the fixed
            # flags above ever pass through cmd.exe's parser; the prompt
            # text itself bypasses shell tokenization entirely.
            shell=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        raise RuntimeError(f"troubleshooter CLI invocation failed: {e}")

    if result.returncode != 0:
        raise RuntimeError(f"claude -p exited {result.returncode}: {result.stderr[:500]}")

    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude -p returned non-JSON output: {e}\n{result.stdout[:500]}")

    if envelope.get("is_error"):
        raise RuntimeError(f"claude -p reported an error: {envelope}")

    decision = envelope.get("structured_output")
    required = ("reasoning", "bl_size", "bl_layers", "bl_ratio")
    if not decision or not all(k in decision for k in required):
        raise RuntimeError(f"claude -p output missing required fields: {envelope.get('result')}")

    return decision


def log_troubleshooter_call(log_path: str, record: dict) -> None:
    """
    Append one JSONL record of a troubleshooter invocation.

    `record` should carry at least: gmsh_output, current_params,
    proposed params, reasoning, and (once known) `outcome`
    ("succeeded" | "failed" | "agent_error"). A recurring pattern here
    is the material for turning an ad hoc agent fix into tomorrow's
    hardcoded rule, same as this session's own findings.
    """
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    record = dict(record)
    record.setdefault("timestamp", time.time())
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
