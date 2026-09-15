"""
Runs gmsh-dependent work in an isolated, spawned subprocess.

Confirmed gotcha (see STATUS.md): calling `gmsh.finalize()` leaves the
current process in a state where a later `subprocess.run(["wsl.exe",
...])` fails with `FileNotFoundError`, even though nothing observable
(PATH, cwd) changes. Since Stage 8 (CalculiX, WSL-based) must run after
Stages 5/6 (gmsh) in the pipeline's own data-dependency order, this can't
be fixed by reordering calls the way Stage 4 was. Running gmsh in a
genuinely fresh, throwaway process -- `multiprocessing`'s `spawn` context
(there is no `fork` on Windows) -- was confirmed empirically to keep the
corruption from ever reaching the parent process, which stays free to
make WSL calls afterward.

Real incident (2026-09-15, GitHub Actions smoke-test workflow): the
workflow ran its Python snippet via `python - <<HEREDOC` (stdin), which
sets `__main__.__file__` to `<stdin>` -- not a real path. The spawned
child crashed during interpreter bootstrap trying to re-import that
non-existent path, before ever reaching `_worker_entry` at all, so it
never called `queue.put`. The parent's un-timed `queue.get()` then
blocked forever, and the job only ended 28 minutes later when GitHub's
own `timeout-minutes: 30` force-killed it -- a genuine solve looks
identical to this from the outside (both "still running"), so the
un-timed wait masked a 10-second crash as a 28-minute hang. `run_isolated`
now takes a bounded `timeout` (queue.get(timeout=...)) so any child that
dies or hangs before reporting fails fast with a clear diagnostic
instead of silently waiting for an external killer.
"""

import multiprocessing as mp
import queue as queue_module
import time

DEFAULT_TIMEOUT_SECONDS = 600  # generous vs. real per-airfoil mesh times (seconds), tiny vs. a 30min job timeout
_POLL_INTERVAL_SECONDS = 0.2


def _worker_entry(fn, args, kwargs, queue):
    try:
        result = fn(*args, **kwargs)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {e}"))


def run_isolated(fn, *args, timeout=DEFAULT_TIMEOUT_SECONDS, **kwargs):
    """
    Run fn(*args, **kwargs) in a fresh spawned subprocess and return its
    result. fn must be a module-level function (picklable by reference)
    and its args/kwargs/return value must be picklable.

    timeout (seconds): if the child hasn't reported a result by then, it
    is terminated and a TimeoutError is raised -- bounded by `timeout`,
    not by whatever external job/process killer happens to be watching
    (see module docstring's real incident). A child that dies WITHOUT
    ever reporting a result (e.g. a bootstrap crash) is detected within
    one poll interval, not the full timeout, and raises a distinct
    RuntimeError -- "crashed immediately" and "still running past
    budget" are different failures and shouldn't look the same.

    Explicitly closes the Queue and Process afterward -- defensive
    cleanup for scaling to hundreds/thousands of calls (one or two per
    airfoil in real usage). Verified directly that 40 consecutive calls
    in a plain script leak nothing observable (every WSL call afterward
    still succeeds), but this costs nothing and removes any doubt at
    1000-airfoil scale.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker_entry, args=(fn, args, kwargs, result_queue))
    try:
        proc.start()
        deadline = time.monotonic() + timeout
        status = payload = None
        got_result = False
        while time.monotonic() < deadline:
            try:
                status, payload = result_queue.get(timeout=_POLL_INTERVAL_SECONDS)
                got_result = True
                break
            except queue_module.Empty:
                if not proc.is_alive():
                    break  # died without ever calling queue.put -- don't keep polling out the clock

        if not got_result:
            if not proc.is_alive():
                proc.join(timeout=5)
                raise RuntimeError(
                    f"isolated subprocess exited (code {proc.exitcode}) without reporting a "
                    f"result -- it likely crashed before user code ran at all (e.g. an "
                    f"interpreter-bootstrap failure re-importing __main__, seen in practice "
                    f"when the parent script itself was run via `python -` from stdin rather "
                    f"than a real .py file)"
                )
            proc.terminate()
            proc.join(timeout=5)
            raise TimeoutError(f"isolated subprocess did not report a result within {timeout}s")
        proc.join()

        if status == "error":
            raise RuntimeError(f"isolated gmsh subprocess failed: {payload}")
        if proc.exitcode != 0:
            raise RuntimeError(f"isolated gmsh subprocess exited with code {proc.exitcode}")
        return payload
    finally:
        result_queue.close()
        result_queue.join_thread()
        if proc.exitcode is None:
            # Only reached if start()/get() itself raised before the
            # normal join() above -- make sure the child isn't left
            # dangling before proc.close() (which requires termination).
            proc.terminate()
            proc.join(timeout=5)
        if proc.exitcode is not None:
            proc.close()
