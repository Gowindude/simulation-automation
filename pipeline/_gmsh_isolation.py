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
"""

import multiprocessing as mp


def _worker_entry(fn, args, kwargs, queue):
    try:
        result = fn(*args, **kwargs)
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {e}"))


def run_isolated(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) in a fresh spawned subprocess and return its
    result. fn must be a module-level function (picklable by reference)
    and its args/kwargs/return value must be picklable.

    Explicitly closes the Queue and Process afterward -- defensive
    cleanup for scaling to hundreds/thousands of calls (one or two per
    airfoil in real usage). Verified directly that 40 consecutive calls
    in a plain script leak nothing observable (every WSL call afterward
    still succeeds), but this costs nothing and removes any doubt at
    1000-airfoil scale.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker_entry, args=(fn, args, kwargs, queue))
    try:
        proc.start()
        status, payload = queue.get()
        proc.join()

        if status == "error":
            raise RuntimeError(f"isolated gmsh subprocess failed: {payload}")
        if proc.exitcode != 0:
            raise RuntimeError(f"isolated gmsh subprocess exited with code {proc.exitcode}")
        return payload
    finally:
        queue.close()
        queue.join_thread()
        if proc.exitcode is None:
            # Only reached if start()/queue.get() itself raised before
            # the normal join() above -- make sure the child isn't left
            # dangling before proc.close() (which requires termination).
            proc.join(timeout=5)
        if proc.exitcode is not None:
            proc.close()
