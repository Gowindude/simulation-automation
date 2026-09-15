"""
Verification tests for pipeline/_gmsh_isolation.py.

Contract (locked before implementation, per .claude/airfoil_pipeline_build_spec.md's
"propose the verification tests ... then agree on them before you build" --
this fix followed a real incident, not speculation: the GitHub Actions
smoke-test workflow's spawned child died during interpreter bootstrap
(re-importing `__main__` from `<stdin>`, a `python -` heredoc gotcha) and
the parent's un-timed `queue.get()` hung for the full 30-minute job
timeout instead of failing with a clear error):
  - Existing behavior unchanged: a normal function call still returns its
    result; a function that raises still surfaces that as a RuntimeError
    naming the original exception.
  - A child that dies WITHOUT ever calling queue.put (bootstrap crash,
    the actual incident) must raise a clear, distinct RuntimeError within
    a bounded time -- not hang.
  - A child that runs longer than `timeout` must be terminated and raise
    a clear TimeoutError within roughly `timeout` (not run to completion).
"""

import time

import pytest

from pipeline._gmsh_isolation import run_isolated


def _add(a, b):
    return a + b


def _raises(msg):
    raise ValueError(msg)


def _sleep_forever():
    time.sleep(3600)


def _die_without_reporting():
    """Simulates the real incident: the child process dies before ever
    reaching queue.put (there, a multiprocessing spawn-context bootstrap
    crash; here, os._exit bypasses _worker_entry's own try/except the
    same way a bootstrap crash would)."""
    import os
    os._exit(1)


class TestRunIsolatedNormalBehavior:
    def test_returns_result(self):
        assert run_isolated(_add, 2, 3) == 5

    def test_propagates_function_exception(self):
        with pytest.raises(RuntimeError, match="ValueError: boom"):
            run_isolated(_raises, "boom")


class TestRunIsolatedTimeout:
    def test_slow_child_is_terminated_and_raises_timeout(self):
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            run_isolated(_sleep_forever, timeout=2)
        elapsed = time.monotonic() - start
        # Must fail close to the requested timeout, not hang toward
        # anything like the 1800s job-timeout scale of the real incident.
        assert elapsed < 15, f"took {elapsed:.1f}s to time out a 2s-timeout call"

    def test_default_timeout_is_bounded_not_infinite(self):
        import inspect
        sig = inspect.signature(run_isolated)
        assert sig.parameters["timeout"].default is not None

    def test_child_dying_without_reporting_raises_promptly_not_hangs(self):
        """The actual incident: a child that exits before queue.put is
        ever called must not hang the parent -- confirmed bounded by
        `timeout`, not by an external process/job killer."""
        start = time.monotonic()
        with pytest.raises(RuntimeError):
            run_isolated(_die_without_reporting, timeout=10)
        elapsed = time.monotonic() - start
        assert elapsed < 15, f"took {elapsed:.1f}s -- child death wasn't detected promptly"
