"""Regression: `run_update_check_background()` respects a hermetic-mode env var.

The daemon thread that `smem`'s CLI kicks off on almost every invocation calls
`urlopen(PYPI_URL, timeout=3)` against pypi.org. `#110`/`#121` closed the
write side of that path (`_isolated_home_dir` in `tests/conftest.py`
redirects `$HOME` for the whole session); this closes the network side, so a
pytest run inside a sandboxed or offline environment doesn't sit through per-test
3-second `OSError` swallows or make silent live PyPI requests.

Contract: any truthy value of `SURREAL_MEMORY_NO_UPDATE_CHECK` (i.e. not empty,
not "0"/"false"/"no", case-insensitive) makes `run_update_check_background()` a
no-op *before* it starts the thread. The env var is set session-wide in
`tests/conftest.py::_isolated_home_dir`, alongside the `$HOME` redirect —
same fixture, same rationale.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from surreal_memory.cli import update_check


class TestUpdateCheckEnvGate:
    """Guards the "no PyPI calls under pytest" contract."""

    def test_env_var_short_circuits_before_thread_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With SURREAL_MEMORY_NO_UPDATE_CHECK=1, no daemon thread is created."""
        monkeypatch.setenv("SURREAL_MEMORY_NO_UPDATE_CHECK", "1")
        with patch("surreal_memory.cli.update_check.threading.Thread") as thread_ctor:
            update_check.run_update_check_background()
        thread_ctor.assert_not_called()

    def test_env_var_missing_still_starts_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the env var, the daemon thread does start — positive control
        that the short-circuit is specifically the env var, not something else."""
        # `_isolated_home_dir` (session, autouse) sets this to "1" for the whole
        # test session; explicitly clear it for this one test to prove the
        # short-circuit is env-gated rather than a permanent no-op.
        monkeypatch.delenv("SURREAL_MEMORY_NO_UPDATE_CHECK", raising=False)
        with patch("surreal_memory.cli.update_check.threading.Thread") as thread_ctor:
            update_check.run_update_check_background()
        thread_ctor.assert_called_once()
        # `daemon=True` in the call so a leaked thread never blocks interpreter shutdown.
        _args, kwargs = thread_ctor.call_args
        assert kwargs.get("daemon") is True

    @pytest.mark.parametrize("falsy", ["", "0", "false", "no", "FALSE", "No"])
    def test_falsy_values_do_not_short_circuit(
        self, monkeypatch: pytest.MonkeyPatch, falsy: str
    ) -> None:
        """Empty, '0', 'false', 'no' (any case) count as "not set"."""
        monkeypatch.setenv("SURREAL_MEMORY_NO_UPDATE_CHECK", falsy)
        with patch("surreal_memory.cli.update_check.threading.Thread") as thread_ctor:
            update_check.run_update_check_background()
        thread_ctor.assert_called_once()

    @pytest.mark.parametrize("truthy", ["1", "yes", "true", "TRUE", "on", "anything"])
    def test_truthy_values_short_circuit(
        self, monkeypatch: pytest.MonkeyPatch, truthy: str
    ) -> None:
        """Anything else — including case variants of yes/true/on and arbitrary
        strings — counts as truthy. Matches how `#121`'s fixture reads env vars."""
        monkeypatch.setenv("SURREAL_MEMORY_NO_UPDATE_CHECK", truthy)
        with patch("surreal_memory.cli.update_check.threading.Thread") as thread_ctor:
            update_check.run_update_check_background()
        thread_ctor.assert_not_called()
