from __future__ import annotations

from osmia.tui import _looks_failed


def test_looks_failed_reads_exit_codes_and_errors() -> None:
    assert not _looks_failed("exit code: 0\nsrc\nREADME.md")
    assert _looks_failed("exit code: 1\nls: cannot access 'nope'")
    assert _looks_failed("exit code: timeout after 120s\nstill running")
    assert _looks_failed("Error: unknown tool 'nope'")
    assert _looks_failed("Traceback (most recent call last):\n  ...")
    assert not _looks_failed("")
    assert not _looks_failed("the build printed errors: 0")
