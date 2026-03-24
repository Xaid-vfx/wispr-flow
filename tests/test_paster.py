"""
Tests for AutoPaster.

These test the logic and fallback behaviour using mocks.
We cannot test actual clipboard/paste in CI without a display server,
so GUI-touching code paths are isolated behind a feature flag and mocked.
"""

import threading
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from output.paster import AutoPaster, _DEFAULT_EXCLUDED


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fake_app(name: str):
    app = MagicMock()
    app.localizedName.return_value = name
    return app


# ── Exclusion list tests (no AppKit needed) ───────────────────────────────────

def test_default_excluded_contains_terminal():
    assert "Terminal" in _DEFAULT_EXCLUDED
    assert "iTerm2" in _DEFAULT_EXCLUDED
    assert "Warp" in _DEFAULT_EXCLUDED


def test_custom_exclusions_merged():
    paster = AutoPaster(excluded_apps={"MyCustomApp"})
    assert "MyCustomApp" in paster._excluded
    assert "Terminal" in paster._excluded  # defaults preserved


# ── Tracker logic (mock NSWorkspace) ─────────────────────────────────────────


def test_terminal_app_is_not_tracked():
    """A terminal app must not overwrite a previously stored target."""
    paster = AutoPaster()
    paster.enabled = True

    notes_app   = make_fake_app("Notes")
    terminal_app = make_fake_app("Terminal")

    # Store a good target first
    with paster._lock:
        paster._target_app = notes_app

    # Simulate what the track loop would do with a terminal app
    name = terminal_app.localizedName()
    if name not in paster._excluded:
        with paster._lock:
            paster._target_app = terminal_app

    # Target should still be Notes
    with paster._lock:
        assert paster._target_app is notes_app


# ── Paste fallback behaviour ──────────────────────────────────────────────────

def test_paste_returns_false_when_disabled():
    paster = AutoPaster()
    paster.enabled = False
    assert paster.paste("hello") is False


def test_paste_returns_false_when_no_target():
    paster = AutoPaster()
    paster.enabled = True
    # _target_app is None by default
    assert paster.paste("hello") is False


def test_paste_attempts_when_target_set():
    """When a target app is set, paste() should activate it and return True."""
    paster = AutoPaster()
    paster.enabled = True

    fake_app = make_fake_app("Notes")
    with paster._lock:
        paster._target_app = fake_app

    fake_pb = MagicMock()
    fake_pb.stringForType_.return_value = "old clipboard"

    with patch("output.paster.NSPasteboard") as mock_pb_class, \
         patch("output.paster.subprocess.run") as mock_run, \
         patch("output.paster.time.sleep"):

        mock_pb_class.generalPasteboard.return_value = fake_pb
        mock_run.return_value = MagicMock(returncode=0)

        result = paster.paste("hello world")

    assert result is True
    fake_app.activateWithOptions_.assert_called_once()
    fake_pb.clearContents.assert_called()
    # Verify the new text was written (first positional arg) regardless of type constant
    written_texts = [call.args[0] for call in fake_pb.setString_forType_.call_args_list]
    assert "hello world " in written_texts


def test_paste_restores_clipboard():
    """Old clipboard contents must be restored after paste."""
    paster = AutoPaster()
    paster.enabled = True

    fake_app = make_fake_app("Notes")
    with paster._lock:
        paster._target_app = fake_app

    fake_pb = MagicMock()
    old_content = "my precious clipboard text"
    fake_pb.stringForType_.return_value = old_content

    calls = []
    def track_set(text, type_):
        calls.append(text)
    fake_pb.setString_forType_ = track_set

    with patch("output.paster.NSPasteboard") as mock_pb_class, \
         patch("output.paster.subprocess.run"), \
         patch("output.paster.time.sleep"):

        mock_pb_class.generalPasteboard.return_value = fake_pb
        paster.paste("dictated text")

    # First call sets new text, second call restores old text
    assert any("dictated text" in c for c in calls)
    assert calls[-1] == old_content


def test_paste_error_returns_false():
    """If activation throws, paste() should return False, not crash."""
    paster = AutoPaster()
    paster.enabled = True

    bad_app = MagicMock()
    bad_app.activateWithOptions_.side_effect = RuntimeError("boom")

    with paster._lock:
        paster._target_app = bad_app

    with patch("output.paster.NSPasteboard"):
        result = paster.paste("hello")

    assert result is False


# ── start / stop ──────────────────────────────────────────────────────────────

def test_stop_is_clean():
    """stop() must return immediately — no background thread."""
    paster = AutoPaster()
    t0 = time.time()
    paster.stop()
    assert time.time() - t0 < 0.1


def test_poll_updates_target_from_main_thread():
    """poll() must store a non-excluded app as the target."""
    paster = AutoPaster()
    paster.enabled = True

    fake_notes = make_fake_app("Notes")
    fake_ws = MagicMock()
    fake_ws.frontmostApplication.return_value = fake_notes

    with patch("output.paster.NSWorkspace") as mock_ws_class:
        mock_ws_class.sharedWorkspace.return_value = fake_ws
        paster.poll()

    with paster._lock:
        assert paster._target_app is fake_notes
    assert paster.target_name == "Notes"


def test_poll_skips_excluded_apps():
    """poll() must not update target when a terminal is frontmost."""
    paster = AutoPaster()
    paster.enabled = True

    fake_notes    = make_fake_app("Notes")
    fake_terminal = make_fake_app("Terminal")

    # Store Notes first
    with paster._lock:
        paster._target_app  = fake_notes
        paster._target_name = "Notes"

    fake_ws = MagicMock()
    fake_ws.frontmostApplication.return_value = fake_terminal

    with patch("output.paster.NSWorkspace") as mock_ws_class:
        mock_ws_class.sharedWorkspace.return_value = fake_ws
        paster.poll()

    # Should still be Notes
    assert paster.target_name == "Notes"
