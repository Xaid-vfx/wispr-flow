"""
macOS Auto-Paste

Tracks the last non-terminal frontmost application, then pastes
transcription results directly into it by:
  1. Activating the target app
  2. Writing text to the clipboard (saving + restoring previous contents)
  3. Simulating Cmd+V via osascript

IMPORTANT: NSWorkspace.frontmostApplication() must be called from the
main thread in a Python process without an AppKit run loop. The background
thread approach is unreliable. Call poll() from the main loop instead.
"""

import subprocess
import threading
import time
from typing import Optional

try:
    from AppKit import (
        NSWorkspace,
        NSPasteboard,
        NSStringPboardType,
        NSApplicationActivateIgnoringOtherApps,
    )
    _APPKIT_AVAILABLE = True
except ImportError:
    _APPKIT_AVAILABLE = False

# Terminal / launcher apps — never paste into these
_DEFAULT_EXCLUDED = {
    "Terminal", "iTerm2", "iTerm", "Alacritty", "Warp",
    "Hyper", "kitty", "WezTerm", "Tabby", "Ghostty",
}

# macOS system processes — skip in auto-detect (not useful paste targets)
_SYSTEM_SKIP = {
    "Finder", "SystemUIServer", "Dock", "Control Center",
    "Notification Center", "Spotlight", "Siri", "loginwindow",
    "WindowServer", "AirPlayUIAgent", "universalaccessd",
    "TextInputMenuAgent", "ScreenSaverEngine",
}

_FOCUS_DELAY  = 0.12   # seconds to wait after activating target app
_PASTE_SETTLE = 0.15   # seconds before restoring old clipboard


def check_accessibility() -> bool:
    """Return True if Accessibility permission is granted."""
    result = subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to get name of first process'],
        capture_output=True,
        timeout=3,
    )
    return result.returncode == 0


def print_accessibility_instructions():
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  Auto-paste needs Accessibility permission.          │")
    print("  │                                                       │")
    print("  │  1. Open System Settings → Privacy & Security        │")
    print("  │  2. Click Accessibility                               │")
    print("  │  3. Enable your terminal app (Terminal / iTerm2 …)   │")
    print("  │  4. Restart the dictation tool                        │")
    print("  └─────────────────────────────────────────────────────┘")
    print()


class AutoPaster:
    """
    Tracks frontmost non-terminal app and pastes text into it.

    poll() must be called periodically from the MAIN thread — AppKit's
    NSWorkspace is unreliable from background threads in non-AppKit processes.

    Alternatively, set a fixed target app with set_target_by_name().
    """

    def __init__(self, excluded_apps: set[str] | None = None):
        self._excluded   = _DEFAULT_EXCLUDED | (excluded_apps or set())
        self._target_app = None
        self._lock       = threading.Lock()
        self.enabled     = _APPKIT_AVAILABLE
        self._target_name: str | None = None  # for display / confirmation

    # ── Public API ────────────────────────────────────────────────────────────

    def find_candidates(self) -> list[str]:
        """
        Return names of all running GUI apps that are viable paste targets.
        Filters out terminals and macOS system processes.
        """
        if not self.enabled:
            return []
        try:
            ws = NSWorkspace.sharedWorkspace()
            names = []
            for app in ws.runningApplications():
                # activationPolicy 0 = regular foreground GUI app
                if app.activationPolicy() != 0:
                    continue
                if not app.isFinishedLaunching():
                    continue
                name = app.localizedName() or ""
                if name and name not in self._excluded and name not in _SYSTEM_SKIP:
                    names.append(name)
            return names
        except Exception:
            return []

    def auto_detect(self) -> str | None:
        """
        Scan running apps and auto-select the best paste target.
        Sets _target_app and returns the app name, or None if ambiguous/none found.

        - If exactly one viable app is running: auto-select it.
        - If multiple: return None (caller decides how to handle).
        """
        if not self.enabled:
            return None
        try:
            ws = NSWorkspace.sharedWorkspace()
            candidates = []
            for app in ws.runningApplications():
                if app.activationPolicy() != 0:
                    continue
                if not app.isFinishedLaunching():
                    continue
                name = app.localizedName() or ""
                if name and name not in self._excluded and name not in _SYSTEM_SKIP:
                    candidates.append((name, app))

            if len(candidates) == 1:
                name, app = candidates[0]
                with self._lock:
                    self._target_app  = app
                    self._target_name = name
                return name
            return None
        except Exception:
            return None

    def poll(self):
        """
        Sample the current frontmost app. Call this from the main thread
        periodically (every ~500ms is fine). Updates _target_app if the
        frontmost app is not in the excluded list.
        """
        if not self.enabled:
            return
        try:
            ws  = NSWorkspace.sharedWorkspace()
            app = ws.frontmostApplication()
            if app is None:
                return
            name = app.localizedName() or ""
            if name and name not in self._excluded:
                with self._lock:
                    if self._target_name != name:
                        self._target_name = name
                        self._target_app  = app
                        print(f"  [paste target → {name}]", flush=True)
        except Exception:
            pass

    def set_target_by_name(self, app_name: str) -> bool:
        """
        Pin the paste target to a specific app by name.
        Useful when the user always dictates into the same app.
        Returns True if the app is currently running.
        """
        if not self.enabled:
            return False
        try:
            ws = NSWorkspace.sharedWorkspace()
            for app in ws.runningApplications():
                if app.localizedName() == app_name:
                    with self._lock:
                        self._target_app  = app
                        self._target_name = app_name
                    print(f"  [paste target pinned → {app_name}]")
                    return True
            print(f"  [paste: '{app_name}' is not running]")
            return False
        except Exception as e:
            print(f"  [paste: error finding '{app_name}': {e}]")
            return False

    @property
    def target_name(self) -> str | None:
        return self._target_name

    def stop(self):
        pass  # no background thread to clean up

    def paste_by_name(self, text: str, app_name: str) -> bool:
        """
        Paste text into an app identified by name, using osascript for both
        activate and keystroke (works from any thread, no NSRunningApplication needed).
        """
        if not self.enabled:
            return False
        try:
            pb = NSPasteboard.generalPasteboard()
            old_text = pb.stringForType_(NSStringPboardType)

            pb.clearContents()
            pb.setString_forType_(text + " ", NSStringPboardType)

            script = (
                f'tell application "{app_name}" to activate\n'
                f'delay {_FOCUS_DELAY}\n'
                f'tell application "System Events" to keystroke "v" using command down'
            )
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=3,
            )

            time.sleep(_PASTE_SETTLE)
            if old_text:
                pb.clearContents()
                pb.setString_forType_(old_text, NSStringPboardType)

            return True
        except Exception as e:
            print(f"  [paste error: {e}]")
            return False

    def paste(self, text: str, app=None) -> bool:
        """
        Paste text into a target app.

        If `app` is provided (an NSRunningApplication), paste there directly.
        Otherwise paste into the tracked target app (rolling tracker / --target).
        Returns True if paste was attempted.
        """
        if not self.enabled:
            return False

        if app is None:
            with self._lock:
                app = self._target_app

        if app is None:
            return False

        try:
            pb = NSPasteboard.generalPasteboard()
            old_text = pb.stringForType_(NSStringPboardType)

            # Activate target and wait for focus transfer
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            time.sleep(_FOCUS_DELAY)

            # Write dictated text (trailing space for natural flow)
            pb.clearContents()
            pb.setString_forType_(text + " ", NSStringPboardType)

            # Paste via osascript (no Quartz dependency)
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to keystroke "v" using command down'],
                capture_output=True,
                timeout=2,
            )

            # Restore previous clipboard contents
            time.sleep(_PASTE_SETTLE)
            if old_text:
                pb.clearContents()
                pb.setString_forType_(old_text, NSStringPboardType)

            return True

        except Exception as e:
            print(f"  [paste error: {e}]")
            return False
