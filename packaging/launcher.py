#!/usr/bin/env python3
"""
py2app entry point for Dictation.app.

A bundled GUI app has no CLI — LaunchServices invokes it with just the
bundle path. This launcher forces hotkey+menubar mode by rewriting argv
before main.py's argparse runs.
"""
import locale
import os
import sys

# GUI-launched processes often inherit a C/POSIX locale, which makes
# Python's default text codec ASCII and crashes read_text() on any
# non-ASCII content. Force a UTF-8 locale before any app code imports.
for loc in ("en_US.UTF-8", "C.UTF-8", "UTF-8"):
    try:
        locale.setlocale(locale.LC_ALL, loc)
        break
    except locale.Error:
        continue

# Force hotkey mode; main.py will fall back to menubar UI via rumps.
sys.argv = [sys.argv[0], "--hotkey"]

# Make repo root importable when launched from inside the .app bundle.
# py2app places source next to this file in Contents/Resources.
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
