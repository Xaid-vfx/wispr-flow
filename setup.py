"""
py2app build recipe for Dictation.app.

Usage:
    ./build_dmg.sh
or manually:
    rm -rf build dist
    python setup.py py2app
    # .app appears at dist/Dictation.app
"""
import sys

# modulegraph walks AST recursively and blows Python's default
# recursion limit on some deps (pyobjc, numpy stubs). Raise it.
sys.setrecursionlimit(10_000)

from setuptools import setup

APP = ["packaging/launcher.py"]

# Force full directory copies for packages that contain non-.py assets
# (native libs for sounddevice, platform backends for pynput, etc).
# NOTE: 'mlx' is a PEP 420 namespace package (no top-level __init__.py);
# py2app's legacy imp.find_module can't resolve it, so we copy it manually
# in build_dmg.sh after py2app finishes.
PACKAGES = [
    "audio",
    "transcription",
    "pipeline",
    "llm",
    "output",
    "mlx_whisper",
    "numpy",
    "rumps",
    "pynput",
    "sounddevice",
    "ollama",
    "certifi",
]

# Modules that aren't in packages above but main.py / deps import
INCLUDES = [
    "config",
    "main",
    "queue",
    "signal",
    "subprocess",
    "threading",
    "dataclasses",
    # mlx submodules (imported dynamically by mlx_whisper)
    "mlx.core",
    "mlx.nn",
    "mlx.optimizers",
    "mlx.utils",
]

PLIST = {
    "CFBundleName":             "Dictation",
    "CFBundleDisplayName":      "Dictation",
    "CFBundleIdentifier":       "com.local.dictation",
    "CFBundleVersion":          "0.2.0",
    "CFBundleShortVersionString":"0.2.0",
    "CFBundlePackageType":      "APPL",
    "LSMinimumSystemVersion":   "11.0",
    "LSUIElement":              True,          # menubar-only, no Dock icon
    "NSHighResolutionCapable":  True,
    "NSMicrophoneUsageDescription":
        "Dictation listens to your voice while you hold the hotkey, then "
        "transcribes it to text.",
    "NSAppleEventsUsageDescription":
        "Dictation uses System Events to activate your target app and paste "
        "transcribed text with Cmd+V.",
    # GUI-launched apps inherit a minimal environment; force UTF-8 so Python's
    # default text codec (used by e.g. Path.read_text()) doesn't fall back to
    # ASCII and crash on non-ASCII file content.
    "LSEnvironment": {
        "PYTHONUTF8":       "1",
        "LANG":             "en_US.UTF-8",
        "LC_ALL":           "en_US.UTF-8",
    },
}

OPTIONS = {
    "argv_emulation": False,
    "arch":           "arm64",         # MLX is Apple Silicon only
    "packages":       PACKAGES,
    "includes":       INCLUDES,
    "plist":          PLIST,
    "strip":          False,           # keep symbols; easier to debug
}

setup(
    name="Dictation",
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
