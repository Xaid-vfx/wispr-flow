"""
Push-to-talk audio recorder.

Listens for a global hotkey. While the key is held:
  - Captures audio into a buffer
  - Snapshots the frontmost app at key-down (before Terminal steals focus)

On key release:
  - Emits the buffered audio + captured app via on_utterance callback

The frontmost-app snapshot is the key advantage over rolling tracking:
it's always captured at the moment the user is in their target app.
"""

import subprocess
import threading
import time
from typing import Callable, Optional, Any

import numpy as np

from audio.capture import AudioCapture
from audio.vad import Utterance

try:
    from pynput import keyboard as _kb
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False

# Map friendly names → pynput Key attributes
_KEY_MAP: dict[str, str] = {
    "right_option": "alt_r",
    "right_cmd":    "cmd_r",
    "right_ctrl":   "ctrl_r",
    "right_shift":  "shift_r",
    "f13":          "f13",
    "f14":          "f14",
    "f15":          "f15",
}


def _frontmost_app_name() -> str | None:
    """Return the name of the frontmost app via osascript (works from any thread)."""
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of first process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        name = result.stdout.strip()
        return name if name else None
    except Exception:
        return None


class HotkeyRecorder:
    """
    Records audio only while a hotkey is held down.

    Callbacks:
        on_recording_start()             — key pressed, mic is live
        on_utterance(utterance, app)     — key released, audio ready
    """

    def __init__(self, config,
                 on_recording_start: Callable | None = None,
                 on_utterance: Callable | None = None):
        self.config = config
        self._on_start    = on_recording_start
        self._on_utterance = on_utterance

        self._capture = AudioCapture(config.audio)

        self._buffer:    list[np.ndarray] = []
        self._held       = threading.Event()   # set while key is down
        self._was_held   = False
        self._press_time = 0.0
        self._target_app_name: str | None = None

        self._stop         = threading.Event()
        self._audio_thread: Optional[threading.Thread] = None
        self._listener     = None

        self.available = _PYNPUT_AVAILABLE

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()
        self._capture.start()
        self._audio_thread = threading.Thread(
            target=self._drain_loop, name="ptt-audio", daemon=True
        )
        self._audio_thread.start()
        self._start_listener()

    def stop(self):
        self._stop.set()
        self._capture.stop()
        if self._listener:
            self._listener.stop()

    # ── Audio drain loop ──────────────────────────────────────────────────────

    def _drain_loop(self):
        """
        Drain AudioCapture queue. Buffer chunks while key is held.
        Emit utterance when key is released.
        """
        while not self._stop.is_set():
            chunk = self._capture.get_chunk(timeout=0.05)
            if chunk is None:
                continue

            held = self._held.is_set()

            # Transition: held → released
            if self._was_held and not held:
                self._emit()

            if held:
                self._buffer.append(chunk)

            self._was_held = held

    def _emit(self):
        if not self._buffer:
            return

        duration = time.monotonic() - self._press_time
        if duration < self.config.hotkey.min_duration:
            self._buffer.clear()
            return   # accidental tap — discard

        audio = np.concatenate(self._buffer)
        self._buffer = []

        utterance = Utterance(
            audio=audio,
            sample_rate=self.config.audio.sample_rate,
            duration=len(audio) / self.config.audio.sample_rate,
        )

        if self._on_utterance:
            self._on_utterance(utterance, self._target_app_name)

    # ── Key listener ──────────────────────────────────────────────────────────

    def _start_listener(self):
        key_attr  = _KEY_MAP.get(self.config.hotkey.key, "alt_r")
        target_key = getattr(_kb.Key, key_attr, _kb.Key.alt_r)
        active     = {"held": False}

        def on_press(key):
            try:
                if key == target_key and not active["held"]:
                    active["held"]         = True
                    # Snapshot frontmost app BEFORE setting the held flag.
                    # osascript runs in its own process — no AppKit thread issues.
                    self._target_app_name  = _frontmost_app_name()
                    self._press_time       = time.monotonic()
                    self._buffer           = []
                    self._held.set()
                    if self._on_start:
                        self._on_start()
            except Exception:
                pass

        def on_release(key):
            try:
                if key == target_key and active["held"]:
                    active["held"] = False
                    self._held.clear()
            except Exception:
                pass

        self._listener = _kb.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()
