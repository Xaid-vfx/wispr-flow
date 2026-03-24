"""
macOS Menubar icon for dictation status (hotkey mode).

States:
  ⌥  — idle, waiting for hotkey press
  ⏺  — recording (mic is live)
  ✓  — done/pasted (resets to idle after 2s)
"""

import rumps

from pipeline.hotkey_pipeline import HotkeyRecordingStarted, HotkeyResult, HotkeyWarning

_IDLE        = "⌥"
_RECORDING   = "⏺"
_DONE        = "✓"
_RESET_AFTER = 2.0   # seconds to show ✓ before returning to idle


class DictationMenubar(rumps.App):
    """
    Thin menubar wrapper around HotkeyPipeline.

    Polls the pipeline event queue via a rumps.Timer so the NSRunLoop
    can run on the main thread (required for AppKit / NSStatusBar).
    """

    def __init__(self, pipeline, config, paster=None):
        super().__init__(name="Dictation", title=_IDLE, quit_button=None)
        self._pipeline     = pipeline
        self._paster       = paster
        self._config       = config
        self._reset_timer  = None

        self.menu = [rumps.MenuItem("Quit", callback=self._on_quit)]

        # Poll pipeline events every 50ms
        rumps.Timer(self._poll, 0.05).start()

    # ── Event polling ─────────────────────────────────────────────────────────

    def _poll(self, _):
        event = self._pipeline.get_event(timeout=0)
        if event is None:
            return

        if isinstance(event, HotkeyRecordingStarted):
            self.title = _RECORDING

        elif isinstance(event, HotkeyResult):
            self._show_result(event)

        elif isinstance(event, HotkeyWarning):
            print(f"\n  ⚠  {event.message}\n", flush=True)

    def _show_result(self, event: HotkeyResult):
        self.title = _DONE

        if self._config.debug:
            print(f"  [raw]  {event.raw}")

        label = "out" if event.t_llm > 0 else "raw"
        print(f"  [{label}]  {event.cleaned}")

        timing = f"whisper {event.t_whisper:.2f}s"
        if event.t_llm > 0:
            timing += f" | llm {event.t_llm:.2f}s"

        paste_status = f"pasted → {event.app_name}" if event.pasted else "paste failed"
        print(f"  ⏱  {event.duration:.1f}s · {timing} · {paste_status}\n", flush=True)

        # Reset icon back to idle after a moment
        if self._reset_timer:
            self._reset_timer.stop()
        self._reset_timer = rumps.Timer(self._reset_idle, _RESET_AFTER)
        self._reset_timer.start()

    def _reset_idle(self, _):
        self.title = _IDLE
        if self._reset_timer:
            self._reset_timer.stop()
            self._reset_timer = None

    # ── Quit ──────────────────────────────────────────────────────────────────

    def _on_quit(self, _):
        print("\nStopping...")
        self._pipeline.stop()
        if self._paster:
            self._paster.stop()
        print("Goodbye.")
        rumps.quit_application()
