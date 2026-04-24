"""
Hotkey (push-to-talk) pipeline.

HotkeyRecorder → utterance_queue → Whisper → (delta: LLM) → paste → event_queue

Unlike the VAD pipeline, each utterance carries its own target_app —
captured at the moment the hotkey was pressed, when the user's app had focus.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from pipeline.command_detector import strip_llm_prefix
from transcription.streaming import StreamingTranscriber


# ── Events (sent to main thread for display) ──────────────────────────────────

@dataclass
class HotkeyRecordingStarted:
    """Hotkey pressed — mic is live."""
    pass


@dataclass
class HotkeyResult:
    raw:       str
    cleaned:   str
    duration:  float
    t_whisper: float
    t_llm:     float
    pasted:    bool
    app_name:  str   # name of the app we pasted into


@dataclass
class HotkeyWarning:
    message: str


# ── Pipeline ──────────────────────────────────────────────────────────────────

class HotkeyPipeline:
    """
    Thin orchestrator for push-to-talk mode.

    The recorder calls back into this pipeline; a single processing thread
    drives Whisper → (delta: LLM) → paste for each utterance.
    """

    def __init__(self, recorder, whisper, rewriter, paster, config):
        self._recorder  = recorder
        self._whisper   = whisper
        self._rewriter  = rewriter
        self._paster    = paster
        self._config    = config

        self._streaming = StreamingTranscriber(whisper, config.audio.sample_rate)

        self._utt_queue: queue.Queue   = queue.Queue()
        self._event_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._proc_thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()

        # Wire recorder callbacks into this pipeline
        self._recorder._on_start     = self._on_recording_start
        self._recorder._on_utterance = self._on_utterance_ready
        self._recorder._on_chunk     = self._on_audio_chunk

        self._proc_thread = threading.Thread(
            target=self._processing_loop, name="hk-proc", daemon=True
        )
        self._proc_thread.start()
        self._recorder.start()

    def stop(self):
        self._stop.set()
        self._recorder.stop()
        if self._proc_thread:
            self._proc_thread.join(timeout=10.0)

    def get_event(self, timeout: float = 0.1):
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Recorder callbacks (called from recorder threads) ─────────────────────

    def _on_recording_start(self):
        self._streaming.start()
        self._event_queue.put(HotkeyRecordingStarted())

    def _on_audio_chunk(self, chunk):
        self._streaming.add_chunk(chunk)

    def _on_utterance_ready(self, utterance, target_app):
        self._utt_queue.put((utterance, target_app))

    # ── Processing thread ─────────────────────────────────────────────────────

    def _processing_loop(self):
        while not self._stop.is_set():
            try:
                utterance, target_app = self._utt_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = self._process(utterance, target_app)
                if result:
                    self._event_queue.put(result)
            except Exception as e:
                self._event_queue.put(HotkeyWarning(f"processing error: {e}"))

    def _process(self, utterance, target_app_name) -> Optional[HotkeyResult]:
        t0  = time.perf_counter()
        raw, _tail_seconds = self._streaming.finalize()
        t_whisper = time.perf_counter() - t0

        if not raw:
            return None

        is_llm, text = strip_llm_prefix(raw)

        t1 = time.perf_counter()
        if is_llm:
            cleaned = self._rewriter.rewrite(text)
            t_llm   = time.perf_counter() - t1
        else:
            cleaned, t_llm = text, 0.0

        # Paste into the app that was focused when the key was pressed
        pasted   = False
        app_name = target_app_name or "unknown"
        if target_app_name and self._paster.enabled:
            pasted = self._paster.paste_by_name(cleaned, target_app_name)

        return HotkeyResult(
            raw=raw,
            cleaned=cleaned,
            duration=utterance.duration,
            t_whisper=t_whisper,
            t_llm=t_llm,
            pasted=pasted,
            app_name=app_name,
        )
