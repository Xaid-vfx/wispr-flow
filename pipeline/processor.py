"""
Async two-thread pipeline.

Thread 1 (vad_thread):        capture → VAD → utterance_queue
Thread 2 (processing_thread): utterance_queue → Whisper → (delta: LLM) → event_queue

The main thread reads from event_queue and handles display.
The microphone is always live — processing never blocks capture.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

from pipeline.command_detector import strip_llm_prefix


# ── Event types sent to the main thread ──────────────────────────────────────

@dataclass
class SpeechStarted:
    """VAD just confirmed speech — show a live indicator immediately."""
    pass


@dataclass
class ProcessingResult:
    raw: str
    cleaned: str
    duration: float      # seconds of audio
    t_whisper: float     # seconds spent in Whisper
    t_llm: float         # seconds spent in LLM


@dataclass
class ProcessingWarning:
    message: str


Event = Union[SpeechStarted, ProcessingResult, ProcessingWarning]


# ── Pipeline ──────────────────────────────────────────────────────────────────

class AsyncPipeline:
    def __init__(self, capture, vad, whisper, rewriter, config):
        self._capture  = capture
        self._vad      = vad
        self._whisper  = whisper
        self._rewriter = rewriter
        self._config   = config

        # utterance_queue: VAD thread → processing thread
        # Bounded so we don't accumulate unbounded backlog if the user talks
        # faster than Whisper can process. If full, oldest is dropped.
        self._utterance_queue: queue.Queue = queue.Queue(maxsize=8)

        # event_queue: both threads → main thread (display only)
        self._event_queue: queue.Queue = queue.Queue()

        self._stop = threading.Event()
        self._vad_thread:        Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()
        self._capture.start()

        self._vad_thread = threading.Thread(
            target=self._vad_loop, name="vad", daemon=True
        )
        self._processing_thread = threading.Thread(
            target=self._processing_loop, name="proc", daemon=True
        )
        self._vad_thread.start()
        self._processing_thread.start()

    def stop(self):
        self._stop.set()
        self._capture.stop()
        if self._vad_thread:
            self._vad_thread.join(timeout=2.0)
        if self._processing_thread:
            # Give it time to finish the current transcription
            self._processing_thread.join(timeout=10.0)

    def get_event(self, timeout: float = 0.1) -> Optional[Event]:
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Thread 1: Audio capture + VAD ────────────────────────────────────────

    def _vad_loop(self):
        was_speaking = False

        while not self._stop.is_set():
            chunk = self._capture.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            utterance = self._vad.process_chunk(chunk)

            # Detect speech-start transition → notify main thread immediately
            now_speaking = self._vad.is_speaking
            if now_speaking and not was_speaking:
                self._event_queue.put(SpeechStarted())
            was_speaking = now_speaking

            if utterance is None:
                continue

            # Utterance complete — hand off to processing thread
            try:
                self._utterance_queue.put_nowait(utterance)
            except queue.Full:
                self._event_queue.put(ProcessingWarning(
                    "processing queue full — one utterance dropped "
                    "(speaking faster than Whisper can process)"
                ))

    # ── Thread 2: Whisper + LLM ───────────────────────────────────────────────

    def _processing_loop(self):
        while not self._stop.is_set():
            try:
                utterance = self._utterance_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                result = self._process(utterance)
                if result is not None:
                    self._event_queue.put(result)
            except Exception as e:
                self._event_queue.put(ProcessingWarning(f"processing error: {e}"))

    def _process(self, utterance) -> Optional[ProcessingResult]:
        t0 = time.perf_counter()
        raw = self._whisper.transcribe(utterance.audio)
        t_whisper = time.perf_counter() - t0

        if not raw:
            return None

        is_llm, text = strip_llm_prefix(raw)

        t1 = time.perf_counter()
        if is_llm:
            cleaned = self._rewriter.rewrite(text)
            t_llm = time.perf_counter() - t1
        else:
            cleaned, t_llm = text, 0.0

        return ProcessingResult(
            raw=raw,
            cleaned=cleaned,
            duration=utterance.duration,
            t_whisper=t_whisper,
            t_llm=t_llm,
        )
