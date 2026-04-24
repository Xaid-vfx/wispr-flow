"""
Streaming Whisper transcription.

While audio is still being recorded, a background worker transcribes
completed audio chunks. On finalize(), only the short tail since the
last committed chunk needs fresh transcription — that's where the
perceived-latency win comes from.

Each chunk is transcribed with the last N words of prior transcript as
initial_prompt, which gives Whisper linguistic context across chunk
boundaries and reduces boundary hallucinations.
"""

import threading
from typing import Optional

import numpy as np


class StreamingTranscriber:
    STEP_SECONDS: float = 2.5      # size of each committed chunk
    MIN_TAIL_SECONDS: float = 0.3  # skip tail if tinier than this (hallucination risk)
    PROMPT_WORDS: int = 20         # words of prior transcript to seed next chunk

    def __init__(self, whisper, sample_rate: int):
        self._whisper = whisper
        self._sample_rate = sample_rate

        self._buffer: list[np.ndarray] = []
        self._committed_samples: int = 0
        self._committed_text: str = ""
        self._lock = threading.Lock()

        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._work_signal = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Reset state and launch the background transcription worker."""
        # Ensure any previous worker is fully stopped before resetting state
        self._stop.set()
        self._work_signal.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5.0)

        with self._lock:
            self._buffer = []
            self._committed_samples = 0
            self._committed_text = ""

        self._stop.clear()
        self._work_signal.clear()
        self._worker = threading.Thread(
            target=self._loop, name="streaming-whisper", daemon=True
        )
        self._worker.start()

    def add_chunk(self, chunk: np.ndarray):
        """Feed one audio chunk. Signals the worker when a full step is ready."""
        step_samples = int(self.STEP_SECONDS * self._sample_rate)
        with self._lock:
            self._buffer.append(chunk)
            total = sum(len(c) for c in self._buffer)
            ready = (total - self._committed_samples) >= step_samples
        if ready:
            self._work_signal.set()

    def finalize(self) -> tuple[str, float]:
        """
        Stop the worker, transcribe the remaining tail, return (full_text, tail_seconds).

        tail_seconds is how much audio still needed transcription after the
        background worker finished — useful for logging / understanding wins.
        """
        self._stop.set()
        self._work_signal.set()
        if self._worker:
            self._worker.join(timeout=30.0)

        with self._lock:
            buf_snapshot = list(self._buffer)
            committed = self._committed_samples
            prior_text = self._committed_text

        total = sum(len(c) for c in buf_snapshot)
        tail_samples = total - committed
        tail_seconds = tail_samples / self._sample_rate

        tail_text = ""
        if tail_samples >= int(self.MIN_TAIL_SECONDS * self._sample_rate):
            audio_all = np.concatenate(buf_snapshot)
            tail_audio = audio_all[committed:]
            prompt = _last_words(prior_text, self.PROMPT_WORDS)
            tail_text = self._whisper.transcribe(tail_audio, initial_prompt=prompt)

        if tail_text:
            full = (prior_text + " " + tail_text).strip()
        else:
            full = prior_text

        return full, tail_seconds

    # ── Worker loop ───────────────────────────────────────────────────────────

    def _loop(self):
        step_samples = int(self.STEP_SECONDS * self._sample_rate)
        while not self._stop.is_set():
            self._work_signal.wait(timeout=0.5)
            self._work_signal.clear()
            if self._stop.is_set():
                break

            with self._lock:
                buf_snapshot = list(self._buffer)
                chunk_start = self._committed_samples
                prior_text = self._committed_text

            total = sum(len(c) for c in buf_snapshot)
            if total - chunk_start < step_samples:
                continue

            chunk_end = chunk_start + step_samples
            audio_all = np.concatenate(buf_snapshot)
            chunk_audio = audio_all[chunk_start:chunk_end]
            prompt = _last_words(prior_text, self.PROMPT_WORDS)

            try:
                text = self._whisper.transcribe(chunk_audio, initial_prompt=prompt)
            except Exception:
                text = ""   # swallow — finalize() will re-transcribe the tail

            with self._lock:
                if text:
                    self._committed_text = (self._committed_text + " " + text).strip()
                self._committed_samples = chunk_end


def _last_words(text: str, n: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[-n:])
