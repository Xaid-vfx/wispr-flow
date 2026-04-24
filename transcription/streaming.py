"""
Streaming Whisper transcription.

While audio is still being recorded, a background worker transcribes
completed audio chunks. On finalize(), only the short tail since the
last committed chunk needs fresh transcription — that's where the
perceived-latency win comes from.

Each chunk is transcribed with the last N words of prior transcript as
initial_prompt, which gives Whisper linguistic context across chunk
boundaries and reduces boundary hallucinations.

Concurrency model
-----------------
start() is called from pynput's key-listener thread and MUST NOT block —
if it blocks, pynput can't dispatch the matching key-release event and
the recorder gets stuck in "held" state. So start() just stamps in a new
_Session and spins up a new worker; any prior session is flagged
abandoned and its worker dies on its own.

A single _whisper_lock serializes all mlx_whisper calls across workers
and finalize, since mlx_whisper isn't safe for concurrent invocation.
"""

import threading

import numpy as np


class _Session:
    """Per-recording state. Each press→release is one session."""
    __slots__ = (
        "buffer", "committed_samples", "committed_text",
        "lock", "work_signal", "abandoned",
    )

    def __init__(self):
        self.buffer: list[np.ndarray] = []
        self.committed_samples: int = 0
        self.committed_text: str = ""
        self.lock = threading.Lock()
        self.work_signal = threading.Event()
        self.abandoned = False


class StreamingTranscriber:
    STEP_SECONDS: float = 2.5      # size of each committed chunk
    MIN_TAIL_SECONDS: float = 0.3  # skip tail if tinier than this (hallucination risk)
    PROMPT_WORDS: int = 20         # words of prior transcript to seed next chunk

    def __init__(self, whisper, sample_rate: int):
        self._whisper = whisper
        self._sample_rate = sample_rate
        self._session: _Session | None = None
        self._whisper_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """
        Begin a new session. Non-blocking — any previous session is marked
        abandoned and its worker exits on its own.
        """
        prev = self._session
        if prev is not None:
            prev.abandoned = True
            prev.work_signal.set()

        new_session = _Session()
        self._session = new_session

        threading.Thread(
            target=self._loop, args=(new_session,),
            name="streaming-whisper", daemon=True,
        ).start()

    def add_chunk(self, chunk: np.ndarray):
        """Feed one audio chunk into the current session."""
        s = self._session
        if s is None or s.abandoned:
            return
        step_samples = int(self.STEP_SECONDS * self._sample_rate)
        with s.lock:
            s.buffer.append(chunk)
            total = sum(len(c) for c in s.buffer)
            ready = (total - s.committed_samples) >= step_samples
        if ready:
            s.work_signal.set()

    def finalize(self) -> tuple[str, float]:
        """
        End the current session, transcribe the remaining tail synchronously,
        return (full_text, tail_seconds). The background worker is marked
        abandoned but not joined — it will exit on its own.
        """
        s = self._session
        self._session = None
        if s is None:
            return "", 0.0

        s.abandoned = True
        s.work_signal.set()

        with s.lock:
            buf_snapshot = list(s.buffer)
            committed = s.committed_samples
            prior_text = s.committed_text

        total = sum(len(c) for c in buf_snapshot)
        tail_samples = total - committed
        tail_seconds = tail_samples / self._sample_rate

        tail_text = ""
        if tail_samples >= int(self.MIN_TAIL_SECONDS * self._sample_rate):
            audio_all = np.concatenate(buf_snapshot)
            tail_audio = audio_all[committed:]
            prompt = _last_words(prior_text, self.PROMPT_WORDS)
            with self._whisper_lock:
                tail_text = self._whisper.transcribe(tail_audio, initial_prompt=prompt)

        if tail_text:
            full = (prior_text + " " + tail_text).strip()
        else:
            full = prior_text
        return full, tail_seconds

    # ── Worker loop ───────────────────────────────────────────────────────────

    def _loop(self, s: _Session):
        step_samples = int(self.STEP_SECONDS * self._sample_rate)
        while not s.abandoned:
            s.work_signal.wait(timeout=0.5)
            s.work_signal.clear()
            if s.abandoned:
                return

            with s.lock:
                buf_snapshot = list(s.buffer)
                chunk_start = s.committed_samples
                prior_text = s.committed_text

            total = sum(len(c) for c in buf_snapshot)
            if total - chunk_start < step_samples:
                continue

            chunk_end = chunk_start + step_samples
            audio_all = np.concatenate(buf_snapshot)
            chunk_audio = audio_all[chunk_start:chunk_end]
            prompt = _last_words(prior_text, self.PROMPT_WORDS)

            try:
                with self._whisper_lock:
                    if s.abandoned:
                        return
                    text = self._whisper.transcribe(chunk_audio, initial_prompt=prompt)
            except Exception:
                text = ""   # finalize() will re-transcribe the tail anyway

            with s.lock:
                if s.abandoned:
                    return
                if text:
                    s.committed_text = (s.committed_text + " " + text).strip()
                s.committed_samples = chunk_end


def _last_words(text: str, n: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[-n:])
