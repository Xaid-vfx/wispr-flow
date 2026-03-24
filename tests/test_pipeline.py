"""
Tests for the async pipeline.

Uses a fake capture and fake Whisper/LLM so we can drive the pipeline
with synthetic audio without real hardware or models.
"""

import queue
import time
import threading
import numpy as np
import pytest

from audio.vad import EnergyVAD
from pipeline.processor import AsyncPipeline, SpeechStarted, ProcessingResult, ProcessingWarning
from pipeline.context_manager import ContextManager
from config import AudioConfig, ContextConfig


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeCapture:
    """Feeds pre-loaded chunks on demand; simulates AudioCapture."""

    def __init__(self, chunks: list[np.ndarray]):
        self._q: queue.Queue = queue.Queue()
        for c in chunks:
            self._q.put(c)

    def start(self): pass
    def stop(self):  pass

    def get_chunk(self, timeout=0.1) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class FakeWhisper:
    def transcribe(self, audio: np.ndarray) -> str:
        return "hello world"


class FakeWhisperDelta:
    """Returns delta-prefixed transcript so the LLM path is exercised."""
    def transcribe(self, audio: np.ndarray) -> str:
        return "delta, hello world"


class FakeWhisperSilent:
    """Always returns empty — simulates hallucination-filtered silence."""
    def transcribe(self, audio: np.ndarray) -> str:
        return ""


class FakeLLM:
    def __init__(self):
        self.config = type("cfg", (), {"enabled": True})()

    def rewrite(self, transcript: str, context: str = "") -> str:
        return transcript.capitalize() + "."


# ── Helpers ───────────────────────────────────────────────────────────────────

CHUNK = 512
SR = 16000

def vad_cfg(**kw) -> AudioConfig:
    c = AudioConfig(
        energy_threshold=0.02,
        speech_start_chunks=4,
        speech_end_chunks=10,
        min_speech_chunks=6,
        max_speech_duration=30.0,
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def silent_chunks(n: int) -> list[np.ndarray]:
    return [np.full(CHUNK, 0.001, dtype=np.float32) for _ in range(n)]


def loud_chunks(n: int, amplitude=0.1) -> list[np.ndarray]:
    t = np.linspace(0, CHUNK / SR, CHUNK)
    base = (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)
    return [base.copy() for _ in range(n)]


def drain_events(pipeline, timeout_total=5.0) -> list:
    """Collect all events until nothing arrives for 0.5s or total timeout hit."""
    events = []
    deadline = time.time() + timeout_total
    while time.time() < deadline:
        ev = pipeline.get_event(timeout=0.5)
        if ev is None:
            break
        events.append(ev)
    return events


def build_pipeline(chunks, whisper=None, llm=None, audio_cfg=None):
    cfg   = audio_cfg or vad_cfg()
    cap   = FakeCapture(chunks)
    vad   = EnergyVAD(cfg)
    wh    = whisper or FakeWhisper()
    rw    = llm or FakeLLM()
    ctx   = ContextManager(ContextConfig(max_sentences=5))

    class _Config:
        debug = False
        class llm:
            enabled = True

    return AsyncPipeline(cap, vad, wh, rw, ctx, _Config())


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_speech_started_event_emitted():
    """SpeechStarted must arrive before any ProcessingResult."""
    chunks = loud_chunks(30) + silent_chunks(20)
    pipeline = build_pipeline(chunks)
    pipeline.start()

    events = drain_events(pipeline, timeout_total=8.0)
    pipeline.stop()

    types = [type(e).__name__ for e in events]
    assert "SpeechStarted" in types
    # SpeechStarted should appear before the first ProcessingResult
    first_started = next(i for i, t in enumerate(types) if t == "SpeechStarted")
    first_result  = next((i for i, t in enumerate(types) if t == "ProcessingResult"), None)
    if first_result is not None:
        assert first_started < first_result


def test_processing_result_emitted_after_utterance():
    """A complete speech + silence sequence must produce a ProcessingResult."""
    chunks = loud_chunks(30) + silent_chunks(20)
    pipeline = build_pipeline(chunks)
    pipeline.start()

    events = drain_events(pipeline, timeout_total=8.0)
    pipeline.stop()

    results = [e for e in events if isinstance(e, ProcessingResult)]
    assert len(results) >= 1
    r = results[0]
    assert r.raw == "hello world"
    assert r.cleaned == "hello world"   # raw-by-default; no delta prefix → no LLM
    assert r.duration > 0
    assert r.t_whisper >= 0
    assert r.t_llm == 0.0


def test_silence_only_produces_no_result():
    """Pure silence must produce zero ProcessingResults."""
    chunks = silent_chunks(100)
    pipeline = build_pipeline(chunks)
    pipeline.start()

    events = drain_events(pipeline, timeout_total=3.0)
    pipeline.stop()

    results = [e for e in events if isinstance(e, ProcessingResult)]
    assert results == []


def test_whisper_silence_filtered():
    """If Whisper returns empty (silence/hallucination), no result is emitted."""
    chunks = loud_chunks(30) + silent_chunks(20)
    pipeline = build_pipeline(chunks, whisper=FakeWhisperSilent())
    pipeline.start()

    events = drain_events(pipeline, timeout_total=8.0)
    pipeline.stop()

    results = [e for e in events if isinstance(e, ProcessingResult)]
    assert results == []


def test_two_utterances_produce_two_results():
    chunks = (
        loud_chunks(20) + silent_chunks(15) +
        loud_chunks(20) + silent_chunks(15)
    )
    pipeline = build_pipeline(chunks)
    pipeline.start()

    events = drain_events(pipeline, timeout_total=12.0)
    pipeline.stop()

    results = [e for e in events if isinstance(e, ProcessingResult)]
    assert len(results) == 2


def test_context_accumulates_across_utterances():
    """Second utterance's context should contain first utterance's output."""
    received_contexts = []

    class CapturingLLM:
        def __init__(self):
            self.config = type("cfg", (), {"enabled": True})()
        def rewrite(self, transcript, context=""):
            received_contexts.append(context)
            return transcript

    chunks = (
        loud_chunks(20) + silent_chunks(15) +
        loud_chunks(20) + silent_chunks(15)
    )
    pipeline = build_pipeline(chunks, whisper=FakeWhisperDelta(), llm=CapturingLLM())
    pipeline.start()
    drain_events(pipeline, timeout_total=12.0)
    pipeline.stop()

    assert len(received_contexts) >= 2
    # Second call's context should contain first result
    assert "hello world" in received_contexts[1]


def test_stop_is_clean():
    """Pipeline must stop without hanging even if processing is mid-flight."""
    chunks = loud_chunks(30) + silent_chunks(5)  # short silence — may not finish
    pipeline = build_pipeline(chunks)
    pipeline.start()
    time.sleep(0.5)

    t0 = time.time()
    pipeline.stop()
    elapsed = time.time() - t0

    # Should stop within the join timeout (10s), not hang
    assert elapsed < 11.0
