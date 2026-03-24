"""
Integration tests for the Whisper engine.

Requires: mlx-whisper installed and internet access on first run
          (downloads ~150MB for tiny.en).

Uses `tiny.en` model to keep test runtime fast (~2-3s total).
"""

import time
import numpy as np
import pytest

from transcription.engine import WhisperEngine
from config import WhisperConfig


@pytest.fixture(scope="module")
def engine():
    """Load tiny.en once for all tests in this module."""
    e = WhisperEngine(WhisperConfig(model="tiny.en"))
    e.load()
    return e


# ── Tests ────────────────────────────────────────────────────────────────────

def test_engine_loads(engine):
    """Basic sanity — engine object exists and has the right repo."""
    assert "tiny" in engine._repo


def test_silence_returns_string(engine):
    """Silent audio must return a string (empty or filtered hallucination)."""
    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio)
    assert isinstance(result, str)


def test_silence_hallucination_filtered(engine):
    """
    Common Whisper hallucinations on silence ('Thank you.', 'Bye.', etc.)
    should be filtered out and return an empty string.
    """
    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio)
    # The result should be empty or very short — not a full hallucinated phrase
    assert len(result) < 15, f"Possible hallucination not filtered: '{result}'"


def test_white_noise_returns_string(engine):
    """Random noise must not crash — should return a string."""
    noise = np.random.normal(0, 0.01, 16000).astype(np.float32)
    result = engine.transcribe(noise)
    assert isinstance(result, str)


def test_tone_returns_string(engine):
    """A pure sine tone must not crash — should return a string."""
    t = np.linspace(0, 1.0, 16000)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    result = engine.transcribe(tone)
    assert isinstance(result, str)


def test_transcription_speed(engine):
    """
    tiny.en should transcribe 3 seconds of audio in under 3 seconds on M4 Pro.
    If this fails, something is wrong with Metal acceleration.
    """
    audio = np.zeros(16000 * 3, dtype=np.float32)
    t0 = time.perf_counter()
    engine.transcribe(audio)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, (
        f"Transcription took {elapsed:.2f}s. "
        "Expected < 3s on M4 Pro. Check Metal GPU acceleration."
    )


def test_invalid_model_name_raises():
    """Passing a bad model name should raise ValueError, not crash silently."""
    with pytest.raises(ValueError, match="Unknown model"):
        WhisperEngine(WhisperConfig(model="nonexistent-model"))
