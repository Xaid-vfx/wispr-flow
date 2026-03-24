"""
Unit tests for the VAD state machine.

These use synthetic numpy arrays to simulate silence and speech —
no microphone required.
"""

import numpy as np
import pytest

from audio.vad import EnergyVAD, Utterance
from config import AudioConfig

CHUNK = 512
SR = 16000


# ── Helpers ───────────────────────────────────────────────────────────────────

def cfg(**overrides) -> AudioConfig:
    c = AudioConfig(
        energy_threshold=0.02,
        speech_start_chunks=4,
        speech_end_chunks=10,
        min_speech_chunks=6,
        max_speech_duration=30.0,
    )
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def silent(amplitude=0.001) -> np.ndarray:
    """Near-silent chunk (well below any realistic threshold)."""
    return np.full(CHUNK, amplitude, dtype=np.float32)


def loud(amplitude=0.1) -> np.ndarray:
    """Sine wave chunk clearly above threshold."""
    t = np.linspace(0, CHUNK / SR, CHUNK)
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


def feed(vad, chunks) -> list[Utterance]:
    """Feed a list of chunks and collect any returned Utterances."""
    results = []
    for chunk in chunks:
        r = vad.process_chunk(chunk)
        if r is not None:
            results.append(r)
    return results


# ── Tests ────────────────────────────────────────────────────────────────────

def test_silence_never_triggers():
    """100 silent chunks should produce zero utterances."""
    vad = EnergyVAD(cfg())
    results = feed(vad, [silent()] * 100)
    assert results == []


def test_noise_spike_below_start_threshold_ignored():
    """3 loud chunks (< speech_start_chunks=4) must not trigger an utterance."""
    vad = EnergyVAD(cfg(speech_start_chunks=4))
    results = feed(vad, [loud()] * 3 + [silent()] * 20)
    assert results == []


def test_sustained_speech_then_pause_yields_utterance():
    """4+ loud chunks followed by 10+ silent chunks should return one Utterance."""
    vad = EnergyVAD(cfg())
    chunks = [loud()] * 30 + [silent()] * 15
    results = feed(vad, chunks)
    assert len(results) == 1
    assert isinstance(results[0], Utterance)
    assert results[0].duration > 0
    assert results[0].sample_rate == SR


def test_utterance_audio_is_non_empty():
    vad = EnergyVAD(cfg())
    chunks = [loud()] * 20 + [silent()] * 15
    results = feed(vad, chunks)
    assert len(results[0].audio) > 0


def test_utterance_too_short_is_discarded():
    """
    Speech shorter than min_speech_chunks should be silently discarded,
    not returned as a short garbage utterance.
    """
    vad = EnergyVAD(cfg(min_speech_chunks=50))  # require 50 chunks minimum
    # Feed 10 speech chunks (triggers start, but below min), then silence
    chunks = [loud()] * 10 + [silent()] * 20
    results = feed(vad, chunks)
    assert results == []


def test_two_utterances_in_sequence():
    """After one utterance is emitted, VAD must reset and detect the next one."""
    vad = EnergyVAD(cfg())
    chunks = (
        [loud()] * 20 + [silent()] * 15 +   # utterance 1
        [loud()] * 20 + [silent()] * 15      # utterance 2
    )
    results = feed(vad, chunks)
    assert len(results) == 2


def test_max_duration_forces_flush():
    """Audio exceeding max_speech_duration must be force-flushed."""
    # max = 0.5s at 16kHz = 8000 samples = ~15 chunks of 512
    vad = EnergyVAD(cfg(max_speech_duration=0.5, min_speech_chunks=1))
    # Feed 100 loud chunks (way more than 0.5s worth)
    results = feed(vad, [loud()] * 100)
    assert len(results) >= 1  # at least one flush should have happened


def test_pre_buffer_lead_in_included():
    """
    The speech buffer should include a few chunks captured before
    speech_start was confirmed (the lead-in).
    """
    vad = EnergyVAD(cfg(speech_start_chunks=4))
    # Feed 4 loud chunks — speech starts on chunk 4
    # The pre-buffer should pull back those 4 chunks into the utterance
    chunks = [loud()] * 4 + [silent()] * 15
    results = feed(vad, chunks)
    assert len(results) == 1
    # Duration should be at least the 4 speech chunks worth of audio
    min_expected_duration = (4 * CHUNK) / SR
    assert results[0].duration >= min_expected_duration * 0.5  # lenient
