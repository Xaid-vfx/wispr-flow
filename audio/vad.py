from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Utterance:
    audio: np.ndarray
    sample_rate: int
    duration: float


class EnergyVAD:
    """
    Energy-based Voice Activity Detection.

    Watches a stream of audio chunks and returns a complete Utterance
    each time it detects that a speech segment has ended (speaker paused).

    State machine:
        IDLE → (speech_start_chunks loud chunks) → SPEAKING
        SPEAKING → (speech_end_chunks quiet chunks) → emit Utterance → IDLE
    """

    def __init__(self, config):
        self.config = config
        self._reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        return self._in_speech

    def process_chunk(self, chunk: np.ndarray) -> Optional[Utterance]:
        """
        Feed one audio chunk. Returns an Utterance when a complete
        speech segment is detected, otherwise returns None.
        """
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        is_speech = rms > self.config.energy_threshold

        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0

        # Transition: IDLE → SPEAKING
        if not self._in_speech and self._consecutive_speech >= self.config.speech_start_chunks:
            self._in_speech = True
            # Pull in the lead-in frames we buffered before speech was confirmed
            lead_in = list(self._pre_buffer)
            self._speech_buffer = lead_in + [chunk]
        elif self._in_speech:
            self._speech_buffer.append(chunk)

            # Safety: force-flush if utterance is too long
            total_samples = sum(len(c) for c in self._speech_buffer)
            if total_samples >= self.config.max_speech_duration * self.config.sample_rate:
                return self._finalize()

        # Transition: SPEAKING → IDLE (end of utterance)
        if self._in_speech and self._consecutive_silence >= self.config.speech_end_chunks:
            total_samples = sum(len(c) for c in self._speech_buffer)
            min_samples = self.config.min_speech_chunks * self.config.chunk_size

            if total_samples >= min_samples:
                return self._finalize()
            else:
                # Too short — likely a click or breath, discard
                self._reset()

        # Rolling pre-speech buffer so we can prepend lead-in on speech start
        self._pre_buffer.append(chunk)
        if len(self._pre_buffer) > self.config.speech_start_chunks + 2:
            self._pre_buffer.pop(0)

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finalize(self) -> Utterance:
        audio = np.concatenate(self._speech_buffer)
        duration = len(audio) / self.config.sample_rate
        utterance = Utterance(
            audio=audio,
            sample_rate=self.config.sample_rate,
            duration=duration,
        )
        self._reset()
        return utterance

    def _reset(self):
        self._in_speech = False
        self._speech_buffer: list[np.ndarray] = []
        self._pre_buffer: list[np.ndarray] = []
        self._consecutive_speech = 0
        self._consecutive_silence = 0
