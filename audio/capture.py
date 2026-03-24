import queue
import numpy as np
import sounddevice as sd


class AudioCapture:
    """
    Captures microphone audio in fixed-size chunks and pushes them
    into a thread-safe queue for downstream processing.
    """

    def __init__(self, config):
        self.config = config
        self._queue: queue.Queue = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"  [audio warning: {status}]")
        # indata shape: (frames, channels) — take mono channel 0
        chunk = indata[:, 0].copy().astype(np.float32)
        self._queue.put(chunk)

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_size,
            dtype=np.float32,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
