import os
import numpy as np
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import mlx_whisper

# Mapping from friendly model names to HuggingFace MLX repo IDs
_MODEL_REPOS = {
    "tiny.en":   "mlx-community/whisper-tiny.en-mlx",
    "base.en":   "mlx-community/whisper-base.en-mlx",
    "small.en":  "mlx-community/whisper-small.en-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large-v3":  "mlx-community/whisper-large-v3-mlx",
}

# Whisper hallucinates these phrases on silence — filter them out
_HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thank you",
    "thanks for watching",
    "bye",
    "bye bye",
    "you",
    ".",
    " .",
    "...",
}


class WhisperEngine:
    def __init__(self, config):
        self.config = config
        self._repo = _MODEL_REPOS.get(config.model)
        if not self._repo:
            raise ValueError(
                f"Unknown model '{config.model}'. "
                f"Choose from: {list(_MODEL_REPOS.keys())}"
            )

    def load(self):
        """Warm up the model so the first real transcription isn't slow."""
        dummy = np.zeros(16000, dtype=np.float32)
        mlx_whisper.transcribe(dummy, path_or_hf_repo=self._repo)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio and return cleaned text, or empty string on silence."""
        kwargs = dict(
            path_or_hf_repo=self._repo,
            language=self.config.language,
        )
        if self.config.initial_prompt:
            kwargs["initial_prompt"] = self.config.initial_prompt
        result = mlx_whisper.transcribe(audio, **kwargs)
        text = result.get("text", "").strip()

        # Filter Whisper hallucinations
        if text.lower().rstrip(".!?, ") in _HALLUCINATION_PHRASES:
            return ""

        return text
