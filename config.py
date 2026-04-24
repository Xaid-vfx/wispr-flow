from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512          # samples per callback (~32ms at 16kHz)

    # Energy-based VAD thresholds
    energy_threshold: float = 0.015  # RMS threshold — tune this for your mic/environment
    speech_start_chunks: int = 4     # consecutive loud chunks to declare speech start
    speech_end_chunks: int = 25      # consecutive quiet chunks to declare speech end (~800ms)
    min_speech_chunks: int = 12      # min chunks for a valid utterance (~380ms)
    max_speech_duration: float = 30.0  # force-flush after this many seconds


@dataclass
class WhisperConfig:
    model: str = "medium.en"     # tiny.en | base.en | small.en | medium.en | large-v3
    language: str = "en"
    initial_prompt: str = ""     # biases Whisper vocabulary/style; empty = no prompt


@dataclass
class LLMConfig:
    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    enabled: bool = False    # default raw; "delta," prefix or "hey dictation rewrite mode" enables LLM
    temperature: float = 0.3
    max_tokens: int = 300


@dataclass
class ContextConfig:
    max_sentences: int = 5


@dataclass
class HotkeyConfig:
    # Key to hold while speaking.
    # Options: right_option | right_cmd | right_ctrl | right_shift | f13 | f14
    key: str = "right_option"
    min_duration: float = 0.3   # discard accidental taps shorter than this (seconds)


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    debug: bool = False   # show raw transcript before LLM rewrite
