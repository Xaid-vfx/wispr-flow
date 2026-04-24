"""
Personal vocabulary loader.

Reads ~/.dictation/vocab.txt (one word or phrase per line, # comments allowed)
and returns a comma-separated string suitable for Whisper's initial_prompt.

Whisper uses initial_prompt as a soft bias toward the given vocabulary —
useful for names, acronyms, and jargon that Whisper would otherwise mishear.

Creates the file with a helpful header on first run so the user can discover
and edit it.
"""

from pathlib import Path

VOCAB_DIR  = Path.home() / ".dictation"
VOCAB_FILE = VOCAB_DIR / "vocab.txt"

_HEADER = """\
# Personal vocabulary for local AI dictation.
# One word or phrase per line. Lines starting with # are ignored.
# Whisper uses these as a bias to transcribe your common words correctly.
#
# Examples — uncomment or replace with your own:
# Zaid
# OAuth
# kubectl
# WhisperFlow
"""


def _ensure_file():
    """Create the vocab file with a helpful header if it doesn't exist."""
    if VOCAB_FILE.exists():
        return
    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    VOCAB_FILE.write_text(_HEADER)


def load_vocab() -> tuple[str, int]:
    """
    Return (prompt_string, entry_count).

    prompt_string is a comma-separated list of vocab entries, ready to pass
    to Whisper as initial_prompt. Empty string if the file has no entries.
    """
    _ensure_file()
    entries = []
    for line in VOCAB_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return ", ".join(entries), len(entries)
