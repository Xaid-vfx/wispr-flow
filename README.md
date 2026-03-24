# wispr-flow

Local AI dictation for macOS. Speak → text appears in any app. Fully private — nothing leaves your machine.

## How it works

- **Whisper** (via MLX, Apple Silicon optimised) transcribes your speech
- By default, raw transcription is pasted instantly
- Say **"delta, ..."** to run the utterance through a local LLM for cleanup/rephrasing
- Push-to-talk: hold a key while speaking, release to paste

## Requirements

- macOS (Apple Silicon recommended)
- [Ollama](https://ollama.com) running locally with a model pulled
- Python 3.11+

## Setup

```bash
# 1. Clone
git clone https://github.com/Xaid-vfx/wispr-flow.git
cd wispr-flow

# 2. Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Pull a model for LLM rewriting (only needed for delta prefix)
ollama pull dolphin3
```

## Run

```bash
# Push-to-talk (hold Right ⌘ while speaking, release to paste)
.venv/bin/python main.py --hotkey --hotkey-key right_cmd

# With a custom prompt to improve Whisper accuracy
.venv/bin/python main.py --hotkey --hotkey-key right_cmd --prompt "Transcribing technical notes."

# Continuous VAD mode (terminal output only)
.venv/bin/python main.py

# Use a more accurate Whisper model
.venv/bin/python main.py --hotkey --hotkey-key right_cmd --model large-v3
```

Grant **Accessibility** permission when prompted (required for auto-paste).

## Usage

| What you say | What happens |
|---|---|
| Anything | Pasted as raw Whisper output |
| `delta, <text>` | LLM rewrites `<text>` before pasting |

## Options

| Flag | Default | Description |
|---|---|---|
| `--hotkey` | off | Push-to-talk mode |
| `--hotkey-key` | `right_option` | Key to hold (`right_cmd`, `right_ctrl`, etc.) |
| `--model` | `medium.en` | Whisper model (`tiny.en` → `large-v3`) |
| `--llm-model` | `dolphin3` | Ollama model for delta rewriting |
| `--prompt` | — | Initial prompt to bias Whisper |
| `--debug` | off | Show raw Whisper transcript |
| `--threshold` | `0.015` | VAD energy threshold |

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```
