# Architecture — Local AI Dictation

A macOS push-to-talk dictation tool, locally run. No cloud. No subscription.
Inspired by WhisperFlow.

---

## The Two Modes

| Mode | How it starts | How it ends | Paste target |
|------|---------------|-------------|--------------|
| **Hotkey** (push-to-talk) | Hold Right ⌥ | Release key | App that was focused *when key was pressed* |
| **Continuous VAD** | App launches | Ctrl+C | Last non-terminal app in focus |

The hotkey mode is the primary mode (matches WhisperFlow's UX). The rest of this document focuses on it.

---

## End-to-End Flow: Hotkey Mode

```
User presses Right ⌥ (while in, say, Notes.app)
        │
        ▼
[pynput Listener]  ──── global keyboard hook (works system-wide)
        │
        ├─ Snapshot frontmost app via osascript → "Notes"
        ├─ Start buffering mic audio
        │
        ▼
[AudioCapture]  ──── sounddevice InputStream @ 16 kHz mono
        │              chunks of 512 samples (~32ms each)
        │              pushed into a thread-safe queue
        │
        ▼  (audio drain loop reads queue while key is held)
[HotkeyRecorder._buffer]  ──── raw np.float32 chunks accumulate
        │
User releases Right ⌥
        │
        ▼
[HotkeyRecorder._emit()]
        ├─ Concatenates all buffered chunks → one np.ndarray
        └─ Puts (utterance, "Notes") into utterance_queue
                │
                ▼
[HotkeyPipeline._processing_loop]  ──── background thread
                │
                ▼
[WhisperEngine.transcribe()]  ──── mlx-whisper (Apple Silicon GPU)
        │  model: whisper-medium.en-mlx (from HuggingFace, cached locally)
        │  input: float32 numpy array
        │  output: raw transcript string  e.g. "delta, fix this up for me"
        │
        ▼
[CommandDetector.strip_llm_prefix()]
        ├─ Does transcript start with "delta"?
        │     YES → strip prefix, pass remainder to LLM
        │     NO  → use transcript as-is (no LLM call)
        │
        ▼  (if "delta" prefix detected)
[LLMRewriter.rewrite()]  ──── Ollama local server (HTTP on localhost:11434)
        │  model: llama3.2:3b
        │  system prompt: grammar/punctuation fixer, no hallucination
        │  context: last 5 cleaned sentences (for pronoun resolution)
        │  output: cleaned text
        │
        ▼
[ContextManager.add()]  ──── rolling window of last 5 cleaned sentences
        │  used in the NEXT utterance's LLM call
        │
        ▼
[AutoPaster.paste_by_name(text, "Notes")]
        │
        ├─ Save current clipboard contents (NSPasteboard)
        ├─ Write transcribed text to clipboard
        ├─ osascript: tell Notes to activate + wait 120ms
        ├─ osascript: System Events → keystroke "v" using command down
        ├─ Wait 150ms
        └─ Restore previous clipboard contents
                │
                ▼
        Text appears in Notes.app  ✓
                │
                ▼
[HotkeyResult event]  ──── sent to event_queue
                │
                ▼
[DictationMenubar / terminal loop]
        ├─ Menubar icon: ⌥ → ⏺ → ✓ → ⌥ (resets after 2s)
        └─ Terminal: prints timing + paste status
```

---

## Component Breakdown

### `main.py` — Entry Point & Mode Router
- Parses CLI args (`--hotkey`, `--paste`, `--model`, etc.)
- Wires together all components and hands off to a mode runner
- `run_hotkey_mode()` is the WhisperFlow-equivalent path

### `config.py` — Configuration Dataclasses
All tuneable parameters in one place:
- `AudioConfig`: sample rate (16kHz), chunk size (512 samples), VAD thresholds
- `WhisperConfig`: model name, language, optional initial prompt
- `LLMConfig`: Ollama model name, base URL, temperature
- `HotkeyConfig`: which key to use, minimum hold duration (anti-tap filter)
- `ContextConfig`: how many previous sentences to keep

### `audio/capture.py` — `AudioCapture`
- Opens a `sounddevice.InputStream` at 16kHz, mono, 512-sample blocks
- The `_callback` runs on sounddevice's audio thread — just copies chunks into a `queue.Queue`
- Consumer calls `get_chunk()` from its own thread (non-blocking with timeout)

### `audio/hotkey_recorder.py` — `HotkeyRecorder`
- The push-to-talk core
- Uses `pynput.keyboard.Listener` for a **global** key listener (works even when terminal is not focused)
- On key press:
  - Calls `osascript` via subprocess to snapshot the frontmost app name (must happen before Terminal steals focus)
  - Sets a `threading.Event` flag that tells the drain loop to buffer audio
- On key release: clears the flag; drain loop detects the transition and calls `_emit()`
- `_emit()` concatenates buffered chunks, discards if too short (accidental tap), fires `on_utterance` callback

### `audio/vad.py` — `EnergyVAD` (continuous mode only)
- A simple state machine: IDLE → SPEAKING → IDLE
- Computes RMS of each chunk; if RMS > threshold for N consecutive chunks → speech started
- If RMS < threshold for M consecutive chunks → speech ended, emit utterance
- Keeps a pre-buffer so leading audio isn't lost

### `transcription/engine.py` — `WhisperEngine`
- Wraps `mlx_whisper.transcribe()`
- Models are downloaded from HuggingFace on first use and cached locally (`~/.cache/huggingface/`)
- `load()` runs a dummy transcription at startup to warm the model (avoids slow first real transcription)
- Filters out known Whisper hallucination phrases ("thank you for watching", etc.)

### `pipeline/command_detector.py` — `strip_llm_prefix`
- Checks if transcript starts with "delta" (the LLM trigger word)
- Returns `(is_llm: bool, remaining_text: str)`
- Tolerates Whisper adding leading quotes or punctuation before "delta"

### `llm/rewriter.py` — `LLMRewriter`
- Connects to Ollama's HTTP API via the `ollama` Python client
- `load()` pings the model with a warm-up request
- `rewrite()` sends a system+user prompt; the LLM cleans grammar, removes filler words, fixes punctuation
- Guards against hallucination: if output is >2.5x longer than input, falls back to raw transcript

### `pipeline/context_manager.py` — `ContextManager`
- A list of the last N cleaned sentences (default 5)
- Passed to the LLM as context so it can resolve "it", "them", "that" across utterances
- `get_context()` returns them joined as a single string

### `pipeline/hotkey_pipeline.py` — `HotkeyPipeline`
- Orchestrates hotkey mode
- Has two queues: `_utt_queue` (recorder → processing thread) and `_event_queue` (processing thread → main thread)
- Single processing thread: dequeues utterances, runs Whisper → maybe LLM → paste → emits `HotkeyResult`
- Main thread only reads from `_event_queue` for display — never blocks on transcription

### `pipeline/processor.py` — `AsyncPipeline` (continuous mode only)
- Two-thread design: Thread 1 = AudioCapture + VAD, Thread 2 = Whisper + LLM
- Mic is always live; capture is never blocked by transcription
- Utterance queue is bounded (max 8) — drops oldest if Whisper can't keep up

### `output/paster.py` — `AutoPaster`
- The macOS-specific paste mechanism
- **`paste_by_name()`** (hotkey mode): uses `osascript` to activate the target app + send Cmd+V
- **`paste()`** (continuous mode): uses `NSRunningApplication.activateWithOptions_()` directly
- Clipboard dance: save old contents → write text → paste → restore old contents
- `poll()` tracks frontmost app via `NSWorkspace` (must be called from main thread — AppKit limitation)

### `output/menubar.py` — `DictationMenubar`
- Uses `rumps` to put an icon in the macOS menubar
- States: `⌥` (idle) → `⏺` (recording) → `✓` (done, resets after 2s)
- A `rumps.Timer` polls the pipeline event queue every 50ms on the main NSRunLoop
- Falls back to a plain terminal event loop if `rumps` is not installed

---

## External Dependencies

| Library | What it does | Why this one |
|---------|--------------|--------------|
| `sounddevice` | Opens microphone stream, fires callback per chunk | Thin wrapper over PortAudio; works on macOS without extras |
| `numpy` | Audio as float32 arrays; RMS computation | Standard; mlx-whisper expects numpy input |
| `mlx-whisper` | Speech-to-text on Apple Silicon | Uses the MLX framework to run on the M-series Neural Engine/GPU — much faster than CPU Whisper |
| `pynput` | Global keyboard listener | The only pure-Python way to listen to keys system-wide on macOS without Quartz |
| `ollama` (Python client) | HTTP client for local Ollama server | Official client; Ollama manages model download/serving |
| `pyobjc` (`AppKit`) | `NSWorkspace`, `NSPasteboard`, `NSRunningApplication` | The macOS Objective-C bridge — required to talk to the system clipboard and app switcher |
| `rumps` | macOS menubar app framework | Wraps NSStatusBar/NSRunLoop boilerplate |

---

## macOS Permissions Required

| Permission | Why |
|------------|-----|
| **Microphone** | `sounddevice` opens the input stream — macOS prompts on first use |
| **Accessibility** | `osascript` + `System Events` needs this to send keystrokes (Cmd+V) to other apps. Grant in System Settings → Privacy & Security → Accessibility for your terminal |
| **Automation** (optional) | Needed if you want `osascript` to `tell Notes/Safari/etc to activate` without prompts |

---

## Infrastructure (what must be running)

| Thing | Purpose |
|-------|---------|
| **Ollama server** (`ollama serve`) | Local LLM inference. Listens on `localhost:11434`. Only needed if you use the "delta" prefix to trigger rewrites |
| **Whisper model** (auto-downloaded) | Fetched from HuggingFace on first run, cached at `~/.cache/huggingface/hub/`. ~500MB for `medium.en` |
| **Llama model** (auto-downloaded by Ollama) | `ollama pull llama3.2:3b` — ~2GB, stored in `~/.ollama/models/` |

---

## Key Design Decisions

**Why snapshot the app at key-press, not key-release?**
When you release the key, macOS has already given focus to the terminal (or wherever). The app name must be captured the instant the key goes down, while the user's app still has focus. `osascript` runs in a separate process so it can query System Events safely from any thread.

**Why `osascript` for Cmd+V instead of a Quartz event?**
Quartz requires the `quartz` Python package (an extra dependency) and needs careful thread management. `osascript` + `System Events` works reliably from any subprocess and only needs Accessibility permission, which users already grant.

**Why restore the clipboard after pasting?**
The user might have something on their clipboard they want to keep. Overwriting it permanently with the transcript would be destructive. The paste-and-restore pattern is the same thing WhisperFlow and other tools do.

**Why MLX Whisper instead of OpenAI's `whisper` package?**
On Apple Silicon Macs, MLX runs on the Neural Engine / GPU. The standard `whisper` package runs on CPU. `medium.en` via MLX is 5-10x faster than CPU, which is the difference between a 1s wait and a 5-8s wait.

**Why a separate LLM trigger word ("delta") instead of always using the LLM?**
LLM adds ~1-2s latency. For quick notes or numbers where you don't need grammar fixing, raw Whisper output is instant and perfectly fine. The trigger word gives you the choice per utterance.
