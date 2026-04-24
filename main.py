#!/usr/bin/env python3
"""
Local AI Dictation — Phase 2 + 4
Modes:
  default        continuous VAD, terminal output
  --paste        continuous VAD, auto-paste to tracked app
  --hotkey       push-to-talk, paste to app that was focused at key-press
"""

import argparse
import signal
import sys
import time

import numpy as np
import sounddevice as sd

from config import Config
from vocab import load_vocab, VOCAB_FILE
from audio.capture import AudioCapture
from audio.vad import EnergyVAD
from audio.hotkey_recorder import HotkeyRecorder
from transcription.engine import WhisperEngine
from llm.rewriter import LLMRewriter
from pipeline.processor import AsyncPipeline, SpeechStarted, ProcessingResult, ProcessingWarning
from pipeline.hotkey_pipeline import HotkeyPipeline, HotkeyRecordingStarted, HotkeyResult, HotkeyWarning
from output.paster import AutoPaster, check_accessibility, print_accessibility_instructions


def parse_args():
    parser = argparse.ArgumentParser(description="Local AI Dictation")
    parser.add_argument("--no-llm",    action="store_true", help="Raw Whisper output only")
    parser.add_argument("--debug",     action="store_true", help="Show raw transcript too")
    parser.add_argument("--model",     default="medium.en",
                        choices=["tiny.en", "base.en", "small.en", "medium.en", "large-v3"])
    parser.add_argument("--llm-model", default="llama3.2:3b")
    parser.add_argument("--threshold", type=float, default=None,
                        help="VAD energy threshold (continuous mode only)")
    parser.add_argument("--prompt", type=str, default=None, metavar="TEXT",
                        help="Initial prompt to bias Whisper (vocabulary, style, names)")
    parser.add_argument("--monitor",   action="store_true",
                        help="Live RMS meter — no transcription")

    # Continuous paste mode
    parser.add_argument("--paste",  action="store_true",
                        help="Auto-paste results into the last focused app")
    parser.add_argument("--target", type=str, default=None, metavar="APP",
                        help='Pin paste to a specific app e.g. --target "Notes"')

    # Push-to-talk mode
    parser.add_argument("--hotkey", action="store_true",
                        help="Push-to-talk: hold key while speaking, release to paste")
    parser.add_argument("--hotkey-key", default="right_option",
                        choices=["right_option", "right_cmd", "right_ctrl", "right_shift",
                                 "f13", "f14", "f15"],
                        help="Which key to hold (default: right_option / Right ⌥)")
    return parser.parse_args()


def run_monitor(threshold: float):
    SCALE, WIDTH = 0.15, 40
    print(f"\n  Live RMS Monitor  (threshold={threshold:.4f})  Ctrl+C to stop")
    print(f"  Bar: 0.0 {'─' * WIDTH} {SCALE}\n")
    try:
        with sd.InputStream(samplerate=16000, channels=1, blocksize=512,
                            dtype=np.float32) as stream:
            while True:
                data, _ = stream.read(512)
                rms   = float(np.sqrt(np.mean(data[:, 0] ** 2)))
                filled = int(min(rms / SCALE, 1.0) * WIDTH)
                bar   = "█" * filled + "░" * (WIDTH - filled)
                t_pos = int(min(threshold / SCALE, 1.0) * WIDTH)
                bar   = bar[:t_pos] + "|" + bar[t_pos + 1:]
                label = "SPEECH" if rms > threshold else "quiet "
                print(f"  {label}  RMS={rms:.4f}  [{bar}]", end="\r")
    except KeyboardInterrupt:
        print()
        sys.exit(0)


def setup_paster(args, config) -> tuple[AutoPaster, bool]:
    """Initialise AutoPaster. Returns (paster, paste_enabled)."""
    paster = AutoPaster()
    enabled = args.paste or args.hotkey or bool(args.target)

    if not enabled:
        return paster, False

    if not paster.enabled:
        print("  Auto-paste requires pyobjc (pip install pyobjc-framework-Cocoa)")
        return paster, False

    if not check_accessibility():
        print_accessibility_instructions()
        print("  Continuing without auto-paste.")
        return paster, False

    if args.target:
        paster.set_target_by_name(args.target)

    return paster, True


def print_banner(mode: str):
    print()
    print("  ╔══════════════════════════════════╗")
    print("  ║     Local AI Dictation  v0.2     ║")
    print(f"  ║  mode: {mode:<26}║")
    print("  ╚══════════════════════════════════╝")
    print()


# ── Hotkey mode ───────────────────────────────────────────────────────────────

def _run_hotkey_terminal(pipeline, config):
    """Terminal-only event loop fallback (used when rumps is unavailable)."""
    running = True
    recording_shown = False

    def on_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)

    try:
        while running:
            event = pipeline.get_event(timeout=0.1)
            if event is None:
                continue

            if isinstance(event, HotkeyRecordingStarted):
                print("  ● recording...", end="\r", flush=True)
                recording_shown = True

            elif isinstance(event, HotkeyResult):
                if recording_shown:
                    print(" " * 30, end="\r")
                    recording_shown = False

                if config.debug:
                    print(f"  [raw]  {event.raw}")

                label = "out" if event.t_llm > 0 else "raw"
                print(f"  [{label}]  {event.cleaned}")

                timing = f"whisper {event.t_whisper:.2f}s"
                if event.t_llm > 0:
                    timing += f" | llm {event.t_llm:.2f}s"

                paste_status = f"pasted → {event.app_name}" if event.pasted else "paste failed"
                print(f"  ⏱  {event.duration:.1f}s · {timing} · {paste_status}\n", flush=True)

            elif isinstance(event, HotkeyWarning):
                print(f"\n  ⚠  {event.message}\n", flush=True)

    except KeyboardInterrupt:
        pass


def run_hotkey_mode(args, config, paster):
    config.hotkey.key = args.hotkey_key

    recorder = HotkeyRecorder(config)
    if not recorder.available:
        print("  pynput not installed. Run: pip install pynput")
        sys.exit(1)

    whisper  = WhisperEngine(config.whisper)
    rewriter = LLMRewriter(config.llm)

    whisper.load()
    rewriter.load()

    pipeline = HotkeyPipeline(recorder, whisper, rewriter, paster, config)

    key_label = {
        "right_option": "Right ⌥ Option",
        "right_cmd":    "Right ⌘ Cmd",
        "right_ctrl":   "Right ⌃ Ctrl",
        "right_shift":  "Right ⇧ Shift",
        "f13": "F13", "f14": "F14", "f15": "F15",
    }.get(args.hotkey_key, args.hotkey_key)

    print_banner("push-to-talk")
    print(f"  Hold [{key_label}] while speaking, release to paste.")
    print(f"  Works in any app — no need to switch to terminal.")

    pipeline.start()

    try:
        from output.menubar import DictationMenubar
        print(f"  Menubar icon active — use ⌥ in menu bar to quit.\n")
        app = DictationMenubar(pipeline, config, paster)
        app.run()   # blocks; quit handled inside DictationMenubar._on_quit
    except ImportError:
        print(f"  Ctrl+C to quit.\n")
        _run_hotkey_terminal(pipeline, config)
        print("\nStopping...")
        pipeline.stop()
        paster.stop()
        print("Goodbye.")


# ── Continuous VAD mode ───────────────────────────────────────────────────────

def run_continuous_mode(args, config, paster, paste_enabled):
    capture  = AudioCapture(config.audio)
    vad      = EnergyVAD(config.audio)
    whisper  = WhisperEngine(config.whisper)
    rewriter = LLMRewriter(config.llm)

    whisper.load()
    rewriter.load()

    pipeline = AsyncPipeline(capture, vad, whisper, rewriter, config)

    if paste_enabled and not args.target and not args.hotkey:
        detected = paster.auto_detect()
        if detected:
            print(f"  [auto-detected paste target → {detected}]")
        else:
            candidates = paster.find_candidates()
            if candidates:
                print(f"  Available targets: {', '.join(candidates)}")
                print(f"  Switch to your target app to select it.")

    mode = "auto-paste" if paste_enabled else "terminal output"
    print_banner(mode)
    print("─" * 50)
    if paste_enabled:
        print("  Listening... speak naturally, results paste automatically.")
    else:
        print("  Listening... (Ctrl+C to stop)")
    print("─" * 50)
    print()

    pipeline.start()
    running = True

    def on_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)
    recording_shown = False
    last_poll = 0.0

    try:
        while running:
            event = pipeline.get_event(timeout=0.1)

            if paste_enabled and not args.target:
                now = time.monotonic()
                if now - last_poll >= 0.5:
                    paster.poll()
                    last_poll = now

            if event is None:
                continue

            if isinstance(event, SpeechStarted):
                print("  ● recording...", end="\r", flush=True)
                recording_shown = True

            elif isinstance(event, ProcessingResult):
                if recording_shown:
                    print(" " * 30, end="\r")
                    recording_shown = False

                if config.debug:
                    print(f"  [raw]  {event.raw}")

                label = "out" if event.t_llm > 0 else "raw"
                print(f"  [{label}]  {event.cleaned}")

                timing = f"whisper {event.t_whisper:.2f}s"
                if event.t_llm > 0:
                    timing += f" | llm {event.t_llm:.2f}s"

                if paste_enabled:
                    ok = paster.paste(event.cleaned)
                    status = f"pasted → {paster.target_name}" if ok else "paste failed — switch to target app first"
                    print(f"  ⏱  {event.duration:.1f}s · {timing} · {status}\n", flush=True)
                else:
                    print(f"  ⏱  {event.duration:.1f}s · {timing}\n", flush=True)

            elif isinstance(event, ProcessingWarning):
                print(f"\n  ⚠  {event.message}\n", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        pipeline.stop()
        paster.stop()
        print("Goodbye.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = Config()

    config.whisper.model = args.model
    config.llm.model     = args.llm_model
    config.debug         = args.debug
    if args.no_llm:
        config.llm.enabled = False

    # Load personal vocabulary as Whisper's initial_prompt. --prompt overrides.
    vocab_prompt, vocab_count = load_vocab()
    if vocab_prompt:
        config.whisper.initial_prompt = vocab_prompt
        print(f"  [vocab] {vocab_count} entries from {VOCAB_FILE}")
    else:
        print(f"  [vocab] no entries yet — edit {VOCAB_FILE} to add words")

    if args.prompt is not None:
        config.whisper.initial_prompt = args.prompt
    if args.threshold is not None:
        config.audio.energy_threshold = args.threshold

    if args.monitor:
        run_monitor(args.threshold or config.audio.energy_threshold)
        return

    paster, paste_enabled = setup_paster(args, config)

    if args.hotkey:
        run_hotkey_mode(args, config, paster)
    else:
        run_continuous_mode(args, config, paster, paste_enabled)


if __name__ == "__main__":
    main()
