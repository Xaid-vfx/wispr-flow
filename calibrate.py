#!/usr/bin/env python3
"""
VAD Calibration Tool

Run this before using the dictation system to find the right
energy threshold for your microphone and environment.

Usage:
    python calibrate.py           # full calibration
    python calibrate.py --monitor # live RMS monitor (no calibration)
"""

import sys
import time
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SIZE = 512


def live_bar(rms: float, threshold: float = None, width: int = 40) -> str:
    """Render a fixed-width bar showing RMS energy."""
    # Scale: 0.0 → 0.15 maps to 0 → width
    scale = 0.15
    filled = int(min(rms / scale, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    if threshold is not None:
        marker = int(min(threshold / scale, 1.0) * width)
        bar = bar[:marker] + "|" + bar[marker + 1:]
    return bar


def monitor_mode():
    """Continuously display mic RMS so you can diagnose detection issues."""
    print()
    print("Live RMS Monitor  (Ctrl+C to stop)")
    print("─" * 50)
    print("  Bar scale: 0.0 ──────────────────── 0.15")
    print()
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            blocksize=CHUNK_SIZE, dtype=np.float32) as stream:
            while True:
                data, _ = stream.read(CHUNK_SIZE)
                rms = float(np.sqrt(np.mean(data[:, 0] ** 2)))
                bar = live_bar(rms)
                print(f"  RMS={rms:.4f}  [{bar}]", end="\r")
    except KeyboardInterrupt:
        print()
        print("Done.")


def calibrate():
    print()
    print("VAD Threshold Calibration")
    print("─" * 50)
    print()
    print("Step 1 of 2: SILENCE — stay completely quiet for 5 seconds...")
    silence_samples = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=CHUNK_SIZE, dtype=np.float32) as stream:
        start = time.time()
        while time.time() - start < 5:
            data, _ = stream.read(CHUNK_SIZE)
            rms = float(np.sqrt(np.mean(data[:, 0] ** 2)))
            silence_samples.append(rms)
            elapsed = time.time() - start
            bar = live_bar(rms)
            print(f"  {elapsed:.1f}s  RMS={rms:.4f}  [{bar}]", end="\r")
        print()

    print()
    print("Step 2 of 2: SPEAK — talk normally for 5 seconds...")
    speech_samples = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=CHUNK_SIZE, dtype=np.float32) as stream:
        start = time.time()
        while time.time() - start < 5:
            data, _ = stream.read(CHUNK_SIZE)
            rms = float(np.sqrt(np.mean(data[:, 0] ** 2)))
            speech_samples.append(rms)
            elapsed = time.time() - start
            bar = live_bar(rms)
            print(f"  {elapsed:.1f}s  RMS={rms:.4f}  [{bar}]", end="\r")
        print()

    silence_samples = np.array(silence_samples)
    speech_samples = np.array(speech_samples)

    silence_mean = float(np.mean(silence_samples))
    silence_std  = float(np.std(silence_samples))
    silence_p95  = float(np.percentile(silence_samples, 95))
    speech_mean  = float(np.mean(speech_samples))
    speech_p25   = float(np.percentile(speech_samples, 25))
    speech_max   = float(np.max(speech_samples))

    # Threshold formula: 3 sigma above noise floor, clipped to stay below speech 25th pct
    threshold_statistical = silence_mean + 3.0 * silence_std
    # Midpoint between silence p95 and speech p25 (the "gap" between the two distributions)
    threshold_midpoint = (silence_p95 + speech_p25) / 2.0
    # Take the lower of the two so we don't overshoot into speech territory
    recommended = min(threshold_statistical, threshold_midpoint)
    # Sanity clamp: never recommend higher than 80% of speech max
    recommended = min(recommended, speech_max * 0.8)
    # Sanity clamp: never recommend lower than 110% of silence mean
    recommended = max(recommended, silence_mean * 1.1)

    separation = speech_mean / silence_mean if silence_mean > 0 else 0

    print()
    print("─" * 50)
    print("Results:")
    print(f"  Noise floor  — mean: {silence_mean:.4f}  95th pct: {silence_p95:.4f}")
    print(f"  Speech level — mean: {speech_mean:.4f}  max:      {speech_max:.4f}")
    print(f"  Separation ratio: {separation:.2f}x  (want > 3x for reliable VAD)")
    print()
    print(f"  ► Recommended threshold: {recommended:.4f}")
    print()

    if separation >= 3.0:
        print("  ✓ Good separation. Energy VAD will work reliably.")
    elif separation >= 1.5:
        print("  △ Marginal separation. VAD will work but may occasionally mis-fire.")
        print("    Try: speak more directly into the mic, or reduce background noise.")
    else:
        print("  ✗ Poor separation (speech barely louder than background noise).")
        print("    Energy VAD will struggle. Causes:")
        print("      - Mic input volume too low (check System Settings → Sound → Input)")
        print("      - Speaking too far from mic or too quietly")
        print("      - Loud environment (fan, AC, traffic)")
        print()
        print("    Immediate fix: run `python calibrate.py --monitor` to see live")
        print("    levels, then adjust your mic input volume until speaking")
        print("    clearly pushes the bar past the midpoint.")

    print()
    print(f"  Update config.py:")
    print(f"    energy_threshold: float = {recommended:.4f}")
    print(f"  Or use as a flag:")
    print(f"    python main.py --threshold {recommended:.4f}")
    print()


if __name__ == "__main__":
    if "--monitor" in sys.argv:
        monitor_mode()
    else:
        calibrate()
