#!/usr/bin/env python3
"""Aeyian TTS fusion-v3: pitch-locked phonetic-print harness.

Goal:
  * V15/legacy supplies pitch, harmonic locations, phase and fine voice texture.
  * qce-ng supplies the *phonetic spectrographic print*.

Important difference from v1/v2:
  * NO DTW based on spectral similarity. That could warp the guide to match the
    carrier's existing pronunciation and therefore preserve the very thing we
    want to replace.
  * qce and carrier speech spans are mapped monotonically by normalized speech
    time, so qce's own phonetic trajectory remains the target.
  * The guide representation is a low-order LPC spectral envelope. Harmonic
    spacing/F0 is deliberately excluded from the guide.
  * Reconstruction modifies only carrier magnitudes. Carrier complex phase and
    harmonic-bin locations are retained, so qce pitch is never transplanted.

At --strength 1.0 this is a deliberately hard test: the speech-defining LPC
shape comes from qce (within a safety clamp), while carrier periodicity remains
V15.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy import signal

N_FFT = 1024
HOP = 128
LPC_ORDER = 18
PRINT_LOW_HZ = 100.0
PRINT_HIGH_HZ = 7000.0
MAX_DELTA_DB = 24.0
EPS = 1e-8


def load(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    x, got_sr = librosa.load(path, sr=sr, mono=True)
    return x.astype(np.float64), got_sr


def speech_bounds(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Return coarse active-speech bounds in seconds."""
    rms = librosa.feature.rms(
        y=x, frame_length=N_FFT, hop_length=HOP, center=True
    )[0]
    db = librosa.amplitude_to_db(rms + EPS, ref=np.max)
    active = np.flatnonzero(db > -35.0)
    if len(active) == 0:
        return 0.0, len(x) / sr
    start = max(0, active[0] * HOP - N_FFT // 2) / sr
    end = min(len(x), active[-1] * HOP + N_FFT // 2) / sr
    return start, end


def lpc_print(x: np.ndarray, sr: int):
    """Return STFT and pitch-suppressed LPC log spectral envelope.

    A low LPC order follows broad vocal-tract/noise-envelope shape while being
    unable to encode the dense harmonic comb that carries F0.
    """
    z = librosa.stft(
        x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window="hann", center=True
    )
    mag = np.maximum(np.abs(z), EPS)
    logmag = np.log(mag)

    padded = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = librosa.util.frame(
        padded, frame_length=N_FFT, hop_length=HOP
    )
    freqs = np.linspace(0.0, sr / 2.0, N_FFT // 2 + 1)
    fit_band = (freqs >= PRINT_LOW_HZ) & (freqs <= PRINT_HIGH_HZ)

    env = np.empty_like(logmag)
    window = np.hanning(N_FFT)

    for i in range(frames.shape[1]):
        frame = frames[:, i] * window
        if np.sqrt(np.mean(frame * frame)) < 1e-5:
            env[:, i] = logmag[:, i]
            continue
        try:
            a = librosa.lpc(frame, order=LPC_ORDER)
            _, h = signal.freqz(
                [1.0], a, worN=N_FFT // 2 + 1, fs=sr
            )
            shape = np.log(np.maximum(np.abs(h), EPS))
            # LPC gives shape but arbitrary gain. Anchor it to this frame only
            # for numerical stability; gain is removed again before transfer.
            offset = np.median(logmag[fit_band, i] - shape[fit_band])
            env[:, i] = shape + offset
        except Exception:
            env[:, i] = logmag[:, i]

    return z, env, freqs


def normalized_shape(env: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    band = (freqs >= PRINT_LOW_HZ) & (freqs <= PRINT_HIGH_HZ)
    return env - env[band].mean(axis=0, keepdims=True)


def map_guide_print(
    guide_shape: np.ndarray,
    carrier_frames: int,
    sr: int,
    carrier_span: tuple[float, float],
    guide_span: tuple[float, float],
) -> np.ndarray:
    """Map qce print by normalized speech time, never spectral similarity."""
    tc = np.arange(carrier_frames) * HOP / sr
    tg = np.arange(guide_shape.shape[1]) * HOP / sr
    cs, ce = carrier_span
    gs, ge = guide_span

    mapped = np.empty_like(tc)
    before = tc < cs
    speech = (tc >= cs) & (tc <= ce)
    after = tc > ce

    mapped[before] = gs + (tc[before] - cs)
    mapped[speech] = gs + (tc[speech] - cs) * (ge - gs) / max(ce - cs, EPS)
    mapped[after] = ge + (tc[after] - ce)

    out = np.empty((guide_shape.shape[0], carrier_frames), dtype=np.float64)
    for k in range(guide_shape.shape[0]):
        out[k] = np.interp(
            mapped, tg, guide_shape[k],
            left=guide_shape[k, 0], right=guide_shape[k, -1]
        )
    return out


def fuse(carrier: np.ndarray, guide: np.ndarray, sr: int, strength: float):
    zc, carrier_env, freqs = lpc_print(carrier, sr)
    _, guide_env, _ = lpc_print(guide, sr)

    carrier_shape = normalized_shape(carrier_env, freqs)
    guide_shape = normalized_shape(guide_env, freqs)
    guide_aligned = map_guide_print(
        guide_shape,
        zc.shape[1],
        sr,
        speech_bounds(carrier, sr),
        speech_bounds(guide, sr),
    )

    logmag = np.log(np.maximum(np.abs(zc), EPS))

    # This is the actual phonetic harness. qce controls the broad per-frame
    # spectral shape; V15 retains its harmonic comb and phase.
    delta = guide_aligned - carrier_shape
    max_ln = np.log(10.0) * MAX_DELTA_DB / 20.0
    delta = np.clip(delta, -max_ln, max_ln)

    # Only the speech-defining band is harnessed. Smooth edge ramps avoid
    # discontinuities; this is not an F0-dependent weighting.
    fw = np.ones_like(freqs)
    fw[freqs < 70.0] = 0.0
    low = (freqs >= 70.0) & (freqs < PRINT_LOW_HZ)
    fw[low] = (freqs[low] - 70.0) / (PRINT_LOW_HZ - 70.0)
    high = (freqs > PRINT_HIGH_HZ) & (freqs < 8000.0)
    fw[high] = (8000.0 - freqs[high]) / (8000.0 - PRINT_HIGH_HZ)
    fw[freqs >= 8000.0] = 0.0

    out_logmag = logmag + strength * fw[:, None] * delta
    out_mag = np.exp(out_logmag)

    # Explicit pitch lock: use V15 complex phase and frequency-bin locations.
    zout = out_mag * np.exp(1j * np.angle(zc))
    y = librosa.istft(
        zout, hop_length=HOP, win_length=N_FFT,
        window="hann", center=True, length=len(carrier)
    )

    # Preserve overall V15 loudness for fair listening comparisons.
    rms_c = np.sqrt(np.mean(carrier * carrier) + EPS)
    rms_y = np.sqrt(np.mean(y * y) + EPS)
    y *= rms_c / rms_y
    return y


def pitch_check(carrier: np.ndarray, output: np.ndarray, sr: int) -> str:
    """Diagnostic only: verify that F0 stayed on the carrier."""
    def track(x):
        f0, _, _ = librosa.pyin(
            x, fmin=70.0, fmax=300.0, sr=sr,
            frame_length=2048, hop_length=HOP
        )
        return f0

    a = track(carrier)
    b = track(output)
    n = min(len(a), len(b))
    valid = np.isfinite(a[:n]) & np.isfinite(b[:n])
    if not np.any(valid):
        return "F0 check: no jointly voiced frames"
    cents = 1200.0 * np.log2(b[:n][valid] / a[:n][valid])
    return (
        f"F0 check: median shift={np.median(cents):.3f} cents; "
        f"median abs shift={np.median(np.abs(cents)):.3f} cents; "
        f"frames={valid.sum()}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carrier", type=Path)
    ap.add_argument("guide", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()
    if not 0.0 <= args.strength <= 1.0:
        ap.error("--strength must be 0..1")

    carrier, sr = load(args.carrier)
    guide, _ = load(args.guide, sr=sr)
    y = fuse(carrier, guide, sr, args.strength)
    sf.write(args.output, y, sr, subtype="PCM_16")
    print(pitch_check(carrier, y, sr))


if __name__ == "__main__":
    main()
