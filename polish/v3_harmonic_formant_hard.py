#!/usr/bin/env python3
"""Aggressive humanesque polish for Fusion V3.

V3 remains the phonetic skeleton. This stage deliberately keeps:
  * V3 timing
  * V3 harmonic locations / F0
  * V3 complex phase

It audibly changes two texture layers:
  1) harmonic-series chassis: explicit per-harmonic amplitude sculpting on voiced
     frames (H1 emphasis, high-harmonic rolloff, odd/even contrast, and strong
     compression of overly rigid comb contrast);
  2) formant bandwidth layer: stronger symmetric frequency-domain damping of the
     broad V3 envelope, without deliberately translating formant centres.

Output is 44.1 kHz, mono, 32-bit float WAV by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from math import gcd

import librosa
import numpy as np
import soundfile as sf
from scipy.fft import dct, idct
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly

EPS = 1e-9
TARGET_SR = 44100
N_FFT = 4096
HOP = 256
KEEP_CEPSTRAL = 54


def load_mono(path: Path):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), int(sr)


def resample_to(x: np.ndarray, sr: int, target_sr: int):
    if sr == target_sr:
        return x
    g = gcd(sr, target_sr)
    return resample_poly(x, target_sr // g, sr // g)


def envelope(logmag: np.ndarray) -> np.ndarray:
    c = dct(logmag, type=2, axis=0, norm="ortho")
    c[KEEP_CEPSTRAL:, :] = 0.0
    return idct(c, type=2, axis=0, norm="ortho")


def voice_track(x: np.ndarray, sr: int, frames: int):
    f0, voiced, prob = librosa.pyin(
        x, fmin=65.0, fmax=320.0, sr=sr,
        frame_length=4096, hop_length=HOP, center=True,
    )
    if prob is None:
        prob = np.asarray(voiced, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    prob = np.nan_to_num(np.asarray(prob, dtype=float), nan=0.0)
    if len(f0) < frames:
        f0 = np.pad(f0, (0, frames-len(f0)), constant_values=np.nan)
        prob = np.pad(prob, (0, frames-len(prob)))
    f0 = f0[:frames]
    prob = prob[:frames]
    weight = np.clip((prob - 0.18) / 0.55, 0.0, 1.0)
    weight = gaussian_filter1d(weight, sigma=1.0)
    return f0, weight


def harmonic_chassis(
    logmag: np.ndarray,
    env: np.ndarray,
    freqs: np.ndarray,
    f0: np.ndarray,
    voiced: np.ndarray,
    amount: float,
    tilt_db_oct: float,
    odd_even_db: float,
    h1_db: float,
) -> np.ndarray:
    """Explicitly reshape V3's voiced harmonic series without moving F0."""
    residual = logmag - env

    # First make the comb itself less rigid. Unlike the previous polish this is
    # intentionally strong: amount=1 moves all voiced fine residual toward a
    # heavily compressed version rather than a tiny interpolation.
    knee = 0.72
    compressed = knee * np.tanh(residual / knee)
    shaped = residual + voiced[None, :] * amount * (compressed - residual)

    hz_per_bin = freqs[1] - freqs[0]
    nyq = freqs[-1]

    # Then explicitly sculpt harmonic peaks. This is the audible chassis.
    for t in range(logmag.shape[1]):
        if voiced[t] < 0.08 or not np.isfinite(f0[t]) or f0[t] <= 0:
            continue
        f = float(f0[t])
        max_h = int(min(80, nyq // f))
        if max_h < 1:
            continue

        local = amount * voiced[t]
        sigma_hz = max(28.0, min(70.0, 0.28 * f))
        sigma_bins = max(1.0, sigma_hz / hz_per_bin)
        x = np.arange(len(freqs), dtype=float)

        for h in range(1, max_h + 1):
            center = h * f
            if center >= nyq:
                break
            b = center / hz_per_bin

            # Keep the fundamental warmer/stronger, then deliberately roll the
            # harmonic series above H2. This changes voice texture, not F0.
            if h == 1:
                db = h1_db
            else:
                octs = max(0.0, np.log2(h / 2.0))
                db = -tilt_db_oct * octs

            # Break perfectly uniform synthetic harmonic balance.
            if h >= 2:
                db += odd_even_db if (h % 2) else -odd_even_db

            gain_ln = (np.log(10.0) / 20.0) * db * local
            if abs(gain_ln) < 1e-6:
                continue
            mask = np.exp(-0.5 * ((x - b) / sigma_bins) ** 2)
            shaped[:, t] += gain_ln * mask

    return env + shaped


def formant_damping(
    current_logmag: np.ndarray,
    freqs: np.ndarray,
    voiced: np.ndarray,
    amount: float,
    sigma_bins: float,
) -> np.ndarray:
    """Broaden/damp broad resonances while keeping frame energy anchored."""
    env_now = envelope(current_logmag)
    residual = current_logmag - env_now
    widened = gaussian_filter1d(env_now, sigma=sigma_bins, axis=0, mode="nearest")
    band = (freqs >= 120.0) & (freqs <= 7500.0)
    widened -= widened[band].mean(axis=0, keepdims=True) - env_now[band].mean(axis=0, keepdims=True)
    mix = amount * voiced[None, :]
    env2 = env_now + mix * (widened - env_now)
    return env2 + residual


def polish(x: np.ndarray, sr: int, preset: str):
    if preset == "hard":
        p = dict(amount=1.0, tilt_db_oct=4.8, odd_even_db=1.8, h1_db=3.0,
                 formant_amount=0.42, formant_sigma=5.0)
    elif preset == "extreme":
        p = dict(amount=1.0, tilt_db_oct=7.5, odd_even_db=3.2, h1_db=5.0,
                 formant_amount=0.68, formant_sigma=8.0)
    else:
        raise ValueError(preset)

    z = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                     window="hann", center=True)
    mag = np.maximum(np.abs(z), EPS)
    phase = np.angle(z)
    logmag = np.log(mag)
    env = envelope(logmag)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    f0, voiced = voice_track(x, sr, z.shape[1])

    shaped = harmonic_chassis(logmag, env, freqs, f0, voiced,
                              p["amount"], p["tilt_db_oct"],
                              p["odd_even_db"], p["h1_db"])
    shaped = formant_damping(shaped, freqs, voiced,
                             p["formant_amount"], p["formant_sigma"])

    out_mag = np.exp(shaped)
    y = librosa.istft(out_mag * np.exp(1j*phase), hop_length=HOP,
                      win_length=N_FFT, window="hann", center=True,
                      length=len(x))

    # Preserve RMS, then only attenuate if needed to keep all common encoders safe.
    rms_x = np.sqrt(np.mean(x*x) + EPS)
    rms_y = np.sqrt(np.mean(y*y) + EPS)
    y *= rms_x / rms_y
    peak = np.max(np.abs(y)) + EPS
    if peak > 0.985:
        y *= 0.985 / peak

    # Verify F0 stayed on V3.
    f0_after, _, _ = librosa.pyin(y, fmin=65.0, fmax=320.0, sr=sr,
                                  frame_length=4096, hop_length=HOP, center=True)
    n = min(len(f0), len(f0_after))
    valid = np.isfinite(f0[:n]) & np.isfinite(f0_after[:n])
    if np.any(valid):
        cents = 1200*np.log2(f0_after[:n][valid] / f0[:n][valid])
        med = float(np.median(cents)); mad = float(np.median(np.abs(cents)))
    else:
        med = mad = float("nan")
    return y, med, mad, int(np.sum(valid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--preset", choices=["hard", "extreme"], default="hard")
    ap.add_argument("--sr", type=int, default=TARGET_SR)
    args = ap.parse_args()

    x, source_sr = load_mono(args.input)
    x = resample_to(x, source_sr, args.sr)
    y, med, mad, frames = polish(x, args.sr, args.preset)
    sf.write(args.output, y.astype(np.float32), args.sr, subtype="FLOAT")
    print(f"preset={args.preset}; source_sr={source_sr}; output_sr={args.sr}; subtype=FLOAT; mono=1")
    print(f"F0 median shift={med:.3f} cents; median abs shift={mad:.3f} cents; frames={frames}")


if __name__ == "__main__":
    main()
