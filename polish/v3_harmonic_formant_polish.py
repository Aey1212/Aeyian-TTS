#!/usr/bin/env python3
"""V3 humanesque polish stage for Aeyian TTS.

V3 remains the phonetic skeleton. This stage deliberately does NOT alter text,
phoneme timing, or F0 trajectory. It changes only two acoustic texture layers:

1. Harmonic texture shaping (main chassis)
   - separate the broad spectral envelope from the fine harmonic residual;
   - gently compress/soften extreme comb contrast on voiced frames;
   - retain the original harmonic locations and complex phase.

2. Formant bandwidth/damping (light top layer)
   - broaden the already-established V3 spectral envelope slightly;
   - preserve formant centres as closely as possible by symmetric frequency
     smoothing and frame-gain re-anchoring.

The master output defaults to 44.1 kHz, mono, 32-bit float WAV. Upsampling does
not invent source bandwidth; it provides a higher-rate/precision processing
container for this and later polish stages.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.fft import dct, idct
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly

EPS = 1e-9
TARGET_SR = 44100
N_FFT = 2048
HOP = 256
# Low-quefrency envelope: broad enough to keep V3's phonetic/formant skeleton,
# too coarse to carry individual F0 harmonics.
KEEP_CEPSTRAL = 42
PEAK_CEILING = 0.985


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), sr


def resample_to(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    from math import gcd
    g = gcd(sr, target_sr)
    return resample_poly(x, target_sr // g, sr // g)


def cepstral_envelope(logmag: np.ndarray) -> np.ndarray:
    c = dct(logmag, type=2, axis=0, norm="ortho")
    c[KEEP_CEPSTRAL:, :] = 0.0
    return idct(c, type=2, axis=0, norm="ortho")


def voiced_weights(x: np.ndarray, sr: int, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return soft voiced weight and V3 F0 track. V3 itself is the only pitch source."""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        x,
        fmin=65.0,
        fmax=320.0,
        sr=sr,
        frame_length=2048,
        hop_length=HOP,
        center=True,
    )
    if voiced_prob is None:
        voiced_prob = np.asarray(voiced_flag, dtype=np.float64)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
    f0 = np.asarray(f0, dtype=np.float64)
    if len(voiced_prob) < frame_count:
        voiced_prob = np.pad(voiced_prob, (0, frame_count - len(voiced_prob)))
        f0 = np.pad(f0, (0, frame_count - len(f0)), constant_values=np.nan)
    voiced_prob = voiced_prob[:frame_count]
    f0 = f0[:frame_count]
    # Do not flicker processing at frame boundaries.
    weight = gaussian_filter1d(np.clip((voiced_prob - 0.25) / 0.55, 0.0, 1.0), sigma=1.0)
    return weight, f0


def shape_harmonic_residual(
    residual: np.ndarray,
    freqs: np.ndarray,
    voiced: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Humanise harmonic contrast without moving harmonic locations.

    The residual is the fine comb/noise structure after removing the broad V3
    envelope. A soft nonlinearity reins in unrealistically sharp peaks/valleys.
    A mild frequency-dependent increase above 2.5 kHz reduces the brittle,
    perfectly-periodic edge without adding breath/noise.
    """
    # Saturating map is identity near zero and progressively compresses only
    # extreme fine-structure contrast.
    knee = 1.45
    softened = knee * np.tanh(residual / knee)

    hi = np.clip((freqs - 2500.0) / 4500.0, 0.0, 1.0)
    local_amount = amount * (0.72 + 0.28 * hi[:, None])
    mix = local_amount * voiced[None, :]
    return residual + mix * (softened - residual)


def damp_formant_bandwidth(
    env: np.ndarray,
    freqs: np.ndarray,
    voiced: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Very gently broaden/damp formant peaks while retaining their centres."""
    if amount <= 0:
        return env

    # Symmetric smoothing broadens narrow resonances instead of translating
    # them. At 44.1 kHz / 2048 FFT, sigma=2.2 is ~47 Hz.
    widened = gaussian_filter1d(env, sigma=2.2, axis=0, mode="nearest")

    # Preserve each frame's mean energy in the speech-defining band.
    band = (freqs >= 120.0) & (freqs <= 7000.0)
    widened -= widened[band].mean(axis=0, keepdims=True) - env[band].mean(axis=0, keepdims=True)

    # Keep this layer intentionally lighter than harmonic shaping.
    mix = amount * voiced[None, :]
    return env + mix * (widened - env)


def polish(
    x: np.ndarray,
    sr: int,
    harmonic_amount: float,
    formant_amount: float,
) -> tuple[np.ndarray, dict[str, float]]:
    z = librosa.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window="hann",
        center=True,
    )
    mag = np.maximum(np.abs(z), EPS)
    phase = np.angle(z)
    logmag = np.log(mag)
    env = cepstral_envelope(logmag)
    residual = logmag - env
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    voiced, f0_before = voiced_weights(x, sr, z.shape[1])
    residual2 = shape_harmonic_residual(residual, freqs, voiced, harmonic_amount)
    env2 = damp_formant_bandwidth(env, freqs, voiced, formant_amount)

    out_mag = np.exp(env2 + residual2)
    zout = out_mag * np.exp(1j * phase)
    y = librosa.istft(
        zout,
        hop_length=HOP,
        win_length=N_FFT,
        window="hann",
        center=True,
        length=len(x),
    )

    # Preserve V3 overall level first, then apply a transparent peak ceiling if
    # texture reshaping created >0 dBFS peaks. 32-bit float itself can represent
    # them, but downstream PCM/MP3 exports should never clip.
    rms_x = np.sqrt(np.mean(x * x) + EPS)
    rms_y = np.sqrt(np.mean(y * y) + EPS)
    y *= rms_x / rms_y
    peak_before_ceiling = float(np.max(np.abs(y)))
    if peak_before_ceiling > PEAK_CEILING:
        y *= PEAK_CEILING / peak_before_ceiling

    # Verification: this stage should not materially move V3 F0.
    f0_after, _, _ = librosa.pyin(
        y,
        fmin=65.0,
        fmax=320.0,
        sr=sr,
        frame_length=2048,
        hop_length=HOP,
        center=True,
    )
    n = min(len(f0_before), len(f0_after))
    valid = np.isfinite(f0_before[:n]) & np.isfinite(f0_after[:n])
    if np.any(valid):
        cents = 1200.0 * np.log2(f0_after[:n][valid] / f0_before[:n][valid])
        median_cents = float(np.median(cents))
        median_abs_cents = float(np.median(np.abs(cents)))
    else:
        median_cents = float("nan")
        median_abs_cents = float("nan")

    return y, {
        "f0_median_shift_cents": median_cents,
        "f0_median_abs_shift_cents": median_abs_cents,
        "voiced_frames": int(np.sum(valid)),
        "peak_before_ceiling": peak_before_ceiling,
        "peak_after_ceiling": float(np.max(np.abs(y))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Fusion V3 WAV")
    ap.add_argument("output", type=Path, help="44.1 kHz FLOAT mono WAV")
    ap.add_argument("--harmonic", type=float, default=0.42, help="harmonic texture amount, 0..1")
    ap.add_argument("--formant", type=float, default=0.12, help="formant damping amount, 0..1")
    ap.add_argument("--sr", type=int, default=TARGET_SR)
    args = ap.parse_args()

    if not 0.0 <= args.harmonic <= 1.0:
        ap.error("--harmonic must be 0..1")
    if not 0.0 <= args.formant <= 1.0:
        ap.error("--formant must be 0..1")

    x, source_sr = load_mono(args.input)
    x = resample_to(x, source_sr, args.sr)
    y, metrics = polish(x, args.sr, args.harmonic, args.formant)
    sf.write(args.output, y.astype(np.float32), args.sr, subtype="FLOAT")

    print(f"source_sr={source_sr}; output_sr={args.sr}; subtype=FLOAT; channels=1")
    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
