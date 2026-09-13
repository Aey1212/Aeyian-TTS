#!/usr/bin/env python3
"""Fusion V4: V3 broad phonetic print + pitch-averaged mid-resolution qce print.

V15 still owns F0, phase, harmonic locations and fine voice texture. qce-ng is
never mixed into the waveform. The extra V4 guide uses mel-band energy shapes,
which average across harmonic spacing, then removes the broad component already
handled by V3. This targets finer spectrographic pronunciation/artifact patterns
without using qce pitch.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
import fuse_v3_phonetic_print as v3

MEL_BANDS = 64
MEL_FMIN = 90.0
MEL_FMAX = 7800.0
MID_MAX_DELTA_DB = 10.0
EPS = 1e-8


def mel_shape(z: np.ndarray, sr: int):
    basis = librosa.filters.mel(
        sr=sr, n_fft=v3.N_FFT, n_mels=MEL_BANDS,
        fmin=MEL_FMIN, fmax=min(MEL_FMAX, sr / 2 - 1), norm="slaney"
    )
    # Integration across mel bands averages over individual harmonics/F0 comb.
    p = np.maximum(basis @ (np.abs(z) ** 2), EPS)
    x = 0.5 * np.log(p)
    x -= x.mean(axis=0, keepdims=True)  # remove loudness
    mf = librosa.mel_frequencies(
        n_mels=MEL_BANDS, fmin=MEL_FMIN,
        fmax=min(MEL_FMAX, sr / 2 - 1)
    )
    return x, mf


def mid_print(x: np.ndarray) -> np.ndarray:
    # V3 LPC owns the broad envelope. Keep only finer phonetic structure.
    broad = gaussian_filter1d(x, sigma=3.2, axis=0, mode="nearest")
    mid = x - broad
    return gaussian_filter1d(mid, sigma=0.65, axis=1, mode="nearest")


def mel_to_fft(delta: np.ndarray, mf: np.ndarray, ff: np.ndarray) -> np.ndarray:
    out = np.empty((len(ff), delta.shape[1]), dtype=np.float64)
    for i in range(delta.shape[1]):
        out[:, i] = np.interp(ff, mf, delta[:, i], left=0.0, right=0.0)
    return out


def band_weight(freqs: np.ndarray) -> np.ndarray:
    w = np.ones_like(freqs)
    w[freqs < 70] = 0
    lo = (freqs >= 70) & (freqs < v3.PRINT_LOW_HZ)
    w[lo] = (freqs[lo] - 70) / (v3.PRINT_LOW_HZ - 70)
    hi = (freqs > v3.PRINT_HIGH_HZ) & (freqs < 8000)
    w[hi] = (8000 - freqs[hi]) / (8000 - v3.PRINT_HIGH_HZ)
    w[freqs >= 8000] = 0
    return w


def fuse(carrier, guide, sr, mid_strength):
    zc, cenv, freqs = v3.lpc_print(carrier, sr)
    zg, genv, _ = v3.lpc_print(guide, sr)
    cspan, gspan = v3.speech_bounds(carrier, sr), v3.speech_bounds(guide, sr)

    # Exact V3 broad qce phonetic-print transfer.
    cshape = v3.normalized_shape(cenv, freqs)
    gshape = v3.normalized_shape(genv, freqs)
    ga = v3.map_guide_print(gshape, zc.shape[1], sr, cspan, gspan)
    d_lpc = ga - cshape
    lim = np.log(10) * v3.MAX_DELTA_DB / 20
    d_lpc = np.clip(d_lpc, -lim, lim)

    # Additional pitch-averaged mid-resolution qce spectrographic print.
    cm, mf = mel_shape(zc, sr)
    gm, _ = mel_shape(zg, sr)
    cm, gm = mid_print(cm), mid_print(gm)
    gma = v3.map_guide_print(gm, zc.shape[1], sr, cspan, gspan)
    d_mid = gma - cm
    mlim = np.log(10) * MID_MAX_DELTA_DB / 20
    d_mid = np.clip(d_mid, -mlim, mlim)
    d_mid = mel_to_fft(d_mid, mf, freqs)

    logmag = np.log(np.maximum(np.abs(zc), EPS))
    outmag = np.exp(logmag + band_weight(freqs)[:, None] * (d_lpc + mid_strength * d_mid))

    # Hard pitch lock: V15 phase and FFT-bin harmonic locations remain untouched.
    zout = outmag * np.exp(1j * np.angle(zc))
    y = librosa.istft(
        zout, hop_length=v3.HOP, win_length=v3.N_FFT,
        window="hann", center=True, length=len(carrier)
    )
    y *= np.sqrt(np.mean(carrier**2) + EPS) / np.sqrt(np.mean(y**2) + EPS)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carrier", type=Path)
    ap.add_argument("guide", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--mid-strength", type=float, default=1.0)
    a = ap.parse_args()
    if a.mid_strength < 0:
        ap.error("--mid-strength must be non-negative")
    carrier, sr = v3.load(a.carrier)
    guide, _ = v3.load(a.guide, sr=sr)
    y = fuse(carrier, guide, sr, a.mid_strength)
    sf.write(a.output, y, sr, subtype="PCM_16")
    print(v3.pitch_check(carrier, y, sr))


if __name__ == "__main__":
    main()
