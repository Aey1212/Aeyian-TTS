#!/usr/bin/env python3
"""Aeyian TTS F3-HQ: numerically lossless/high-rate render of canonical F3.

Same F3 phonetic-print architecture, but:
  * decode O and M once to float64
  * exact 4x polyphase upsample to 88.2 kHz (for 22.05 kHz sources)
  * keep all fusion calculations in float64/complex128
  * scale FFT/hop with sample rate so F3's time/frequency resolution is preserved
  * write IEEE 64-bit float WAV (libsndfile subtype DOUBLE)

This does not invent source bandwidth. It prevents intermediate PCM16 quantization and
lossy encoding before the L3 voice-conversion stage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy import signal

BASE_SR = 22050
UPSAMPLE = 4
TARGET_SR = BASE_SR * UPSAMPLE
BASE_N_FFT = 1024
BASE_HOP = 128
N_FFT = BASE_N_FFT * UPSAMPLE
HOP = BASE_HOP * UPSAMPLE
LPC_ORDER = 18
PRINT_LOW_HZ = 100.0
PRINT_HIGH_HZ = 7000.0
MAX_DELTA_DB = 24.0
EPS = 1e-12


def read_native(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim != 1:
        x = np.mean(x, axis=-1, dtype=np.float64)
    return np.asarray(x, dtype=np.float64), int(sr)


def upsample_exact(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == TARGET_SR:
        return x.astype(np.float64, copy=False), sr
    if TARGET_SR % sr != 0:
        raise ValueError(f"Expected integer upsample into {TARGET_SR} Hz; got {sr} Hz")
    factor = TARGET_SR // sr
    y = signal.resample_poly(x, factor, 1, window=("kaiser", 14.0))
    return np.asarray(y, dtype=np.float64), TARGET_SR


def speech_bounds(x: np.ndarray, sr: int) -> tuple[float, float]:
    rms = librosa.feature.rms(y=x, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    db = librosa.amplitude_to_db(rms + EPS, ref=np.max)
    active = np.flatnonzero(db > -35.0)
    if len(active) == 0:
        return 0.0, len(x) / sr
    start = max(0, active[0] * HOP - N_FFT // 2) / sr
    end = min(len(x), active[-1] * HOP + N_FFT // 2) / sr
    return start, end


def lpc_print(x: np.ndarray, sr: int):
    z = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                     window="hann", center=True)
    z = np.asarray(z, dtype=np.complex128)
    mag = np.maximum(np.abs(z), EPS)
    logmag = np.log(mag)

    padded = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = librosa.util.frame(padded, frame_length=N_FFT, hop_length=HOP)
    freqs = np.linspace(0.0, sr / 2.0, N_FFT // 2 + 1, dtype=np.float64)
    fit_band = (freqs >= PRINT_LOW_HZ) & (freqs <= PRINT_HIGH_HZ)

    env = np.empty_like(logmag, dtype=np.float64)
    window = np.hanning(N_FFT).astype(np.float64)

    for i in range(frames.shape[1]):
        frame = np.asarray(frames[:, i], dtype=np.float64) * window
        if np.sqrt(np.mean(frame * frame)) < 1e-5:
            env[:, i] = logmag[:, i]
            continue
        try:
            a = librosa.lpc(frame, order=LPC_ORDER)
            _, h = signal.freqz([1.0], a, worN=N_FFT // 2 + 1, fs=sr)
            shape = np.log(np.maximum(np.abs(h), EPS))
            offset = np.median(logmag[fit_band, i] - shape[fit_band])
            env[:, i] = shape + offset
        except Exception:
            env[:, i] = logmag[:, i]
    return z, env, freqs


def normalized_shape(env: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    band = (freqs >= PRINT_LOW_HZ) & (freqs <= PRINT_HIGH_HZ)
    return env - env[band].mean(axis=0, keepdims=True)


def map_guide_print(guide_shape: np.ndarray, carrier_frames: int, sr: int,
                    carrier_span: tuple[float, float], guide_span: tuple[float, float]) -> np.ndarray:
    tc = np.arange(carrier_frames, dtype=np.float64) * HOP / sr
    tg = np.arange(guide_shape.shape[1], dtype=np.float64) * HOP / sr
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
        out[k] = np.interp(mapped, tg, guide_shape[k],
                           left=guide_shape[k, 0], right=guide_shape[k, -1])
    return out


def fuse(carrier: np.ndarray, guide: np.ndarray, sr: int, strength: float) -> np.ndarray:
    zc, carrier_env, freqs = lpc_print(carrier, sr)
    _, guide_env, _ = lpc_print(guide, sr)
    carrier_shape = normalized_shape(carrier_env, freqs)
    guide_shape = normalized_shape(guide_env, freqs)
    guide_aligned = map_guide_print(guide_shape, zc.shape[1], sr,
                                    speech_bounds(carrier, sr), speech_bounds(guide, sr))
    logmag = np.log(np.maximum(np.abs(zc), EPS))
    delta = guide_aligned - carrier_shape
    max_ln = np.log(10.0) * MAX_DELTA_DB / 20.0
    delta = np.clip(delta, -max_ln, max_ln)

    fw = np.ones_like(freqs, dtype=np.float64)
    fw[freqs < 70.0] = 0.0
    low = (freqs >= 70.0) & (freqs < PRINT_LOW_HZ)
    fw[low] = (freqs[low] - 70.0) / (PRINT_LOW_HZ - 70.0)
    high = (freqs > PRINT_HIGH_HZ) & (freqs < 8000.0)
    fw[high] = (8000.0 - freqs[high]) / (8000.0 - PRINT_HIGH_HZ)
    fw[freqs >= 8000.0] = 0.0

    out_logmag = logmag + strength * fw[:, None] * delta
    out_mag = np.exp(out_logmag)
    zout = np.asarray(out_mag * np.exp(1j * np.angle(zc)), dtype=np.complex128)
    y = librosa.istft(zout, hop_length=HOP, win_length=N_FFT,
                      window="hann", center=True, length=len(carrier))
    y = np.asarray(y, dtype=np.float64)
    rms_c = np.sqrt(np.mean(carrier * carrier) + EPS)
    rms_y = np.sqrt(np.mean(y * y) + EPS)
    y *= rms_c / rms_y
    return y


def pitch_check(carrier: np.ndarray, output: np.ndarray, sr: int) -> str:
    def track(x):
        f0, _, _ = librosa.pyin(x, fmin=70.0, fmax=300.0, sr=sr,
                                frame_length=2048 * UPSAMPLE, hop_length=HOP)
        return f0
    a = track(carrier); b = track(output)
    n = min(len(a), len(b))
    valid = np.isfinite(a[:n]) & np.isfinite(b[:n])
    if not np.any(valid):
        return "F0 check: no jointly voiced frames"
    cents = 1200.0 * np.log2(b[:n][valid] / a[:n][valid])
    return (f"F0 check: median shift={np.median(cents):.3f} cents; "
            f"median abs shift={np.median(np.abs(cents)):.3f} cents; frames={valid.sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carrier", type=Path)
    ap.add_argument("guide", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()
    if not 0.0 <= args.strength <= 1.0:
        ap.error("--strength must be 0..1")

    c0, csr = read_native(args.carrier)
    g0, gsr = read_native(args.guide)
    carrier, sr = upsample_exact(c0, csr)
    guide, gsr2 = upsample_exact(g0, gsr)
    if gsr2 != sr:
        raise ValueError("carrier and guide HQ sample rates differ")

    y = fuse(carrier, guide, sr, args.strength)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, y, sr, subtype="DOUBLE")
    print(f"wrote {args.output}: {sr} Hz mono IEEE float64, {len(y)/sr:.6f} s")
    print(pitch_check(carrier, y, sr))

if __name__ == "__main__":
    main()
