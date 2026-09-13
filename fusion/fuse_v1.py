#!/usr/bin/env python3
"""Aeyian TTS fusion-v1.

Use a legacy/V15-family render as the *carrier* and a modern qce-ng render as
an acoustic pronunciation guide.  The qce waveform is never mixed into the
output.  Instead, its slowly-varying spectral envelope guides a small,
clamped correction of the carrier while the carrier phase/fine harmonic
structure is preserved.

This is deliberately conservative.  It is a first experiment in keeping the
legacy voice texture while borrowing qce-ng's clearer articulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft, istft, resample_poly
from scipy.spatial.distance import cdist


N_FFT = 1024
HOP = 128
KEEP_CEPSTRAL_COEFFS = 30
MAX_CORRECTION_DB = 3.0


def load_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        scale = max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / scale
    else:
        x = x.astype(np.float64)
    return sr, x


def save_pcm16(path: Path, sr: int, x: np.ndarray) -> None:
    peak = np.max(np.abs(x)) + 1e-12
    if peak > 0.985:
        x = x * (0.985 / peak)
    wavfile.write(path, sr, np.int16(np.clip(x, -1.0, 1.0) * 32767.0))


def spectral_envelope(log_mag: np.ndarray) -> np.ndarray:
    """Low-quefrency DCT envelope; removes harmonic/fine spectral detail."""
    cep = dct(log_mag, type=2, axis=0, norm="ortho")
    cep[KEEP_CEPSTRAL_COEFFS:, :] = 0.0
    return idct(cep, type=2, axis=0, norm="ortho")


def pooled_features(env: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Build coarse spectral-shape features for DTW alignment."""
    use = (freqs >= 100.0) & (freqs <= 4500.0)
    e = env[use]
    bands = np.array_split(np.arange(e.shape[0]), 24)
    feat = np.vstack([e[idx].mean(axis=0) for idx in bands]).T
    feat -= feat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-9
    return feat / norm


def dtw_map(carrier_feat: np.ndarray, guide_feat: np.ndarray) -> np.ndarray:
    """Map every carrier frame to a guide frame using banded monotonic DTW."""
    nc, ng = len(carrier_feat), len(guide_feat)
    cost = cdist(carrier_feat, guide_feat, metric="cosine")

    inf = np.inf
    dp = np.full((nc, ng), inf, dtype=np.float64)
    back = np.zeros((nc, ng), dtype=np.uint8)

    # Keep the path near the expected global timing ratio.  Wide enough to
    # permit phrase/phoneme timing differences, narrow enough to avoid matching
    # repeated vowels to the wrong word.
    band = max(40, int(0.12 * ng))

    for i in range(nc):
        center = int(round(i * (ng - 1) / max(1, nc - 1)))
        lo = max(0, center - band)
        hi = min(ng, center + band + 1)
        for j in range(lo, hi):
            c = cost[i, j]
            if i == 0 and j == 0:
                dp[i, j] = c
                continue
            best = inf
            code = 0
            if i > 0 and j > 0 and dp[i - 1, j - 1] < best:
                best = dp[i - 1, j - 1]
                code = 1
            if i > 0 and dp[i - 1, j] + 0.08 < best:
                best = dp[i - 1, j] + 0.08
                code = 2
            if j > 0 and dp[i, j - 1] + 0.08 < best:
                best = dp[i, j - 1] + 0.08
                code = 3
            if np.isfinite(best):
                dp[i, j] = c + best
                back[i, j] = code

    i, j = nc - 1, ng - 1
    if not np.isfinite(dp[i, j]):
        raise RuntimeError("DTW failed to reach the final frame")

    path: list[tuple[int, int]] = []
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        code = back[i, j]
        if code == 1:
            i -= 1; j -= 1
        elif code == 2:
            i -= 1
        elif code == 3:
            j -= 1
        else:
            raise RuntimeError(f"Broken DTW traceback at {i}, {j}")
    path.reverse()

    buckets: list[list[int]] = [[] for _ in range(nc)]
    for ci, gj in path:
        buckets[ci].append(gj)

    mapping = np.zeros(nc, dtype=np.int32)
    last = 0
    for ci, js in enumerate(buckets):
        if js:
            last = int(round(float(np.mean(js))))
        mapping[ci] = last
    return mapping


def voiced_weight(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Softly prefer spectral guidance on voiced/sonorant material."""
    use = (freqs >= 100.0) & (freqs <= 5000.0)
    m = np.maximum(mag[use], 1e-10)
    flatness = np.exp(np.mean(np.log(m), axis=0)) / (np.mean(m, axis=0) + 1e-10)
    rms = np.sqrt(np.mean(m * m, axis=0))
    floor = np.percentile(rms, 18)

    # 1 on strongly harmonic frames, falling smoothly toward 0 for noise.
    harmonic = np.clip((0.55 - flatness) / 0.40, 0.0, 1.0)
    energy = np.clip((rms - floor) / (2.5 * floor + 1e-10), 0.0, 1.0)
    w = harmonic * energy
    return gaussian_filter1d(w, sigma=1.5)


def fuse(carrier: np.ndarray, guide: np.ndarray, sr: int, strength: float) -> np.ndarray:
    noverlap = N_FFT - HOP
    f, _, zc = stft(carrier, fs=sr, window="hann", nperseg=N_FFT,
                    noverlap=noverlap, boundary="zeros", padded=True)
    _, _, zg = stft(guide, fs=sr, window="hann", nperseg=N_FFT,
                    noverlap=noverlap, boundary="zeros", padded=True)

    mc = np.maximum(np.abs(zc), 1e-9)
    mg = np.maximum(np.abs(zg), 1e-9)
    lc = np.log(mc)
    lg = np.log(mg)

    ec = spectral_envelope(lc)
    eg = spectral_envelope(lg)

    mapping = dtw_map(pooled_features(ec, f), pooled_features(eg, f))
    eg_aligned = eg[:, mapping]

    # Compare only spectral *shape*, not loudness.  This prevents the guide's
    # volume/envelope from replacing the carrier's natural dynamics.
    delta = eg_aligned - ec
    delta -= delta.mean(axis=0, keepdims=True)
    delta = gaussian_filter1d(delta, sigma=1.2, axis=1)

    max_ln = np.log(10.0) * MAX_CORRECTION_DB / 20.0
    delta = np.clip(delta, -max_ln, max_ln)

    # Do not alter DC/very-low-frequency energy or high-frequency texture.
    freq_weight = np.ones_like(f)
    freq_weight[f < 80.0] = 0.0
    low = (f >= 80.0) & (f < 180.0)
    freq_weight[low] = (f[low] - 80.0) / 100.0
    high = (f > 3800.0) & (f < 5500.0)
    freq_weight[high] = (5500.0 - f[high]) / 1700.0
    freq_weight[f >= 5500.0] = 0.0

    time_weight = voiced_weight(mc, f)
    correction = strength * delta * freq_weight[:, None] * time_weight[None, :]

    # Preserve the carrier phase and fine harmonic structure.  qce-ng audio is
    # never added to the output.
    zout = zc * np.exp(correction)
    _, y = istft(zout, fs=sr, window="hann", nperseg=N_FFT,
                 noverlap=noverlap, input_onesided=True, boundary=True)
    return y[: len(carrier)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("carrier", type=Path, help="legacy/V15-family WAV")
    ap.add_argument("guide", type=Path, help="qce-ng clarity-guide WAV")
    ap.add_argument("output", type=Path)
    ap.add_argument("--strength", type=float, default=0.22,
                    help="guide strength; 0.22 is the conservative v1 default")
    args = ap.parse_args()

    sr_c, carrier = load_mono(args.carrier)
    sr_g, guide = load_mono(args.guide)
    if sr_g != sr_c:
        from math import gcd
        g = gcd(sr_g, sr_c)
        guide = resample_poly(guide, sr_c // g, sr_g // g)

    y = fuse(carrier, guide, sr_c, args.strength)
    save_pcm16(args.output, sr_c, y)


if __name__ == "__main__":
    main()
