#!/usr/bin/env python3
"""Aeyian TTS fusion-v2: hard qce-guided envelope transplant.

The legacy/V15-family render is the carrier. qce-ng is used only as an
acoustic articulation guide. Its waveform is never added to the output.

Unlike fusion-v1's small EQ-like correction, v2 explicitly decomposes the
carrier magnitude spectrum into:
    carrier = slow_envelope + fine_residual
and reconstructs it with an interpolated/aligned qce-ng slow envelope:
    output = fine_residual(carrier) + mix(carrier_env, qce_env)

At strength=1.0, the slow spectral envelope is qce-guided while the carrier's
phase and fine harmonic/noise residual are retained. This is intentionally a
harsh test of the fusion hypothesis, not a conservative polish pass.
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
KEEP_CEPSTRAL_COEFFS = 34
MAX_DELTA_DB = 12.0
EPS = 1e-9


def load_mono(path: Path):
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        lim = max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / lim
    else:
        x = x.astype(np.float64)
    return sr, x


def save_pcm16(path: Path, sr: int, x: np.ndarray):
    x = np.nan_to_num(x)
    peak = np.max(np.abs(x)) + EPS
    if peak > 0.985:
        x = x * (0.985 / peak)
    wavfile.write(path, sr, np.int16(np.clip(x, -1, 1) * 32767))


def envelope(log_mag: np.ndarray) -> np.ndarray:
    c = dct(log_mag, type=2, axis=0, norm="ortho")
    c[KEEP_CEPSTRAL_COEFFS:, :] = 0.0
    return idct(c, type=2, axis=0, norm="ortho")


def coarse_features(env: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    use = (freqs >= 100) & (freqs <= 5000)
    e = env[use]
    bands = np.array_split(np.arange(e.shape[0]), 28)
    feat = np.vstack([e[b].mean(axis=0) for b in bands]).T
    feat -= feat.mean(axis=1, keepdims=True)
    feat /= np.linalg.norm(feat, axis=1, keepdims=True) + EPS
    return feat


def dtw_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = len(a), len(b)
    costs = cdist(a, b, metric="cosine")
    dp = np.full((na, nb), np.inf)
    back = np.zeros((na, nb), np.uint8)
    band = max(48, int(0.16 * nb))

    for i in range(na):
        center = round(i * (nb - 1) / max(1, na - 1))
        lo, hi = max(0, center-band), min(nb, center+band+1)
        for j in range(lo, hi):
            c = costs[i, j]
            if i == 0 and j == 0:
                dp[i, j] = c
                continue
            choices = []
            if i and j: choices.append((dp[i-1, j-1], 1))
            if i: choices.append((dp[i-1, j] + 0.06, 2))
            if j: choices.append((dp[i, j-1] + 0.06, 3))
            best, code = min(choices, key=lambda t: t[0])
            if np.isfinite(best):
                dp[i, j] = c + best
                back[i, j] = code

    i, j = na-1, nb-1
    if not np.isfinite(dp[i, j]):
        raise RuntimeError("DTW alignment failed")
    path=[]
    while True:
        path.append((i,j))
        if i == 0 and j == 0: break
        code = back[i,j]
        if code == 1: i-=1; j-=1
        elif code == 2: i-=1
        elif code == 3: j-=1
        else: raise RuntimeError(f"broken DTW path at {i},{j}")
    path.reverse()

    buckets=[[] for _ in range(na)]
    for i,j in path: buckets[i].append(j)
    out=np.zeros(na, dtype=np.int32)
    last=0
    for i, js in enumerate(buckets):
        if js: last=int(round(np.mean(js)))
        out[i]=last
    return out


def fuse(carrier: np.ndarray, guide: np.ndarray, sr: int, strength: float) -> np.ndarray:
    noverlap=N_FFT-HOP
    f,_,zc=stft(carrier, fs=sr, window="hann", nperseg=N_FFT,
                 noverlap=noverlap, boundary="zeros", padded=True)
    _,_,zg=stft(guide, fs=sr, window="hann", nperseg=N_FFT,
                 noverlap=noverlap, boundary="zeros", padded=True)

    mc=np.maximum(np.abs(zc), EPS)
    mg=np.maximum(np.abs(zg), EPS)
    lc=np.log(mc); lg=np.log(mg)
    ec=envelope(lc); eg=envelope(lg)

    mapping=dtw_map(coarse_features(ec,f), coarse_features(eg,f))
    ega=eg[:,mapping]

    # Preserve carrier frame loudness. Guide controls spectral SHAPE, not gain.
    ega = ega - ega.mean(axis=0, keepdims=True) + ec.mean(axis=0, keepdims=True)

    # Guide trajectory is smoothed only slightly in time; unlike v1 this is not
    # reduced to a tiny local correction.
    ega = gaussian_filter1d(ega, sigma=0.65, axis=1)

    # Protect against pathological bins while allowing a genuinely strong test.
    max_ln=np.log(10.0)*MAX_DELTA_DB/20.0
    delta=np.clip(ega-ec, -max_ln, max_ln)
    target_env=ec + strength*delta

    # Preserve extreme low frequency and top-end carrier texture, but let qce
    # control the speech-defining band very strongly.
    fw=np.ones_like(f)
    fw[f<70]=0
    ramp=(f>=70)&(f<120); fw[ramp]=(f[ramp]-70)/50
    ramp=(f>6000)&(f<7500); fw[ramp]=(7500-f[ramp])/1500
    fw[f>=7500]=0
    target_env = ec + fw[:,None]*(target_env-ec)

    # Carrier fine residual = harmonics/noise detail around its own envelope.
    residual = lc - ec
    out_logmag = target_env + residual
    out_mag = np.exp(out_logmag)

    # qce phase is NEVER used. Keep the carrier complex phase exactly.
    zout = out_mag * np.exp(1j*np.angle(zc))
    _,y=istft(zout, fs=sr, window="hann", nperseg=N_FFT,
              noverlap=noverlap, input_onesided=True, boundary=True)
    y=y[:len(carrier)]

    # Match carrier RMS so comparisons aren't biased by loudness.
    rms_c=np.sqrt(np.mean(carrier**2)+EPS)
    rms_y=np.sqrt(np.mean(y**2)+EPS)
    y*=rms_c/rms_y
    return y


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("carrier", type=Path)
    ap.add_argument("guide", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--strength", type=float, default=0.85)
    args=ap.parse_args()
    if not 0 <= args.strength <= 1.0:
        ap.error("--strength must be between 0 and 1")

    sr,c=load_mono(args.carrier)
    gs,g=load_mono(args.guide)
    if gs != sr:
        from math import gcd
        q=gcd(sr,gs)
        g=resample_poly(g, sr//q, gs//q)
    y=fuse(c,g,sr,args.strength)
    save_pcm16(args.output,sr,y)

if __name__ == "__main__":
    main()
