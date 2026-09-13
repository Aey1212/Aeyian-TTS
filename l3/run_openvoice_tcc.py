#!/usr/bin/env python3
"""L3a: raw OpenVoice V2 tone-color conversion of Fusion F3.

F3 is the pronunciation/performance authority. OpenVoice receives no Commune
text and performs no TTS. It extracts a source-speaker embedding from F3, a
target-speaker embedding from a human reference, then runs direct spectrogram
voice conversion.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_mono(path: Path, sr: int | None = None):
    y, got_sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), got_sr


def f0_metrics(source: Path, converted: Path):
    src, sr = load_mono(source, 22050)
    out, _ = load_mono(converted, 22050)
    hop = 256
    f0s, _, _ = librosa.pyin(src, fmin=65, fmax=320, sr=sr, hop_length=hop)
    f0o, _, _ = librosa.pyin(out, fmin=65, fmax=320, sr=sr, hop_length=hop)
    n = min(len(f0s), len(f0o))
    a, b = f0s[:n], f0o[:n]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return {"joint_voiced_frames": int(valid.sum())}
    cents = 1200.0 * np.log2(b[valid] / a[valid])
    corr = float(np.corrcoef(a[valid], b[valid])[0, 1]) if valid.sum() > 2 else float('nan')
    return {
        "joint_voiced_frames": int(valid.sum()),
        "median_f0_shift_cents": float(np.median(cents)),
        "median_abs_f0_shift_cents": float(np.median(np.abs(cents))),
        "f0_correlation": corr,
    }


def duration_metrics(source: Path, converted: Path):
    src, sr_s = sf.read(source, always_2d=False)
    out, sr_o = sf.read(converted, always_2d=False)
    ds = len(src) / sr_s
    do = len(out) / sr_o
    return {
        "source_duration_s": ds,
        "converted_duration_s": do,
        "duration_ratio": do / ds if ds else float('nan'),
    }


def make_float_master(src: Path, out: Path, target_sr: int = 44100):
    y, sr = sf.read(src, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float64)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        y = resample_poly(y, target_sr // g, sr // g)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.985:
        y *= 0.985 / peak
    sf.write(out, y.astype(np.float32), target_sr, subtype='FLOAT')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--openvoice-root', type=Path, required=True)
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--target', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--master', type=Path, required=True)
    ap.add_argument('--metrics', type=Path, required=True)
    ap.add_argument('--tau', type=float, default=0.3)
    args = ap.parse_args()

    sys.path.insert(0, str(args.openvoice_root))
    from openvoice.api import ToneColorConverter

    cfg = args.checkpoint_dir / 'config.json'
    ckpt = args.checkpoint_dir / 'checkpoint.pth'
    converter = ToneColorConverter(str(cfg), device='cpu')
    # Explicitly disable watermark after construction if present. The CI setup
    # patches OpenVoice so missing wavmark does not block this raw experiment.
    converter.load_ckpt(str(ckpt))
    converter.watermark_model = None

    src_se = converter.extract_se(str(args.source))
    tgt_se = converter.extract_se(str(args.target))
    converter.convert(
        audio_src_path=str(args.source),
        src_se=src_se,
        tgt_se=tgt_se,
        output_path=str(args.output),
        tau=args.tau,
        message='L3a',
    )

    make_float_master(args.output, args.master, 44100)
    metrics = {
        "method": "OpenVoice V2 ToneColorConverter",
        "tau": args.tau,
        **duration_metrics(args.source, args.output),
        **f0_metrics(args.source, args.output),
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
