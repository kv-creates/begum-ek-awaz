#!/usr/bin/env python3
"""
test_accuracy.py — Evaluate Begum INT8 TFLite: TPR/FPR + FAPH on ambient noise.

- Loads training/artifacts/begum_int8.tflite (falls back to begum_fp32.keras).
- Slides 1 s windows (100 ms hop => 10 Hz, matching firmware duty-cycle) over:
    * data/test/positive|negative|noise  -> TPR / FPR
    * data/ambient_validation/*.wav      -> false triggers -> FAPH
- Decision: dequantized begum prob + 8-point sliding average > 0.55,
  with 3 s cooldown, exactly like firmware.
- Asserts false triggers < 1 per 24 h of ambient audio (scales observed rate).
  Prints TPR, FPR, FAPH. Exit code 0 on pass, 1 on fail.

Usage:
    python test_accuracy.py --data_dir ../training/data --artifacts ../training/artifacts
Requires Python 3.11+.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
THRESHOLD = 0.55
HISTORY = 8
HOP = 1600          # 100 ms @16k
WIN = 16000         # 1 s
COOLDOWN_WIN = 30   # 30 x 100ms = 3 s

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_model import wav_to_logmel  # noqa: E402


def load_tflite(artifacts: Path):
    import tensorflow as tf
    tfl = artifacts / "begum_int8.tflite"
    if tfl.exists():
        print(f"Loading INT8 model: {tfl} ({tfl.stat().st_size} bytes)")
        it = tf.lite.Interpreter(model_path=str(tfl))
        it.allocate_tensors()
        return ("tflite", it)
    for cand in (artifacts / "begum_fp32.keras", artifacts / "begum_fp32.h5"):
        if cand.exists():
            print(f"Loading FP32 model: {cand}")
            m = tf.keras.models.load_model(str(cand), compile=False)
            return ("keras", m)
    raise FileNotFoundError(f"No model in {artifacts}")


def wav_to_pcm(path: str) -> np.ndarray:
    import soundfile as sf
    import librosa
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=SAMPLE_RATE).astype(np.float32)
    return y


def pcm_window_to_logmel(window_1s: np.ndarray) -> np.ndarray:
    import soundfile as sf
    import tempfile, os
    # reuse exact training preprocessing by round-tripping through temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        sf.write(tmp, window_1s.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
        return wav_to_logmel(tmp)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def infer_prob(backend, feat: np.ndarray) -> float:
    kind, h = backend
    if kind == "keras":
        return float(h.predict(feat[np.newaxis, ...], verbose=0)[0, 1])
    it = h
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    scale, zp = inp["quantization"]
    if scale == 0:
        scale = 0.1
    q = np.clip(np.round(feat[np.newaxis, ...] / scale + zp), -128, 127).astype(np.int8)
    it.set_tensor(inp["index"], q)
    it.invoke()
    raw = it.get_tensor(out["index"]).astype(np.float32).reshape(-1)
    osc, ozp = out["quantization"]
    if osc != 0:
        raw = (raw - ozp) * osc
    # softmax (outputs may be logits or probs; softmax handles both when calibrated)
    m = float(np.max(raw))
    e = np.exp(raw - m)
    return float(e[1] / (np.sum(e) + 1e-9))


def score_file(backend, pcm: np.ndarray) -> tuple:
    """Slide over PCM, return (triggered: bool, max_avg_prob: float)."""
    hist = []
    max_avg = 0.0
    cooldown = 0
    triggered = False
    # pad short files
    if len(pcm) < WIN:
        pcm = np.pad(pcm, (0, WIN - len(pcm)))
    for start in range(0, len(pcm) - WIN + 1, HOP):
        w = pcm[start:start + WIN]
        feat = pcm_window_to_logmel(w)
        p = infer_prob(backend, feat)
        hist.append(p)
        if len(hist) > HISTORY:
            hist.pop(0)
        avg = float(np.mean(hist))
        max_avg = max(max_avg, avg)
        if cooldown > 0:
            cooldown -= 1
            continue
        if avg >= THRESHOLD:
            triggered = True
            cooldown = COOLDOWN_WIN
    return triggered, max_avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../training/data")
    ap.add_argument("--artifacts", default="../training/artifacts")
    ap.add_argument("--max_per_class", type=int, default=400)
    args = ap.parse_args()

    data = Path(args.data_dir)
    art = Path(args.artifacts)
    backend = load_tflite(art)

    def lst(split, cls):
        return sorted(glob.glob(str(data / split / cls / "*.wav")))[:args.max_per_class]

    # ---- TPR on positives ----
    pos = lst("test", "positive")
    tp = 0
    for f in pos:
        trig, _ = score_file(backend, wav_to_pcm(f))
        tp += 1 if trig else 0
    tpr = tp / max(1, len(pos))

    # ---- FPR on negatives + noise ----
    neg = lst("test", "negative") + lst("test", "noise")
    fp = 0
    for f in neg:
        trig, _ = score_file(backend, wav_to_pcm(f))
        fp += 1 if trig else 0
    fpr = fp / max(1, len(neg))

    print(f"TPR: {tpr*100:.2f}% ({tp}/{len(pos)})")
    print(f"FPR: {fpr*100:.3f}% ({fp}/{len(neg)})")

    # ---- FAPH on ambient_validation (30 min shipped, scaled to 24 h) ----
    amb_files = sorted(glob.glob(str(data / "ambient_validation" / "*.wav")))
    if not amb_files:
        print("WARNING: no ambient_validation files; FAPH check skipped (FAIL).")
        sys.exit(1)
    total_sec = 0.0
    false_trig = 0
    for f in amb_files:
        pcm = wav_to_pcm(f)
        total_sec += len(pcm) / SAMPLE_RATE
        # count ALL triggers with cooldown (score_file counts >=1; recount properly)
        hist = []
        cooldown = 0
        if len(pcm) < WIN:
            continue
        for start in range(0, len(pcm) - WIN + 1, HOP):
            w = pcm[start:start + WIN]
            p = infer_prob(backend, pcm_window_to_logmel(w))
            hist.append(p)
            if len(hist) > HISTORY:
                hist.pop(0)
            avg = float(np.mean(hist))
            if cooldown > 0:
                cooldown -= 1
                continue
            if avg >= THRESHOLD:
                false_trig += 1
                cooldown = COOLDOWN_WIN
    hours = total_sec / 3600.0
    faph = false_trig / hours if hours > 0 else float("inf")
    per24h = faph * 24.0
    print(f"Ambient: {total_sec/60:.1f} min, false triggers={false_trig}")
    print(f"FAPH: {faph:.4f}/hour  (~{per24h:.3f} per 24h)")
    print(f"TPR={tpr:.4f} FPR={fpr:.5f} FAPH={faph:.5f}")

    ok = True
    if tpr < 0.95:
        print(f"FAIL: TPR {tpr:.3f} < 0.95 target")
        ok = False
    if per24h >= 1.0:
        print(f"FAIL: {per24h:.3f} false triggers per 24h >= 1")
        ok = False
    if ok:
        print("PASS: TPR>95% and <1 false trigger per 24h.")
    else:
        print("EVALUATION FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
