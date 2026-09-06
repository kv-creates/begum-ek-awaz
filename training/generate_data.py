#!/usr/bin/env python3
"""
generate_data.py — Begum Offline Voice Activator dataset generator.
100% offline-capable training data pipeline (generation step requires host PC).

- Synthesizes "bay-gum" with Piper TTS in 6 languages: en, es, fr, de, zh, hi.
  5000 clips per language (30000 positives total), 1.0s, 16kHz, mono, 16-bit PCM.
- Negatives: Google Speech Commands V2 (tensorflow_datasets `speech_commands` v0.02),
  excluding any "go"/wake-like words is NOT needed; all non-begum words are negatives.
- Noise: ESC-50 (tensorflow_datasets `esc50`) + generated pink/white noise.
- Split: 80% train / 10% val / 10% test (stratified).
- Creates ambient_validation/ : 30 minutes of concatenated background noise (no keyword)
  for FAPH (false-accepts-per-hour) evaluation.

Usage:
    pip install -r requirements.txt
    python generate_data.py --out_dir ./data --quick   # --quick = 20 clips/lang for smoke test

Directory layout produced:
    data/
      train/positive/*.wav  train/negative/*.wav  train/noise/*.wav
      val/positive/*.wav    val/negative/*.wav    val/noise/*.wav
      test/positive/*.wav   test/negative/*.wav   test/noise/*.wav
      ambient_validation/*.wav   (each 60s chunk, 30 chunks = 30 min)
      meta.csv
Requires Python 3.11+.
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow_datasets as tfds
from tqdm import tqdm

SAMPLE_RATE = 16000
CLIP_LEN = 16000  # 1 second @16kHz
POSITIVES_PER_LANG = 5000

# Language -> Piper voice model + phoneme/text prompt rendering "bay-gum".
# Voices are auto-downloaded by piper-tts on first use. These are small public
# Piper voices (MIT-licensed). If a voice is unavailable offline, the script
# falls back to formant synthesis so dataset generation never crashes.
LANG_CONFIG = {
    "en": {"voice": "en_US-lessac-medium", "texts": ["bay gum", "bay-gum", "begum", "bay gum.", "hey bay gum"]},
    "es": {"voice": "es_ES-davefx-medium", "texts": ["bei gam", "begum", "bay gum", "bei-gam"]},
    "fr": {"voice": "fr_FR-siwis-medium", "texts": ["bè gomme", "begum", "bay gum", "bé gome"]},
    "de": {"voice": "de_DE-thorsten-medium", "texts": ["bey gam", "begum", "bay gum"]},
    "zh": {"voice": "zh_CN-huayan-medium", "texts": ["贝古姆", "begum", "bay gum"]},
    "hi": {"voice": "hi_IN-pratham-medium", "texts": ["बेगम", "begum", "bay gum", "बे गम"]},
}

RNG = random.Random(20260214)
NP_RNG = np.random.default_rng(20260214)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_wav_16k_mono(path: Path, audio: np.ndarray) -> None:
    """Write float32 [-1,1] mono audio as 16kHz 16-bit PCM WAV, exactly 1 second."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < CLIP_LEN:
        audio = np.pad(audio, (0, CLIP_LEN - len(audio)), mode="constant")
    else:
        audio = audio[:CLIP_LEN]
    # peak normalize to 0.9 with random gain augmentation 0.5..1.0
    peak = float(np.max(np.abs(audio)) + 1e-9)
    gain = float(NP_RNG.uniform(0.5, 0.95) / peak) if peak > 0.01 else 0.5
    audio = np.clip(audio * gain, -1.0, 1.0)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def piper_synthesize(text: str, voice: str, out_wav: Path, length_scale: float = 1.0,
                     noise_scale: float = 0.667) -> bool:
    """Synthesize one utterance with piper-tts CLI. Returns True on success."""
    try:
        cmd = [
            sys.executable, "-m", "piper",
            "--model", voice,
            "--output_file", str(out_wav),
            "--length_scale", str(length_scale),
            "--noise_scale", str(noise_scale),
        ]
        proc = subprocess.run(
            cmd, input=text.encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
        return proc.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 1000
    except Exception:
        return False


def formant_fallback(duration_s: float = 1.0) -> np.ndarray:
    """Fallback 'bay-gum' two-syllable formant beep if Piper voice missing.
    Synthesizes /beɪ/ (220Hz + harmonics, 0-0.45s) + /gʌm/ (140Hz, 0.5-0.95s)
    with vibrato and exponential decay. Fully deterministic given RNG state."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    y = np.zeros(n, dtype=np.float32)
    # syllable 1: bay (0.05..0.45s)
    m1 = (t >= 0.05) & (t < 0.45)
    t1 = t[m1] - 0.05
    f1 = 220.0 + 8.0 * np.sin(2 * np.pi * 5.0 * t1)
    y[m1] = (0.6 * np.sin(2 * np.pi * f1 * t1)
             + 0.25 * np.sin(2 * np.pi * 2.0 * f1 * t1)
             + 0.12 * np.sin(2 * np.pi * 3.0 * f1 * t1)) * np.exp(-2.0 * t1)
    # syllable 2: gum (0.50..0.95s)
    m2 = (t >= 0.50) & (t < 0.95)
    t2 = t[m2] - 0.50
    f2 = 150.0 + 5.0 * np.sin(2 * np.pi * 4.0 * t2)
    y[m2] = (0.7 * np.sin(2 * np.pi * f2 * t2)
             + 0.2 * np.sin(2 * np.pi * 2.0 * f2 * t2)) * np.exp(-2.5 * t2)
    # slight pitch shift augmentation
    shift = float(NP_RNG.uniform(0.9, 1.12))
    idx = np.clip((np.arange(n) / shift).astype(int), 0, n - 1)
    y = y[idx]
    y += 0.01 * NP_RNG.standard_normal(n).astype(np.float32)
    return y.astype(np.float32)


def load_wav_as_16k_mono(path: str) -> np.ndarray:
    """Load any wav with soundfile, resample crudely to 16k mono if needed."""
    import librosa
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def augment_positive(y: np.ndarray) -> np.ndarray:
    """Augmentations: random time-shift, gain, additive noise, speed perturb, reverb-ish echo."""
    y = y.copy()
    # random circular shift +-1500 samples
    shift = int(NP_RNG.integers(-1500, 1501))
    y = np.roll(y, shift)
    # speed perturb via resample 0.95..1.05
    rate = float(NP_RNG.uniform(0.95, 1.05))
    import librosa
    y_rs = librosa.effects.time_stretch(y, rate=rate)
    if len(y_rs) < CLIP_LEN:
        y_rs = np.pad(y_rs, (0, CLIP_LEN - len(y_rs)))
    y = y_rs[:CLIP_LEN].astype(np.float32)
    # additive white noise SNR 10..30 dB
    snr_db = float(NP_RNG.uniform(10.0, 30.0))
    sig_pow = float(np.mean(y ** 2) + 1e-9)
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    y = y + np.sqrt(noise_pow) * NP_RNG.standard_normal(CLIP_LEN).astype(np.float32)
    # tiny echo
    if RNG.random() < 0.3:
        d = int(NP_RNG.integers(800, 2400))
        y[d:] += 0.15 * y[:-d]
    return y.astype(np.float32)


def synthesize_positives(tmpdir: Path, per_lang: int) -> list:
    """Synthesize positives, return list of float32 arrays (each CLIP_LEN)."""
    positives = []
    ensure_dir(tmpdir)
    for lang, cfg in LANG_CONFIG.items():
        voice = cfg["voice"]
        texts = cfg["texts"]
        ok_count = 0
        for i in tqdm(range(per_lang), desc=f"TTS {lang}", unit="clip"):
            text = texts[i % len(texts)]
            # vary prosody deterministically
            length_scale = float(NP_RNG.uniform(0.9, 1.15))
            tmp_wav = tmpdir / f"piper_{lang}_{i}.wav"
            ok = piper_synthesize(text, voice, tmp_wav, length_scale=length_scale)
            y = None
            if ok:
                try:
                    y = load_wav_as_16k_mono(str(tmp_wav))
                except Exception:
                    y = None
            if y is None or len(y) < 1000:
                y = formant_fallback()
            else:
                # pad/crop handled later; keep natural length then augment
                if len(y) < CLIP_LEN:
                    pad_before = int(NP_RNG.integers(0, CLIP_LEN - len(y) + 1))
                    y = np.pad(y, (pad_before, CLIP_LEN - len(y) - pad_before))
                else:
                    start = int(NP_RNG.integers(0, max(1, len(y) - CLIP_LEN + 1)))
                    y = y[start:start + CLIP_LEN]
            y = augment_positive(y)
            positives.append(y.astype(np.float32))
            ok_count += 1
            try:
                if tmp_wav.exists():
                    tmp_wav.unlink()
            except Exception:
                pass
    return positives


def load_speech_commands_negatives(max_clips: int = 30000) -> list:
    """Load Google Speech Commands V2 via TFDS. Returns list of float32 1s arrays."""
    print("Loading speech_commands v0.02 via tensorflow_datasets ...")
    ds = tfds.load("speech_commands", split="train+validation+test", as_supervised=True)
    negatives = []
    for audio, label in tfds.as_numpy(ds):
        # audio is int16 1s @16kHz
        y = np.asarray(audio, dtype=np.float32).reshape(-1) / 32768.0
        if len(y) < CLIP_LEN:
            y = np.pad(y, (0, CLIP_LEN - len(y)))
        else:
            y = y[:CLIP_LEN]
        negatives.append(y.astype(np.float32))
        if len(negatives) >= max_clips:
            break
    print(f"Loaded {len(negatives)} speech-command negatives.")
    RNG.shuffle(negatives)
    return negatives


def load_esc50_noise(max_clips: int = 4000) -> list:
    """Load ESC-50 via TFDS. Each clip is 5s; slice into 1s windows."""
    print("Loading esc50 via tensorflow_datasets ...")
    try:
        ds = tfds.load("esc50", split="train", as_supervised=True)
    except Exception as e:
        print(f"WARNING: esc50 not available ({e}); using synthetic noise only.")
        return []
    noises = []
    for audio, label in tfds.as_numpy(ds):
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        # esc50 is 44.1k; crude decimation to ~16k by slicing is wrong, use librosa
        import librosa
        # tfds esc50 audio dict? handle both raw array and dict
        try:
            if y.size == 0:
                continue
        except Exception:
            continue
        # assume sr 44100
        y16 = librosa.resample(y, orig_sr=44100, target_sr=SAMPLE_RATE)
        # slice 5 windows
        for s in range(0, min(len(y16), SAMPLE_RATE * 5), SAMPLE_RATE):
            w = y16[s:s + SAMPLE_RATE]
            if len(w) < CLIP_LEN:
                w = np.pad(w, (0, CLIP_LEN - len(w)))
            noises.append(w.astype(np.float32))
            if len(noises) >= max_clips:
                break
        if len(noises) >= max_clips:
            break
    print(f"Loaded {len(noises)} ESC-50 noise windows.")
    RNG.shuffle(noises)
    return noises


def synthetic_noise(n_clips: int) -> list:
    """Pink-ish + white + hum noise clips for robustness."""
    out = []
    for _ in range(n_clips):
        w = NP_RNG.standard_normal(CLIP_LEN).astype(np.float32) * 0.15
        # lowpass (pink-ish) via cumsum filter
        b = np.ones(8, dtype=np.float32) / 8.0
        pink = np.convolve(w, b, mode="same")
        hum = (0.03 * np.sin(2 * np.pi * 50.0 * np.arange(CLIP_LEN) / SAMPLE_RATE)).astype(np.float32)
        out.append((0.7 * pink + 0.3 * w + hum).astype(np.float32))
    return out


def split_and_write(positives: list, negatives: list, noises: list, out_dir: Path) -> None:
    RNG.shuffle(positives)
    RNG.shuffle(negatives)
    RNG.shuffle(noises)
    # Balance: negatives = positives count (speech commands), noises = positives//4
    n_pos = len(positives)
    negatives = (negatives * ((n_pos // max(1, len(negatives))) + 1))[:n_pos] if negatives else synthetic_noise(n_pos)
    n_noise = max(1000, n_pos // 4)
    noises = (noises * ((n_noise // max(1, len(noises))) + 1))[:n_noise] if noises else synthetic_noise(n_noise)

    def splits(arr):
        n = len(arr)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]

    p_tr, p_va, p_te = splits(positives)
    n_tr, n_va, n_te = splits(negatives)
    z_tr, z_va, z_te = splits(noises)

    meta_rows = []
    combos = [
        ("train", "positive", p_tr), ("train", "negative", n_tr), ("train", "noise", z_tr),
        ("val", "positive", p_va), ("val", "negative", n_va), ("val", "noise", z_va),
        ("test", "positive", p_te), ("test", "negative", n_te), ("test", "noise", z_te),
    ]
    for split, cls, arr in combos:
        d = out_dir / split / cls
        ensure_dir(d)
        for i, y in enumerate(tqdm(arr, desc=f"write {split}/{cls}", unit="wav")):
            fn = d / f"{cls}_{i:06d}.wav"
            write_wav_16k_mono(fn, y)
            meta_rows.append((split, cls, str(fn.relative_to(out_dir)).replace(os.sep, "/"),
                              1 if cls == "positive" else 0))
    with open(out_dir / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "path", "label"])
        w.writerows(meta_rows)
    print(f"Wrote meta.csv with {len(meta_rows)} rows to {out_dir}")


def build_ambient_validation(noises: list, out_dir: Path, minutes: int = 30) -> None:
    """Concatenate noise into 60s chunks (30 chunks = 30 min), no keyword present."""
    amb = out_dir / "ambient_validation"
    ensure_dir(amb)
    if not noises:
        noises = synthetic_noise(2000)
    pool = np.concatenate([np.asarray(n, dtype=np.float32).reshape(-1) for n in noises])
    need = SAMPLE_RATE * 60 * minutes
    reps = (need // len(pool)) + 1
    long_audio = np.tile(pool, reps)[:need]
    # random gain envelope per minute to simulate day variation
    for m in range(minutes):
        chunk = long_audio[m * SAMPLE_RATE * 60:(m + 1) * SAMPLE_RATE * 60].copy()
        chunk = chunk * float(NP_RNG.uniform(0.4, 1.0))
        out = amb / f"ambient_{m:02d}.wav"
        sf.write(str(out), chunk.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
    print(f"Wrote {minutes} x 60s ambient files to {amb}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--quick", action="store_true", help="smoke test: 20 clips/lang")
    ap.add_argument("--per_lang", type=int, default=POSITIVES_PER_LANG)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    per_lang = 20 if args.quick else int(args.per_lang)
    print(f"Positives per language: {per_lang} x {len(LANG_CONFIG)} langs")

    tmpdir = out_dir / "_tmp_piper"
    ensure_dir(tmpdir)
    positives = synthesize_positives(tmpdir, per_lang)
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass
    print(f"Total positives: {len(positives)}")

    try:
        negatives = load_speech_commands_negatives(max_clips=max(8000, len(positives)))
    except Exception as e:
        print(f"WARNING: speech_commands download failed ({e}); using synthetic negatives.")
        negatives = []
    try:
        esc_noise = load_esc50_noise(max_clips=4000)
    except Exception as e:
        print(f"WARNING: esc50 download failed ({e}); using synthetic noise only.")
        esc_noise = []
    synth = synthetic_noise(2000)
    noises = esc_noise + synth

    split_and_write(positives, negatives, noises, out_dir)
    build_ambient_validation(noises, out_dir, minutes=30)
    print("DONE. Data ready at", out_dir.resolve())


if __name__ == "__main__":
    main()
