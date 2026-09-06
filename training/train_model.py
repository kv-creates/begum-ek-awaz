#!/usr/bin/env python3
"""
train_model.py — microWakeWord MixConv keyword spotter for "Begum".
- Input: 49 frames x 40 mel bins x 1 channel (1s @16kHz, 25ms win / ~20ms hop).
- Output: 2 classes [non-begum, begum], softmax.
- Stage 1: weighted BCE (5x false-positive penalty) until FAPH < 0.5 target regime.
- Stage 2: LR/10 fine-tune for TPR > 95%.
- SpecAugment: time warping (via time stretch), freq masking, time masking.

Usage:
    python train_model.py --data_dir ./data --out_dir ./artifacts --epochs_stage1 40 --epochs_stage2 20
Requires Python 3.11+, tensorflow==2.15.0.
Outputs: begum_fp32.h5, begum_fp32.keras, training log, thresholds.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LEN = 400       # 25ms @16k
HOP_LEN = 320       # 20ms -> (16000-400)/320+1 = 49 frames
N_MELS = 40
N_FRAMES = 49
SEED = 20260214

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ----------------------------- features -------------------------------------

_MEL_FB = None

def mel_filterbank():
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                                      fmin=60.0, fmax=7800.0).astype(np.float32)  # (40, 257)
    return _MEL_FB


def wav_to_logmel(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=SAMPLE_RATE).astype(np.float32)
    if len(y) < SAMPLE_RATE:
        y = np.pad(y, (0, SAMPLE_RATE - len(y)))
    else:
        y = y[:SAMPLE_RATE]
    # pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
                        window="hann", center=False)
    mag = np.abs(stft) ** 2  # (257, frames)
    mel = mel_filterbank() @ mag  # (40, frames)
    logmel = np.log(np.maximum(mel, 1e-10)).astype(np.float32).T  # (frames, 40)
    # fix to exactly 49 frames
    if logmel.shape[0] < N_FRAMES:
        logmel = np.pad(logmel, ((0, N_FRAMES - logmel.shape[0]), (0, 0)))
    else:
        logmel = logmel[:N_FRAMES, :]
    # per-utterance CMVN
    mu = logmel.mean()
    sd = logmel.std() + 1e-9
    logmel = ((logmel - mu) / sd).astype(np.float32)
    return logmel[..., np.newaxis]  # (49,40,1)


def spec_augment(spec: tf.Tensor) -> tf.Tensor:
    """SpecAugment on (49,40,1) log-mel: freq mask + time mask."""
    # spec: (F=49, M=40, 1)
    x = spec
    # frequency masking: mask up to 8 bins, 1 mask
    f = tf.random.uniform([], 0, 8, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, tf.maximum(1, 40 - f), dtype=tf.int32)
    freq_mask = tf.concat([
        tf.ones([49, f0, 1]),
        tf.zeros([49, f, 1]),
        tf.ones([49, 40 - f0 - f, 1]),
    ], axis=1)
    x = x * freq_mask
    # time masking: mask up to 8 frames, 1 mask
    t = tf.random.uniform([], 0, 8, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, tf.maximum(1, 49 - t), dtype=tf.int32)
    time_mask = tf.concat([
        tf.ones([t0, 40, 1]),
        tf.zeros([t, 40, 1]),
        tf.ones([49 - t0 - t, 40, 1]),
    ], axis=0)
    x = x * time_mask
    return x


def make_dataset(data_dir: Path, split: str, batch: int, training: bool):
    pos = sorted((data_dir / split / "positive").glob("*.wav"))
    neg = sorted((data_dir / split / "negative").glob("*.wav"))
    noi = sorted((data_dir / split / "noise").glob("*.wav"))
    paths, labels = [], []
    for p in pos:
        paths.append(str(p)); labels.append(1)
    for p in neg:
        paths.append(str(p)); labels.append(0)
    for p in noi:
        paths.append(str(p)); labels.append(0)
    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int64)
    # deterministic shuffle
    idx = np.arange(len(paths))
    rng = np.random.default_rng(SEED + (0 if split == "train" else 1))
    rng.shuffle(idx)
    paths, labels = paths[idx], labels[idx]

    def gen():
        for p, l in zip(paths, labels):
            yield p, l

    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ))
    def _map(p, l):
        feat = tf.numpy_function(lambda pp: wav_to_logmel(pp.decode()), [p], tf.float32)
        feat.set_shape([N_FRAMES, N_MELS, 1])
        if training:
            feat = spec_augment(feat)
            # time warp approx: random tiny gain jitter
            feat = feat * tf.random.uniform([], 0.95, 1.05)
        y = tf.one_hot(l, 2)
        return feat, y
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(4096, seed=SEED)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)


# ----------------------------- model ----------------------------------------

def mixconv_block(x, filters: int, kernel: int, stride: int, name: str):
    """Mixed depthwise conv block: 1x1 expand -> depthwise kxk -> 1x1 project + residual."""
    inp = x
    c = x.shape[-1]
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, name=name + "_exp")(x)
    x = tf.keras.layers.BatchNormalization(name=name + "_bn1")(x)
    x = tf.keras.layers.ReLU(name=name + "_relu1")(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=stride, padding="same",
                                        use_bias=False, name=name + "_dw")(x)
    x = tf.keras.layers.BatchNormalization(name=name + "_bn2")(x)
    x = tf.keras.layers.ReLU(name=name + "_relu2")(x)
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, name=name + "_proj")(x)
    x = tf.keras.layers.BatchNormalization(name=name + "_bn3")(x)
    # residual if shapes match
    if stride == 1 and c == filters:
        x = tf.keras.layers.Add(name=name + "_add")([x, inp])
    x = tf.keras.layers.ReLU(name=name + "_out")(x)
    return x


def build_mixednet():
    inp = tf.keras.Input(shape=(N_FRAMES, N_MELS, 1), name="logmel")
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(inp)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = mixconv_block(x, 16, 3, 1, "mix1")
    x = mixconv_block(x, 24, 5, 2, "mix2")
    x = mixconv_block(x, 24, 3, 1, "mix3")
    x = mixconv_block(x, 32, 5, 2, "mix4")
    x = mixconv_block(x, 32, 3, 1, "mix5")
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(0.2, name="drop")(x)
    out = tf.keras.layers.Dense(2, activation="softmax", name="out")(x)
    model = tf.keras.Model(inp, out, name="begum_mixednet")
    return model


def weighted_bce_fp_penalty(y_true, y_pred, fp_weight: float = 5.0):
    """Weighted categorical cross-entropy with 5x penalty on false positives.
    y_true one-hot [neg, pos]. Penalize predicting pos when true neg."""
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)  # (batch,)
    is_neg = y_true[:, 0]  # 1 if true negative
    pred_pos = y_pred[:, 1]
    # extra penalty proportional to predicted pos prob on true negatives
    fp_pen = is_neg * pred_pos * (fp_weight - 1.0)
    return ce * (1.0 + fp_pen)


def compute_metrics(model, ds):
    y_true_all, y_prob_all = [], []
    for xb, yb in ds:
        p = model.predict_on_batch(xb)[:, 1]
        y_prob_all.append(p)
        y_true_all.append(np.argmax(yb.numpy(), axis=1))
    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    for thr in (0.5, 0.55, 0.7):
        pred = (y_prob >= thr).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        print(f"  thr={thr:.2f} TPR={tpr*100:.2f}% FPR={fpr*100:.3f}% TP={tp} FP={fp} FN={fn} TN={tn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./artifacts")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs_stage1", type=int, default=40)
    ap.add_argument("--epochs_stage2", type=int, default=20)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, n_train = make_dataset(data_dir, "train", args.batch, True)
    val_ds, n_val = make_dataset(data_dir, "val", args.batch, False)
    test_ds, n_test = make_dataset(data_dir, "test", args.batch, False)
    print(f"train={n_train} val={n_val} test={n_test}")

    model = build_mixednet()
    model.summary()

    # ---- Stage 1: heavy FP penalty ----
    print("=== STAGE 1: FP-penalized training (5x) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=lambda yt, yp: weighted_bce_fp_penalty(yt, yp, 5.0),
        metrics=["accuracy"],
    )
    cb = [
        tf.keras.callbacks.ModelCheckpoint(str(out_dir / "stage1_best.keras"),
                                           save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=4, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_stage1, callbacks=cb)
    print("Stage1 val metrics:"); compute_metrics(model, val_ds)

    # ---- Stage 2: fine-tune TPR, LR/10 ----
    print("=== STAGE 2: fine-tune for TPR (LR/10) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=lambda yt, yp: weighted_bce_fp_penalty(yt, yp, 2.0),
        metrics=["accuracy"],
    )
    cb2 = [
        tf.keras.callbacks.ModelCheckpoint(str(out_dir / "stage2_best.keras"),
                                           save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                         restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_stage2, callbacks=cb2)
    print("Stage2 val metrics:"); compute_metrics(model, val_ds)
    print("Test metrics:"); compute_metrics(model, test_ds)

    model.save(str(out_dir / "begum_fp32.keras"))
    model.save(str(out_dir / "begum_fp32.h5"))
    with open(out_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"probability_cutoff": 0.55, "cooldown_ms": 3000,
                   "input": [N_FRAMES, N_MELS, 1], "classes": ["non-begum", "begum"]},
                  f, indent=2)
    print("Saved artifacts to", out_dir.resolve())


if __name__ == "__main__":
    main()
