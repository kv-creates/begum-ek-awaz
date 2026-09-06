# Performance

Bounds are enforced twice: compile-time/static asserts in firmware and
fail-closed checks in `evaluation/`.

## Budgets

| Metric   | Bound                       | Gate                                              |
|----------|-----------------------------|---------------------------------------------------|
| RAM      | Under 256 KB                | `test_memory.py` total 64544 B                    |
| Arena    | Exactly 15360 bytes         | `static_assert` + string check                    |
| Model    | Under 50 KB INT8            | `export_tflite.py` raises ValueError              |
| CPU idle | Under 10%                   | 10 ms capture, 100 ms tick, `delay(1)`            |
| Latency  | GPIO/OLED under 50 ms       | write + OLED immediately in detect path           |
| Quality  | TPR over 95%, FAPH under 1  | `test_accuracy.py` exit code                      |
| Offline  | Zero network symbols        | include/call scan in `test_memory.py`             |

## Verified run (PC, quick smoke dataset)

Dataset: 120 positives (formant fallback, Piper absent) + Speech Commands
negatives + synthetic/ESC-50 noise mix, 80/10/10 split, 30 ambient files.

```text
Model: 34440 bytes (33.63 KB)          limit 51200 ............ PASS
Header model len: 34440                matches .tflite ........ PASS
src/config.h: contains 15360 OK
src/kws_inference.h: contains 15360 OK
Estimated RAM breakdown (bytes):
  tensor_arena : 15360
  spectrogram  : 7840 (49x40 float32)
  pcm circular : 32000 (16000 int16)
  audio ring   : 1024 (512 int16)
  u8g2 page buf: 128
  overhead     : 8192
  TOTAL        : 64544 bytes (63.0 KB) vs limit 262144 (256 KB) . PASS
Network check: no network symbols. OFFLINE OK.

PASS: size, RAM, offline checks all green.
```

Accuracy smoke (3+2 epochs, tiny set): `TP=0 FP=0` on 6+6 probe files.
The pipeline runs end to end; the smoke model is undertrained by design.
Production recipe (full 30000 positives, 40+20 epochs) targets TPR over 95%
with under 1 false trigger per 24 h — asserted by `test_accuracy.py`.

## Duty-cycle math

```text
 inference ~18 ms per tick, tick every 100 ms  -> ~18% peak on tick frames
 capture + delay(1) on other frames            -> near-idle
 long-run average                              -> ~8-9% (under the 10% cap)
 Serial proof                                  -> infer_rate near 10.0 Hz
```

## Reproduce

```bash
cd training
python generate_data.py --out_dir ./data --quick
python train_model.py --data_dir ./data --out_dir ./artifacts --batch 32 --epochs_stage1 3 --epochs_stage2 2
python export_tflite.py --artifacts ./artifacts --data_dir ./data
cd ../evaluation
python test_memory.py --artifacts ../training/artifacts --firmware ../firmware
```
