# Release v1.0 checklist

## Must be green before tagging v1.0

- [ ] Full dataset generated: `generate_data.py` without `--quick` (5000/lang).
- [ ] Full training: `--epochs_stage1 40 --epochs_stage2 20`, TPR over 95%.
- [ ] `export_tflite.py` model under 50 KB, header overwritten (no dummy).
- [ ] `test_accuracy.py` PASS: under 1 false trigger per 24 h.
- [ ] `test_memory.py` PASS: arena 15360, RAM under 256 KB, offline clean.
- [ ] `pio run` builds with zero warnings on `d1_mini`.
- [ ] Bench test: boot log, detection under 50 ms, cooldown 3 s, rate near 10 Hz.
- [ ] OLED: all four states photographed (Listening, Detected, Cooldown, Boot).

## Tagging

```bash
git log --oneline | wc -l     # expect 20+ on the way to v1.0
pio run
cd evaluation
python test_memory.py --artifacts ../training/artifacts --firmware ../firmware
git tag -a v1.0 -m "v1.0 offline begum voice activator"
git push origin v1.0
```

## Roadmap (post v1.0)

| Item | Why | Owner |
|------|-----|-------|
| On-device threshold trim via serial | room tuning without reflash | firmware |
| Second keyword slot | same arena, new head | training |
| Noise-profile capture mode | 60 s room sample into ambient set | docs |
| 160 MHz power profile | latency vs current draw table | firmware |
| Multilingual TPR per language | en/es/fr/de/zh/hi breakdown | evaluation |
