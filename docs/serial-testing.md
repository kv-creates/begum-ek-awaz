# Serial testing

The ESP8266 stays offline in every test. The phone or PC is only a
115200 8N1 serial monitor. Nothing is transmitted or received over radio.

## Phone (USB OTG)

1. Flash from PC, then move the board to a USB power bank.
2. Connect phone to board with an OTG adapter + USB cable.
3. Open Serial USB Terminal (Android) or Serial Terminal Pro (iOS/Android).
4. Set 115200 8N1, no flow control, open `ttyUSB0` (or `ttyACM0`).
5. Press RESET on the board.

## PC

```bash
cd firmware
pio device monitor -b 115200
```

Same output as the phone path.

## Boot transcript (expected, values vary)

```text
=== BEGUM OFFLINE VOICE ACTIVATOR ===
Free Heap: 41880 bytes
Model Size: 33.63 KB
Tensor Arena: 15360 bytes
OLED: Initialized
Audio: Initialized
Audio: source=INMP441 I2S DOUT=12 WS=13 SCK=14
KWS: Ready
Free Heap: 41232 bytes
Listening...
```

Dummy-model boot (before running `export_tflite.py`):

```text
KWS: model invalid (dummy 20-byte placeholder?)
KWS: run training/export_tflite.py, continuing with prob=0
```

## Detection transcript (say "bay-gum" ~30 cm from mic)

```text
[12345] BEGUM DETECTED! Confidence: 0.8712
[12346] LED ON, OLED Updated
[15346] Cooldown: 3s
[15346] Ready.
```

What to watch at the same moment:

- OLED flips to `BEGUM DETECTED!` + `conf: 87%`.
- Built-in LED (GPIO2) lights for exactly 3 s.
- OLED counts `Cooldown: 3s / 2s / 1s`, then back to `Listening...`.
- GPIO/OLED react within 50 ms of the word ending.

## Rate transcript (duty-cycle proof, every 10 s)

```text
[20000] infer_rate=10.02 Hz frames=1023 infers=102 last_prob=0.0412 heap=41232
[30000] infer_rate=9.98 Hz frames=2021 infers=202 last_prob=0.0388 heap=41200
```

Rules of thumb:

| Reading            | Meaning                                   |
|--------------------|-------------------------------------------|
| near 10.0 Hz       | 10 ms capture / 100 ms tick is healthy    |
| near 100 Hz        | duty-cycle broken (tick every frame)      |
| 0.0 Hz             | inference stalled, check audio + model    |
| falling heap       | leak hunt: arena and buffers are static   |

## Troubleshooting matrix

| Symptom                | Check                                                        |
|------------------------|--------------------------------------------------------------|
| Garbled text           | 115200 8N1, another OTG cable                               |
| OLED blank             | VCC to 3V3, SCL D1/GPIO5, SDA D2/GPIO4, addr 0x3C           |
| Always 0.0000 prob     | DOUT D6/GPIO12, WS D7/GPIO13, SCK D5/GPIO14, L/R to GND      |
| Dummy-model warning    | run export_tflite.py, rebuild, reflash                      |
| No trigger on speech   | 30 cm distance, quiet room, threshold 0.55 in config.h       |
