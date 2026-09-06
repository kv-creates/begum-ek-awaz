# OLED states

128x64 SSD1306, U8g2 page buffer. Four states, drawn with text primitives
plus a small microphone glyph on the idle screen.

## 1. Listening (idle)

```text
 +------------------------------------------------------+
 |BEGUM                                                 |
 |                                                      |
 | Listening...                               +----+    |
 |                                            |    |    |
 |                                            |    |    |
 | say: bay-gum                               +----+    |
 |                                               ||    |
 +------------------------------------------------------+
```

## 2. Detected (keyword, with confidence)

```text
 +------------------------------------------------------+
 |! KEYWORD !                                           |
 |                                                      |
 | BEGUM                                                |
 | DETECTED!                                            |
 |                                                      |
 | conf: 87%                                            |
 +------------------------------------------------------+
```

Shown together with GPIO2 LOW (LED ON) and the Serial detection line.

## 3. Cooldown (3 s lockout, 1 s steps)

```text
 +------------------------------------------------------+
 |BEGUM                                                 |
 |                                                      |
 | Cooldown: 3s                                         |
 |                                                      |
 | please wait...                                       |
 |                                                      |
 +------------------------------------------------------+
```

Counts 3, 2, 1, then returns to Listening. Triggers are ignored while shown.

## 4. Boot / status

```text
 +------------------------------------------------------+
 |BEGUM boot OK                                         |
 |                                                      |
 | RAM 41232 B                                          |
 | infer 10.0 Hz                                        |
 | Listening...                                         |
 +------------------------------------------------------+
```

Generic two-line helper `oled_status(line1, line2)` covers error paths
(for example `AUDIO FAIL` / `check wiring`).
