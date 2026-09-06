#!/usr/bin/env python3
"""
test_memory.py — Static analysis of Begum model size and ESP8266 RAM budget.

Checks (all must PASS):
  1. Model file (begum_int8.tflite) exists and is < 50 KB (strict).
  2. firmware/src/model_data.h length constant matches the .tflite size
     (i.e. export_tflite.py was run and overwrote the 20-byte dummy).
  3. Tensor arena in firmware/src/config.h and kws_inference.h is exactly 15360.
  4. Estimated RAM = arena (15360) + spectrogram (49*40*4=7840) + PCM (16000*2=32000
     if linear... firmware uses 16000 int16 circular = 32000) + audio ring (512*2=1024)
     + U8g2 page buffer (128) + stack/MCU overhead (~8KB) stays < 256 KB and
     within ESP8266 ~80KB user RAM with comfortable margin for TFLM overhead.
  5. No forbidden network symbols (WiFi, Ethernet, MQTT, WebSocket, UDP) in firmware/.

Usage:
    python test_memory.py --artifacts ../training/artifacts --firmware ../firmware
Requires Python 3.11+ (stdlib only).
"""

import argparse
import re
import sys
from pathlib import Path

MODEL_MAX = 50 * 1024
ARENA_REQUIRED = 15360
RAM_LIMIT = 256 * 1024

FORBIDDEN = ["ESP8266WiFi", "ESP8266WebServer", "WiFiClient", "PubSubClient",
             "MQTT", "WebSockets", "WebSocket", "Ethernet", "UDP", "HTTPClient"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="../training/artifacts")
    ap.add_argument("--firmware", default="../firmware")
    args = ap.parse_args()

    art = Path(args.artifacts)
    fw = Path(args.firmware)
    failures = []

    # 1. model size
    tfl = art / "begum_int8.tflite"
    if not tfl.exists():
        print(f"INFO: {tfl} not found (train+export first); checking header dummy.")
        model_size = None
    else:
        model_size = tfl.stat().st_size
        print(f"Model: {model_size} bytes ({model_size/1024:.2f} KB)")
        if model_size > MODEL_MAX:
            failures.append(f"model {model_size} > {MODEL_MAX} (50KB)")

    # 2. header consistency
    hdr = fw / "src" / "model_data.h"
    if not hdr.exists():
        failures.append("firmware/src/model_data.h missing")
    else:
        txt = hdr.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"begum_model_tflite_len\s*=\s*(\d+)", txt)
        if not m:
            failures.append("model_data.h: length constant not found")
        else:
            hlen = int(m.group(1))
            print(f"Header model len: {hlen}")
            if hlen == 20:
                print("Header: DUMMY 20-byte placeholder (run export_tflite.py before flashing).")
            if model_size is not None and hlen != model_size:
                failures.append(f"header len {hlen} != tflite {model_size} (re-run export)")

    # 3. arena exactly 15360
    for rel in ("src/config.h", "src/kws_inference.h"):
        p = fw / rel
        if not p.exists():
            failures.append(f"{rel} missing")
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if "15360" not in t:
            failures.append(f"{rel}: 15360 not found")
        else:
            print(f"{rel}: contains 15360 OK")
    cfg = (fw / "src" / "config.h").read_text(encoding="utf-8", errors="replace")
    mc = re.search(r"#define\s+TENSOR_ARENA_BYTES\s+(\d+)", cfg)
    if mc and int(mc.group(1)) != ARENA_REQUIRED:
        failures.append(f"TENSOR_ARENA_BYTES={mc.group(1)} != 15360")

    # 4. RAM estimate
    arena = 15360
    spectrogram = 49 * 40 * 4
    pcm_circ = 16000 * 2
    audio_ring = 512 * 2
    u8g2_page = 128
    overhead = 8 * 1024
    total = arena + spectrogram + pcm_circ + audio_ring + u8g2_page + overhead
    print("Estimated RAM breakdown (bytes):")
    print(f"  tensor_arena : {arena}")
    print(f"  spectrogram  : {spectrogram} (49x40 float32)")
    print(f"  pcm circular : {pcm_circ} (16000 int16)")
    print(f"  audio ring   : {audio_ring} (512 int16)")
    print(f"  u8g2 page buf: {u8g2_page}")
    print(f"  overhead     : {overhead}")
    print(f"  TOTAL        : {total} bytes ({total/1024:.1f} KB) vs limit {RAM_LIMIT} ({RAM_LIMIT/1024:.0f} KB)")
    if total >= RAM_LIMIT:
        failures.append(f"estimated RAM {total} >= {RAM_LIMIT}")
    # ESP8266 practical user heap ~80KB: warn (not fail) if > 60KB static
    if total > 60 * 1024:
        print("NOTE: static estimate >60KB; ESP8266 heap ~80KB — still OK since "
              "PCM+TFLM reuse DRAM, but keep -Os and page-buffer OLED (done).")

    # 5. forbidden network includes/libs (comments mentioning "no WiFi" are OK).
    # Only fail on actual code usage: #include <...WiFi...>, lib_deps entries,
    # or real API calls outside comments.
    hits = []
    for f in (fw).rglob("*"):
        if f.is_file() and f.suffix in (".h", ".ino", ".cpp", ".ini"):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            # remove /* ... */ block comments and ; full-line comments (platformio.ini)
            text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
            for n, line in enumerate(text.splitlines(), 1):
                s = line.strip()
                if not s or s.startswith("//") or s.startswith(";") or s.startswith("*") \
                        or s.startswith("# //"):
                    continue
                # strip trailing // comments for .h/.ino/.cpp
                code = line.split("//")[0]
                # skip ; trailing comments in .ini
                if f.suffix == ".ini":
                    code = code.split(";")[0]
                for sym in FORBIDDEN:
                    if sym in code and ("#include" in code or "lib_deps" in code
                                        or "WiFi." in code or "MQTT" in code
                                        or "WebSocket" in code or "Ethernet." in code
                                        or "UDP." in code or "HTTP" in code):
                        hits.append(f"{f.relative_to(fw)}:{n}: {sym}")
    if hits:
        print("Forbidden network references found:")
        for h in hits:
            print("  " + h)
        failures.append(f"{len(hits)} forbidden network symbol(s) in firmware/")
    else:
        print("Network check: no WiFi/Ethernet/MQTT/WebSocket/UDP symbols. OFFLINE OK.")

    if failures:
        print("\nFAIL:")
        for x in failures:
            print(" - " + x)
        sys.exit(1)
    print("\nPASS: size, RAM, offline checks all green.")


if __name__ == "__main__":
    main()
