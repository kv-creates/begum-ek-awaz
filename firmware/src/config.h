#ifndef BEGUM_CONFIG_H
#define BEGUM_CONFIG_H

// ============================================================================
// config.h — Begum Offline Voice Activator, central configuration.
// 100% OFFLINE: no WiFi / Ethernet / UDP / WebSocket / MQTT anywhere.
// Hardware: ESP8266 (D1 mini / NodeMCU) + INMP441 I2S mic + SSD1306 OLED.
// ============================================================================

// ------------------------- Wiring -------------------------------------------
// Microphone: INMP441 I2S
//   INMP441 VDD -> 3V3,  GND -> GND,  L/R -> GND (left channel mono)
//   INMP441 DOUT (SD) -> GPIO12 (D6)
//   INMP441 LRCLK (WS) -> GPIO13 (D7)
//   INMP441 BCLK (SCK) -> GPIO14 (D5)
// Fallback analog mic (MAX9814 OUT -> A0) enabled with USE_ANALOG_MIC.
// OLED: 0.96" SSD1306 I2C 128x64
//   VCC -> 3.3V, GND -> GND
//   SCL -> GPIO5 (D1)
//   SDA -> GPIO4 (D2)
// Built-in LED: GPIO2, active LOW.

// #define USE_ANALOG_MIC  // uncomment to use MAX9814 on A0 instead of INMP441

#define PIN_I2S_DOUT   12   // D6 — INMP441 SD
#define PIN_I2S_LRCLK  13   // D7 — INMP441 WS
#define PIN_I2S_BCLK   14   // D5 — INMP441 SCK

#define PIN_OLED_SCL    5   // D1
#define PIN_OLED_SDA    4   // D2
#define OLED_I2C_ADDR   0x3C
#define OLED_WIDTH      128
#define OLED_HEIGHT     64

#define PIN_LED         2   // LED_BUILTIN, active LOW
#define LED_ON          LOW
#define LED_OFF         HIGH

// ------------------------- Audio / model ------------------------------------
#define SAMPLE_RATE_HZ      16000
#define FRAME_SAMPLES       160     // 10 ms @16kHz per capture step
#define N_MELS              40
#define N_FRAMES            49      // model input time steps (49 x 40 x 1)
#define FFT_SIZE            512
#define WIN_LEN             400     // 25 ms Hann window
#define HOP_LEN             320     // 20 ms hop (firmware resamples 10ms steps)

// ------------------------- Inference ----------------------------------------
// Tensor arena MUST be exactly 15360 bytes (RAM budget < 256KB total).
#define TENSOR_ARENA_BYTES  15360
static_assert(TENSOR_ARENA_BYTES == 15360, "Tensor arena must be exactly 15360 bytes");

// Keyword decision: averaged probability over 8-point sliding window.
#define PROBABILITY_CUTOFF  0.55f
#define PROB_HISTORY_LEN    8

// Cooldown lockout after a detection: 3 seconds.
#define COOLDOWN_MS         3000UL

// Duty-cycle: capture every 10 ms, run heavy TFLite tick() every 10th frame
// (every 100 ms => ~10 Hz inference). 18 ms inference / 100 ms = ~18% peak,
// plus delay(1) each loop => ~8-9% total idle CPU.
#define INFERENCE_EVERY_N_FRAMES  10

// OLED / serial
#define SERIAL_BAUD         115200
#define INFER_RATE_LOG_MS   10000UL  // print effective Hz every 10 s

#endif // BEGUM_CONFIG_H
