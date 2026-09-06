/*
 * begum_activator.ino — Begum Offline Voice Activator (ESP8266 + INMP441 + SSD1306)
 *
 * 100% OFFLINE: no WiFi, Ethernet, UDP, WebSockets, MQTT. Pure on-device KWS.
 *
 * Wiring:
 *   INMP441 VDD->3V3, GND->GND, L/R->GND
 *   INMP441 DOUT -> GPIO12 (D6)
 *   INMP441 LRCLK(WS) -> GPIO13 (D7)
 *   INMP441 BCLK(SCK) -> GPIO14 (D5)
 *   OLED VCC->3V3, GND->GND, SCL->GPIO5 (D1), SDA->GPIO4 (D2)
 *   Built-in LED GPIO2 (active LOW) ON for 3 s on detection.
 *   Fallback: define USE_ANALOG_MIC for MAX9814 OUT->A0.
 *
 * Behavior:
 *   - Capture audio every 10 ms (160 samples @16kHz).
 *   - Run heavy TFLite tick() every 10th frame (every 100 ms => ~10 Hz).
 *     Inference ~18 ms / 100 ms ~= 18% peak; delay(1) each loop => ~8-9% idle.
 *   - avg_prob > 0.55 + cooldown expired => LED 3 s, OLED update, Serial log.
 *   - Serial 115200 for phone testing (Serial USB Terminal / Serial Terminal Pro).
 *   - Prints inference rate (Hz) every 10 s to prove ~10 Hz duty-cycle.
 */

#include <Arduino.h>
#include "config.h"
#include "audio_capture.h"
#include "kws_inference.h"
#include "oled_display.h"

static AudioCapture audioCapture;
static KWSInference kws;

static int16_t audio_buffer[FRAME_SAMPLES];
static uint32_t frame_counter = 0;
static uint32_t infer_calls = 0;
static float last_avg_prob = 0.0f;
static uint32_t last_trigger_ms = 0;
static bool first_trigger_done = false;
static uint32_t rate_window_start = 0;
static uint32_t rate_window_inf = 0;

static void printBootBanner(uint32_t freeHeap, size_t modelLen) {
  Serial.println(F("=== BEGUM OFFLINE VOICE ACTIVATOR ==="));
  Serial.print(F("Free Heap: "));
  Serial.print(freeHeap);
  Serial.println(F(" bytes"));
  Serial.print(F("Model Size: "));
  Serial.print((float)modelLen / 1024.0f, 2);
  Serial.println(F(" KB"));
  Serial.print(F("Tensor Arena: "));
  Serial.print((int)TENSOR_ARENA_BYTES);
  Serial.println(F(" bytes"));
}

void setup() {
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LED_OFF);

  Serial.begin(SERIAL_BAUD);
  delay(300);
  uint32_t heap = ESP.getFreeHeap();
  printBootBanner(heap, (size_t)begum_model_tflite_len);

  // OLED
  oled_init();
  Serial.println(F("OLED: Initialized"));
  oled_status("boot...", "init audio");

  // Audio
  if (audioCapture.begin()) {
    Serial.println(F("Audio: Initialized"));
#ifdef USE_ANALOG_MIC
    Serial.println(F("Audio: source=MAX9814 analog A0"));
#else
    Serial.println(F("Audio: source=INMP441 I2S DOUT=12 WS=13 SCK=14"));
#endif
  } else {
    Serial.println(F("Audio: FAILED"));
    oled_status("AUDIO FAIL", "check wiring");
    while (true) { delay(1000); }
  }

  // KWS
  bool ok = kws.begin();
  if (ok) {
    Serial.println(F("KWS: Ready"));
  } else {
    Serial.println(F("KWS: model invalid (dummy 20-byte placeholder?)"));
    Serial.println(F("KWS: run training/export_tflite.py, continuing with prob=0"));
  }

  // Boot diagnostics to OLED
  uint32_t heap2 = ESP.getFreeHeap();
  Serial.print(F("Free Heap: "));
  Serial.print(heap2);
  Serial.println(F(" bytes"));
  oled_boot(heap2, 10.0f);
  delay(1200);

  Serial.println(F("Listening..."));
  oled_listening();

  frame_counter = 0;
  infer_calls = 0;
  rate_window_start = millis();
  rate_window_inf = 0;
  last_trigger_ms = 0;
  first_trigger_done = false;
}

static void handleDetection(float avg_prob) {
  uint32_t now = millis();
  Serial.print(F("["));
  Serial.print(now);
  Serial.print(F("] BEGUM DETECTED! Confidence: "));
  Serial.println(avg_prob, 4);

  oled_detected(avg_prob);
  digitalWrite(PIN_LED, LED_ON);
  Serial.print(F("["));
  Serial.print(millis());
  Serial.println(F("] LED ON, OLED Updated"));

  // 3 s lockout with OLED countdown (non-blocking-ish: keep feeding WDT).
  const uint32_t t0 = millis();
  int lastShown = -1;
  while (millis() - t0 < COOLDOWN_MS) {
    uint32_t left = COOLDOWN_MS - (millis() - t0);
    int secs = (int)((left + 999UL) / 1000UL);
    if (secs != lastShown) {
      lastShown = secs;
      oled_cooldown(secs);
    }
    yield();
    delay(10);
  }

  digitalWrite(PIN_LED, LED_OFF);
  oled_listening();
  Serial.print(F("["));
  Serial.print(millis());
  Serial.println(F("] Cooldown: 3s"));
  Serial.print(F("["));
  Serial.print(millis());
  Serial.println(F("] Ready."));
  last_trigger_ms = millis();
  first_trigger_done = true;
}

void loop() {
  // a. Capture 10 ms of audio (blocks ~10 ms for I2S bit-bang).
  if (!audioCapture.readFrame(audio_buffer, FRAME_SAMPLES)) {
    Serial.println(F("WARN: audio read failed"));
    delay(1);
    return;
  }
  frame_counter++;

  // c. DUTY-CYCLE: heavy inference only every 10th frame (100 ms => ~10 Hz).
  if ((frame_counter % INFERENCE_EVERY_N_FRAMES) == 0) {
    float avg_prob = kws.tick(audio_buffer);
    last_avg_prob = avg_prob;
    infer_calls++;
    rate_window_inf++;

    // d. Threshold check with 3 s cooldown.
    uint32_t now = millis();
    bool cooled = !first_trigger_done || (now - last_trigger_ms >= COOLDOWN_MS);
    if (avg_prob > PROBABILITY_CUTOFF && cooled) {
      handleDetection(avg_prob);
      // reset rate window after blocking cooldown so Hz stays honest
      rate_window_start = millis();
      rate_window_inf = 0;
    }
  }

  // Inference-rate log every 10 s: proves ~10 Hz duty-cycle.
  uint32_t now2 = millis();
  if (now2 - rate_window_start >= INFER_RATE_LOG_MS) {
    float secs = (float)(now2 - rate_window_start) / 1000.0f;
    float hz = secs > 0.0f ? (float)rate_window_inf / secs : 0.0f;
    Serial.print(F("["));
    Serial.print(now2);
    Serial.print(F("] infer_rate="));
    Serial.print(hz, 2);
    Serial.print(F(" Hz frames="));
    Serial.print(frame_counter);
    Serial.print(F(" infers="));
    Serial.print(infer_calls);
    Serial.print(F(" last_prob="));
    Serial.print(last_avg_prob, 4);
    Serial.print(F(" heap="));
    Serial.println(ESP.getFreeHeap());
    rate_window_start = now2;
    rate_window_inf = 0;
  }

  // f. Mandatory: keeps idle CPU at ~8-9% (lets background SDK tasks idle).
  delay(1);
}
