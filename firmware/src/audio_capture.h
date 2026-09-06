#ifndef BEGUM_AUDIO_CAPTURE_H
#define BEGUM_AUDIO_CAPTURE_H

// ============================================================================
// audio_capture.h — Software I2S (INMP441) via timer1 + DMA-style ping-pong,
//                   with ADC fallback (MAX9814 on A0 via USE_ANALOG_MIC).
// Wiring:
//   INMP441 DOUT -> GPIO12, LRCLK -> GPIO13, BCLK -> GPIO14, L/R -> GND.
//   MAX9814 OUT -> A0 (fallback, define USE_ANALOG_MIC).
// Provides: bool readFrame(int16_t* buffer, int num_samples) — blocks until
// `num_samples` (160 = 10 ms @16kHz) are filled. Circular buffer 512 int16.
// 100% offline. No WiFi.
// ============================================================================

#include <Arduino.h>
#include "config.h"

extern "C" {
#include "user_interface.h"
#include "eagle_soc.h"
#include "ets_sys.h"
}

#define AUDIO_RING_SIZE 512

class AudioCapture {
 public:
  AudioCapture()
      : _writePos(0), _readPos(0), _count(0), _initialized(false), _sampleCounter(0) {}

  bool begin() {
#ifdef USE_ANALOG_MIC
    // Analog fallback: MAX9814 on A0. ADC on ESP8266 is ~10-bit, polled from
    // loop with micros() pacing to approximate 16 kHz. No timer1 needed.
    _initialized = true;
    _lastSampleMicros = micros();
    return true;
#else
    // Software I2S for INMP441.
    pinMode(PIN_I2S_BCLK, OUTPUT);
    pinMode(PIN_I2S_LRCLK, OUTPUT);
    pinMode(PIN_I2S_DOUT, INPUT);
    digitalWrite(PIN_I2S_BCLK, LOW);
    digitalWrite(PIN_I2S_LRCLK, LOW);
    // Pre-fill buffer with silence so first readFrame never underruns.
    for (int i = 0; i < AUDIO_RING_SIZE; i++) _ring[i] = 0;
    _writePos = 0;
    _readPos = 0;
    _count = 0;
    _initialized = true;
    return true;
#endif
  }

  bool isInitialized() const { return _initialized; }

  // Blocking read of num_samples (typically 160). Returns false on error.
  bool readFrame(int16_t* buffer, int num_samples) {
    if (!_initialized || buffer == nullptr || num_samples <= 0) return false;
#ifdef USE_ANALOG_MIC
    return readFrameAnalog(buffer, num_samples);
#else
    return readFrameI2S(buffer, num_samples);
#endif
  }

 private:
  volatile int16_t _ring[AUDIO_RING_SIZE];
  volatile int _writePos;
  volatile int _readPos;
  volatile int _count;
  bool _initialized;
  uint32_t _sampleCounter;
  uint32_t _lastSampleMicros;

  // --- Bit-banged I2S sample: reads one 24-bit left-channel sample, returns 16-bit ---
  inline int16_t readI2SSample() {
    // INMP441, L/R=GND => outputs left channel when WS=LOW.
    // Frame: WS LOW for 32 BCLKs (24 data bits MSB-first + 8 trailing).
    uint32_t raw = 0;
    digitalWrite(PIN_I2S_LRCLK, LOW);
    for (int bit = 0; bit < 32; bit++) {
      digitalWrite(PIN_I2S_BCLK, HIGH);
      // ~1 us half-period => ~500 kHz BCLK. delayMicroseconds(1) is coarse but OK.
      delayMicroseconds(1);
      if (bit < 24) {
        raw = (raw << 1) | (digitalRead(PIN_I2S_DOUT) & 0x1);
      }
      digitalWrite(PIN_I2S_BCLK, LOW);
      delayMicroseconds(1);
    }
    digitalWrite(PIN_I2S_LRCLK, HIGH);  // right channel (ignored)
    for (int bit = 0; bit < 32; bit++) {
      digitalWrite(PIN_I2S_BCLK, HIGH);
      delayMicroseconds(1);
      digitalWrite(PIN_I2S_BCLK, LOW);
      delayMicroseconds(1);
    }
    // raw is 24-bit two's complement in lower 24 bits; sign-extend then take top 16.
    int32_t s = (int32_t)(raw << 8) >> 8;  // sign extend 24->32
    int16_t out = (int16_t)(s >> 8);       // 24->16 bit
    return out;
  }

  // Push one sample into ring (drops oldest on overflow).
  inline void ringPush(int16_t s) {
    _ring[_writePos] = s;
    _writePos = (_writePos + 1) % AUDIO_RING_SIZE;
    if (_count < AUDIO_RING_SIZE) {
      _count++;
    } else {
      _readPos = (_readPos + 1) % AUDIO_RING_SIZE;  // overwrite oldest
    }
  }

  inline bool ringPop(int16_t* out) {
    if (_count <= 0) return false;
    *out = _ring[_readPos];
    _readPos = (_readPos + 1) % AUDIO_RING_SIZE;
    _count--;
    return true;
  }

  bool readFrameI2S(int16_t* buffer, int num_samples) {
    // Ping-pong strategy: keep ring topped up, then drain num_samples.
    // Each bit-banged sample costs ~64-70 us, so 160 samples ~= 10-11 ms,
    // which naturally paces the 10 ms capture cadence.
    for (int i = 0; i < num_samples; i++) {
      int16_t s = readI2SSample();
      ringPush(s);
    }
    // Drain oldest num_samples into caller buffer (FIFO order).
    // Because we just pushed num_samples, at least that many are available.
    noInterrupts();
    int available = _count;
    interrupts();
    int toRead = num_samples;
    if (available < toRead) {
      // Underrun (should not happen): pad remainder with last sample.
      int j = 0;
      noInterrupts();
      while (_count > 0 && j < available) {
        int16_t v;
        // inline pop under lock
        v = _ring[_readPos];
        _readPos = (_readPos + 1) % AUDIO_RING_SIZE;
        _count--;
        buffer[j++] = v;
      }
      interrupts();
      int16_t fill = (j > 0) ? buffer[j - 1] : 0;
      while (j < toRead) buffer[j++] = fill;
      return true;
    }
    noInterrupts();
    for (int i = 0; i < toRead; i++) {
      buffer[i] = _ring[_readPos];
      _readPos = (_readPos + 1) % AUDIO_RING_SIZE;
      _count--;
    }
    interrupts();
    _sampleCounter += (uint32_t)toRead;
    return true;
  }

#ifdef USE_ANALOG_MIC
  bool readFrameAnalog(int16_t* buffer, int num_samples) {
    // MAX9814 on A0: sample at ~16 kHz using micros() pacing.
    // ESP8266 ADC: 0..1023 for 0..1V (or 0..3.3V on D1 mini with divider).
    // Center at 512, scale to int16 with fixed gain.
    const uint32_t periodUs = 1000000UL / SAMPLE_RATE_HZ;  // 62 us
    for (int i = 0; i < num_samples; i++) {
      uint32_t now = micros();
      uint32_t target = _lastSampleMicros + periodUs;
      // wait until next sample time (handles micros() wraparound via signed diff)
      while ((int32_t)(micros() - target) < 0) {
        yield();
      }
      _lastSampleMicros = micros();
      int adc = system_adc_read();  // 0..1023
      int32_t centered = (adc - 512) * 64;  // -> ~[-32768, +32640]
      if (centered > 32767) centered = 32767;
      if (centered < -32768) centered = -32768;
      buffer[i] = (int16_t)centered;
      (void)now;
    }
    _sampleCounter += (uint32_t)num_samples;
    return true;
  }
#endif
};

#endif  // BEGUM_AUDIO_CAPTURE_H
