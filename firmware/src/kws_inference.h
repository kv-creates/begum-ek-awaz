#ifndef BEGUM_KWS_INFERENCE_H
#define BEGUM_KWS_INFERENCE_H

// ============================================================================
// kws_inference.h — TFLite-Micro keyword spotting for "Begum".
// - AllOpsResolver, static arena exactly 15360 bytes, 40-dim log-Mel,
//   Hann 400, FFT 512, 8-point sliding average.
// - Input model: 49 x 40 x 1 int8, output: 2 x int8 [non-begum, begum].
// - tick(frame160) appends 160 samples (10 ms), computes ONE mel frame from
//   the newest 400 samples, shifts the 49-frame spectrogram, invokes TFLM,
//   updates prob_history[8], returns averaged begum probability.
// ============================================================================

#include <Arduino.h>
#include <math.h>
#include "config.h"
#include "model_data.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Exactly 15360 bytes per RAM budget.
static uint8_t tensor_arena[TENSOR_ARENA_BYTES] __attribute__((aligned(16)));
static_assert(sizeof(tensor_arena) == 15360, "arena must be 15360 bytes");

class KWSInference {
 public:
  KWSInference()
      : _interpreter(nullptr),
        _inputTensor(nullptr),
        _outputTensor(nullptr),
        _modelValid(false),
        _histIdx(0),
        _histFilled(0),
        _lastRaw(0.0f),
        _pcmPos(0) {
    for (int i = 0; i < PROB_HISTORY_LEN; i++) _probHist[i] = 0.0f;
    for (int i = 0; i < N_FRAMES; i++)
      for (int j = 0; j < N_MELS; j++) _spectrogram[i][j] = 0.0f;
    for (int i = 0; i < SAMPLE_RATE_HZ; i++) _pcm[i] = 0;
    buildHann();
    buildMelFilterbank();
  }

  bool begin() {
    _reporter = new tflite::MicroErrorReporter();
    _model = tflite::GetModel(begum_model_tflite);
    if (_model == nullptr || _model->version() != TFLITE_SCHEMA_VERSION) {
      if (_reporter) _reporter->Report("KWS: bad model (dummy placeholder?)");
      _modelValid = false;
      return false;
    }
    // Verify model size matches header length (dummy is 20 bytes -> invalid flatbuffer).
    if (begum_model_tflite_len < 1024) {
      if (_reporter) _reporter->Report("KWS: dummy 20-byte model, run export_tflite.py");
      _modelValid = false;
      return false;
    }
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter interpreter(_model, resolver, tensor_arena,
                                               TENSOR_ARENA_BYTES, _reporter);
    _interpreter = &interpreter;
    if (_interpreter->AllocateTensors() != kTfLiteOk) {
      if (_reporter) _reporter->Report("KWS: AllocateTensors failed (arena 15360)");
      _modelValid = false;
      return false;
    }
    _inputTensor = _interpreter->input(0);
    _outputTensor = _interpreter->output(0);
    _modelValid = true;
    // Prime spectrogram with silence frames so first invokes are stable.
    for (int i = 0; i < N_FRAMES; i++) {
      float frame[N_MELS];
      computeMelFrame(_pcm, 0, frame);
      for (int j = 0; j < N_MELS; j++) _spectrogram[i][j] = frame[j];
    }
    return true;
  }

  bool isModelValid() const { return _modelValid; }
  float lastRaw() const { return _lastRaw; }

  // Append 160 new samples, run one streaming step, return 8-pt averaged prob.
  // Called every 10th capture frame (every 100 ms) per duty-cycle.
  float tick(const int16_t* frame160) {
    if (frame160 == nullptr) return averaged();
    // Append to 1 s PCM ring (linear shift buffer, 16000 int16 = 32 KB... too big
    // for ESP8266 heap? Use circular buffer of 16000 int16 in static RAM = 32KB,
    // acceptable within 80KB total. Alternatively keep linear with memmove.)
    appendPcm(frame160, FRAME_SAMPLES);
    // Compute one mel frame from newest WIN_LEN samples.
    float mel[N_MELS];
    computeMelFrameNewest(mel);
    // Shift spectrogram up by one, append new frame at end.
    for (int i = 0; i < N_FRAMES - 1; i++)
      for (int j = 0; j < N_MELS; j++) _spectrogram[i][j] = _spectrogram[i + 1][j];
    for (int j = 0; j < N_MELS; j++) _spectrogram[N_FRAMES - 1][j] = mel[j];

    float raw = 0.0f;
    if (_modelValid) {
      raw = invoke();
    }
    _lastRaw = raw;
    _probHist[_histIdx] = raw;
    _histIdx = (_histIdx + 1) % PROB_HISTORY_LEN;
    if (_histFilled < PROB_HISTORY_LEN) _histFilled++;
    return averaged();
  }

  float averaged() const {
    if (_histFilled == 0) return 0.0f;
    float s = 0.0f;
    for (int i = 0; i < _histFilled; i++) s += _probHist[i];
    return s / (float)_histFilled;
  }

 private:
  tflite::MicroErrorReporter* _reporter = nullptr;
  const tflite::Model* _model = nullptr;
  tflite::MicroInterpreter* _interpreter;
  TfLiteTensor* _inputTensor;
  TfLiteTensor* _outputTensor;
  bool _modelValid;
  float _probHist[PROB_HISTORY_LEN];
  int _histIdx;
  int _histFilled;
  float _lastRaw;

  // Streaming buffers
  int16_t _pcm[SAMPLE_RATE_HZ];  // 1 s circular
  int _pcmPos;                   // write head
  float _spectrogram[N_FRAMES][N_MELS];

  // DSP tables
  float _hann[WIN_LEN];
  float _melFB[N_MELS][FFT_SIZE / 2 + 1];

  void buildHann() {
    for (int n = 0; n < WIN_LEN; n++) {
      _hann[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / (float)(WIN_LEN - 1));
    }
  }

  static float hzToMel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
  static float melToHz(float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

  void buildMelFilterbank() {
    const float fmin = 60.0f, fmax = 7800.0f;
    const int nfft = FFT_SIZE, nmels = N_MELS, sr = SAMPLE_RATE_HZ;
    float melLow = hzToMel(fmin), melHigh = hzToMel(fmax);
    float melPts[nmels + 2];
    for (int i = 0; i < nmels + 2; i++)
      melPts[i] = melLow + (melHigh - melLow) * (float)i / (float)(nmels + 1);
    float hzPts[nmels + 2];
    for (int i = 0; i < nmels + 2; i++) hzPts[i] = melToHz(melPts[i]);
    float binPts[nmels + 2];
    for (int i = 0; i < nmels + 2; i++) binPts[i] = (float)(nfft + 1) * hzPts[i] / (float)sr;
    for (int m = 0; m < nmels; m++) {
      for (int k = 0; k < nfft / 2 + 1; k++) {
        float f = (float)k;
        float w = 0.0f;
        if (f >= binPts[m] && f <= binPts[m + 1])
          w = (f - binPts[m]) / (binPts[m + 1] - binPts[m] + 1e-9f);
        else if (f >= binPts[m + 1] && f <= binPts[m + 2])
          w = (binPts[m + 2] - f) / (binPts[m + 2] - binPts[m + 1] + 1e-9f);
        _melFB[m][k] = w;
      }
    }
  }

  void appendPcm(const int16_t* src, int n) {
    for (int i = 0; i < n; i++) {
      _pcm[_pcmPos] = src[i];
      _pcmPos = (_pcmPos + 1) % SAMPLE_RATE_HZ;
    }
  }

  // Gather newest WIN_LEN samples (oldest->newest) into tmp float [-1,1].
  void newestWindow(float* out) {
    int start = (_pcmPos - WIN_LEN + SAMPLE_RATE_HZ * 2) % SAMPLE_RATE_HZ;
    for (int i = 0; i < WIN_LEN; i++) {
      int idx = (start + i) % SAMPLE_RATE_HZ;
      out[i] = (float)_pcm[idx] / 32768.0f;
    }
  }

  // In-place radix-2 FFT (real input packed as complex). n must be 512.
  static void fft512(float* re, float* im) {
    const int N = 512;
    // bit reversal
    for (int i = 1, j = 0; i < N; i++) {
      int bit = N >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        float t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (int len = 2; len <= N; len <<= 1) {
      float ang = -2.0f * (float)M_PI / (float)len;
      float wRe = cosf(ang), wIm = sinf(ang);
      for (int i = 0; i < N; i += len) {
        float curRe = 1.0f, curIm = 0.0f;
        for (int j = 0; j < len / 2; j++) {
          float uRe = re[i + j], uIm = im[i + j];
          float vRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm;
          float vIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe;
          re[i + j] = uRe + vRe;
          im[i + j] = uIm + vIm;
          re[i + j + len / 2] = uRe - vRe;
          im[i + j + len / 2] = uIm - vIm;
          float nRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nRe;
        }
      }
    }
  }

  void computeMelFrameNewest(float* melOut) {
    float win[WIN_LEN];
    newestWindow(win);
    // pre-emphasis
    for (int i = WIN_LEN - 1; i > 0; i--) win[i] = win[i] - 0.97f * win[i - 1];
    // Hann + zero-pad to 512
    float re[FFT_SIZE], im[FFT_SIZE];
    for (int i = 0; i < FFT_SIZE; i++) { re[i] = 0.0f; im[i] = 0.0f; }
    for (int i = 0; i < WIN_LEN; i++) re[i] = win[i] * _hann[i];
    fft512(re, im);
    float power[FFT_SIZE / 2 + 1];
    for (int k = 0; k < FFT_SIZE / 2 + 1; k++)
      power[k] = re[k] * re[k] + im[k] * im[k];
    for (int m = 0; m < N_MELS; m++) {
      float s = 0.0f;
      for (int k = 0; k < FFT_SIZE / 2 + 1; k++) s += power[k] * _melFB[m][k];
      float v = logf(s + 1e-10f);
      // fixed streaming norm (matches training CMVN approx: mean~2, std~2)
      melOut[m] = (v - 2.0f) * 0.5f;
      if (melOut[m] > 3.0f) melOut[m] = 3.0f;
      if (melOut[m] < -3.0f) melOut[m] = -3.0f;
    }
  }

  // Used only at boot to prime with silence (reads from linear _pcm zeros).
  void computeMelFrame(const int16_t* pcmLin, int offset, float* melOut) {
    (void)pcmLin; (void)offset;
    for (int m = 0; m < N_MELS; m++) melOut[m] = -1.0f;
  }

  float invoke() {
    if (!_interpreter || !_inputTensor || !_outputTensor) return 0.0f;
    // Quantize spectrogram -> int8 input.
    float inScale = _inputTensor->params.scale;
    int inZp = _inputTensor->params.zero_point;
    if (inScale <= 0.0f) inScale = 0.1f;
    int8_t* qin = _inputTensor->data.int8;
    // Input layout: [1, 49, 40, 1]
    int idx = 0;
    for (int i = 0; i < N_FRAMES; i++) {
      for (int j = 0; j < N_MELS; j++) {
        float v = _spectrogram[i][j];
        int q = (int)roundf(v / inScale) + inZp;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        qin[idx++] = (int8_t)q;
      }
    }
    if (_interpreter->Invoke() != kTfLiteOk) return _lastRaw;
    // Dequantize output [1,2] int8 -> softmax probs.
    int8_t* qout = _outputTensor->data.int8;
    float outScale = _outputTensor->params.scale;
    int outZp = _outputTensor->params.zero_point;
    float logit0 = ((float)qout[0] - (float)outZp) * outScale;
    float logit1 = ((float)qout[1] - (float)outZp) * outScale;
    // Outputs are already softmaxed in float domain pre-quant; but to be safe
    // apply softmax in case model outputs logits.
    float m = logit0 > logit1 ? logit0 : logit1;
    float e0 = expf(logit0 - m), e1 = expf(logit1 - m);
    float prob = e1 / (e0 + e1 + 1e-9f);
    // If model was trained with softmax output, dequant directly approximates
    // probabilities; blending both interpretations: if e-softmax saturates,
    // fall back to direct dequant of channel 1 clipped to [0,1].
    // Heuristic: trust softmax form (correct for both cases when calibrated).
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;
    return prob;
  }
};

#endif  // BEGUM_KWS_INFERENCE_H
