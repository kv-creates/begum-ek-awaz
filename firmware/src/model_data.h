#ifndef BEGUM_MODEL_DATA_H
#define BEGUM_MODEL_DATA_H

// DUMMY placeholder (20 bytes). training/export_tflite.py OVERWRITES this file
// with the real ~50KB INT8 quantized TFLite model byte array.
// Do not edit: the exporter replaces the entire array + length below.
#include <cstdint>

const unsigned int begum_model_tflite_len = 20;
const alignas(16) unsigned char begum_model_tflite[20] = {
  0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x00, 0x00, 0x42, 0x45,
  0x47, 0x55, 0x4d, 0x2d, 0x44, 0x55, 0x4d, 0x4d, 0x59, 0x00
};

#endif // BEGUM_MODEL_DATA_H
