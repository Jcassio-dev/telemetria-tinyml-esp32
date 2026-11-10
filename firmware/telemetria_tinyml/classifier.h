// Envolve o TensorFlow Lite Micro. Compila no ESP32 (biblioteca tflm_esp32)
// e no PC (tests/host/), com o mesmo código.
#pragma once
#include <cstdint>
#include <cstdlib>  // abort(), usado pelos cabeçalhos do TFLM
#include <cstring>

#include "model_config.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Memória de trabalho do modelo (ativações). O valor real usado é medido em
// runtime por arena_used_bytes() e reportado na inicialização.
constexpr size_t kArenaSize = 16 * 1024;

class Classifier {
 public:
  // Retorna false se o modelo não carregar (versão de schema, operação faltando ou arena pequena).
  bool begin() {
    model_ = tflite::GetModel(g_model);
    if (model_->version() != TFLITE_SCHEMA_VERSION) return false;

    // Só as 7 operações que o modelo usa, para economizar flash.
    if (resolver_.AddConv2D() != kTfLiteOk || resolver_.AddExpandDims() != kTfLiteOk ||
        resolver_.AddMaxPool2D() != kTfLiteOk || resolver_.AddReshape() != kTfLiteOk ||
        resolver_.AddMean() != kTfLiteOk || resolver_.AddFullyConnected() != kTfLiteOk ||
        resolver_.AddSoftmax() != kTfLiteOk) {
      return false;
    }

    interpreter_ = new tflite::MicroInterpreter(model_, resolver_, arena_, kArenaSize);
    if (interpreter_->AllocateTensors() != kTfLiteOk) return false;

    input_ = interpreter_->input(0);
    output_ = interpreter_->output(0);
    return input_->type == kTfLiteInt8 && output_->type == kTfLiteInt8 &&
           input_->bytes == static_cast<size_t>(kWindowSize * kNumChannels);
  }

  int8_t* input_buffer() { return input_->data.int8; }

  // Roda a inferência sobre o que estiver em input_buffer().
  bool invoke() { return interpreter_->Invoke() == kTfLiteOk; }

  // Classe com maior probabilidade e a probabilidade dela.
  int top(float* confidence) const {
    int best = 0;
    for (int i = 1; i < kNumClasses; ++i) {
      if (output_->data.int8[i] > output_->data.int8[best]) best = i;
    }
    if (confidence) *confidence = (output_->data.int8[best] - kOutputZeroPoint) * kOutputScale;
    return best;
  }

  size_t arena_used_bytes() const { return interpreter_ ? interpreter_->arena_used_bytes() : 0; }

 private:
  const tflite::Model* model_ = nullptr;
  tflite::MicroMutableOpResolver<7> resolver_;
  tflite::MicroInterpreter* interpreter_ = nullptr;
  TfLiteTensor* input_ = nullptr;
  TfLiteTensor* output_ = nullptr;
  alignas(16) uint8_t arena_[kArenaSize];
};
