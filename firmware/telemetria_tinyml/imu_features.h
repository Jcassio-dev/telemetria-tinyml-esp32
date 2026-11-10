// Lógica de pré-processamento, sem dependência de Arduino.
// Compila no ESP32 e no PC (ver tests/host/), para comparar com o Python.
//
// Replica src/features.py com a placa deitada na horizontal (eixo z para cima):
//   a_h = |(ax, ay)| sem a gravidade e o viés  (m/s²)
//   g_z = gz                                     (rad/s, positivo = esquerda)
//   g_h = |(gx, gy)|                             (rad/s)
#pragma once
#include <cmath>
#include <cstdint>

#include "model_config.h"

struct ImuSample {      // unidades SI, como entregues pela Adafruit_MPU6050
  float ax, ay, az;     // m/s²
  float gx, gy, gz;     // rad/s
};

// Remove o viés medido com o veículo parado (inclui a pequena inclinação da placa).
class Calibration {
 public:
  void add(const ImuSample& s) {
    sum_.ax += s.ax; sum_.ay += s.ay; sum_.az += s.az;
    sum_.gx += s.gx; sum_.gy += s.gy; sum_.gz += s.gz;
    ++n_;
  }
  bool ready() const { return n_ > 0; }
  ImuSample bias() const {
    if (n_ == 0) return ImuSample{0, 0, 0, 0, 0, 0};
    return ImuSample{sum_.ax / n_, sum_.ay / n_, sum_.az / n_,
                     sum_.gx / n_, sum_.gy / n_, sum_.gz / n_};
  }
 private:
  ImuSample sum_{0, 0, 0, 0, 0, 0};
  int n_ = 0;
};

// Converte uma leitura do IMU nos canais do modelo, na ordem a_h, g_z, g_h.
inline void compute_features(const ImuSample& s, const ImuSample& bias, float out[kNumChannels]) {
  const float ax = s.ax - bias.ax, ay = s.ay - bias.ay;
  const float gx = s.gx - bias.gx, gy = s.gy - bias.gy, gz = s.gz - bias.gz;
  out[0] = std::sqrt(ax * ax + ay * ay);
  out[1] = gz;
  out[2] = std::sqrt(gx * gx + gy * gy);
}

// Buffer circular com as últimas kWindowSize amostras de atributos.
class WindowBuffer {
 public:
  void push(const float f[kNumChannels]) {
    for (int c = 0; c < kNumChannels; ++c) data_[head_][c] = f[c];
    head_ = (head_ + 1) % kWindowSize;
    if (count_ < kWindowSize) ++count_;
  }
  bool full() const { return count_ == kWindowSize; }

  // Copia a janela em ordem cronológica, normaliza e quantiza para INT8,
  // exatamente como o conversor do TFLite espera: q = round(x / s) + z.
  void quantize_into(int8_t* dst) const {
    for (int t = 0; t < kWindowSize; ++t) {
      const int i = (head_ + t) % kWindowSize;  // mais antiga primeiro
      for (int c = 0; c < kNumChannels; ++c) {
        const float x = data_[i][c] / kChannelScale[c];
        long q = std::lround(x / kInputScale) + kInputZeroPoint;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        dst[t * kNumChannels + c] = static_cast<int8_t>(q);
      }
    }
  }

  // Média de um canal bruto na janela (usada para separar frenagem de aceleração).
  float mean(int channel) const {
    float s = 0;
    for (int i = 0; i < kWindowSize; ++i) s += data_[i][channel];
    return s / kWindowSize;
  }

 private:
  float data_[kWindowSize][kNumChannels] = {};
  int head_ = 0;
  int count_ = 0;
};

inline float dequantize_output(int8_t q) { return (q - kOutputZeroPoint) * kOutputScale; }

// Transforma previsões a cada 0,5 s em eventos únicos (um por manobra):
//  - exige 2 previsões iguais seguidas acima do limiar (evita ruído isolado);
//  - não repete a mesma classe enquanto a manobra continua (episódio ativo);
//  - o episódio termina após 2 previsões "normal" seguidas;
//  - mantém um intervalo mínimo entre eventos.
// A mesma lógica está em src/stream.py, para simular e avaliar no PC.
class EventFilter {
 public:
  EventFilter(float min_conf, uint32_t refractory_ms) : min_conf_(min_conf), refractory_ms_(refractory_ms) {}

  // Retorna o índice da classe a reportar, ou -1 se nada deve ser emitido.
  int update(int cls, float conf, uint32_t now_ms) {
    const bool candidate = cls != 0 && conf >= min_conf_;  // 0 = normal
    if (!candidate) {
      if (++quiet_ >= 2) active_cls_ = -1;
      last_cls_ = -1;
      return -1;
    }
    quiet_ = 0;
    const bool confirmed = cls == last_cls_;
    last_cls_ = cls;
    if (!confirmed || cls == active_cls_) return -1;
    if (has_last_event_ && now_ms - last_event_ms_ < refractory_ms_) return -1;
    active_cls_ = cls;
    last_event_ms_ = now_ms;
    has_last_event_ = true;
    return cls;
  }

 private:
  float min_conf_;
  uint32_t refractory_ms_;
  int last_cls_ = -1;
  int active_cls_ = -1;
  int quiet_ = 0;
  uint32_t last_event_ms_ = 0;
  bool has_last_event_ = false;
};
