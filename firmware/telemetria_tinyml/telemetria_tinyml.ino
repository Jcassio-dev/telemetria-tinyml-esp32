// Nó de telemetria veicular com TinyML (ESP32 + MPU6050 + DHT + PIR).
//
// O ESP32 lê o IMU a 50 Hz, calcula 3 atributos invariantes ao rumo e,
// a cada 0,5 s, roda uma CNN 1D quantizada em INT8 (TensorFlow Lite Micro)
// sobre os últimos 2 s. Só eventos são enviados, em JSON, pela serial.
//
// Bibliotecas (Gerenciador de Bibliotecas do Arduino IDE):
//   - tflm_esp32 (Simone Salerno / eloquentarduino)
//   - Adafruit MPU6050 (+ Adafruit Unified Sensor, Adafruit BusIO)
//   - DHT sensor library (Adafruit)
//
// Ligações (ESP32 DevKit V1):
//   MPU6050: VCC 3V3, GND, SDA GPIO21, SCL GPIO22
//   DHT22:   VCC 3V3, GND, DATA GPIO4 (com resistor de 10k para 3V3, se o módulo não tiver)
//   PIR:     VCC 5V (VIN), GND, OUT GPIO27
//
// Montagem: placa do MPU6050 deitada na horizontal, chip para cima.
// Para separar frenagem de aceleração, aponte o eixo X impresso no módulo
// para a frente do veículo (ver kForwardAxisAligned).

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <Wire.h>
#include <tflm_esp32.h>  // precisa vir antes de classifier.h

#include "classifier.h"
#include "imu_features.h"

// ---------- configuração ----------
constexpr int kPinPir = 27;
constexpr int kPinDht = 4;
#define DHT_TYPE DHT22  // troque para DHT11 se for o seu sensor

constexpr bool kForwardAxisAligned = true;     // eixo X do MPU aponta para a frente?
constexpr uint32_t kSamplePeriodUs = 20000;    // 50 Hz, igual ao treino
constexpr int kInferEverySamples = 25;         // inferência a cada 0,5 s
constexpr int kCalibrationSamples = 100;       // 2 s parado na inicialização
constexpr float kMinConfidence = 0.85f;  // ver curva de limiares no README
constexpr uint32_t kEventRefractoryMs = 1000;
constexpr uint32_t kTelemetryPeriodMs = 10000;
constexpr uint32_t kPresenceRefractoryMs = 30000;
constexpr float kStillAccel = 0.4f;            // m/s², limiar de "veículo parado"
constexpr float kStillGyro = 0.05f;            // rad/s

// ---------- estado ----------
Adafruit_MPU6050 mpu;
DHT dht(kPinDht, DHT_TYPE);
Classifier classifier;
WindowBuffer window;
EventFilter events(kMinConfidence, kEventRefractoryMs);
ImuSample bias;
float ax_ring[kWindowSize] = {};               // aceleração no eixo X, para o sinal da manobra
int ax_head = 0;
int samples_since_infer = 0;
uint32_t next_sample_us = 0;
uint32_t last_telemetry_ms = 0;
uint32_t last_presence_ms = 0;
bool presence_reported = false;
float temp_c = NAN, humidity = NAN;

ImuSample read_imu() {
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  return ImuSample{a.acceleration.x, a.acceleration.y, a.acceleration.z,
                   g.gyro.x, g.gyro.y, g.gyro.z};
}

float mean_ax() {
  float s = 0;
  for (float v : ax_ring) s += v;
  return s / kWindowSize;
}

void print_number_or_null(const char* key, float v) {
  if (isnan(v)) Serial.printf(",\"%s\":null", key);
  else Serial.printf(",\"%s\":%.1f", key, v);
}

void emit_event(const char* name, float conf, uint32_t latency_us) {
  Serial.printf("{\"tipo\":\"evento\",\"t_ms\":%lu,\"evento\":\"%s\",\"confianca\":%.2f,\"latencia_us\":%lu",
                (unsigned long)millis(), name, conf, (unsigned long)latency_us);
  print_number_or_null("temp_c", temp_c);
  print_number_or_null("umidade", humidity);
  Serial.printf(",\"presenca\":%s}\n", digitalRead(kPinPir) ? "true" : "false");
}

void setup() {
  Serial.begin(115200);
  delay(500);
  pinMode(kPinPir, INPUT);
  dht.begin();

  Wire.begin();
  Wire.setClock(400000);
  if (!mpu.begin()) {
    Serial.println("{\"tipo\":\"erro\",\"mensagem\":\"MPU6050 não encontrado\"}");
    while (true) delay(1000);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  if (!classifier.begin()) {
    Serial.println("{\"tipo\":\"erro\",\"mensagem\":\"falha ao iniciar o modelo TFLite Micro\"}");
    while (true) delay(1000);
  }

  // Calibração: mede o viés com o veículo parado (inclui a gravidade e a inclinação da placa).
  Serial.println("{\"tipo\":\"info\",\"mensagem\":\"calibrando: mantenha parado por 2 s\"}");
  Calibration calib;
  for (int i = 0; i < kCalibrationSamples; ++i) {
    calib.add(read_imu());
    delay(kSamplePeriodUs / 1000);
  }
  bias = calib.bias();
  temp_c = dht.readTemperature();
  humidity = dht.readHumidity();

  Serial.printf("{\"tipo\":\"info\",\"modelo_bytes\":%u,\"arena_usada_bytes\":%u,\"canais\":%d,\"janela\":%d}\n",
                g_model_len, (unsigned)classifier.arena_used_bytes(), kNumChannels, kWindowSize);
  next_sample_us = micros();
}

void loop() {
  // Amostragem a 50 Hz com período fixo (sem acumular atraso).
  if ((int32_t)(micros() - next_sample_us) < 0) return;
  next_sample_us += kSamplePeriodUs;

  const ImuSample s = read_imu();
  float f[kNumChannels];
  compute_features(s, bias, f);
  window.push(f);
  ax_ring[ax_head] = s.ax - bias.ax;
  ax_head = (ax_head + 1) % kWindowSize;

  const uint32_t now = millis();

  if (window.full() && ++samples_since_infer >= kInferEverySamples) {
    samples_since_infer = 0;
    window.quantize_into(classifier.input_buffer());

    const uint32_t t0 = micros();
    const bool ok = classifier.invoke();
    const uint32_t latency_us = micros() - t0;
    if (!ok) return;

    float conf = 0;
    const int cls = classifier.top(&conf);
    const int confirmed = events.update(cls, conf, now);
    if (confirmed >= 0) {
      const char* name = kClassNames[confirmed];
      if (confirmed == 4 && kForwardAxisAligned) {  // longitudinal
        name = mean_ax() < 0 ? "freada_brusca" : "aceleracao_brusca";
      }
      emit_event(name, conf, latency_us);
    }

    // Antifurto simples por regra: presença na cabine com o veículo parado.
    const bool still = window.mean(0) < kStillAccel && fabsf(window.mean(1)) < kStillGyro;
    const bool presence = digitalRead(kPinPir);
    if (presence && still && !presence_reported && now - last_presence_ms > kPresenceRefractoryMs) {
      emit_event("presenca_veiculo_parado", 1.0f, 0);
      last_presence_ms = now;
      presence_reported = true;
    }
    if (!presence) presence_reported = false;
  }

  // Telemetria periódica do ambiente (o DHT é lento, por isso só a cada 10 s).
  if (now - last_telemetry_ms >= kTelemetryPeriodMs) {
    last_telemetry_ms = now;
    temp_c = dht.readTemperature();
    humidity = dht.readHumidity();
    Serial.printf("{\"tipo\":\"telemetria\",\"t_ms\":%lu", (unsigned long)now);
    print_number_or_null("temp_c", temp_c);
    print_number_or_null("umidade", humidity);
    Serial.printf(",\"presenca\":%s}\n", digitalRead(kPinPir) ? "true" : "false");
  }
}
