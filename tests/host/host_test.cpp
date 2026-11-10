// Roda o código do firmware no PC para comparar com o Python.
//
// Modo "features": lê amostras brutas do IMU (6 floats por linha, mais o viés na
// primeira linha) e imprime os 3 atributos calculados por features.h.
//
// Modo "infer": lê janelas de atributos (kWindowSize linhas de kNumChannels floats
// por janela), passa cada uma pelo WindowBuffer + Classifier, igual ao ESP32,
// e imprime "classe confiança" por janela.
#include <cstdio>
#include <cstring>
#include <vector>

#include "classifier.h"
#include "imu_features.h"

static Classifier classifier;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "uso: host_test features|infer < entrada\n");
    return 2;
  }

  if (std::strcmp(argv[1], "features") == 0) {
    ImuSample bias, s;
    if (std::scanf("%f %f %f %f %f %f", &bias.ax, &bias.ay, &bias.az, &bias.gx, &bias.gy, &bias.gz) != 6) return 1;
    while (std::scanf("%f %f %f %f %f %f", &s.ax, &s.ay, &s.az, &s.gx, &s.gy, &s.gz) == 6) {
      float f[kNumChannels];
      compute_features(s, bias, f);
      std::printf("%.6f %.6f %.6f\n", f[0], f[1], f[2]);
    }
    return 0;
  }

  if (!classifier.begin()) {
    std::fprintf(stderr, "falha ao iniciar o modelo\n");
    return 1;
  }
  std::fprintf(stderr, "arena_used_bytes=%zu model_bytes=%u\n", classifier.arena_used_bytes(), g_model_len);

  WindowBuffer buffer;
  float row[kNumChannels];
  int t = 0;
  while (true) {
    for (int c = 0; c < kNumChannels; ++c) {
      if (std::scanf("%f", &row[c]) != 1) return 0;
    }
    buffer.push(row);
    if (++t == kWindowSize) {  // uma janela completa
      buffer.quantize_into(classifier.input_buffer());
      if (!classifier.invoke()) return 1;
      float conf = 0;
      const int cls = classifier.top(&conf);
      std::printf("%d %.6f\n", cls, conf);
      t = 0;
    }
  }
}
