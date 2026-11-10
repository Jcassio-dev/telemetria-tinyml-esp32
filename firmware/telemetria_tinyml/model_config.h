// Gerado por src/train.py. Não edite à mão.
#pragma once

constexpr int kWindowSize = 100;
constexpr int kNumChannels = 3;
constexpr int kNumClasses = 5;
// Ordem dos canais: a_h, g_z, g_h
constexpr float kChannelScale[kNumChannels] = {1.46287191f, 0.331406146f, 0.161417946f};
constexpr float kInputScale = 0.0477908924f;
constexpr int kInputZeroPoint = -45;
constexpr float kOutputScale = 0.00390625f;
constexpr int kOutputZeroPoint = -128;
const char* const kClassNames[kNumClasses] = {"normal", "curva_esquerda", "curva_direita", "troca_faixa", "longitudinal"};
