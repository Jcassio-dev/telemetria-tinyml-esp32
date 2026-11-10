"""Atributos invariantes ao rumo do veículo.

O dataset está no referencial da Terra (x = leste, y = norte, z = cima).
Sem saber para onde o carro aponta, usamos atributos que não dependem disso:

    a_h  = sqrt(ax² + ay²)   magnitude da aceleração linear horizontal (m/s²)
    a_v  = az                aceleração linear vertical (m/s²)
    g_z  = gz                velocidade de guinada (rad/s, positivo = esquerda)
    g_h  = sqrt(gx² + gy²)   magnitude da rotação horizontal (rad/s)

No ESP32, com a placa deitada na horizontal (z para cima), os mesmos quatro
valores saem direto do MPU6050, em qualquer giro da placa em torno do eixo z.
O firmware replica exatamente esta função (ver firmware/telemetria_tinyml/imu_features.h).
"""
import numpy as np

SAMPLE_RATE_HZ = 50
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)  # 100 amostras
N_CHANNELS = 4
CHANNEL_NAMES = ["a_h", "a_v", "g_z", "g_h"]

CLASSES = ["normal", "curva_esquerda", "curva_direita", "troca_faixa", "longitudinal"]

# Mapeamento dos rótulos originais do dataset para as 5 classes do projeto.
LABEL_MAP = {
    "evento_nao_agressivo": "normal",
    "curva_esquerda_agressiva": "curva_esquerda",
    "curva_direita_agressiva": "curva_direita",
    "troca_faixa_esquerda_agressiva": "troca_faixa",
    "troca_faixa_direita_agressiva": "troca_faixa",
    "freada_agressiva": "longitudinal",
    "aceleracao_agressiva": "longitudinal",
}


def compute_features(lin_acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """Converte (N, 3) de aceleração linear e (N, 3) de giroscópio em (N, 4)."""
    lin_acc = np.asarray(lin_acc, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    if lin_acc.shape != gyro.shape or lin_acc.ndim != 2 or lin_acc.shape[1] != 3:
        raise ValueError("esperado lin_acc e gyro com formato (N, 3) iguais")
    a_h = np.hypot(lin_acc[:, 0], lin_acc[:, 1])
    a_v = lin_acc[:, 2]
    g_z = gyro[:, 2]
    g_h = np.hypot(gyro[:, 0], gyro[:, 1])
    return np.stack([a_h, a_v, g_z, g_h], axis=1).astype(np.float32)


def normalize(windows: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Divide cada canal por uma escala fixa (calculada no treino e embutida no firmware)."""
    return (windows / scale.reshape(1, 1, -1)).astype(np.float32)
