"""Testes dos atributos: formato, física e invariância ao rumo."""
import numpy as np
import pytest

from features import N_CHANNELS, compute_features
from train import mirror
from features import CLASSES


def rotate_z(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return v @ r.T


def test_formato_da_saida():
    acc, gyr = np.zeros((100, 3)), np.zeros((100, 3))
    assert compute_features(acc, gyr).shape == (100, N_CHANNELS)


def test_rejeita_formato_errado():
    with pytest.raises(ValueError):
        compute_features(np.zeros((100, 2)), np.zeros((100, 2)))


def test_invariante_ao_rumo_do_veiculo():
    """Girar o veículo em torno do eixo vertical não muda os atributos."""
    rng = np.random.default_rng(0)
    acc, gyr = rng.normal(size=(200, 3)), rng.normal(size=(200, 3))
    base = compute_features(acc, gyr)
    for angle in (0.3, 1.57, 3.14):
        rotated = compute_features(rotate_z(acc, angle), rotate_z(gyr, angle))
        np.testing.assert_allclose(rotated, base, atol=1e-5)


def test_curva_a_esquerda_tem_guinada_positiva():
    gyr = np.tile([0.0, 0.0, 0.5], (100, 1))  # rotação anti-horária vista de cima
    feats = compute_features(np.zeros((100, 3)), gyr)
    assert np.all(feats[:, 2] > 0)


def test_espelhamento_troca_esquerda_e_direita():
    channels = ["a_h", "g_z", "g_h"]
    X = np.ones((2, 100, 3), dtype=np.float32)
    y = np.array([CLASSES.index("curva_esquerda"), CLASSES.index("longitudinal")])
    Xm, ym = mirror(X, y, channels)
    assert len(Xm) == 4
    assert ym[2] == CLASSES.index("curva_direita")      # esquerda virou direita
    assert ym[3] == CLASSES.index("longitudinal")       # longitudinal não muda
    assert np.all(Xm[2, :, 1] == -1)                     # sinal da guinada invertido
    assert np.all(Xm[2, :, 0] == 1)                      # magnitudes intactas
