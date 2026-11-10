"""Testa o código C++ do firmware compilado no PC contra o Python.

Precisa da pasta src da biblioteca Arduino tflm_esp32:
    TFLM_SRC=/caminho/tflm_esp32/src pytest tests/test_firmware.py
Sem TFLM_SRC, os testes são pulados.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from features import CHANNEL_NAMES, normalize

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "tests" / "host"
pytestmark = pytest.mark.skipif(not os.environ.get("TFLM_SRC"), reason="defina TFLM_SRC para compilar o firmware no PC")


@pytest.fixture(scope="module")
def binary():
    subprocess.run(["bash", str(HOST / "build.sh")], check=True, capture_output=True)
    return str(HOST / "host_test")


def test_atributos_iguais_ao_python(binary):
    rng = np.random.default_rng(1)
    bias = rng.normal(size=6)
    raw = rng.normal(size=(50, 6))
    text = " ".join(f"{v:.7f}" for v in bias) + "\n" + "\n".join(" ".join(f"{v:.7f}" for v in r) for r in raw)
    out = subprocess.run([binary, "features"], input=text, capture_output=True, text=True, check=True).stdout
    cpp = np.array([[float(x) for x in line.split()] for line in out.strip().splitlines()])
    a, g = raw[:, :3] - bias[:3], raw[:, 3:] - bias[3:]
    expected = np.stack([np.hypot(a[:, 0], a[:, 1]), g[:, 2], np.hypot(g[:, 0], g[:, 1])], axis=1)
    np.testing.assert_allclose(cpp, expected, atol=1e-4)


def test_inferencia_identica_ao_tflite_de_referencia(binary):
    report = json.loads((ROOT / "artifacts" / "report.json").read_text())
    d = np.load(ROOT / "artifacts" / "windows.npz")
    idx = [CHANNEL_NAMES.index(c) for c in report["channels"]]
    X = d["X"][:150, :, idx]
    text = "\n".join(" ".join(f"{v:.7f}" for v in row) for w in X for row in w)
    out = subprocess.run([binary, "infer"], input=text, capture_output=True, text=True, check=True).stdout
    cpp = np.array([int(line.split()[0]) for line in out.strip().splitlines()])

    it = tf.lite.Interpreter(model_path=str(ROOT / "artifacts" / "model_int8.tflite"),
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i["quantization"]
    Xn = normalize(X, np.array(report["scale"], dtype=np.float32))
    py = []
    for w in Xn:
        it.set_tensor(i["index"], np.clip(np.round(w[None] / s + z), -128, 127).astype(np.int8))
        it.invoke()
        py.append(int(np.argmax(it.get_tensor(o["index"])[0])))
    assert (cpp == np.array(py)).mean() == 1.0
