"""Testes do modelo exportado: formato INT8, tamanho e fidelidade ao FP32."""
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from features import CHANNEL_NAMES, CLASSES, WINDOW_SIZE, normalize

ART = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model_int8.tflite").exists(),
                                reason="rode src/train.py antes")


@pytest.fixture(scope="module")
def report():
    return json.loads((ART / "report.json").read_text())


def interpreter(name):
    it = tf.lite.Interpreter(model_path=str(ART / name),
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    return it


def test_entrada_e_saida_sao_int8(report):
    it = interpreter("model_int8.tflite")
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    assert inp["dtype"] == np.int8 and out["dtype"] == np.int8
    assert tuple(inp["shape"]) == (1, WINDOW_SIZE, len(report["channels"]))
    assert tuple(out["shape"]) == (1, len(CLASSES))


def test_cabe_com_folga_no_esp32():
    assert (ART / "model_int8.tflite").stat().st_size < 64 * 1024


def test_int8_concorda_com_fp32(report):
    """A quantização não pode mudar a decisão do modelo na grande maioria das janelas."""
    d = np.load(ART / "windows.npz")
    idx = [CHANNEL_NAMES.index(c) for c in report["channels"]]
    X = normalize(d["X"][:, :, idx], np.array(report["scale"], dtype=np.float32))[:200]

    fp32, int8 = interpreter("model_fp32.tflite"), interpreter("model_int8.tflite")
    fi, fo = fp32.get_input_details()[0], fp32.get_output_details()[0]
    qi, qo = int8.get_input_details()[0], int8.get_output_details()[0]
    s, z = qi["quantization"]
    agree = 0
    for w in X:
        fp32.set_tensor(fi["index"], w[None]); fp32.invoke()
        int8.set_tensor(qi["index"], np.clip(np.round(w[None] / s + z), -128, 127).astype(np.int8)); int8.invoke()
        agree += np.argmax(fp32.get_tensor(fo["index"])) == np.argmax(int8.get_tensor(qo["index"]))
    assert agree / len(X) >= 0.95
