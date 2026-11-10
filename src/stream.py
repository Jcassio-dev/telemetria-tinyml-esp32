"""Execução contínua do modelo, como no ESP32: janela de 2 s, inferência a cada 0,5 s.

Usado pelo simulador (gateway/simulate.py) e pela avaliação por viagem (src/evaluate_stream.py).
O EventFilter é a mesma lógica do firmware (firmware/telemetria_tinyml/imu_features.h).
"""
import numpy as np
import tensorflow as tf

from features import CLASSES, WINDOW_SIZE

INFER_EVERY = 25          # amostras (0,5 s), igual ao firmware
MIN_CONF = 0.85
REFRACTORY_MS = 1000


class EventFilter:
    def __init__(self, min_conf=MIN_CONF, refractory_ms=REFRACTORY_MS):
        self.min_conf, self.refractory_ms = min_conf, refractory_ms
        self.last_cls, self.active_cls, self.quiet, self.last_event_ms = -1, -1, 0, None

    def update(self, cls, conf, now_ms):
        candidate = cls != 0 and conf >= self.min_conf
        if not candidate:
            self.quiet += 1
            if self.quiet >= 2:
                self.active_cls = -1
            self.last_cls = -1
            return -1
        self.quiet = 0
        confirmed = cls == self.last_cls
        self.last_cls = cls
        if not confirmed or cls == self.active_cls:
            return -1
        if self.last_event_ms is not None and now_ms - self.last_event_ms < self.refractory_ms:
            return -1
        self.active_cls, self.last_event_ms = cls, now_ms
        return cls


def run_stream(tflite_bytes, feats_normalized, grid, min_conf=MIN_CONF):
    """Roda o modelo INT8 sobre uma viagem inteira e devolve [(tempo_s, classe, confiança)].

    Usa os kernels de referência do TFLite, que dão o mesmo resultado do TFLite Micro.
    """
    interp = tf.lite.Interpreter(model_content=tflite_bytes,
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]

    filt, events = EventFilter(min_conf=min_conf), []
    for end in range(WINDOW_SIZE, len(feats_normalized) + 1, INFER_EVERY):
        w = feats_normalized[end - WINDOW_SIZE:end]
        interp.set_tensor(inp["index"], np.clip(np.round(w / s_in + z_in), -128, 127).astype(np.int8)[None])
        interp.invoke()
        logits = interp.get_tensor(out["index"])[0].astype(np.int32)  # evita estouro do int8
        cls = int(np.argmax(logits))
        conf = float((logits[cls] - z_out) * s_out)
        t = float(grid[end - 1])
        ev = filt.update(cls, conf, int(t * 1000))
        if ev >= 0:
            events.append((t, CLASSES[ev], conf))
    return events
