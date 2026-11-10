"""Treina a CNN 1D, quantiza para INT8 e compara com FP32.

Avaliação: validação cruzada em 5 partes, agrupada por evento (StratifiedGroupKFold),
para que janelas do mesmo evento nunca fiquem no treino e no teste ao mesmo tempo.
Em cada parte, o MESMO modelo é avaliado em três formas:
    Keras FP32  ->  TFLite FP32  ->  TFLite INT8 (quantização inteira completa)

Depois treina o modelo final com todos os dados e exporta os artefatos para o ESP32.

Uso:
    python src/train.py --channels a_h,g_z,g_h
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from features import CHANNEL_NAMES, CLASSES, WINDOW_SIZE, normalize

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
EPOCHS = 80
BATCH = 32
GZ = CHANNEL_NAMES.index("g_z")


# ---------- dados ----------

def select_channels(X, channels):
    idx = [CHANNEL_NAMES.index(c) for c in channels]
    return X[:, :, idx]


def mirror(X, y, channels):
    """Aumento de dados físico: inverter o sinal da guinada espelha a manobra.

    Curva à esquerda vira curva à direita e vice-versa; as demais classes não mudam.
    Isso dobra os exemplos de curva sem inventar dados.
    """
    if "g_z" not in channels:
        return X, y
    k = channels.index("g_z")
    Xm = X.copy()
    Xm[:, :, k] *= -1
    ym = y.copy()
    left, right = CLASSES.index("curva_esquerda"), CLASSES.index("curva_direita")
    ym[y == left], ym[y == right] = right, left
    return np.concatenate([X, Xm]), np.concatenate([y, ym])


def jitter(X, rng):
    """Ruído leve e variação de escala, para tolerar diferenças entre sensores."""
    scale = rng.uniform(0.9, 1.1, size=(len(X), 1, X.shape[2]))
    noise = rng.normal(0, 0.05, size=X.shape)
    return (X * scale + noise).astype(np.float32)


# ---------- modelo ----------

def build_model(n_channels):
    tf.keras.utils.set_random_seed(SEED)
    inp = tf.keras.Input(shape=(WINDOW_SIZE, n_channels))
    x = tf.keras.layers.Conv1D(8, 5, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit(model, X, y, rng):
    Xa, ya = np.concatenate([X, jitter(X, rng)]), np.concatenate([y, y])
    counts = np.bincount(ya, minlength=len(CLASSES))
    class_weight = {i: len(ya) / (len(CLASSES) * c) for i, c in enumerate(counts) if c}
    model.fit(Xa, ya, epochs=EPOCHS, batch_size=BATCH, verbose=0,
              class_weight=class_weight, shuffle=True)
    return model


# ---------- conversão para TFLite ----------

def to_tflite(model, rep_data=None):
    """FP32 se rep_data for None; senão INT8 com entrada e saída inteiras."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if rep_data is not None:
        def representative():
            for i in range(min(len(rep_data), 300)):
                yield [rep_data[i:i + 1]]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        # O TFLite Micro ainda não suporta quantização por canal na camada densa
        # (FULLY_CONNECTED). Sem isso, o modelo falha ao iniciar no ESP32.
        converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    return converter.convert()


def tflite_predict(tflite_bytes, X):
    """Roda o modelo TFLite janela a janela, como no microcontrolador."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    preds, times = [], []
    for x in X:
        x = x[None]
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        t = time.perf_counter()
        interp.invoke()
        times.append(time.perf_counter() - t)
        preds.append(np.argmax(interp.get_tensor(out["index"])[0]))
    return np.array(preds), float(np.median(times) * 1e6)


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(len(CLASSES))), zero_division=0))


# ---------- principal ----------

def cross_validate(X, y, groups, channels):
    rng = np.random.default_rng(SEED)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, all_true, all_int8 = [], [], []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        scale = X[tr].reshape(-1, X.shape[2]).std(0)
        Xtr, Xte = normalize(X[tr], scale), normalize(X[te], scale)
        Xtr_m, ytr_m = mirror(Xtr, y[tr], channels)

        model = fit(build_model(len(channels)), Xtr_m, ytr_m, rng)
        p_keras = np.argmax(model.predict(Xte, verbose=0), axis=1)
        p_fp32, _ = tflite_predict(to_tflite(model), Xte)
        p_int8, _ = tflite_predict(to_tflite(model, Xtr), Xte)

        row = {"fold": fold, "n_test": int(len(te)),
               "f1_keras": macro_f1(y[te], p_keras),
               "f1_tflite_fp32": macro_f1(y[te], p_fp32),
               "f1_tflite_int8": macro_f1(y[te], p_int8),
               "int8_agrees_fp32": float((p_int8 == p_fp32).mean())}
        rows.append(row)
        all_true.append(y[te]); all_int8.append(p_int8)
        print(f"  parte {fold}: F1 FP32={row['f1_tflite_fp32']:.3f}  "
              f"INT8={row['f1_tflite_int8']:.3f}  concordância={row['int8_agrees_fp32']:.1%}")
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_int8),
                          labels=list(range(len(CLASSES))))
    return rows, cm


def summarize(rows, key):
    v = np.array([r[key] for r in rows])
    return float(v.mean()), float(v.std())


def to_c_array(data: bytes, name: str) -> str:
    lines = [", ".join(f"0x{b:02x}" for b in data[i:i + 12]) for i in range(0, len(data), 12)]
    body = ",\n  ".join(lines)
    return (f"// Gerado por src/train.py. Não edite à mão.\n#pragma once\n#include <cstdint>\n\n"
            f"alignas(16) const unsigned char {name}[] = {{\n  {body}\n}};\n"
            f"const unsigned int {name}_len = {len(data)};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "artifacts/windows.npz"))
    ap.add_argument("--channels", default="a_h,a_v,g_z,g_h")
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    ap.add_argument("--firmware", default=str(ROOT / "firmware/telemetria_tinyml"))
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    channels = args.channels.split(",")
    d = np.load(args.data)
    X, y, groups = select_channels(d["X"], channels), d["y"], d["groups"]
    print(f"Canais: {channels} | janelas: {len(X)}")

    rows, cm = cross_validate(X, y, groups, channels)
    report = {"channels": channels, "folds": rows,
              "confusion_matrix_int8": cm.tolist(), "classes": CLASSES}
    for k in ("f1_keras", "f1_tflite_fp32", "f1_tflite_int8", "int8_agrees_fp32"):
        report[k] = summarize(rows, k)
    print("Média: F1 FP32 = %.3f ± %.3f | F1 INT8 = %.3f ± %.3f | concordância = %.1f%%" % (
        *report["f1_tflite_fp32"], *report["f1_tflite_int8"], 100 * report["int8_agrees_fp32"][0]))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(channels)
    if args.no_export:
        (out / f"cv_{tag}.json").write_text(json.dumps(report, indent=2))
        return

    # Modelo final com todos os dados.
    rng = np.random.default_rng(SEED)
    scale = X.reshape(-1, X.shape[2]).std(0)
    Xn = normalize(X, scale)
    Xm, ym = mirror(Xn, y, channels)
    model = fit(build_model(len(channels)), Xm, ym, rng)
    fp32, int8 = to_tflite(model), to_tflite(model, Xn)
    _, lat_fp32 = tflite_predict(fp32, Xn[:200])
    _, lat_int8 = tflite_predict(int8, Xn[:200])

    (out / "model_fp32.tflite").write_bytes(fp32)
    (out / "model_int8.tflite").write_bytes(int8)
    model.save(out / "model.keras")

    interp = tf.lite.Interpreter(model_content=int8); interp.allocate_tensors()
    in_q = interp.get_input_details()[0]["quantization"]
    out_q = interp.get_output_details()[0]["quantization"]

    # Memória ocupada só pelos pesos: FP32 usa 4 bytes por parâmetro; no INT8,
    # os kernels usam 1 byte e os biases continuam em int32 (4 bytes).
    n_bias = sum(int(w.shape[0]) for w in model.weights if "bias" in w.path)
    n_params = int(model.count_params())
    report.update({
        "params": n_params,
        "weights_bytes": {"fp32": n_params * 4, "int8": (n_params - n_bias) + n_bias * 4},
        "size_bytes": {"tflite_fp32": len(fp32), "tflite_int8": len(int8)},
        "host_latency_us_median": {"tflite_fp32": lat_fp32, "tflite_int8": lat_int8},
        "scale": scale.tolist(), "input_quant": list(in_q), "output_quant": list(out_q),
    })
    (out / "report.json").write_text(json.dumps(report, indent=2))

    fw = Path(args.firmware); fw.mkdir(parents=True, exist_ok=True)
    (fw / "model_data.h").write_text(to_c_array(int8, "g_model"))
    cfg = [ "// Gerado por src/train.py. Não edite à mão.", "#pragma once", "",
            f"constexpr int kWindowSize = {WINDOW_SIZE};",
            f"constexpr int kNumChannels = {len(channels)};",
            f"constexpr int kNumClasses = {len(CLASSES)};",
            "// Ordem dos canais: " + ", ".join(channels),
            "constexpr float kChannelScale[kNumChannels] = {" + ", ".join(f"{v:.9g}f" for v in scale) + "};",
            f"constexpr float kInputScale = {in_q[0]:.9g}f;",
            f"constexpr int kInputZeroPoint = {int(in_q[1])};",
            f"constexpr float kOutputScale = {out_q[0]:.9g}f;",
            f"constexpr int kOutputZeroPoint = {int(out_q[1])};",
            "const char* const kClassNames[kNumClasses] = {" + ", ".join(f'"{c}"' for c in CLASSES) + "};", ""]
    (fw / "model_config.h").write_text("\n".join(cfg))

    print(f"Parâmetros: {report['params']} | FP32: {len(fp32)} B | INT8: {len(int8)} B "
          f"({len(fp32) / len(int8):.1f}x menor)")
    print(f"Latência no PC (mediana): FP32 {lat_fp32:.0f} µs | INT8 {lat_int8:.0f} µs")


if __name__ == "__main__":
    main()
