"""Simula o ESP32 reproduzindo uma viagem do dataset, sem hardware.

Usa o mesmo modelo INT8, a mesma janela (2 s), o mesmo passo (0,5 s) e o mesmo
filtro de eventos do firmware, e imprime o mesmo JSON que o ESP32 manda pela serial.
Temperatura, umidade e presença saem como null/false, porque o dataset não tem esses sensores.

Atenção: o modelo final foi treinado com todas as viagens, então esta simulação
serve para demonstrar o fluxo de ponta a ponta, não para medir desempenho.
Para medir, use src/evaluate_stream.py (treina sem a viagem avaliada).

Uso:
    python gateway/simulate.py --trip 21 | python gateway/gateway.py --stdin
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import CHANNEL_NAMES  # noqa: E402
from prepare_data import load_trip  # noqa: E402
from stream import run_stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="21")
    ap.add_argument("--data", default=str(ROOT / "data/driverBehaviorDataset/data"))
    ap.add_argument("--model", default=str(ROOT / "artifacts/model_int8.tflite"))
    ap.add_argument("--report", default=str(ROOT / "artifacts/report.json"))
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    idx = [CHANNEL_NAMES.index(c) for c in report["channels"]]
    scale = np.array(report["scale"], dtype=np.float32)

    grid, feats, _ = load_trip(Path(args.data) / args.trip)
    events = run_stream(Path(args.model).read_bytes(), feats[:, idx] / scale, grid)
    for t, name, conf in events:
        print(json.dumps({"tipo": "evento", "t_ms": int(t * 1000), "evento": name,
                          "confianca": round(conf, 2), "latencia_us": None,
                          "temp_c": None, "umidade": None, "presenca": False}), flush=True)


if __name__ == "__main__":
    main()
