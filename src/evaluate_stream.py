"""Avaliação realista: deixa uma viagem inteira de fora e roda o modelo nela em fluxo contínuo.

Para cada viagem do dataset:
  1. treina o modelo só com as OUTRAS viagens e quantiza para INT8;
  2. roda o modelo na viagem deixada de fora, como no ESP32 (janela de 2 s a cada 0,5 s,
     com o mesmo filtro de eventos);
  3. compara os eventos emitidos com os rótulos do dataset.

Métricas:
  - detecção: % das manobras agressivas rotuladas com ao menos um evento emitido durante elas;
  - classe correta: % das manobras detectadas em que algum evento tinha a classe certa;
  - alarmes falsos por minuto: eventos emitidos fora de qualquer manobra agressiva rotulada.
    É uma medida conservadora: trechos sem rótulo podem conter manobras reais não anotadas.

Isso é possível porque cada classe aparece em pelo menos duas viagens.

Uso:
    python src/evaluate_stream.py --channels a_h,g_z,g_h
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

from features import CHANNEL_NAMES, LABEL_MAP, normalize
from prepare_data import load_trip
from stream import run_stream
from train import SEED, build_model, fit, mirror, select_channels, to_tflite

MATCH_BEFORE, MATCH_AFTER = 0.5, 2.0   # s: tolerância em torno de cada manobra rotulada


ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=str(ROOT / "artifacts/windows.npz"))
    ap.add_argument("--data", default=str(ROOT / "data/driverBehaviorDataset/data"))
    ap.add_argument("--channels", default="a_h,g_z,g_h")
    ap.add_argument("--out", default=str(ROOT / "artifacts/stream_eval.json"))
    ap.add_argument("--thresholds", default="0.7",
                    help="limiares de confiança separados por vírgula; o primeiro é o usado no firmware")
    args = ap.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]

    channels = args.channels.split(",")
    idx = [CHANNEL_NAMES.index(c) for c in channels]
    d = np.load(args.windows)
    X, y, trips = select_channels(d["X"], channels), d["y"], d["trips"]

    models = {}
    for trip in sorted(set(trips)):
        tr = trips != trip
        rng = np.random.default_rng(SEED)
        scale = X[tr].reshape(-1, X.shape[2]).std(0)
        Xtr = normalize(X[tr], scale)
        Xm, ym = mirror(Xtr, y[tr], channels)
        model = fit(build_model(len(channels)), Xm, ym, rng)
        models[trip] = (to_tflite(model, Xtr), scale)

    trip_data = {trip: load_trip(Path(args.data) / trip) for trip in models}
    report = {"channels": channels, "limiar_firmware": thresholds[0], "por_limiar": {}}
    for thr in thresholds:
        print(f"--- limiar de confiança {thr:.2f}")
        results = []
        totals = {"rotuladas": 0, "detectadas": 0, "classe_certa": 0, "falsos": 0, "minutos": 0.0}
        for trip, (int8, scale) in models.items():
            grid, feats, gt = trip_data[trip]
            events = run_stream(int8, feats[:, idx] / scale, grid, min_conf=thr)
            row = score_trip(trip, events, grid, gt)
            results.append(row)
            totals["rotuladas"] += row["manobras_rotuladas"]
            totals["detectadas"] += row["detectadas"]
            totals["classe_certa"] += row["classe_certa"]
            totals["falsos"] += row["alarmes_falsos"]
            totals["minutos"] += row["minutos"]
            print(f"  viagem {trip}: {row['detectadas']}/{row['manobras_rotuladas']} manobras detectadas, "
                  f"{row['classe_certa']} com a classe certa, {row['alarmes_falsos']} alarmes falsos "
                  f"em {row['minutos']:.1f} min ({row['alarmes_falsos_por_min']:.2f}/min)")
        summary = {
            "deteccao": totals["detectadas"] / totals["rotuladas"],
            "classe_certa_entre_detectadas": totals["classe_certa"] / max(totals["detectadas"], 1),
            "alarmes_falsos_por_min": totals["falsos"] / totals["minutos"],
        }
        print("  TOTAL: detecção {deteccao:.0%} | classe certa {classe_certa_entre_detectadas:.0%} | "
              "alarmes falsos {alarmes_falsos_por_min:.2f}/min".format(**summary))
        report["por_limiar"][f"{thr:.2f}"] = {"por_viagem": results, "total": summary}
    Path(args.out).write_text(json.dumps(report, indent=2))


def score_trip(trip, events, grid, gt):
    """Compara os eventos emitidos com as manobras agressivas rotuladas de uma viagem."""
    aggressive = gt[gt["evento"] != "evento_nao_agressivo"]
    spans = [(r["inicio"] - MATCH_BEFORE, r["fim"] + MATCH_AFTER, LABEL_MAP[r["evento"]])
             for _, r in aggressive.iterrows()]
    detected = right = 0
    for a, b, label in spans:
        inside = [c for t, c, _ in events if a <= t <= b]
        detected += bool(inside)
        right += label in inside
    false_alarms = sum(1 for t, _, _ in events if not any(a <= t <= b for a, b, _ in spans))
    minutes = float((grid[-1] - grid[0]) / 60)
    return {"viagem": str(trip), "minutos": round(minutes, 2), "manobras_rotuladas": len(spans),
            "detectadas": detected, "classe_certa": right, "eventos_emitidos": len(events),
            "alarmes_falsos": false_alarms, "alarmes_falsos_por_min": round(false_alarms / minutes, 2)}


if __name__ == "__main__":
    main()
