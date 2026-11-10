"""Gera janelas rotuladas a partir do Driver Behavior Dataset.

Passos:
1. Lê aceleração linear e giroscópio de cada viagem (referencial da Terra).
2. Reamostra os dois sensores numa grade comum de 50 Hz.
3. Calcula os 4 atributos invariantes ao rumo (src/features.py).
4. Recorta janelas de 2 s:
   - dentro de cada evento rotulado, com passo de 0,25 s;
   - fora dos eventos (margem de 1 s), rotuladas como "normal".
5. Salva X, y e o identificador do evento (para validação agrupada).

Uso:
    python src/prepare_data.py --data data/driverBehaviorDataset/data --out artifacts/windows.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import (CLASSES, LABEL_MAP, SAMPLE_RATE_HZ, WINDOW_SIZE,
                      compute_features)

ROOT = Path(__file__).resolve().parents[1]

STRIDE_IN_EVENT = 0.25          # s, passo das janelas dentro de um evento
BACKGROUND_STRIDE = 2.0         # s, passo das janelas "normal" fora de eventos
BACKGROUND_MARGIN = 1.0         # s, distância mínima de qualquer evento
RNG = np.random.default_rng(42)


def load_trip(trip_dir: Path):
    meta = json.loads((trip_dir / "viagem.json").read_text())
    t0 = meta["firstCollectionUptimeNanos"]

    def read(name):
        df = pd.read_csv(trip_dir / f"{name}.csv")
        t = (df["uptimeNanos"].to_numpy() - t0) / 1e9
        return t, df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    t_acc, acc = read("aceleracaoLinear_terra")
    t_gyr, gyr = read("giroscopio_terra")

    start = max(t_acc[0], t_gyr[0])
    end = min(t_acc[-1], t_gyr[-1])
    grid = np.arange(start, end, 1.0 / SAMPLE_RATE_HZ)
    acc_r = np.stack([np.interp(grid, t_acc, acc[:, i]) for i in range(3)], axis=1)
    gyr_r = np.stack([np.interp(grid, t_gyr, gyr[:, i]) for i in range(3)], axis=1)

    gt = pd.read_csv(trip_dir / "groundTruth.csv", skipinitialspace=True)
    gt.columns = [c.strip() for c in gt.columns]
    return grid, compute_features(acc_r, gyr_r), gt


def window_at(grid, feats, center):
    """Janela de 2 s centrada em `center` (segundos), ou None se sair da viagem."""
    i = int(round((center - grid[0]) * SAMPLE_RATE_HZ)) - WINDOW_SIZE // 2
    if i < 0 or i + WINDOW_SIZE > len(feats):
        return None
    return feats[i:i + WINDOW_SIZE]


def build(data_dir: Path):
    X, y, groups, trips, is_bg = [], [], [], [], []
    event_id = 0
    for trip_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        grid, feats, gt = load_trip(trip_dir)

        for _, ev in gt.iterrows():
            label = CLASSES.index(LABEL_MAP[ev["evento"]])
            mid = (ev["inicio"] + ev["fim"]) / 2
            # Centros espalhados pelo evento; eventos curtos geram ao menos 1 janela.
            first, last = ev["inicio"] + 0.5, ev["fim"] - 0.5
            centers = np.arange(first, last + 1e-9, STRIDE_IN_EVENT) if last > first else [mid]
            for c in centers:
                w = window_at(grid, feats, c)
                if w is not None:
                    X.append(w); y.append(label); groups.append(event_id)
                    trips.append(trip_dir.name); is_bg.append(False)
            event_id += 1

        # Janelas "normal" longe de qualquer evento rotulado.
        busy = [(r["inicio"] - BACKGROUND_MARGIN, r["fim"] + BACKGROUND_MARGIN) for _, r in gt.iterrows()]
        half = WINDOW_SIZE / SAMPLE_RATE_HZ / 2
        for c in np.arange(grid[0] + half, grid[-1] - half, BACKGROUND_STRIDE):
            if any(a - half < c < b + half for a, b in busy):
                continue
            w = window_at(grid, feats, c)
            if w is not None:
                # Cada janela de fundo é seu próprio grupo (não há vazamento entre elas).
                X.append(w); y.append(CLASSES.index("normal")); groups.append(event_id)
                trips.append(trip_dir.name); is_bg.append(True)
                event_id += 1

    return (np.stack(X).astype(np.float32), np.array(y), np.array(groups),
            np.array(trips), np.array(is_bg))


def balance_background(X, y, groups, trips, is_bg, max_ratio=1.0):
    """Limita as janelas de fundo para a classe "normal" não dominar o treino.

    As janelas de eventos "não agressivos" rotulados no dataset são mantidas;
    só o fundo (trechos sem rótulo) é amostrado.
    """
    counts = np.bincount(y[~is_bg], minlength=len(CLASSES))
    cap = max(int(max(counts[1:]) * max_ratio) - int(counts[0]), 0)
    bg = np.where(is_bg)[0]
    keep_bg = RNG.choice(bg, size=min(cap, len(bg)), replace=False)
    keep = np.sort(np.concatenate([np.where(~is_bg)[0], keep_bg]))
    return X[keep], y[keep], groups[keep], trips[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/driverBehaviorDataset/data"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/windows.npz"))
    args = ap.parse_args()

    X, y, groups, trips, is_bg = build(Path(args.data))
    X, y, groups, trips = balance_background(X, y, groups, trips, is_bg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, groups=groups, trips=trips)

    print(f"Janelas: {X.shape} (amostras, tempo, canais)")
    for i, name in enumerate(CLASSES):
        n_win = int((y == i).sum())
        n_ev = len(set(groups[y == i]))
        print(f"  {name:15s} {n_win:5d} janelas de {n_ev:4d} eventos/trechos")


if __name__ == "__main__":
    main()
