#!/usr/bin/env bash
# Baixa o Driver Behavior Dataset (Ferreira Júnior et al., 2017) para data/driverBehaviorDataset.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d driverBehaviorDataset ]; then
  git clone --depth 1 https://github.com/jair-jr/driverBehaviorDataset.git
fi
echo "Dataset em: $(pwd)/driverBehaviorDataset/data"
