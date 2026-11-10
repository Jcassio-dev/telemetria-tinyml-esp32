#!/usr/bin/env bash
# Compila o TensorFlow Lite Micro (da biblioteca Arduino tflm_esp32) e o código
# do firmware para o PC, gerando tests/host/host_test.
#
# Uso: TFLM_SRC=/caminho/para/tflm_esp32/src tests/host/build.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="${TFLM_SRC:?defina TFLM_SRC com a pasta src da biblioteca tflm_esp32}"
OBJ="$HERE/obj"
mkdir -p "$OBJ"
FLAGS="-std=c++17 -O2 -w -DTF_LITE_STATIC_MEMORY -I$SRC -I$SRC/third_party/flatbuffers/include -I$SRC/third_party/gemmlowp -I$SRC/third_party/ruy"

if [ ! -f "$OBJ/libtflm_host.a" ]; then
  echo "Compilando TFLM para o PC (só na primeira vez)..."
  find "$SRC/tensorflow" \( -name "*.cpp" -o -name "*.cc" \) | grep -vE "_test\.|test_helper|/testing/|benchmark" |
  while read -r f; do
    g++ $FLAGS -c "$f" -o "$OBJ/$(echo "$f" | md5sum | cut -c1-12).o"
  done
  ar rcs "$OBJ/libtflm_host.a" "$OBJ"/*.o
fi

g++ $FLAGS -I"$ROOT/firmware/telemetria_tinyml" "$HERE/host_test.cpp" "$OBJ/libtflm_host.a" -o "$HERE/host_test"
echo "OK: $HERE/host_test"
