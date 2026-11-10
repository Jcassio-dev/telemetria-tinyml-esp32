"""Gateway: recebe os eventos JSON do ESP32 pela serial e os entrega ao Grafana.

O ESP32 já faz a inferência; o gateway só valida, registra em CSV e, se
configurado, grava no InfluxDB (que o Grafana lê).

Uso:
    python gateway/gateway.py --port /dev/ttyUSB0          # Linux
    python gateway/gateway.py --port COM3                  # Windows
    python gateway/simulate.py | python gateway/gateway.py --stdin   # sem hardware

InfluxDB (opcional, ver docker-compose.yml):
    --influx-url http://localhost:8086 --influx-token TOKEN --influx-org tinyml --influx-bucket telemetria
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

FIELDS = ["recebido_em", "tipo", "t_ms", "evento", "confianca", "latencia_us", "temp_c", "umidade", "presenca"]


def parse_line(line: str):
    """Converte uma linha da serial num dicionário; ignora o que não for JSON válido."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    return msg if isinstance(msg, dict) and "tipo" in msg else None


def to_line_protocol(msg: dict, now_ns: int) -> str:
    """Formato de linha do InfluxDB: medida,tags campos timestamp."""
    measurement = "evento" if msg["tipo"] == "evento" else "telemetria"
    tags = f",evento={msg['evento']}" if msg.get("evento") else ""
    fields = []
    for key in ("confianca", "latencia_us", "temp_c", "umidade"):
        if msg.get(key) is not None:
            fields.append(f"{key}={float(msg[key])}")
    if "presenca" in msg:
        fields.append(f"presenca={'true' if msg['presenca'] else 'false'}")
    if measurement == "evento":
        fields.append("contagem=1i")
    return f"{measurement}{tags} {','.join(fields)} {now_ns}" if fields else ""


class InfluxWriter:
    def __init__(self, url, token, org, bucket):
        self.endpoint = f"{url.rstrip('/')}/api/v2/write?org={org}&bucket={bucket}&precision=ns"
        self.headers = {"Authorization": f"Token {token}", "Content-Type": "text/plain"}

    def write(self, line: str):
        req = urllib.request.Request(self.endpoint, data=line.encode(), headers=self.headers, method="POST")
        try:
            urllib.request.urlopen(req, timeout=3)
        except OSError as e:  # não derruba o gateway se o banco cair
            print(f"[influx] falha ao gravar: {e}", file=sys.stderr)


def lines_from(args):
    if args.stdin:
        yield from sys.stdin
        return
    import serial  # pyserial
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        while True:
            raw = ser.readline()
            if raw:
                yield raw.decode(errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--stdin", action="store_true", help="ler linhas da entrada padrão (simulação)")
    ap.add_argument("--csv", default="eventos.csv")
    ap.add_argument("--influx-url"); ap.add_argument("--influx-token")
    ap.add_argument("--influx-org", default="tinyml"); ap.add_argument("--influx-bucket", default="telemetria")
    args = ap.parse_args()

    influx = InfluxWriter(args.influx_url, args.influx_token, args.influx_org, args.influx_bucket) \
        if args.influx_url and args.influx_token else None

    path = Path(args.csv)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for line in lines_from(args):
            msg = parse_line(line)
            if msg is None:
                continue
            if msg["tipo"] in ("info", "erro"):
                print(f"[{msg['tipo']}] {msg}")
                continue
            writer.writerow({"recebido_em": time.strftime("%Y-%m-%d %H:%M:%S"), **msg})
            f.flush()
            if msg["tipo"] == "evento":
                lat = msg.get("latencia_us")
                print(f"EVENTO {msg['evento']:<24} confiança {msg.get('confianca') or 0:.2f} "
                      f"| inferência {'-' if lat is None else lat} µs")
            if influx:
                lp = to_line_protocol(msg, time.time_ns())
                if lp:
                    influx.write(lp)


if __name__ == "__main__":
    main()
