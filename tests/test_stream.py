"""Testes do filtro de eventos (mesma lógica do firmware)."""
from features import CLASSES
from stream import EventFilter

CURVA = CLASSES.index("curva_esquerda")
LONG = CLASSES.index("longitudinal")


def feed(filt, seq, step_ms=500):
    return [filt.update(c, conf, i * step_ms) for i, (c, conf) in enumerate(seq)]


def test_previsao_isolada_nao_gera_evento():
    out = feed(EventFilter(), [(CURVA, 0.99), (0, 0.99), (0, 0.99)])
    assert all(o == -1 for o in out)


def test_uma_manobra_longa_gera_um_evento_so():
    out = feed(EventFilter(), [(CURVA, 0.99)] * 8)
    assert [o for o in out if o >= 0] == [CURVA]


def test_confianca_baixa_e_ignorada():
    out = feed(EventFilter(min_conf=0.85), [(CURVA, 0.6)] * 6)
    assert all(o == -1 for o in out)


def test_duas_manobras_separadas_geram_dois_eventos():
    seq = [(CURVA, 0.99)] * 3 + [(0, 0.99)] * 3 + [(CURVA, 0.99)] * 3
    out = feed(EventFilter(), seq)
    assert [o for o in out if o >= 0] == [CURVA, CURVA]


def test_troca_de_classe_gera_novo_evento():
    seq = [(LONG, 0.99)] * 3 + [(CURVA, 0.99)] * 3
    out = feed(EventFilter(), seq)
    assert [o for o in out if o >= 0] == [LONG, CURVA]
