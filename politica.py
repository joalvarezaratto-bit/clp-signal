"""
¿Cómo afectan las ELECCIONES presidenciales chilenas al USD/CLP?

Chile elige presidente cada 4 años (2ª vuelta en diciembre, asume en marzo).
Analizamos con datos cómo se movió el dólar alrededor de cada elección, y si el
signo del ganador (izquierda / derecha) importó.

Eventos (2ª vuelta) y orientación del ganador:
  2013-12-15  Bachelet   izquierda
  2017-12-17  Piñera     derecha
  2021-12-19  Boric      izquierda
  2025-12-14  (2025)     — (asume mar-2026)
"""
import datetime as dt
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}

ELECCIONES = [
    ("2013-12-15", "Bachelet", "izquierda"),
    ("2017-12-17", "Piñera", "derecha"),
    ("2021-12-19", "Boric", "izquierda"),
    ("2025-12-14", "2025", "?"),
]
# hitos políticos NO electorales que también sacudieron al peso
OTROS = [
    ("2019-10-18", "Estallido social"),
    ("2022-09-04", "Rechazo 1ª constitución"),
]


def cargar():
    p1 = int(dt.datetime(2012, 1, 1).timestamp()); p2 = int(dt.datetime(2026, 8, 5).timestamp())
    for h in ("query1", "query2"):
        try:
            r = requests.get(f"https://{h}.finance.yahoo.com/v8/finance/chart/USDCLP=X",
                             params={"interval": "1d", "period1": p1, "period2": p2}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, c = res["timestamp"], res["indicators"]["quote"][0]["close"]
            d = {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"): c[i]
                 for i in range(len(ts)) if c[i] is not None}
            return d
        except Exception:
            continue
    return {}


def _cerca(serie, fechas, objetivo, offset_dias):
    """Precio ~offset_dias hábiles desde 'objetivo' (índice en la lista de fechas)."""
    o = dt.datetime.strptime(objetivo, "%Y-%m-%d")
    idx = min(range(len(fechas)), key=lambda i: abs((dt.datetime.strptime(fechas[i], "%Y-%m-%d") - o).days))
    j = max(0, min(len(fechas) - 1, idx + offset_dias))
    return serie[fechas[j]], fechas[j]


def correr():
    d = cargar()
    fechas = sorted(d)
    print(f"Datos USD/CLP: {fechas[0]} → {fechas[-1]}\n")

    print("== Movimiento del dólar alrededor de cada ELECCIÓN (2ª vuelta) ==")
    print(f"{'Elección':24} {'-3 meses':>10} {'-1 mes':>8} {'+1 mes':>8} {'+3 meses':>9} {'+6 meses':>9}")
    print("-" * 76)
    for fecha, quien, signo in ELECCIONES:
        if fecha > fechas[-1]:
            continue
        p0, _ = _cerca(d, fechas, fecha, 0)
        def mv(off):
            p, _ = _cerca(d, fechas, fecha, off)
            return (p / p0 - 1) * 100
        et = f"{fecha[:7]} {quien} ({signo})"
        print(f"{et:24} {mv(-63):>+9.1f}% {mv(-21):>+7.1f}% {mv(21):>+7.1f}% {mv(63):>+8.1f}% {mv(126):>+8.1f}%")
    print("  (% del dólar respecto al día de la elección; + = dólar sube = peso se debilita)\n")

    # promedio por orientación del ganador
    print("== ¿Importa quién gana? (promedio +3 meses tras la elección) ==")
    for signo in ("izquierda", "derecha"):
        movs = []
        for fecha, quien, s in ELECCIONES:
            if s == signo and fecha < fechas[-1]:
                p0, _ = _cerca(d, fechas, fecha, 0)
                p3, _ = _cerca(d, fechas, fecha, 63)
                movs.append((p3 / p0 - 1) * 100)
        if movs:
            print(f"  Ganó {signo:10}: dólar {np.mean(movs):+.1f}% en promedio (n={len(movs)})")
    print()

    # otros shocks políticos
    print("== Otros shocks políticos (no electorales) ==")
    for fecha, nombre in OTROS:
        if fecha > fechas[-1]:
            continue
        p0, _ = _cerca(d, fechas, fecha, 0)
        p1m, _ = _cerca(d, fechas, fecha, 21)
        p3m, _ = _cerca(d, fechas, fecha, 63)
        print(f"  {fecha} {nombre:24}: +1 mes {(p1m/p0-1)*100:+.1f}%  ·  +3 meses {(p3m/p0-1)*100:+.1f}%")
    return d, fechas


if __name__ == "__main__":
    correr()
