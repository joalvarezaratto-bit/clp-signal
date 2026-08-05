"""
¿Qué TEMPORALIDAD de señal funciona mejor? Barrido del lookback del momentum
del cobre (1 día a 2 meses), medido en AMBOS períodos para robustez.

También prueba la frecuencia de rebalanceo (cada cuántos días refresca).
"""
import datetime as dt
import numpy as np
import requests
import lab
import oos_test as oos
import test_viejo as tv

UA = {"User-Agent": "Mozilla/5.0"}
SYMS = {"clp": "USDCLP=X", "cobre": "HG=F", "dxy": "DX-Y.NYB", "brl": "USDBRL=X"}


def _fetch(sym):
    p1 = int(dt.datetime(2009, 1, 1).timestamp()); p2 = int(dt.datetime(2026, 8, 6).timestamp())
    for h in ("query1", "query2"):
        try:
            r = requests.get(f"https://{h}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "period1": p1, "period2": p2}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, c = res["timestamp"], res["indicators"]["quote"][0]["close"]
            return {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"): c[i]
                    for i in range(len(ts)) if c[i] is not None}
        except Exception:
            continue
    return {}


def correr():
    m = {k: _fetch(s) for k, s in SYMS.items()}
    com = sorted(set.intersection(*[set(d) for d in m.values()]))
    arr = {k: tv._limpiar(np.array([m[k][d] for d in com])) for k in m}
    clp = arr["clp"]; n = len(clp)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    # split entre viejo (2009-2021) y reciente (2021-2026)
    corte = next(i for i, d in enumerate(com) if d >= "2021-08-01")
    print(f"{n} días. Viejo: {com[65]}→{com[corte]} | Reciente: {com[corte]}→{com[-1]}\n")

    def sh(pos, a, b):
        _, _, s, _ = lab.metricas(lab.simular(pos, clp, a, b))
        return s

    print("== Barrido del lookback del momentum del cobre ==")
    print(f"{'Lookback señal':16} {'Sharpe VIEJO':>13} {'Sharpe RECIENTE':>16}  {'promedio':>9}")
    print("-" * 60)
    mejor = None
    for w in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        pos = oos.est_combo_dxy(clp, arr, w, 0.01)
        s_v = sh(pos, 65, corte)
        s_r = sh(pos, corte, n - 1)
        prom = (s_v + s_r) / 2
        etq = f"{w} día{'s' if w > 1 else ''}"
        marca = "  ← mejor" if (mejor is None or prom > mejor[1]) else ""
        if mejor is None or prom > mejor[1]:
            mejor = (w, prom)
        print(f"{etq:16} {s_v:>+13.2f} {s_r:>+16.2f}  {prom:>+8.2f}{marca}")
    print(f"\n  Mejor temporalidad (robusta en ambos): {mejor[0]} días\n")

    # frecuencia de rebalanceo (mantener la señal N días)
    print("== Frecuencia de rebalanceo (con lookback óptimo) ==")
    base = oos.est_combo_dxy(clp, arr, mejor[0], 0.01)
    for cada in [1, 2, 3, 5, 10]:
        pos = base.copy()
        for i in range(1, n):
            if i % cada != 0:
                pos[i] = pos[i - 1]   # solo actualiza cada 'cada' días
        s_v = sh(pos, 65, corte); s_r = sh(pos, corte, n - 1)
        print(f"  rebalanceo cada {cada:>2} día(s): viejo {s_v:+.2f}  reciente {s_r:+.2f}")


if __name__ == "__main__":
    correr()
