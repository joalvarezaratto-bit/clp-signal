"""
¿Cerrar ANTES (intradía) cuando el flujo se da vuelta fuerte MEJORA la estrategia?

Modela el stop como orden INTRADÍA (se dispara con el máx/mín del día, no con el
cierre) y prueba distintos niveles. Un stop más ajustado = cierra antes en
movimientos grandes. También prueba FLIP (dar vuelta la posición tras un stop).
Todo en AMBOS períodos (2009-2021 holdout + 2021-2026).
"""
import datetime as dt
import numpy as np
import requests
import estrategia as E
import test_viejo as tv


def _fetch(sym):
    p1 = int(dt.datetime(2009, 1, 1).timestamp()); p2 = int(dt.datetime(2026, 8, 6).timestamp())
    for h in ("query1", "query2"):
        try:
            r = requests.get(f"https://{h}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "period1": p1, "period2": p2},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            return {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"):
                    {"c": q["close"][i], "h": q["high"][i] or q["close"][i], "l": q["low"][i] or q["close"][i]}
                    for i in range(len(ts)) if q["close"][i] is not None}
        except Exception:
            continue
    return {}


def simular(sig, clp, hi, lo, atr, L, flip, corte, ini=65, cost=4.4):
    """Simula con stop INTRADÍA a L×ATR. flip=True: tras el stop, da vuelta la
    posición. Devuelve curvas separadas viejo/reciente."""
    n = len(clp)
    eq = 1.0; curva = [1.0]
    pos = 0.0; stop = 0.0; bloq = 0.0
    for i in range(ini + 1, n):
        r = 0.0; c = 0.0
        if pos != 0:
            # ¿tocó el stop intradía?
            if (pos > 0 and lo[i] <= stop) or (pos < 0 and hi[i] >= stop):
                r = pos * (stop / clp[i - 1] - 1)   # salida al stop
                c += cost / 10000
                bloq = np.sign(pos); pos = 0.0
            else:
                r = pos * (clp[i] / clp[i - 1] - 1)
        eq *= (1 + r - c)
        # decisión al cierre
        s = np.sign(sig[i])
        if bloq != 0 and s != bloq:
            bloq = 0.0
        target = s if bloq == 0 else 0.0
        if flip and bloq != 0 and pos == 0:
            target = -bloq; bloq = 0.0   # dar vuelta tras el stop
        if target != np.sign(pos):
            eq -= eq * (cost / 10000 * abs(target - np.sign(pos)))
            pos = target
            if pos != 0:
                stop = clp[i] - np.sign(pos) * L * atr[i]
        curva.append(eq)
    c = np.array(curva)
    j = corte - ini
    def met(x):
        rr = np.diff(np.log(x)); return rr.mean() / (rr.std() + 1e-9) * np.sqrt(252)
    return met(c[:j]), met(c[j:])


def correr():
    m = {k: _fetch(s) for k, s in E.SYMS.items()}
    com = sorted(set.intersection(*[set(d) for d in m.values()]))
    arr = {k: tv._limpiar(np.array([m[k][d]["c"] for d in com])) for k in m}
    clp = arr["clp"]; n = len(clp)
    hi = np.array([m["clp"][d]["h"] for d in com]); lo = np.array([m["clp"][d]["l"] for d in com])
    rng = np.minimum(hi - lo, clp * 0.08); hi = clp + rng / 2; lo = clp - rng / 2
    tr = hi - lo; atr = np.copy(tr)
    for i in range(1, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    sig = E._combo_dxy(clp, arr, 2, 0.01)
    corte = next(i for i, d in enumerate(com) if d >= "2021-08-01")
    print(f"{n} días. Señal 2d. Sharpe por período:\n")
    print(f"{'Configuración':34} {'VIEJO':>8} {'RECIENTE':>10}")
    print("-" * 56)
    for L, flip, nom in [(3.0, False, "Stop 3×ATR (actual, amplio)"),
                         (2.0, False, "Stop 2×ATR intradía"),
                         (1.5, False, "Stop 1.5×ATR intradía"),
                         (1.0, False, "Stop 1×ATR intradía (ajustado)"),
                         (1.5, True, "Stop 1.5×ATR + FLIP (da vuelta)"),
                         (1.0, True, "Stop 1×ATR + FLIP")]:
        sv, sr = simular(sig, clp, hi, lo, atr, L, flip, corte)
        print(f"{nom:34} {sv:>+8.2f} {sr:>+10.2f}")
    print("\n(mayor Sharpe en AMBOS = mejor. Ojo: stop muy ajustado suele dar whipsaw)")


if __name__ == "__main__":
    correr()
