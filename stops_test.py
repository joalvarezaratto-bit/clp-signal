"""
¿Agregar STOP-LOSS y TAKE-PROFIT mejora la estrategia? (probado OOS)

La versión base cierra por señal. Acá agregamos un stop (X×ATR en contra) y un
take-profit (Y×ATR a favor) y vemos si sube el Sharpe o si empeora (whipsaw).
"""
import datetime as dt
import numpy as np
import requests
import lab
import oos_test as oos
import rentabilidad as R


def _atr(com, clp, nn=14):
    """ATR aproximado en pesos, desde el rango diario de Yahoo."""
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/USDCLP=X",
                             params={"interval": "1d", "range": "5y"}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            rng = {}
            for i in range(len(ts)):
                if q["close"][i] is None:
                    continue
                d = dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d")
                rng[d] = (q["high"][i] or q["close"][i]) - (q["low"][i] or q["close"][i])
            tr = np.array([rng.get(d, clp[i] * 0.008) for i, d in enumerate(com)])
            atr = np.copy(tr)
            for i in range(1, len(tr)):
                atr[i] = (atr[i - 1] * (nn - 1) + tr[i]) / nn
            return atr
        except Exception:
            continue
    return clp * 0.008


UA = {"User-Agent": "Mozilla/5.0"}


def con_stops(base, clp, atr, stop_atr=None, tp_atr=None):
    """Aplica stop/TP sobre la posición base. Tras saltar, queda fuera hasta que
    la señal base cambie de dirección."""
    n = len(clp); out = np.zeros(n)
    cur, entry, fuera = 0, None, False
    for i in range(n):
        s = base[i]
        if np.sign(s) != np.sign(cur):     # la señal cambió de lado -> nuevo trade
            cur = s; entry = clp[i] if s != 0 else None; fuera = False
        if cur != 0 and entry and not fuera:
            mv = (clp[i] - entry) / entry
            umbral = atr[i] / entry
            if stop_atr and ((cur > 0 and mv <= -stop_atr * umbral) or
                             (cur < 0 and mv >= stop_atr * umbral)):
                fuera = True
            elif tp_atr and ((cur > 0 and mv >= tp_atr * umbral) or
                             (cur < 0 and mv <= -tp_atr * umbral)):
                fuera = True
        out[i] = 0 if fuera else cur
    return out


def correr():
    com, arr = lab.cargar()
    clp = arr["clp"]; n = len(clp)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    base = R.posicion_final(com, arr, clp)   # combo v2 + actividad
    atr = _atr(com, clp)
    ini = 65; split = ini + int((n - ini) * 0.6)

    def ev(pos):
        _, _, sh1, _ = lab.metricas(lab.simular(pos, clp, ini, split))
        _, c, sh2, dd = lab.metricas(lab.simular(pos, clp, split, n - 1))
        return sh1, sh2, c, dd

    print(f"{n} días. TEST desde {com[split]}\n")
    print(f"{'Configuración':30} {'Sh.train':>9} {'Sh.TEST':>8} {'CAGR':>7} {'DD':>7}")
    print("-" * 68)
    configs = [("SIN stop (base, por señal)", None, None),
               ("Stop 1.5×ATR", 1.5, None),
               ("Stop 2×ATR", 2.0, None),
               ("Stop 3×ATR", 3.0, None),
               ("Stop 2×ATR + TP 3×ATR", 2.0, 3.0),
               ("Solo TP 2×ATR", None, 2.0)]
    for nom, sl, tp in configs:
        pos = base if (sl is None and tp is None) else con_stops(base, clp, atr, sl, tp)
        sh1, sh2, c, dd = ev(pos)
        print(f"{nom:30} {sh1:>+9.2f} {sh2:>+8.2f} {c:>+6.1f}% {dd:>6.1f}%")
    print("\nSi ninguno supera al 'SIN stop', el stop por señal ya es suficiente.")


if __name__ == "__main__":
    correr()
