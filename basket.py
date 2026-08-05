"""
¿El concepto 'momentum del commodity → su moneda' GENERALIZA?

Cada moneda de commodity debería seguir a su materia prima principal:
  CLP,PEN ← cobre · CAD,NOK,MXN ← petróleo · ZAR ← oro · BRL ← cobre/soja
Probamos la MISMA estrategia (momentum 2d del commodity) en cada una, OOS.
Si varias funcionan, un BASKET diversifica el riesgo de un solo factor.
"""
import datetime as dt
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}

PARES = [
    ("USD/CLP", "USDCLP=X", "HG=F", "cobre"),
    ("USD/PEN", "USDPEN=X", "HG=F", "cobre"),
    ("USD/CAD", "USDCAD=X", "CL=F", "petróleo"),
    ("USD/NOK", "USDNOK=X", "CL=F", "petróleo"),
    ("USD/MXN", "USDMXN=X", "CL=F", "petróleo"),
    ("USD/ZAR", "USDZAR=X", "GC=F", "oro"),
    ("USD/BRL", "USDBRL=X", "HG=F", "cobre"),
]


def _fetch(sym):
    for h in ("query1", "query2"):
        try:
            r = requests.get(f"https://{h}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "range": "5y"}, headers=UA, timeout=25)
            res = r.json()["chart"]["result"][0]
            ts, c = res["timestamp"], res["indicators"]["quote"][0]["close"]
            return {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"): c[i]
                    for i in range(len(ts)) if c[i] is not None}
        except Exception:
            continue
    return {}


def estrategia(fx, com_d):
    """Momentum 2d del commodity → posición en el par USD (commodity sube → moneda
    fuerte → USDXXX baja → corto USD). Devuelve (pnl_series, fechas)."""
    fechas = sorted(set(fx) & set(com_d))
    if len(fechas) < 300:
        return None, None
    px = np.clip(np.array([fx[d] for d in fechas]), 1e-9, None)
    cm = np.array([com_d[d] for d in fechas])
    rc = np.concatenate([[0], np.diff(np.log(cm))])
    rf = np.concatenate([[0], np.diff(np.log(px))])
    pos = np.zeros(len(fechas))
    for i in range(2, len(fechas)):
        mov = rc[i - 1:i + 1].sum()
        pos[i] = -1 if mov > 0.005 else (1 if mov < -0.005 else 0)   # commodity sube -> corto USD
    return pos, rf


def correr():
    coms = {"HG=F": _fetch("HG=F"), "CL=F": _fetch("CL=F"), "GC=F": _fetch("GC=F")}
    print(f"{'Par':10} {'Commodity':11} {'Sh.train':>9} {'Sh.TEST':>9} {'¿sirve?':>10}")
    print("-" * 54)
    pnls = {}
    for nom, fxs, cs, cnom in PARES:
        fx = _fetch(fxs)
        if not fx:
            print(f"{nom:10} {cnom:11}  (sin datos)")
            continue
        pos, rf = estrategia(fx, coms[cs])
        if pos is None:
            print(f"{nom:10} {cnom:11}  (pocos datos)")
            continue
        n = len(pos); split = int(n * 0.6)
        pnl = pos[:-1] * rf[1:] - 4.4 / 10000 * (np.abs(np.diff(np.concatenate([[0], pos]))))[:-1]

        def sh(a, b):
            x = pnl[a:b]; return x.mean() / (x.std() + 1e-9) * np.sqrt(252)
        s_tr, s_te = sh(2, split), sh(split, n - 1)
        v = "SÍ ✓" if (s_te > 0.4 and s_tr > 0.2) else "no"
        print(f"{nom:10} {cnom:11} {s_tr:>+9.2f} {s_te:>+9.2f} {v:>10}")
        pnls[nom] = pnl[split:n - 1]

    # basket: promedio de las que sirven
    if len(pnls) >= 2:
        ln = min(len(v) for v in pnls.values())
        combo = np.mean([v[-ln:] for v in pnls.values()], axis=0)
        sh_b = combo.mean() / (combo.std() + 1e-9) * np.sqrt(252)
        print(f"\n  BASKET (promedio de todas, OOS): Sharpe {sh_b:+.2f}  "
              f"— la diversificación suele subir el Sharpe vs cada una sola")


if __name__ == "__main__":
    correr()
