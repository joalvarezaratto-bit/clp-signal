"""
¿Existe una estrategia MENSUAL que sirva para el USD/CLP?

Marco mensual: ~12 trades/año, costos mínimos. Prueba los factores clásicos de
FX a frecuencia mensual, todo OOS (train mitad vieja, test mitad nueva):
  - Momentum (3, 6, 12 meses)
  - Value / reversión a la media (desvío del promedio de 12m)
  - Momentum del cobre mensual
  - DXY mensual
Historia 2009-2026 (~200 meses).
"""
import datetime as dt
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}
SYMS = {"clp": "USDCLP=X", "cobre": "HG=F", "dxy": "DX-Y.NYB"}


def _fetch(sym):
    p1 = int(dt.datetime(2009, 1, 1).timestamp()); p2 = int(dt.datetime(2026, 8, 6).timestamp())
    for h in ("query1", "query2"):
        try:
            r = requests.get(f"https://{h}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "period1": p1, "period2": p2}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, c = res["timestamp"], res["indicators"]["quote"][0]["close"]
            return {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m"): c[i]
                    for i in range(len(ts)) if c[i] is not None}   # último del mes gana
        except Exception:
            continue
    return {}


def cargar_mensual():
    m = {k: _fetch(s) for k, s in SYMS.items()}
    meses = sorted(set(m["clp"]) & set(m["cobre"]) & set(m["dxy"]))
    return meses, {k: np.array([m[k][mm] for mm in meses]) for k in m}


def simular(pos, px, ini, fin, cost_bps=5):
    r = np.concatenate([[0], np.diff(np.log(px))])
    eq = 1.0; curva = [1.0]; prev = 0.0
    for i in range(ini, fin):
        eq -= eq * (cost_bps / 10000 * abs(pos[i] - prev))
        eq *= (1 + pos[i] * r[i + 1])
        prev = pos[i]; curva.append(eq)
    c = np.array(curva); rr = np.diff(np.log(c))
    sh = rr.mean() / (rr.std() + 1e-9) * np.sqrt(12) if rr.std() > 0 else 0
    cagr = (c[-1] ** (12 / len(c)) - 1) * 100
    return sh, cagr


def correr():
    meses, arr = cargar_mensual()
    clp = arr["clp"]; n = len(clp)
    print(f"{n} meses: {meses[0]} → {meses[-1]}\n")
    split = int(n * 0.55)

    def mom(x, k, signo):
        p = np.zeros(n)
        for i in range(k, n):
            p[i] = signo * np.sign(np.log(x[i] / x[i - k]))
        return p

    def value(x, k=12):
        p = np.zeros(n)
        for i in range(k, n):
            ma = x[i - k:i].mean()
            p[i] = -np.sign(x[i] - ma)   # sobre la media -> corto (fade)
        return p

    ests = {
        "Momentum 3m (USD/CLP)": mom(clp, 3, +1),
        "Momentum 6m": mom(clp, 6, +1),
        "Momentum 12m": mom(clp, 12, +1),
        "Value / reversión 12m": value(clp, 12),
        "Momentum cobre 3m (inverso)": mom(arr["cobre"], 3, -1),
        "Momentum DXY 3m (directo)": mom(arr["dxy"], 3, +1),
        "Comprar y mantener": np.ones(n),
    }

    print(f"{'Estrategia mensual':30} {'Sh.train':>9} {'Sh.TEST':>8} {'CAGR test':>10}")
    print("-" * 62)
    res = []
    for nom, pos in ests.items():
        sh_tr, _ = simular(pos, clp, 12, split)
        sh_te, c_te = simular(pos, clp, split, n - 1)
        res.append((nom, sh_tr, sh_te, c_te))
    for nom, sh_tr, sh_te, c in sorted(res, key=lambda x: -x[2]):
        marca = "✓" if (sh_te > 0.5 and sh_tr > 0.3) else ""
        print(f"{nom:30} {sh_tr:>+9.2f} {sh_te:>+8.2f} {c:>+8.1f}%  {marca}")
    print("\nSharpe test > 0.5 y positivo en train = candidato mensual con edge.")


if __name__ == "__main__":
    correr()
