"""
BACKTEST HONESTO de una estrategia OPERABLE de USD/CLP (con gestión de riesgo).

Estrategia = MEAN-REVERSION del valor justo (la única con edge medido: cuando el
dólar se aparta >1σ de lo que justifican sus motores cobre+DXY+real, apostamos a
que corrige).

  - Entrada SHORT USD/CLP  cuando z ≥ +ENTRY  (dólar caro → apuesta a que baja)
  - Entrada LONG  USD/CLP  cuando z ≤ −ENTRY  (dólar barato → apuesta a que sube)
  - Salida: z vuelve a ~0 (objetivo), o STOP (2×ATR en contra), o máx. días.
  - Tamaño por RIESGO: se arriesga 1% del capital por trade (según distancia al stop).
  - Costos: spread del USD/CLP (exótico) por cada lado.

Sin lookahead: la señal del día i usa solo datos ≤ i. Reporta métricas reales y
las compara con comprar-y-mantener. Prueba también 1ª mitad vs 2ª mitad.
"""
import math
import datetime as dt
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}
SYMS = {"clp": "USDCLP=X", "cobre": "HG=F", "dxy": "DX-Y.NYB", "brl": "USDBRL=X"}

# --- parámetros de la estrategia (sensatos, sin sobreajustar) ---
ENTRY = 1.0        # desviación (σ) para entrar
EXIT = 0.25        # z para tomar ganancia (volvió a su valor justo)
STOP_ATR = 2.0     # stop = 2× ATR en contra
MAX_HOLD = 20      # días máximos en una posición
RISK_PCT = 0.01    # riesgo 1% del capital por trade
MAX_LEV = 3.0      # apalancamiento máximo (tope de tamaño)
COST_BPS = 6       # costo por lado (spread USD/CLP), en puntos base


def _fetch(sym, rng="5y"):
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "range": rng}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            out = {}
            for i in range(len(ts)):
                c = q["close"][i]
                if c is None:
                    continue
                d = dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d")
                out[d] = {"c": c, "h": q["high"][i] or c, "l": q["low"][i] or c}
            return out
        except Exception:
            continue
    return {}


def cargar():
    m = {k: _fetch(s) for k, s in SYMS.items()}
    com = sorted(set(m["clp"]) & set(m["cobre"]) & set(m["dxy"]) & set(m["brl"]))
    clp = np.array([m["clp"][d]["c"] for d in com])
    hi = np.array([m["clp"][d]["h"] for d in com])
    lo = np.array([m["clp"][d]["l"] for d in com])
    drivers = {k: np.array([m[k][d]["c"] for d in com]) for k in ("cobre", "dxy", "brl")}
    return com, clp, hi, lo, drivers


def valor_z(clp, drivers, i, win=60):
    """Residual z del USD/CLP vs cobre+DXY+real (regresión), solo datos ≤ i."""
    if i < win:
        return 0.0
    y = clp[i - win + 1:i + 1]
    X = np.column_stack([drivers["cobre"][i - win + 1:i + 1],
                         drivers["dxy"][i - win + 1:i + 1],
                         drivers["brl"][i - win + 1:i + 1], np.ones(win)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return 0.0
    resid = y - X @ beta
    sd = resid.std()
    return 0.0 if sd == 0 else float(resid[-1] / sd)


def atr(clp, hi, lo, i, n=14):
    if i < n:
        return None
    trs = []
    for k in range(i - n + 1, i + 1):
        trs.append(max(hi[k] - lo[k], abs(hi[k] - clp[k - 1]), abs(lo[k] - clp[k - 1])))
    return float(np.mean(trs))


def simular(com, clp, hi, lo, drivers, ini=65, fin=None):
    fin = fin or len(clp) - 1
    equity = 1.0
    curva = []
    pos = 0
    entry = stop = size = 0.0
    hold = 0
    trades = []
    for i in range(ini, fin + 1):
        # 1) si hay posición: aplicar retorno del día y ver salidas
        if pos != 0:
            r = clp[i] / clp[i - 1] - 1
            equity *= (1 + pos * size * r)
            hold += 1
            z = valor_z(clp, drivers, i)
            salida = None
            if pos == 1 and clp[i] <= stop:
                salida = "stop"
            elif pos == -1 and clp[i] >= stop:
                salida = "stop"
            elif abs(z) <= EXIT:
                salida = "objetivo"
            elif hold >= MAX_HOLD:
                salida = "tiempo"
            if salida:
                equity *= (1 - COST_BPS / 10000 * size)
                trades.append({"salida": salida, "dir": pos})
                pos = 0
        # 2) si está plano: ver entrada
        if pos == 0:
            z = valor_z(clp, drivers, i)
            a = atr(clp, hi, lo, i)
            if a and z >= ENTRY:
                pos = -1
            elif a and z <= -ENTRY:
                pos = 1
            if pos != 0:
                entry = clp[i]
                stopdist = STOP_ATR * a
                stop = entry - pos * stopdist
                size = min(MAX_LEV, RISK_PCT / (stopdist / entry))
                equity *= (1 - COST_BPS / 10000 * size)
                hold = 0
        curva.append(equity)
    return np.array(curva), trades


def metricas(curva, com, ini, clp):
    rets = np.diff(np.log(curva))
    total = (curva[-1] - 1) * 100
    dias = len(curva)
    cagr = ((curva[-1]) ** (252 / dias) - 1) * 100
    sharpe = (rets.mean() / (rets.std() + 1e-9) * math.sqrt(252)) if rets.std() > 0 else 0
    peak = np.maximum.accumulate(curva)
    dd = ((curva - peak) / peak).min() * 100
    bh = (clp[ini + dias - 1] / clp[ini] - 1) * 100
    return {"total": total, "cagr": cagr, "sharpe": sharpe, "dd": dd, "bh": bh, "dias": dias}


def correr():
    print("Descargando 5 años...")
    com, clp, hi, lo, drivers = cargar()
    n = len(clp)
    print(f"{n} días: {com[0]} → {com[-1]}\n")

    curva, trades = simular(com, clp, hi, lo, drivers)
    m = metricas(curva, com, 65, clp)

    print("== ESTRATEGIA mean-reversion del valor justo (con stop + sizing + costos) ==")
    print(f"  Trades: {len(trades)}")
    if trades:
        ganan = "?"
        salidas = {}
        for t in trades:
            salidas[t["salida"]] = salidas.get(t["salida"], 0) + 1
        print(f"  Salidas: {salidas}")
    print(f"  Retorno total:   {m['total']:+.1f}%   (CAGR {m['cagr']:+.1f}%/año)")
    print(f"  Comprar/mantener:{m['bh']:+.1f}%")
    print(f"  Sharpe:          {m['sharpe']:+.2f}")
    print(f"  Máx. caída (DD): {m['dd']:.1f}%")
    print()

    # robustez: 1ª mitad vs 2ª mitad
    mid = 65 + (n - 66) // 2
    c1, t1 = simular(com, clp, hi, lo, drivers, ini=65, fin=mid)
    c2, t2 = simular(com, clp, hi, lo, drivers, ini=mid, fin=n - 1)
    print("== Robustez (mitades) ==")
    print(f"  1ª mitad: {(c1[-1]-1)*100:+.1f}%  ({len(t1)} trades)")
    print(f"  2ª mitad: {(c2[-1]-1)*100:+.1f}%  ({len(t2)} trades)")
    print("  (para confiar, debería ganar en AMBAS, no solo en una)")
    return curva, trades


if __name__ == "__main__":
    correr()
