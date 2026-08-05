"""
LA PRUEBA DE LA VERDAD: estrategia CONGELADA sobre 2009-2021 (nunca visto).

Mismos parámetros exactos que la versión final (win=3, thr=0.01, actividad,
stop 3×ATR). CERO reoptimización. Si funciona acá, el edge es creíble de verdad;
si se derrumba, estábamos sobreajustados a 2021-2026.
"""
import datetime as dt
import numpy as np
import requests
import lab
import oos_test as oos
import stops_test as st

UA = {"User-Agent": "Mozilla/5.0"}
SYMS = {"clp": "USDCLP=X", "cobre": "HG=F", "dxy": "DX-Y.NYB", "brl": "USDBRL=X"}


def _fetch(sym):
    p1 = int(dt.datetime(2009, 1, 1).timestamp())
    p2 = int(dt.datetime(2021, 8, 1).timestamp())
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "period1": p1, "period2": p2},
                             headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            out = {}
            for i in range(len(ts)):
                if q["close"][i] is None:
                    continue
                d = dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d")
                out[d] = {"c": q["close"][i], "h": q["high"][i] or q["close"][i],
                          "l": q["low"][i] or q["close"][i]}
            return out
        except Exception:
            continue
    return {}


def _limpiar(x, cap=0.10):
    """Quita bad ticks: clipa retornos diarios absurdos (>10%) y reconstruye."""
    r = np.clip(np.diff(np.log(x)), -cap, cap)
    out = np.empty(len(x)); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] * np.exp(r[i - 1])
    return out


def correr():
    m = {k: _fetch(s) for k, s in SYMS.items()}
    com = sorted(set.intersection(*[set(d) for d in m.values()]))
    arr = {k: _limpiar(np.array([m[k][d]["c"] for d in com], dtype=float)) for k in m}
    clp = arr["clp"]; n = len(clp)
    hi = np.array([m["clp"][d]["h"] for d in com])
    lo = np.array([m["clp"][d]["l"] for d in com])
    # capear rangos absurdos (bad ticks en máx/mín) a 8% del precio
    rng_raw = np.minimum(hi - lo, clp * 0.08)
    hi = clp + rng_raw / 2
    lo = clp - rng_raw / 2
    print(f"{n} días vírgenes: {com[0]} → {com[-1]}\n")

    # --- estrategia CONGELADA (mismos params exactos) ---
    base_combo = oos.est_combo_dxy(clp, arr, 3, 0.01)
    rango = (hi - lo) / clp * 100
    act = np.ones(n)
    for i in range(20, n):
        med = np.median(rango[i - 20:i]) or 1
        act[i] = np.clip(rango[i] / med, 0.5, 1.5)
    base = base_combo * act
    # ATR y stop 3×ATR
    tr = hi - lo
    atr = np.copy(tr)
    for i in range(1, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    pos = st.con_stops(base, clp, atr, 3.0, None)

    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    cur = lab.simular(pos, clp, 65, n - 1)
    _, cagr, sh, dd = lab.metricas(cur)
    total = (cur[-1] - 1) * 100

    print("===== RESULTADO en 2009-2021 (nunca visto, params congelados) =====")
    print(f"  Retorno total:  {total:+.1f}%  (~{len(cur)/252:.0f} años)")
    print(f"  CAGR:           {cagr:+.1f}%/año")
    print(f"  Sharpe:         {sh:+.2f}")
    print(f"  Máx. caída:     {dd:.1f}%")
    print(f"  Referencia 2021-2026: CAGR ~18%, Sharpe ~1.8\n")

    # año por año
    fechas = com[65:65 + len(cur)]
    ret = np.concatenate([[0], np.diff(np.log(cur))])
    print("  Año por año:")
    for y in sorted(set(f[:4] for f in fechas)):
        idx = [i for i, f in enumerate(fechas) if f[:4] == y]
        if len(idx) < 20:
            continue
        ry = (np.exp(np.sum(ret[idx[0]:idx[-1] + 1])) - 1) * 100
        print(f"    {y}: {ry:+6.1f}%")

    print("\n  VEREDICTO:", "CREÍBLE ✓ (aguanta fuera de muestra)" if sh > 0.6
          else ("dudoso" if sh > 0.2 else "SOBREAJUSTADO ✗ (era espejismo de 2021-26)"))
    return com, cur


if __name__ == "__main__":
    correr()
