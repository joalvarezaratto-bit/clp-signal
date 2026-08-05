"""
¿La estrategia rinde PEOR en las zonas políticas calientes? (con datos)

Compara el rendimiento de la estrategia y la correlación cobre-peso DENTRO vs
FUERA de las ventanas políticas (estallido 2019, proceso constitucional 2022).
Si adentro rinde peor Y el cobre se desconecta, confirma que la política es un
riesgo para la estrategia (basada en cobre).
"""
import datetime as dt
import numpy as np
import requests
import lab
import oos_test as oos
import stops_test as st
import test_viejo as tv

UA = {"User-Agent": "Mozilla/5.0"}
SYMS = {"clp": "USDCLP=X", "cobre": "HG=F", "dxy": "DX-Y.NYB", "brl": "USDBRL=X"}

# ventanas políticas calientes (peso se movió por política, no por cobre)
HOT = [
    ("2019-10-18", "2020-02-01", "Estallido social"),
    ("2021-11-01", "2022-03-31", "Elección Boric + asunción"),
    ("2022-06-01", "2022-09-30", "Constitución (peso récord 1050)"),
]


def _fetch(sym):
    p1 = int(dt.datetime(2012, 1, 1).timestamp()); p2 = int(dt.datetime(2026, 8, 5).timestamp())
    for h in ("query1", "query2"):
        try:
            r = requests.get(f"https://{h}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "period1": p1, "period2": p2}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            out = {}
            for i in range(len(ts)):
                if q["close"][i] is None:
                    continue
                d = dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d")
                out[d] = {"c": q["close"][i], "h": q["high"][i] or q["close"][i], "l": q["low"][i] or q["close"][i]}
            return out
        except Exception:
            continue
    return {}


def correr():
    m = {k: _fetch(s) for k, s in SYMS.items()}
    com = sorted(set.intersection(*[set(d) for d in m.values()]))
    arr = {k: tv._limpiar(np.array([m[k][d]["c"] for d in com])) for k in m}
    clp = arr["clp"]; n = len(clp)
    hi = np.array([m["clp"][d]["h"] for d in com]); lo = np.array([m["clp"][d]["l"] for d in com])
    rng = np.minimum(hi - lo, clp * 0.08); hi = clp + rng / 2; lo = clp - rng / 2

    # estrategia congelada
    base = oos.est_combo_dxy(clp, arr, 3, 0.01)
    ra = (hi - lo) / clp * 100; act = np.ones(n)
    for i in range(20, n):
        med = np.median(ra[i - 20:i]) or 1; act[i] = np.clip(ra[i] / med, 0.5, 1.5)
    tr = hi - lo; atr = np.copy(tr)
    for i in range(1, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    pos = st.con_stops(base * act, clp, atr, 3.0, None)

    # retorno diario de la estrategia (con vol-target)
    ret = np.concatenate([[0], np.diff(np.log(clp))])
    pnl = np.zeros(n)
    for i in range(65, n - 1):
        vol = ret[max(0, i - 20):i].std() * np.sqrt(252)
        scale = min(3.0, 0.10 / max(vol, 0.02))
        pnl[i] = pos[i] * scale * ret[i + 1] * 100

    # marcar días políticos
    hot_mask = np.zeros(n, bool)
    for a, b, _ in HOT:
        for i, dd in enumerate(com):
            if a <= dd <= b:
                hot_mask[i] = True

    val = np.arange(65, n - 1)
    hot = val[hot_mask[65:n - 1]]
    normal = val[~hot_mask[65:n - 1]]

    def stats(idx):
        p = pnl[idx]; op = p[pos[idx] != 0]
        sh = op.mean() / (op.std() + 1e-9) * np.sqrt(252) if len(op) else 0
        return op.mean(), sh, (np.sign(pos[idx][pos[idx] != 0]) == np.sign(ret[idx + 1][pos[idx] != 0])).mean() * 100

    print("== Rendimiento de la estrategia: político vs normal ==")
    for nom, idx in [("Días POLÍTICOS calientes", hot), ("Días normales", normal)]:
        mu, sh, hit = stats(idx)
        print(f"  {nom:26}: retorno {mu:+.3f}%/día  Sharpe {sh:+.2f}  acierto {hit:.0f}%  ({len(idx)} días)")

    # correlación cobre-peso dentro vs fuera
    rc = np.concatenate([[0], np.diff(np.log(arr["cobre"]))])
    def corr(idx):
        a, b = ret[idx], rc[idx]
        return np.corrcoef(a, b)[0, 1] if len(a) > 10 else 0
    print("\n== ¿El peso se desconecta del cobre en política? ==")
    print(f"  Correlación cobre-peso, días normales:  {corr(normal):+.2f}")
    print(f"  Correlación cobre-peso, días políticos: {corr(hot):+.2f}")
    print("  (si baja en días políticos, el cobre 'miente' y la señal falla)")

    # detalle por ventana
    print("\n== Detalle por evento (retorno de la estrategia) ==")
    for a, b, nombre in HOT:
        idx = np.array([i for i in range(65, n - 1) if a <= com[i] <= b])
        if len(idx):
            print(f"  {nombre:34}: {pnl[idx].sum():+.1f}% acumulado ({len(idx)} días)")


if __name__ == "__main__":
    correr()
