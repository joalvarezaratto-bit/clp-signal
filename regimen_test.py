"""
¿Podemos DETECTAR cuándo la estrategia va a funcionar (cobre tendencial) vs no
(cobre lateral)? Y usar eso para operar más/menos.

Indicador: EFFICIENCY RATIO del cobre (qué tan direccional es su movimiento).
  ER = |cambio neto en N días| / suma de |cambios diarios|  → 0 (lateral) a 1 (tendencia).
Alta ER = cobre trendeando = bueno para momentum.
"""
import datetime as dt
import numpy as np
import requests
import estrategia as E
import lab
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


def eff_ratio(x, i, nn=20):
    if i < nn:
        return 0.5
    neto = abs(x[i] - x[i - nn])
    camino = np.sum(np.abs(np.diff(x[i - nn:i + 1])))
    return neto / camino if camino > 0 else 0


def correr():
    m = {k: _fetch(s) for k, s in E.SYMS.items()}
    com = sorted(set.intersection(*[set(d) for d in m.values()]))
    arr = {k: tv._limpiar(np.array([m[k][d]["c"] for d in com])) for k in m}
    clp = arr["clp"]; n = len(clp)
    hi = np.array([m["clp"][d]["h"] for d in com]); lo = np.array([m["clp"][d]["l"] for d in com])
    rng = np.minimum(hi - lo, clp * 0.08); hi = clp + rng / 2; lo = clp - rng / 2
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4

    pos, _ = E.posicion(com, arr, hi, lo)
    ret = np.concatenate([[0], np.diff(np.log(clp))])
    er = np.array([eff_ratio(arr["cobre"], i) for i in range(n)])   # régimen del cobre

    # retorno de la estrategia por tercil de régimen
    val = np.arange(65, n - 1)
    pnl = np.array([pos[i] * min(3, 0.10 / max(np.std(ret[max(0, i-20):i]) * np.sqrt(252), 0.02)) * ret[i + 1] * 100 for i in val])
    reg = er[65:n - 1]
    q1, q2 = np.percentile(reg, [40, 70])
    print("== Rendimiento de la estrategia según el RÉGIMEN del cobre ==")
    for nom, mask in [("Lateral (cobre choppy)", reg <= q1),
                      ("Intermedio", (reg > q1) & (reg <= q2)),
                      ("Tendencial (cobre direccional)", reg > q2)]:
        p = pnl[mask]; op = p[p != 0]
        sh = op.mean() / (op.std() + 1e-9) * np.sqrt(252) if len(op) else 0
        print(f"  {nom:32}: {p.sum():+7.1f}% acum.  Sharpe {sh:+.2f}  ({mask.sum()} días)")
    print("  (si 'tendencial' rinde mucho más, el ER detecta el buen régimen)\n")

    # ¿el régimen de HOY predice? test: operar solo/más en régimen tendencial (OOS)
    corte = next(i for i, d in enumerate(com) if d >= "2021-08-01")

    def sh(p, a, b):
        _, _, s, _ = lab.metricas(lab.simular(p, clp, a, b)); return s
    print("== ¿Filtrar/escalar por régimen mejora? (OOS ambos períodos) ==")
    base = pos
    solo_tend = np.where(er > q1, pos, 0)             # no operar en lateral
    escala_reg = pos * np.clip(er / np.median(er), 0.5, 1.5)   # más grande si más tendencial
    print(f"  {'Base (sin filtro régimen)':32} viejo {sh(base,65,corte):+.2f}  reciente {sh(base,corte,n-1):+.2f}")
    print(f"  {'Solo si cobre no-lateral':32} viejo {sh(solo_tend,65,corte):+.2f}  reciente {sh(solo_tend,corte,n-1):+.2f}")
    print(f"  {'Escalar tamaño por régimen':32} viejo {sh(escala_reg,65,corte):+.2f}  reciente {sh(escala_reg,corte,n-1):+.2f}")
    print(f"\n  ER del cobre HOY: {er[-1]:.2f}  ({'TENDENCIAL' if er[-1]>q2 else 'lateral' if er[-1]<=q1 else 'intermedio'})")


if __name__ == "__main__":
    correr()
