"""
EXPERIMENTO: ¿el edge depende de la ACTIVIDAD/volumen del mercado?

El USD/CLP no tiene volumen (OTC), pero usamos proxies:
  - Rango diario (máx−mín)/precio  → actividad del propio peso
  - Volumen del cobre (HG=F)        → actividad del driver
  - Volatilidad reciente            → régimen (se agrupa, es algo predecible)

Preguntas:
  1) ¿La estrategia rinde distinto en días activos vs tranquilos?
  2) ¿Sirve operar/agrandar SOLO cuando hay actividad? (validado OOS)
"""
import datetime as dt
import numpy as np
import requests
import lab
import oos_test as oos

UA = {"User-Agent": "Mozilla/5.0"}


def _fetch_full(sym):
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "range": "5y"}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            out = {}
            for i in range(len(ts)):
                if q["close"][i] is None:
                    continue
                d = dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d")
                out[d] = {"c": q["close"][i], "h": q["high"][i] or q["close"][i],
                          "l": q["low"][i] or q["close"][i], "v": q["volume"][i] or 0}
            return out
        except Exception:
            continue
    return {}


def correr():
    com, arr = lab.cargar()
    clp = arr["clp"]; n = len(clp)
    clp_full = _fetch_full("USDCLP=X")
    cu_full = _fetch_full("HG=F")
    rango = np.array([(clp_full.get(d, {}).get("h", clp[i]) - clp_full.get(d, {}).get("l", clp[i]))
                      / clp[i] * 100 for i, d in enumerate(com)])
    cu_vol = np.array([cu_full.get(d, {}).get("v", 0) for d in com], dtype=float)

    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    pos = oos.est_combo_dxy(clp, arr, 3, 0.01)
    ret = np.concatenate([[0], np.diff(np.log(clp))])
    ini = 65

    # rendimiento de la estrategia por TERCIL de actividad (rango del día previo)
    print("== 1) ¿Dónde vive el edge? (retorno de la estrategia por actividad) ==")
    dpos = pos[ini:n - 1]
    dret = ret[ini + 1:n] * 100        # retorno del día siguiente
    dact = rango[ini:n - 1]            # rango del día en que se decide
    pnl = dpos * dret                  # ganancia de la estrategia ese día
    q1, q2 = np.percentile(dact, [33, 66])
    for nom, m in [("Tranquilo (rango bajo)", dact <= q1),
                   ("Normal", (dact > q1) & (dact <= q2)),
                   ("Activo (rango alto)", dact > q2)]:
        activos = m & (dpos != 0)
        print(f"  {nom:26}: retorno medio {pnl[activos].mean():+.3f}%/día  "
              f"acierto {(np.sign(dpos[activos])==np.sign(dret[activos])).mean()*100:.0f}%  "
              f"({activos.sum()} días op.)")
    print("  (si el retorno medio es mayor en 'activo', el edge vive en la actividad)\n")

    # 2) sizing por actividad: agrandar cuando hay más actividad (validado OOS)
    print("== 2) ¿Sirve escalar el tamaño por actividad? (OOS) ==")
    split = ini + int((n - ini) * 0.6)

    def evalu(p):
        _, _, sh1, _ = lab.metricas(lab.simular(p, clp, ini, split))
        _, c, sh2, dd = lab.metricas(lab.simular(p, clp, split, n - 1))
        return sh1, sh2, c, dd

    # factor de actividad: rango relativo a su mediana móvil (0.5x..1.5x)
    act_rel = np.ones(n)
    for i in range(20, n):
        med = np.median(rango[i - 20:i]) or 1
        act_rel[i] = np.clip(rango[i] / med, 0.5, 1.5)
    base = evalu(pos)
    con_act = evalu(pos * act_rel)
    solo_act = evalu(np.where(rango > np.median(rango), pos, 0))  # operar solo días activos
    print(f"  {'Base (combo v2)':32} Sharpe test {base[1]:+.2f}  CAGR {base[2]:+.1f}%  DD {base[3]:.1f}%")
    print(f"  {'× factor actividad (sizing)':32} Sharpe test {con_act[1]:+.2f}  CAGR {con_act[2]:+.1f}%  DD {con_act[3]:.1f}%")
    print(f"  {'Solo días de rango alto':32} Sharpe test {solo_act[1]:+.2f}  CAGR {solo_act[2]:+.1f}%  DD {solo_act[3]:.1f}%")

    # 3) ¿la actividad de hoy predice la de mañana? (agrupamiento)
    print("\n== 3) ¿Se puede anticipar la actividad? ==")
    corr = np.corrcoef(rango[ini:n - 1], rango[ini + 1:n])[0, 1]
    print(f"  Correlación rango hoy ↔ rango mañana: {corr:+.2f}  "
          f"({'SÍ se agrupa/predice' if corr > 0.3 else 'poco predecible'})")
    if cu_vol.sum() > 0:
        cv = np.corrcoef(cu_vol[ini:n - 1], rango[ini + 1:n])[0, 1]
        print(f"  Volumen cobre hoy ↔ rango USD/CLP mañana: {cv:+.2f}")


if __name__ == "__main__":
    correr()
