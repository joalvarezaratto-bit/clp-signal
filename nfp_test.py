"""
¿El USD/CLP reacciona a los datos de empleo de EE.UU. (NFP)?

El NFP sale el 1er viernes de cada mes a las 8:30 ET (~10:30 Chile) = ¡Chile
ABIERTO! Así que el peso puede reaccionar en vivo (distinto al Fed, que sale con
Chile cerrado). ¿Hay más volatilidad? ¿sesgo? ¿patrón operable?
"""
import datetime as dt
import calendar
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}


def fechas_nfp(desde=2015, hasta=2026):
    """Primer viernes de cada mes."""
    out = []
    for y in range(desde, hasta + 1):
        for mth in range(1, 13):
            for day in range(1, 8):
                if dt.date(y, mth, day).weekday() == 4:   # viernes
                    out.append(f"{y}-{mth:02d}-{day:02d}")
                    break
    return out


def cargar():
    p1 = int(dt.datetime(2015, 1, 1).timestamp()); p2 = int(dt.datetime(2026, 8, 5).timestamp())
    for h in ("query1", "query2"):
        try:
            r = requests.get(f"https://{h}.finance.yahoo.com/v8/finance/chart/USDCLP=X",
                             params={"interval": "1d", "period1": p1, "period2": p2}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, c = res["timestamp"], res["indicators"]["quote"][0]["close"]
            return {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"): c[i]
                    for i in range(len(ts)) if c[i] is not None}
        except Exception:
            continue
    return {}


def correr():
    d = cargar()
    fechas = sorted(d)
    px = np.array([d[f] for f in fechas])
    ret = np.concatenate([[0], np.diff(np.log(px))]) * 100
    def idx_de(fecha):
        o = dt.datetime.strptime(fecha, "%Y-%m-%d")
        return min(range(len(fechas)), key=lambda i: abs((dt.datetime.strptime(fechas[i], "%Y-%m-%d") - o).days))
    nfp_all = [f for f in fechas_nfp() if fechas[0] <= f <= fechas[-1]]
    nidx = sorted(set(idx_de(f) for f in nfp_all))   # día hábil más cercano (arregla feriados/huecos)
    print(f"{len(nidx)} datos de empleo (NFP) sobre {len(fechas)} días\n")

    vol_nfp = np.mean([abs(ret[i]) for i in nidx])
    vol_normal = np.mean(np.abs(ret[5:]))
    print("== 1) ¿Más volatilidad el día del NFP? (Chile abierto) ==")
    print(f"  |mov| día NFP:    {vol_nfp:.2f}%")
    print(f"  |mov| día normal: {vol_normal:.2f}%")
    print(f"  → {vol_nfp/vol_normal:.1f}x lo normal\n")

    print("== 2) ¿Sesgo direccional? (+ = dólar sube) ==")
    print(f"  Retorno medio día NFP:      {np.mean([ret[i] for i in nidx]):+.3f}%")
    print(f"  Retorno medio día siguiente:{np.mean([ret[i+1] for i in nidx if i+1<len(ret)]):+.3f}%")
    sub = sum(1 for i in nidx if ret[i] > 0)
    print(f"  Reparto: dólar subió {sub}/{len(nidx)} ({sub/len(nidx)*100:.0f}%)\n")

    # 3) ¿el signo del día NFP predice el día siguiente? (momentum/reversión del shock)
    dia = np.array([ret[i] for i in nidx[:-1]])
    sig = np.array([ret[i + 1] for i in nidx[:-1]])
    c = np.corrcoef(dia, sig)[0, 1]
    print("== 3) ¿El movimiento del día NFP sigue al día siguiente? ==")
    print(f"  corr(día NFP, día siguiente): {c:+.2f}  ({'momentum' if c>0.1 else 'reversión' if c<-0.1 else 'nada'})")

    # 4) ¿la estrategia de momentum del cobre 'confunde' el shock del NFP?
    #    comparar acierto de seguir-la-tendencia en día NFP vs normal
    print("\n== 4) Comportamiento del cobre en días NFP (¿mejor o peor para la señal?) ==")
    # aproximación: autocorrelación del propio peso en NFP vs normal
    todos = np.array(range(5, len(ret) - 1))
    nset = set(nidx)
    n_ac = np.corrcoef([ret[i - 1] for i in nidx if i - 1 >= 0 and i + 1 < len(ret)],
                       [ret[i] for i in nidx if i - 1 >= 0 and i + 1 < len(ret)])[0, 1]
    print(f"  (día NFP = evento grande, mayor volatilidad -> más riesgo de whipsaw)")


if __name__ == "__main__":
    correr()
