"""
¿Hay PATRONES en el USD/CLP alrededor de las reuniones del FED (FOMC)?

El FOMC decide ~8 veces/año en fechas fijas. Analizamos el comportamiento del
peso: volatilidad en el día de la decisión, sesgo direccional, y 'drift' antes.
Fechas de decisión 2015-2024 (80 reuniones, conocidas).
"""
import datetime as dt
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}

FOMC = [
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
]


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
    idxmap = {f: i for i, f in enumerate(fechas)}

    def idx_de(fecha):
        o = dt.datetime.strptime(fecha, "%Y-%m-%d")
        return min(range(len(fechas)), key=lambda i: abs((dt.datetime.strptime(fechas[i], "%Y-%m-%d") - o).days))

    fomc_idx = [idx_de(f) for f in FOMC if f <= fechas[-1] and f >= fechas[0]]
    print(f"{len(fomc_idx)} reuniones FOMC sobre {len(fechas)} días de USD/CLP\n")

    # 1) volatilidad: día FOMC vs normal
    vol_fomc = np.mean([abs(ret[i]) for i in fomc_idx])
    vol_dia_sig = np.mean([abs(ret[i + 1]) for i in fomc_idx if i + 1 < len(ret)])
    vol_normal = np.mean(np.abs(ret[5:]))
    print("== 1) ¿Sube la volatilidad en torno al FOMC? ==")
    print(f"  |mov| día del FOMC:      {vol_fomc:.2f}%")
    print(f"  |mov| día siguiente:     {vol_dia_sig:.2f}%")
    print(f"  |mov| día normal:        {vol_normal:.2f}%")
    print(f"  → el día del FOMC se mueve {vol_fomc/vol_normal:.1f}x lo normal\n")

    # 2) sesgo direccional
    print("== 2) ¿Hay sesgo direccional? (+ = dólar sube / peso débil) ==")
    print(f"  Retorno medio día FOMC:      {np.mean([ret[i] for i in fomc_idx]):+.3f}%")
    print(f"  Retorno medio día siguiente: {np.mean([ret[i+1] for i in fomc_idx if i+1<len(ret)]):+.3f}%")
    print(f"  (normal: {np.mean(ret[5:]):+.3f}%)\n")

    # 3) drift previo (5 días antes -> día antes)
    print("== 3) ¿'Drift' antes del FOMC? ==")
    pre = [np.sum(ret[i - 5:i]) for i in fomc_idx if i >= 5]
    print(f"  Movimiento acumulado 5 días PREVIOS al FOMC: {np.mean(pre):+.3f}%  (mediana {np.median(pre):+.3f}%)")
    post = [np.sum(ret[i + 1:i + 4]) for i in fomc_idx if i + 4 < len(ret)]
    print(f"  Movimiento 3 días DESPUÉS:                   {np.mean(post):+.3f}%\n")

    # 4) reparto: cuántas veces subió vs bajó el día del FOMC
    subieron = sum(1 for i in fomc_idx if ret[i] > 0)
    print("== 4) Reparto del día del FOMC ==")
    print(f"  Dólar SUBIÓ: {subieron}/{len(fomc_idx)} ({subieron/len(fomc_idx)*100:.0f}%)  ·  "
          f"BAJÓ: {len(fomc_idx)-subieron} ({(1-subieron/len(fomc_idx))*100:.0f}%)")


if __name__ == "__main__":
    correr()
