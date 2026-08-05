"""
¿El USD/CLP reacciona al CPI (inflación de EE.UU.)?

El CPI sale ~8:30 ET (~10:30 Chile, Chile ABIERTO), a mitad de mes. Fechas
exactas traídas de Investing.com. Mismo análisis que el NFP.
"""
import datetime as dt
import re
import time
import numpy as np
import requests

H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36",
     "X-Requested-With": "XMLHttpRequest", "Content-Type": "application/x-www-form-urlencoded",
     "Referer": "https://www.investing.com/economic-calendar/", "Accept": "*/*"}


def fechas_cpi(años):
    S = requests.Session()
    fechas = set()
    for y in años:
        # 6 semestres... fetch por medio año para no perder eventos, con reintentos
        cnt = 0
        for d1, d2 in [(f"{y}-01-01", f"{y}-06-30"), (f"{y}-07-01", f"{y}-12-31")]:
            h = ""
            for _ in range(6):
                try:
                    r = S.post("https://www.investing.com/economic-calendar/Service/getCalendarFilteredData", headers=H,
                               data={"country[]": "5", "importance[]": "3", "timeZone": "0", "timeFilter": "timeRemain",
                                     "currentTab": "custom", "dateFrom": d1, "dateTo": d2,
                                     "submitFilters": "1", "limit_from": "0"}, timeout=25)
                    j = r.json()
                    if isinstance(j, dict) and j.get("data"):
                        h = j["data"]; break
                except Exception:
                    pass
                time.sleep(2)
            bloques = re.split(r'id="theDay(\d+)"', h)
            for k in range(1, len(bloques) - 1, 2):
                f = dt.datetime.utcfromtimestamp(int(bloques[k])).strftime("%Y-%m-%d")
                for nom in re.findall(r"<a[^>]*>([^<]+)</a>", bloques[k + 1]):
                    if re.search(r"\bCPI \(MoM\)", nom):
                        if f not in fechas:
                            cnt += 1
                        fechas.add(f)
            time.sleep(1)
        print(f"  {y}: {cnt} fechas de CPI")
    return sorted(fechas)


def cargar():
    p1 = int(dt.datetime(2018, 1, 1).timestamp()); p2 = int(dt.datetime(2026, 8, 5).timestamp())
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/USDCLP=X",
                             params={"interval": "1d", "period1": p1, "period2": p2},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, c = res["timestamp"], res["indicators"]["quote"][0]["close"]
            return {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"): c[i]
                    for i in range(len(ts)) if c[i] is not None}
        except Exception:
            continue
    return {}


def correr():
    print("Trayendo fechas de CPI de Investing (2018-2025)...")
    cpi = fechas_cpi(range(2018, 2026))
    print(f"{len(cpi)} fechas de CPI obtenidas\n")
    d = cargar()
    fechas = sorted(d)
    px = np.array([d[f] for f in fechas]); ret = np.concatenate([[0], np.diff(np.log(px))]) * 100

    def idx_de(f):
        o = dt.datetime.strptime(f, "%Y-%m-%d")
        return min(range(len(fechas)), key=lambda i: abs((dt.datetime.strptime(fechas[i], "%Y-%m-%d") - o).days))
    nidx = sorted(set(idx_de(f) for f in cpi if fechas[0] <= f <= fechas[-1]))
    print(f"{len(nidx)} días de CPI sobre {len(fechas)} días de USD/CLP\n")

    vn = np.mean([abs(ret[i]) for i in nidx]); vnorm = np.mean(np.abs(ret[5:]))
    print("== 1) ¿Más volatilidad el día del CPI? ==")
    print(f"  |mov| día CPI: {vn:.2f}%  ·  normal: {vnorm:.2f}%  →  {vn/vnorm:.1f}x\n")
    print("== 2) ¿Sesgo direccional? ==")
    print(f"  Retorno medio día CPI:      {np.mean([ret[i] for i in nidx]):+.3f}%")
    print(f"  Retorno medio día siguiente:{np.mean([ret[i+1] for i in nidx if i+1<len(ret)]):+.3f}%")
    sub = sum(1 for i in nidx if ret[i] > 0)
    print(f"  Dólar subió: {sub}/{len(nidx)} ({sub/len(nidx)*100:.0f}%)\n")
    dia = np.array([ret[i] for i in nidx[:-1]]); sig = np.array([ret[i + 1] for i in nidx[:-1]])
    print("== 3) ¿Continúa al día siguiente? ==")
    print(f"  corr(día CPI, día siguiente): {np.corrcoef(dia, sig)[0,1]:+.2f}")


if __name__ == "__main__":
    correr()
