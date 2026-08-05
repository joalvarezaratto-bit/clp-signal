"""
LABORATORIO INTRADÍA del USD/CLP — ¿hay patrón dentro del día?

Datos: velas de 60 min, 3 meses (lo máximo confiable de Yahoo). OJO: muestra
chica → alto riesgo de sobreajuste. Se analiza SOLO en horario líquido chileno
(UTC 13-18) para evitar el 'precio rancio' de las horas cerradas.

Explora: momentum vs reversión intradía, lead del cobre por hora, y patrón por
hora del día.
"""
import datetime as dt
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}


def _fetch(sym, interval="60m", rng="3mo"):
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": interval, "range": rng}, headers=UA, timeout=25)
            res = r.json()["chart"]["result"][0]
            ts, c = res["timestamp"], res["indicators"]["quote"][0]["close"]
            return {ts[i]: c[i] for i in range(len(ts)) if c[i] is not None}
        except Exception:
            continue
    return {}


def correr():
    clp = _fetch("USDCLP=X"); cu = _fetch("HG=F"); dxy = _fetch("DX-Y.NYB")
    com = sorted(set(clp) & set(cu) & set(dxy))
    print(f"{len(com)} velas de 1h comunes ({dt.datetime.utcfromtimestamp(com[0]):%Y-%m-%d} → "
          f"{dt.datetime.utcfromtimestamp(com[-1]):%Y-%m-%d})\n")
    pc = np.array([clp[t] for t in com]); pcu = np.array([cu[t] for t in com])
    pd = np.array([dxy[t] for t in com])
    horas = np.array([dt.datetime.utcfromtimestamp(t).hour for t in com])
    rc = np.diff(np.log(pc)); rcu = np.diff(np.log(pcu)); rd = np.diff(np.log(pd))
    h = horas[1:]
    liq = (h >= 13) & (h < 18)   # horario líquido Chile (~9-14)

    def corr(a, b):
        if len(a) < 20 or a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    print("== 1) ¿Momentum o reversión intradía del USD/CLP? (hora a hora) ==")
    print(f"  Autocorrelación ret[t]↔ret[t+1], TODAS las horas: {corr(rc[:-1], rc[1:]):+.2f}")
    print(f"  Autocorrelación, solo horario LÍQUIDO:            {corr(rc[:-1][liq[:-1]], rc[1:][liq[:-1]]):+.2f}")
    print("  (+ = momentum, − = reversión, ~0 = nada)\n")

    print("== 2) ¿El cobre se adelanta al peso por 1 hora? (horario líquido) ==")
    print(f"  corr(cobre[t], peso[t+1]) líquido: {corr(rcu[:-1][liq[:-1]], rc[1:][liq[:-1]]):+.2f}")
    print(f"  corr(peso[t], cobre[t+1]) control: {corr(rc[:-1][liq[:-1]], rcu[1:][liq[:-1]]):+.2f}")
    print(f"  corr(DXY[t], peso[t+1]) líquido:   {corr(rd[:-1][liq[:-1]], rc[1:][liq[:-1]]):+.2f}\n")

    print("== 3) Patrón por hora del día (UTC) ==")
    print("  hora  mov.medio  |mov| (volatilidad)")
    for hh in range(9, 22):
        m = h == hh
        if m.sum() > 5:
            print(f"   {hh:02d}h   {rc[m].mean()*100:+.3f}%     {np.abs(rc[m]).mean()*100:.3f}%")
    print("  (Chile ≈ UTC−4: 13h UTC = 9h Chile = apertura)")


if __name__ == "__main__":
    correr()
