"""
IDEA: el peso REACCIONA en la apertura a lo que el cobre/DXY hicieron de NOCHE.

Chile cierra de noche; el cobre (COMEX) y el DXY siguen moviéndose en horario
EE.UU. Al abrir Chile, el peso se ajusta a esa info nueva. Si el cobre subió de
noche → el peso debería abrir/operar más fuerte (USD/CLP baja).

Es 1 operación al día (bajo costo). Se mide con velas de 60m.
Sesión Chile ≈ UTC 13-18. 'Nocturno' = desde el cierre de ayer hasta la apertura.
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
    # organizar por (fecha, hora)
    def h(t):
        return dt.datetime.utcfromtimestamp(t)
    # apertura de sesión = primera vela con hora>=13; cierre = última con hora<=17
    dias = {}
    for t in com:
        d = h(t).strftime("%Y-%m-%d")
        dias.setdefault(d, []).append(t)
    fechas = sorted(dias)

    filas = []   # (cobre_noche, dxy_noche, peso_sesion)
    prev_close_t = None
    for d in fechas:
        ses = [t for t in dias[d] if 13 <= h(t).hour <= 17]
        if len(ses) < 2:
            continue
        t_open, t_close = ses[0], ses[-1]
        if prev_close_t is not None:
            # movimiento nocturno = desde cierre sesión previa hasta apertura de hoy
            cu_noche = np.log(cu[t_open] / cu[prev_close_t])
            dxy_noche = np.log(dxy[t_open] / dxy[prev_close_t])
            peso_ses = np.log(clp[t_close] / clp[t_open])   # lo que hace el peso EN la sesión
            filas.append((cu_noche, dxy_noche, peso_ses))
        prev_close_t = t_close

    a = np.array(filas)
    print(f"{len(a)} sesiones analizadas ({fechas[0]} → {fechas[-1]})\n")
    cun, dxn, ps = a[:, 0], a[:, 1], a[:, 2]

    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if len(x) > 10 and x.std() > 0 else 0

    print("== ¿El movimiento NOCTURNO predice la sesión del peso? ==")
    print(f"  Cobre nocturno → peso sesión:  {corr(cun, ps):+.2f}  (esperado NEGATIVO: cobre sube→peso sube→USD/CLP baja)")
    print(f"  DXY nocturno   → peso sesión:  {corr(dxn, ps):+.2f}  (esperado POSITIVO)")
    print()

    # estrategia: al abrir, si el cobre subió de noche -> corto USD la sesión
    print("== Estrategia: reaccionar en la apertura (1 trade/sesión) ==")
    for costo in (0, 4.4):
        señal = -np.sign(cun)   # cobre sube -> corto USD (peso sube)
        pnl = señal * ps * 100 - (costo / 10000 * 100)   # 1 entrada+salida por sesión
        op = pnl
        tot = pnl.sum()
        sh = op.mean() / (op.std() + 1e-9) * np.sqrt(252)
        hit = (np.sign(señal) == np.sign(ps)).mean() * 100
        print(f"  costo {costo:>3} pb: retorno {tot:+.1f}%  Sharpe {sh:+.2f}  acierto {hit:.0f}%")
    # combinado cobre+DXY
    señal2 = -np.sign(cun) + np.sign(dxn)   # cobre inverso + DXY directo
    señal2 = np.sign(señal2)
    pnl2 = señal2 * ps * 100 - (4.4 / 10000 * 100)
    print(f"  combinado cobre+DXY (4.4pb): retorno {pnl2.sum():+.1f}%  Sharpe {pnl2.mean()/(pnl2.std()+1e-9)*np.sqrt(252):+.2f}")


if __name__ == "__main__":
    correr()
