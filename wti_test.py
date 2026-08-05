"""
¿El PETRÓLEO (WTI) sirve para el USD/CLP?

Chile IMPORTA petróleo → petróleo caro = peso débil = USD/CLP sube (relación
DIRECTA, opuesta al cobre). Probamos: correlación, redundancia con el cobre,
momentum standalone (OOS) y como agregado a la estrategia ganadora.
"""
import datetime as dt
import numpy as np
import requests
import lab
import oos_test as oos
import experimentos_variables as ev


def _fetch(sym):
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "range": "5y"}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, c = res["timestamp"], res["indicators"]["quote"][0]["close"]
            return {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"): c[i]
                    for i in range(len(ts)) if c[i] is not None}
        except Exception:
            continue
    return {}


UA = {"User-Agent": "Mozilla/5.0"}


def correr():
    com0, arr0 = lab.cargar()
    wti_raw = _fetch("CL=F")
    com = [d for d in com0 if d in wti_raw]
    idx = [i for i, d in enumerate(com0) if d in wti_raw]
    arr = {k: arr0[k][idx] for k in arr0}
    arr["wti"] = np.array([wti_raw[d] for d in com])
    clp = arr["clp"]; n = len(clp)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    ini = 65; split = ini + int((n - ini) * 0.6)
    print(f"{n} días. WTI hoy: {arr['wti'][-1]:.1f} USD\n")

    r = {k: np.diff(np.log(arr[k])) for k in ("clp", "wti", "cobre", "dxy")}
    def corr(a, b, w=None):
        if w:
            a, b = a[-w:], b[-w:]
        return float(np.corrcoef(a, b)[0, 1])
    print("== Correlaciones de retornos ==")
    print(f"  WTI ~ USD/CLP (40d): {corr(r['clp'], r['wti'], 40):+.2f}   (1y: {corr(r['clp'], r['wti'], 252):+.2f})")
    print(f"  WTI ~ cobre:         {corr(r['wti'], r['cobre'], 252):+.2f}   <- si alto, redundante con cobre")
    print(f"  WTI ~ DXY:           {corr(r['wti'], r['dxy'], 252):+.2f}\n")

    def ev_(pos):
        _, _, sh1, _ = lab.metricas(lab.simular(pos, clp, ini, split))
        _, c, sh2, dd = lab.metricas(lab.simular(pos, clp, split, n - 1))
        return sh1, sh2, c, dd

    # WTI momentum standalone (directo: petróleo sube -> largo USD)
    grid = [(w, t) for w in (3, 5, 8) for t in (0.005, 0.01, 0.02)]
    mejor, best = None, -9
    for w, t in grid:
        sh, _, _, _ = ev_(ev.mom(arr["wti"], w, t, +1))
        if sh > best:
            best, mejor = sh, (w, t)
    sh1, sh2, c, dd = ev_(ev.mom(arr["wti"], *mejor, +1))
    print("== WTI como señal standalone (OOS) ==")
    print(f"  Sharpe train {sh1:+.2f} → TEST {sh2:+.2f}  (CAGR {c:+.1f}%)  "
          f"{'SIRVE ✓' if sh2 > 0.7 else 'no sirve ✗'}\n")

    # agregar WTI a la estrategia ganadora (combo v2) como confirmación
    base = oos.est_combo_dxy(clp, arr, 3, 0.01)
    _, base_te, base_c, base_dd = ev_(base)
    r_wti = np.concatenate([[0], np.diff(np.log(arr["wti"]))])
    conf = base.copy()
    for i in range(5, n):
        wti_mov = r_wti[i - 4:i + 1].sum()
        # largo USD (base>0) confirma si petróleo sube; corto si petróleo baja
        if base[i] > 0 and wti_mov < 0:
            conf[i] = 0
        elif base[i] < 0 and wti_mov > 0:
            conf[i] = 0
    _, sh_c, c_c, dd_c = ev_(conf)
    print("== Agregar WTI a la estrategia ganadora ==")
    print(f"  BASE (combo v2):        Sharpe test {base_te:+.2f}  CAGR {base_c:+.1f}%  DD {base_dd:.1f}%")
    print(f"  + confirmación WTI:     Sharpe test {sh_c:+.2f}  CAGR {c_c:+.1f}%  DD {dd_c:.1f}%  "
          f"{'🏆 mejora' if sh_c > base_te + 0.05 else ('= igual' if abs(sh_c-base_te) <= 0.05 else 'peor')}")


if __name__ == "__main__":
    correr()
