"""
EXAMEN DE RENTABILIDAD — estrategia final (combo v2 + sizing por actividad).

Analiza a fondo: retorno, Sharpe, Sortino, drawdowns, año a año, mensual,
% meses ganadores, y qué significa en pesos. Enfatiza el OUT-OF-SAMPLE (creíble).
"""
import datetime as dt
import numpy as np
import requests
import lab
import oos_test as oos

UA = {"User-Agent": "Mozilla/5.0"}


def _rango(com, clp):
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/USDCLP=X",
                             params={"interval": "1d", "range": "5y"}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            hl = {}
            for i in range(len(ts)):
                if q["close"][i] is None:
                    continue
                d = dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d")
                hl[d] = (q["high"][i] or q["close"][i]) - (q["low"][i] or q["close"][i])
            return np.array([hl.get(d, 0) / clp[i] * 100 for i, d in enumerate(com)])
        except Exception:
            continue
    return np.ones(len(com))


def posicion_final(com, arr, clp):
    """Combo v2 escalado por actividad (la mejor versión)."""
    base = oos.est_combo_dxy(clp, arr, 3, 0.01)
    rango = _rango(com, clp)
    act = np.ones(len(clp))
    for i in range(20, len(clp)):
        med = np.median(rango[i - 20:i]) or 1
        act[i] = np.clip(rango[i] / med, 0.5, 1.5)
    return base * act


def analizar(curva, fechas, etiqueta):
    r = np.diff(np.log(curva))
    total = (curva[-1] - 1) * 100
    dias = len(curva)
    cagr = (curva[-1] ** (252 / dias) - 1) * 100
    vol = r.std() * np.sqrt(252) * 100
    sharpe = r.mean() / (r.std() + 1e-9) * np.sqrt(252)
    neg = r[r < 0]
    sortino = r.mean() / (neg.std() + 1e-9) * np.sqrt(252)
    peak = np.maximum.accumulate(curva)
    ddser = (curva - peak) / peak
    maxdd = ddser.min() * 100
    en_dd = (ddser < -0.005).mean() * 100
    winday = (r > 0).mean() * 100
    pf = r[r > 0].sum() / (-r[r < 0].sum() + 1e-9)
    print(f"\n===== {etiqueta} =====")
    print(f"  Retorno total:      {total:+.1f}%   ({dias} días, ~{dias/252:.1f} años)")
    print(f"  CAGR (anualizado):  {cagr:+.1f}%")
    print(f"  Volatilidad:        {vol:.1f}%")
    print(f"  Sharpe:             {sharpe:+.2f}")
    print(f"  Sortino:            {sortino:+.2f}   (castiga solo caídas)")
    print(f"  Máx. drawdown:      {maxdd:.1f}%")
    print(f"  Tiempo en pérdida:  {en_dd:.0f}%")
    print(f"  Días ganadores:     {winday:.0f}%   ·  Profit factor: {pf:.2f}")
    # mensual
    meses = {}
    for i, f in enumerate(fechas[1:]):
        meses.setdefault(f[:7], []).append(r[i])
    mret = {m: (np.exp(np.sum(v)) - 1) * 100 for m, v in meses.items()}
    vals = list(mret.values())
    print(f"  Meses positivos:    {sum(1 for v in vals if v>0)/len(vals)*100:.0f}%  "
          f"(mejor {max(vals):+.1f}%, peor {min(vals):+.1f}%)")
    return cagr, sharpe, maxdd


def correr():
    com, arr = lab.cargar()
    clp = arr["clp"]; n = len(clp)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    pos = posicion_final(com, arr, clp)
    ini = 65; split = ini + int((n - ini) * 0.6)

    # curvas
    cur_full = lab.simular(pos, clp, ini, n - 1)
    cur_oos = lab.simular(pos, clp, split, n - 1)
    f_full = com[ini:ini + len(cur_full)]
    f_oos = com[split:split + len(cur_oos)]

    analizar(cur_full, f_full, "PERÍODO COMPLETO (2021-2026)")
    analizar(cur_oos, f_oos, "OUT-OF-SAMPLE (2024-2026, lo CREÍBLE)")

    # año a año
    print("\n===== Rentabilidad AÑO A AÑO =====")
    ret = np.concatenate([[0], np.diff(np.log(cur_full))])
    for y in sorted(set(f[:4] for f in f_full)):
        idx = [i for i, f in enumerate(f_full) if f[:4] == y]
        if len(idx) < 20:
            continue
        ry = (np.exp(np.sum(ret[idx[0]:idx[-1] + 1])) - 1) * 100
        print(f"  {y}: {ry:+6.1f}%")

    # a distinto apalancamiento (sobre la base 10% vol)
    print("\n===== Rentabilidad vs apalancamiento (OOS, Sharpe fijo) =====")
    cagr_oos = (cur_oos[-1] ** (252 / len(cur_oos)) - 1) * 100
    peak = np.maximum.accumulate(cur_oos); dd_oos = ((cur_oos - peak) / peak).min() * 100
    for lev in (1, 2, 3, 4):
        print(f"  {lev}x → CAGR ~{cagr_oos*lev:+.0f}%/año   ·   caída esperada ~{dd_oos*lev:.0f}%")

    # en pesos
    print("\n===== En pesos (ejemplo, capital $5.000.000, OOS ~2 años) =====")
    for lev in (1, 2, 3):
        final = 5_000_000 * (1 + cagr_oos * lev / 100) ** 1.8
        print(f"  {lev}x: $5.000.000 → ${final:,.0f}  (en ~1,8 años)")
    print("\n  OJO: rentabilidad pasada NO garantiza futura. El paper-trading en vivo")
    print("  confirma si el edge sigue. Apalancar sube retorno Y riesgo de ruina juntos.")
    return com, cur_full, f_full


if __name__ == "__main__":
    correr()
