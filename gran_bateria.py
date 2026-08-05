"""
GRAN BATERÍA — exploración sistemática de TODO lo no probado.

Ideas: asimetría largo/corto, mineras de cobre (FCX/SCCO) como señal adelantada,
ratio cobre/oro, multi-timeframe, histéresis, estacionalidad, take-profit,
piramidación. Metodología estricta: una variante solo "gana" si supera a la BASE
en AMBOS períodos (2009-2021 y 2021-2026). Base = combo v2 (2d) + actividad + stop.
"""
import datetime as dt
import numpy as np
import requests
import estrategia as E
import lab
import test_viejo as tv

UA = {"User-Agent": "Mozilla/5.0"}
EXTRA = {"fcx": "FCX", "scco": "SCCO", "ech": "ECH", "oro": "GC=F"}


def _fetch(sym):
    p1 = int(dt.datetime(2009, 1, 1).timestamp()); p2 = int(dt.datetime(2026, 8, 7).timestamp())
    for h in ("query1", "query2"):
        try:
            r = requests.get(f"https://{h}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "period1": p1, "period2": p2},
                             headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            return {dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d"):
                    {"c": q["close"][i], "h": q["high"][i] or q["close"][i], "l": q["low"][i] or q["close"][i]}
                    for i in range(len(ts)) if q["close"][i] is not None}
        except Exception:
            continue
    return {}


def cargar():
    m = {k: _fetch(s) for k, s in E.SYMS.items()}
    for k, s in EXTRA.items():
        m[k] = _fetch(s)
    # fechas comunes de los 4 principales; extras se alinean con forward-fill
    com = sorted(set.intersection(*[set(m[k]) for k in E.SYMS]))
    arr = {k: tv._limpiar(np.array([m[k][d]["c"] for d in com])) for k in E.SYMS}
    for k in EXTRA:
        serie = []; ult = None
        for d in com:
            if d in m[k]:
                ult = m[k][d]["c"]
            serie.append(ult if ult is not None else np.nan)
        s = np.array(serie, dtype=float)
        # rellenar NaN iniciales con primer valor válido
        first = s[~np.isnan(s)][0] if np.any(~np.isnan(s)) else 1.0
        s[np.isnan(s)] = first
        arr[k] = tv._limpiar(s)
    hi = np.array([m["clp"][d]["h"] for d in com]); lo = np.array([m["clp"][d]["l"] for d in com])
    clp = arr["clp"]
    rng = np.minimum(hi - lo, clp * 0.08)
    return com, arr, clp + rng / 2, clp - rng / 2


def aplicar_actividad_stop(base, clp, hi, lo):
    """Pipeline estándar: sizing por actividad + stop 3×ATR sobre cualquier señal."""
    n = len(clp)
    r = (hi - lo) / clp * 100
    act = np.ones(n)
    for i in range(20, n):
        med = np.median(r[i - 20:i]) or 1
        act[i] = np.clip(r[i] / med, 0.5, 1.5)
    esc = base * act
    tr = hi - lo; atr = np.copy(tr)
    for i in range(1, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    out = np.zeros(n); cur, ent, fu = 0, None, False
    for i in range(n):
        s = esc[i]
        if np.sign(s) != np.sign(cur):
            cur = s; ent = clp[i] if s != 0 else None; fu = False
        if cur != 0 and ent and not fu:
            mv = (clp[i] - ent) / ent
            if (cur > 0 and mv <= -3 * atr[i] / ent) or (cur < 0 and mv >= 3 * atr[i] / ent):
                fu = True
        out[i] = 0 if fu else cur
    return out


def correr():
    com, arr, hi, lo = cargar()
    clp = arr["clp"]; n = len(clp)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    corte = next(i for i, d in enumerate(com) if d >= "2021-08-01")

    def sh(pos, a, b):
        _, _, s, _ = lab.metricas(lab.simular(pos, clp, a, b)); return s

    def rets(k):
        return np.concatenate([[0], np.diff(np.log(arr[k]))])

    rc, rd = rets("cobre"), rets("dxy")
    base_signal = E._combo_dxy(clp, arr, 2, 0.01)
    BASE = aplicar_actividad_stop(base_signal, clp, hi, lo)
    b_v, b_r = sh(BASE, 65, corte), sh(BASE, corte, n - 1)
    print(f"{n} días. BASE: viejo {b_v:+.2f} · reciente {b_r:+.2f}\n")
    print(f"{'Experimento':40} {'VIEJO':>7} {'RECIENTE':>9}  veredicto")
    print("-" * 76)

    def probar(nom, señal, pipeline=True):
        pos = aplicar_actividad_stop(señal, clp, hi, lo) if pipeline else señal
        v, r = sh(pos, 65, corte), sh(pos, corte, n - 1)
        gana = v > b_v + 0.03 and r > b_r + 0.03
        alguno = v > b_v + 0.03 or r > b_r + 0.03
        ver = "🏆 GANA ambos" if gana else ("mixto" if alguno else "peor/igual")
        print(f"{nom:40} {v:>+7.2f} {r:>+9.2f}  {ver}")
        return v, r

    # ---- 1) ASIMETRÍA: solo cortos / solo largos ----
    probar("Solo CORTOS de dólar (cobre sube)", np.where(base_signal < 0, base_signal, 0))
    probar("Solo LARGOS de dólar (cobre baja)", np.where(base_signal > 0, base_signal, 0))

    # ---- 2) MINERAS como señal adelantada (FCX+SCCO promedio, mom 2d) ----
    for k, nom in [("fcx", "Señal por FCX (minera) 2d"), ("scco", "Señal por SCCO 2d")]:
        rm = rets(k)
        sig = np.zeros(n)
        for i in range(2, n):
            mov = rm[i - 1:i + 1].sum()
            sig[i] = -1 if mov > 0.005 else (1 if mov < -0.005 else 0)
        probar(nom, sig)
    # mineras como CONFIRMACIÓN del combo
    rm = (rets("fcx") + rets("scco")) / 2
    conf = base_signal.copy()
    for i in range(2, n):
        m2 = rm[i - 1:i + 1].sum()
        if base_signal[i] < 0 and m2 < 0:
            conf[i] = 0
        elif base_signal[i] > 0 and m2 > 0:
            conf[i] = 0
    probar("Combo + confirmación mineras", conf)

    # ---- 3) RATIO COBRE/ORO (crecimiento global) ----
    ratio = arr["cobre"] / arr["oro"]
    rr = np.concatenate([[0], np.diff(np.log(ratio))])
    sig = np.zeros(n)
    for i in range(2, n):
        mov = rr[i - 1:i + 1].sum()
        sig[i] = -1 if mov > 0.005 else (1 if mov < -0.005 else 0)
    probar("Señal por ratio cobre/oro 2d", sig)

    # ---- 4) MULTI-TIMEFRAME: 2d Y 13d de acuerdo ----
    sig13 = np.zeros(n)
    for i in range(13, n):
        mov = rc[i - 12:i + 1].sum()
        sig13[i] = -1 if mov > 0.01 else (1 if mov < -0.01 else 0)
    mtf = np.where(np.sign(base_signal) == np.sign(sig13), base_signal, 0)
    probar("Multi-timeframe (2d ∧ 13d)", mtf)

    # ---- 5) HISTÉRESIS: entrar fuerte (1.5%), salir suave (0.2%) ----
    sig = np.zeros(n); s = 0
    for i in range(2, n):
        mov = rc[i - 1:i + 1].sum()
        if s == 0:
            s = -1 if mov > 0.015 else (1 if mov < -0.015 else 0)
        else:
            if abs(mov) < 0.002 or np.sign(-mov) != s and abs(mov) > 0.01:
                s = 0
        sig[i] = s
    # aplicar mismos filtros valor justo + dxy
    for i in range(n):
        z = E.valor_z(clp, arr, i) if i >= 60 else 0
        d5 = rd[max(0, i - 4):i + 1].sum()
        if sig[i] > 0 and (z > 1 or d5 < 0):
            sig[i] = 0
        elif sig[i] < 0 and (z < -1 or d5 > 0):
            sig[i] = 0
    probar("Histéresis (entrar 1.5%, salir suave)", sig)

    # ---- 6) ESTACIONALIDAD: día de la semana ----
    dows = np.array([dt.datetime.strptime(d, "%Y-%m-%d").weekday() for d in com])
    ret = np.concatenate([[0], np.diff(np.log(clp))])
    print("\n  Día de la semana (retorno medio del USD/CLP, pb/día):")
    for dow, nom in enumerate(["lun", "mar", "mié", "jue", "vie"]):
        m = dows[1:] == dow
        print(f"    {nom}: {ret[1:][m].mean()*10000:+.1f} pb  ({m.sum()} días)")

    # ---- 7) TAKE-PROFIT 2×ATR (revalidar en ambos) ----
    tr = hi - lo; atr = np.copy(tr)
    for i in range(1, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    r_act = (hi - lo) / clp * 100
    act = np.ones(n)
    for i in range(20, n):
        med = np.median(r_act[i - 20:i]) or 1
        act[i] = np.clip(r_act[i] / med, 0.5, 1.5)
    esc = base_signal * act
    out = np.zeros(n); cur, ent, fu = 0, None, False
    for i in range(n):
        s = esc[i]
        if np.sign(s) != np.sign(cur):
            cur = s; ent = clp[i] if s != 0 else None; fu = False
        if cur != 0 and ent and not fu:
            mv = (clp[i] - ent) / ent
            if (cur > 0 and mv <= -3 * atr[i] / ent) or (cur < 0 and mv >= 3 * atr[i] / ent):
                fu = True   # stop
            elif (cur > 0 and mv >= 2 * atr[i] / ent) or (cur < 0 and mv <= -2 * atr[i] / ent):
                fu = True   # take-profit
        out[i] = 0 if fu else cur
    print()
    probar("Stop 3×ATR + take-profit 2×ATR", out, pipeline=False)

    # ---- 8) PIRAMIDAR: doblar si el trade va ganando >1×ATR ----
    pir = np.copy(BASE)
    cur, ent = 0, None
    for i in range(n):
        s = np.sign(BASE[i])
        if s != np.sign(cur):
            cur = s; ent = clp[i] if s != 0 else None
        if cur != 0 and ent:
            mv = (clp[i] - ent) / ent
            if (cur > 0 and mv > atr[i] / ent) or (cur < 0 and mv < -atr[i] / ent):
                pir[i] = BASE[i] * 1.5   # agrandar ganadora
    probar("Piramidar ganadoras (+50%)", pir, pipeline=False)


if __name__ == "__main__":
    correr()
