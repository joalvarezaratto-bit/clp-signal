"""
ESTRATEGIA FINAL (validada 17 años) — todo en un módulo limpio para producción.

  Momentum del cobre (2d) + filtro valor justo + confirmación DXY
  × sizing por actividad (rango diario) + stop-loss 3×ATR.

Validada out-of-sample 2009-2021: CAGR +9%, Sharpe 1,2. 17 años: Sharpe 1,49.
Datos: Yahoo (failover query1/query2), sin API key.
"""
import datetime as dt
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}
SYMS = {"clp": "USDCLP=X", "cobre": "HG=F", "dxy": "DX-Y.NYB", "brl": "USDBRL=X"}
STOP_ATR = 3.0


def _limpiar(x, cap=0.10):
    r = np.clip(np.diff(np.log(np.maximum(x, 1e-9))), -cap, cap)
    out = np.empty(len(x)); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] * np.exp(r[i - 1])
    return out


def _fetch(sym, rng="2y"):
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "range": rng}, headers=UA, timeout=25)
            res = r.json()["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            out = {}
            for i in range(len(ts)):
                if q["close"][i] is None:
                    continue
                d = dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d")
                out[d] = {"c": q["close"][i], "h": q["high"][i] or q["close"][i],
                          "l": q["low"][i] or q["close"][i]}
            return out
        except Exception:
            continue
    return {}


def cargar():
    m = {k: _fetch(s) for k, s in SYMS.items()}
    if not all(m.values()):
        return None
    com = sorted(set.intersection(*[set(d) for d in m.values()]))
    arr = {k: _limpiar(np.array([m[k][d]["c"] for d in com], dtype=float)) for k in m}
    hi = np.array([m["clp"][d]["h"] for d in com]); lo = np.array([m["clp"][d]["l"] for d in com])
    rng = np.minimum(hi - lo, arr["clp"] * 0.08)
    hi = arr["clp"] + rng / 2; lo = arr["clp"] - rng / 2
    return com, arr, hi, lo


# --------------------------- indicadores ---------------------------
def valor_z(clp, arr, i, win=60):
    if i < win:
        return 0.0
    y = clp[i - win + 1:i + 1]
    X = np.column_stack([arr["cobre"][i - win + 1:i + 1], arr["dxy"][i - win + 1:i + 1],
                         arr["brl"][i - win + 1:i + 1], np.ones(win)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return 0.0
    resid = y - X @ beta
    return 0.0 if resid.std() == 0 else float(resid[-1] / resid.std())


def _combo_dxy(clp, arr, w=2, t=0.01):
    """Cobre-momentum + filtro valor justo + confirmación DXY."""
    n = len(clp); pos = np.zeros(n); s = 0
    rc = np.concatenate([[0], np.diff(np.log(arr["cobre"]))])
    rd = np.concatenate([[0], np.diff(np.log(arr["dxy"]))])
    for i in range(n):
        mov = rc[i - w + 1:i + 1].sum() if i >= w else 0
        z = valor_z(clp, arr, i)
        base = -1 if mov > t else (1 if mov < -t else 0)   # cobre inverso
        if s == 0:
            s = base
        elif abs(z) <= 0.25:
            s = 0
        # filtro valor justo (no operar contra el fair value extremo)
        cand = s
        if cand > 0 and z > 1:
            cand = 0
        elif cand < 0 and z < -1:
            cand = 0
        # confirmación DXY
        dxy_mov = rd[i - 4:i + 1].sum() if i >= 5 else 0
        if cand > 0 and dxy_mov < 0:
            cand = 0
        elif cand < 0 and dxy_mov > 0:
            cand = 0
        pos[i] = cand
    return pos


def posicion(com, arr, hi, lo):
    """Posición final por día (con actividad y stop). Devuelve (pos, contexto_hoy)."""
    clp = arr["clp"]; n = len(clp)
    base = _combo_dxy(clp, arr)
    # sizing por actividad
    rango = (hi - lo) / clp * 100
    act = np.ones(n)
    for i in range(20, n):
        med = np.median(rango[i - 20:i]) or 1
        act[i] = np.clip(rango[i] / med, 0.5, 1.5)
    escalada = base * act
    # stop 3×ATR
    tr = hi - lo; atr = np.copy(tr)
    for i in range(1, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    out = np.zeros(n); cur, entry, fuera = 0, None, False
    for i in range(n):
        s = escalada[i]
        if np.sign(s) != np.sign(cur):
            cur = s; entry = clp[i] if s != 0 else None; fuera = False
        if cur != 0 and entry and not fuera:
            mv = (clp[i] - entry) / entry
            if (cur > 0 and mv <= -STOP_ATR * atr[i] / entry) or \
               (cur < 0 and mv >= STOP_ATR * atr[i] / entry):
                fuera = True
        out[i] = 0 if fuera else cur

    # contexto de hoy
    rc = np.concatenate([[0], np.diff(np.log(arr["cobre"]))])
    rd = np.concatenate([[0], np.diff(np.log(arr["dxy"]))])
    ctx = {"mom_cobre": rc[-2:].sum() * 100, "mom_dxy": rd[-5:].sum() * 100,
           "z": valor_z(clp, arr, n - 1), "act": act[-1],
           "vol": np.diff(np.log(clp))[-20:].std() * np.sqrt(252)}
    return out, ctx
