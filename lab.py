"""
LABORATORIO de estrategias USD/CLP — probar varias y ver si alguna es rentable.

Cada estrategia decide una posición (-1 corto dólar / 0 fuera / +1 largo dólar)
sin mirar el futuro. Un motor común las simula con el MISMO riesgo (volatilidad
objetivo 10% anual, costos, tope de apalancamiento) para poder compararlas justo.

Reporta por estrategia: retorno, CAGR, Sharpe, máx. caída, y 1ª vs 2ª mitad
(para detectar suerte). Guarda un gráfico de curvas de capital.

Correr:  python3 lab.py
Experimentar: agrega tu estrategia en ESTRATEGIAS y vuelve a correr.
"""
import math
import datetime as dt
import numpy as np
import requests

UA = {"User-Agent": "Mozilla/5.0"}
SYMS = {"clp": "USDCLP=X", "cobre": "HG=F", "dxy": "DX-Y.NYB", "brl": "USDBRL=X"}

TARGET_VOL = 0.10   # volatilidad anual objetivo (para comparar justo)
COST_BPS = 6        # costo por cambio de posición (spread USD/CLP)
MAX_LEV = 3.0


# ============================ datos ============================
def _fetch(sym, rng="5y"):
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1d", "range": rng}, headers=UA, timeout=30)
            res = r.json()["chart"]["result"][0]
            ts, q = res["timestamp"], res["indicators"]["quote"][0]
            out = {}
            for i in range(len(ts)):
                if q["close"][i] is None:
                    continue
                d = dt.datetime.utcfromtimestamp(ts[i]).strftime("%Y-%m-%d")
                out[d] = q["close"][i]
            return out
        except Exception:
            continue
    return {}


def cargar():
    m = {k: _fetch(s) for k, s in SYMS.items()}
    com = sorted(set(m["clp"]) & set(m["cobre"]) & set(m["dxy"]) & set(m["brl"]))
    arr = {k: np.array([m[k][d] for d in com], dtype=float) for k in m}
    return com, arr


# ============================ indicadores ============================
def sma(x, n, i):
    return x[i - n + 1:i + 1].mean() if i + 1 >= n else None


def rsi(x, i, n=14):
    if i < n:
        return 50.0
    d = np.diff(x[:i + 1]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ag, ap = up[:n].mean(), dn[:n].mean()
    for k in range(n, len(d)):
        ag = (ag * (n - 1) + up[k]) / n
        ap = (ap * (n - 1) + dn[k]) / n
    return 100.0 if ap == 0 else 100 - 100 / (1 + ag / ap)


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


# ============================ estrategias ============================
# Cada una devuelve un array de posiciones (-1/0/+1), sin mirar el futuro.
def est_meanrev(clp, arr):
    """Mean-reversion del valor justo: fade cuando el dólar se aparta >1σ."""
    n = len(clp); pos = np.zeros(n); s = 0
    for i in range(n):
        z = valor_z(clp, arr, i)
        if s == 0:
            s = -1 if z >= 1 else (1 if z <= -1 else 0)
        elif abs(z) <= 0.25:
            s = 0
        pos[i] = s
    return pos


def est_tendencia(clp, arr):
    """Seguir tendencia: largo dólar si sube (medias 20/50)."""
    n = len(clp); pos = np.zeros(n)
    for i in range(n):
        s20, s50 = sma(clp, 20, i), sma(clp, 50, i)
        if s20 and s50:
            pos[i] = 1 if clp[i] > s20 > s50 else (-1 if clp[i] < s20 < s50 else 0)
    return pos


def est_rsi(clp, arr):
    """RSI: comprar dólar en sobreventa, vender en sobrecompra."""
    n = len(clp); pos = np.zeros(n); s = 0
    for i in range(n):
        r = rsi(clp, i)
        if r <= 30:
            s = 1
        elif r >= 70:
            s = -1
        elif 45 <= r <= 55:
            s = 0
        pos[i] = s
    return pos


def est_cobre(clp, arr):
    """Contrarian al cobre: cobre sube -> corto dólar (relación inversa)."""
    n = len(clp); pos = np.zeros(n)
    r = np.concatenate([[0], np.diff(np.log(arr["cobre"]))])
    for i in range(5, n):
        mov = r[i - 4:i + 1].sum()
        pos[i] = -1 if mov > 0.01 else (1 if mov < -0.01 else 0)
    return pos


def est_carry(clp, arr):
    """Carry: mantener pesos siempre (corto dólar). Ilustra el riesgo de tendencia."""
    return np.full(len(clp), -1.0)


def est_buyhold(clp, arr):
    """Comprar y mantener el dólar."""
    return np.ones(len(clp))


ESTRATEGIAS = {
    "Mean-reversion (valor justo)": est_meanrev,
    "Tendencia (medias 20/50)": est_tendencia,
    "RSI sobrecompra/venta": est_rsi,
    "Contrarian al cobre": est_cobre,
    "Carry (mantener pesos)": est_carry,
    "Comprar y mantener USD": est_buyhold,
}


# ============================ simulador común ============================
def simular(pos, clp, ini=65, fin=None, diff=None, markup=1.0):
    """Vol-targeting a ~10% vol anual. Costos por cambio + SWAP opcional.
    diff = array (tasa_Chile − tasa_EEUU) en % anual; markup = recargo del bróker."""
    fin = fin or len(clp) - 1
    ret = np.concatenate([[0], np.diff(np.log(clp))])
    equity = 1.0
    curva = [1.0]
    notional_prev = 0.0
    for i in range(ini, fin):
        vol = ret[max(0, i - 20):i].std() * math.sqrt(252)
        scale = min(MAX_LEV, TARGET_VOL / max(vol, 0.02))
        notional = pos[i] * scale
        equity -= equity * (COST_BPS / 10000 * abs(notional - notional_prev))
        equity *= (1 + notional * ret[i + 1])   # posición de hoy gana el retorno de mañana
        if diff is not None and notional != 0:   # swap/rollover del día
            carry = -notional * diff[i] / 100 / 365 - abs(notional) * markup / 100 / 365
            equity *= (1 + carry)
        notional_prev = notional
        curva.append(equity)
    return np.array(curva)


def metricas(curva):
    rets = np.diff(np.log(curva))
    dias = len(curva)
    total = (curva[-1] - 1) * 100
    cagr = (curva[-1] ** (252 / dias) - 1) * 100
    sharpe = rets.mean() / (rets.std() + 1e-9) * math.sqrt(252) if rets.std() > 0 else 0
    peak = np.maximum.accumulate(curva)
    dd = ((curva - peak) / peak).min() * 100
    return total, cagr, sharpe, dd


def correr():
    print("Descargando 5 años de datos...")
    com, arr = cargar()
    clp = arr["clp"]; n = len(clp)
    print(f"{n} días: {com[0]} → {com[-1]}\n")
    mid = 65 + (n - 66) // 2

    print(f"{'Estrategia':30} {'Retorno':>8} {'CAGR':>7} {'Sharpe':>7} {'MáxDD':>7}  {'1ª/2ª mitad':>14}")
    print("-" * 82)
    curvas = {}
    for nombre, fn in ESTRATEGIAS.items():
        pos = fn(clp, arr)
        curva = simular(pos, clp)
        t, c, sh, dd = metricas(curva)
        c1 = (simular(pos, clp, 65, mid)[-1] - 1) * 100
        c2 = (simular(pos, clp, mid, n - 1)[-1] - 1) * 100
        ambas = "✓" if c1 > 0 and c2 > 0 else " "
        print(f"{nombre:30} {t:+7.1f}% {c:+6.1f}% {sh:+7.2f} {dd:6.1f}%  {c1:+5.1f}/{c2:+5.1f} {ambas}")
        curvas[nombre] = curva
    print("\nSharpe > 0.5 y positivo en AMBAS mitades = candidato con edge real.")
    return com, curvas


if __name__ == "__main__":
    correr()
