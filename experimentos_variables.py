"""
EXPERIMENTO: ¿otras variables agregan edge más allá del cobre?

Prueba momentum de varias variables como señal, y algunas como FILTRO del combo
base. Todo validado OUT-OF-SAMPLE (optimiza en train, mide en test nunca visto).
Si una variable solo ayuda in-sample -> sobreajuste, se descarta.
"""
import datetime as dt
import numpy as np
import requests
import lab
import oos_test as oos

UA = {"User-Agent": "Mozilla/5.0"}
EXTRA = {"tnx": "^TNX", "spx": "^GSPC", "vix": "^VIX", "oro": "GC=F"}


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


def cargar_todo():
    base = {k: lab._fetch(s) for k, s in lab.SYMS.items()}
    ext = {k: _fetch(s) for k, s in EXTRA.items()}
    todos = {**base, **ext}
    com = sorted(set.intersection(*[set(d) for d in todos.values()]))
    arr = {k: np.array([todos[k][d] for d in com], dtype=float) for k in todos}
    return com, arr


def mom(serie, win, thr, direccion):
    """Señal de momentum: direccion=+1 si (indicador sube -> largo USD), -1 si inverso."""
    n = len(serie); pos = np.zeros(n)
    r = np.concatenate([[0], np.diff(np.log(np.maximum(serie, 1e-9)))])
    for i in range(win, n):
        mv = r[i - win + 1:i + 1].sum()
        pos[i] = direccion * (1 if mv > thr else (-1 if mv < -thr else 0))
    return pos


def oos_sharpe(pos, clp, ini, split, n):
    _, _, sh_tr, _ = lab.metricas(lab.simular(pos, clp, ini, split))
    _, c_te, sh_te, dd = lab.metricas(lab.simular(pos, clp, split, n - 1))
    return sh_tr, sh_te, c_te, dd


def correr():
    com, arr = cargar_todo()
    clp = arr["clp"]; n = len(clp)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    ini = 65; split = 65 + int((n - 66) * 0.6)
    print(f"{n} días. TRAIN→{com[split]} | TEST {com[split]}→{com[-1]}\n")

    grid = [(w, t) for w in (3, 5, 8) for t in (0.005, 0.01, 0.02)]

    # --- 1) cada variable como señal standalone ---
    print("== Momentum de cada variable como señal (OUT-OF-SAMPLE) ==")
    print(f"{'Variable':22} {'Sharpe train':>12} {'Sharpe TEST':>12} {'CAGR test':>10}  veredicto")
    vars_dir = [("Cobre (inverso)", "cobre", -1), ("DXY (directo)", "dxy", +1),
                ("Real BRL (directo)", "brl", +1), ("Bono 10Y (directo)", "tnx", +1),
                ("S&P 500 (inverso)", "spx", -1), ("Oro (inverso)", "oro", -1)]
    for nombre, k, d in vars_dir:
        mejor, mejor_sh = None, -9
        for w, t in grid:
            sh_tr, _, _, _ = oos_sharpe(mom(arr[k], w, t, d), clp, ini, split, n)
            if sh_tr > mejor_sh:
                mejor_sh, mejor = sh_tr, (w, t)
        sh_tr, sh_te, c_te, _ = oos_sharpe(mom(arr[k], *mejor, d), clp, ini, split, n)
        vd = "SIRVE ✓" if sh_te > 0.7 else ("dudoso" if sh_te > 0.3 else "no sirve ✗")
        print(f"{nombre:22} {sh_tr:>+12.2f} {sh_te:>+12.2f} {c_te:>+9.1f}%  {vd}")

    # --- 2) base cobre-combo + confirmación / filtro de otra variable ---
    print("\n== Combo base + agregar otra variable (¿mejora el TEST?) ==")
    base = oos.est_combo(clp, arr, 3, 0.01)
    _, base_te, base_c, base_dd = oos_sharpe(base, clp, ini, split, n)
    print(f"{'BASE (cobre+valor justo)':34} Sharpe test {base_te:+.2f}  CAGR {base_c:+.1f}%  DD {base_dd:.1f}%")

    # 2a) filtro VIX: no operar si el VIX está muy alto (pánico)
    vix = arr["vix"]
    for umbral in (25, 30, 35):
        pos = base.copy()
        pos[vix > umbral] = 0
        _, sh_te, c_te, dd = oos_sharpe(pos, clp, ini, split, n)
        print(f"{'  + filtro VIX < '+str(umbral):34} Sharpe test {sh_te:+.2f}  CAGR {c_te:+.1f}%  DD {dd:.1f}%")

    # 2b) confirmación DXY: operar solo si el DXY confirma la dirección
    r_dxy = np.concatenate([[0], np.diff(np.log(arr["dxy"]))])
    pos = base.copy()
    for i in range(5, n):
        dxy_mov = r_dxy[i - 4:i + 1].sum()
        # base=+1 (largo USD) requiere DXY subiendo; base=-1 requiere DXY bajando
        if base[i] == 1 and dxy_mov < 0:
            pos[i] = 0
        elif base[i] == -1 and dxy_mov > 0:
            pos[i] = 0
    _, sh_te, c_te, dd = oos_sharpe(pos, clp, ini, split, n)
    print(f"{'  + confirmación DXY':34} Sharpe test {sh_te:+.2f}  CAGR {c_te:+.1f}%  DD {dd:.1f}%")

    print("\nSolo vale la pena agregar algo si SUBE el Sharpe TEST vs la base.")


if __name__ == "__main__":
    correr()
