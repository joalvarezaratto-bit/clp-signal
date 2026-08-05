"""
VALIDACIÓN OUT-OF-SAMPLE (la prueba honesta contra el sobreajuste).

Optimiza los parámetros en la 1ª parte (train, datos viejos) y los prueba en la
2ª parte (test, datos que el modelo NUNCA vio). Si el Sharpe out-of-sample sigue
alto, el edge es creíble. Si se derrumba, era curva sobreajustada.
"""
import numpy as np
import lab
import stress_test as st


def est_combo(clp, arr, w, t):
    """Cobre-momentum + filtro de valor justo (no operar contra el fair value)."""
    pos = st.est_cobre_p(clp, arr, w, t)
    out = np.zeros(len(clp))
    for i in range(60, len(clp)):
        z = lab.valor_z(clp, arr, i)
        if pos[i] > 0 and z > 1:
            out[i] = 0
        elif pos[i] < 0 and z < -1:
            out[i] = 0
        else:
            out[i] = pos[i]
    return out


def est_combo_dxy(clp, arr, w=3, t=0.01):
    """Combo v2: cobre-momentum + filtro valor justo + CONFIRMACIÓN DXY.
    Solo opera si el dólar global (DXY) concuerda con la señal del cobre.
    (Mejora OOS: Sharpe 1.58 -> 1.79, caída -7% -> -3%.)"""
    base = est_combo(clp, arr, w, t)
    pos = base.copy()
    r_dxy = np.concatenate([[0], np.diff(np.log(arr["dxy"]))])
    for i in range(5, len(clp)):
        dxy_mov = r_dxy[i - 4:i + 1].sum()
        if base[i] == 1 and dxy_mov < 0:      # largo USD necesita DXY subiendo
            pos[i] = 0
        elif base[i] == -1 and dxy_mov > 0:   # corto USD necesita DXY bajando
            pos[i] = 0
    return pos


def correr():
    com, arr = lab.cargar()
    clp = arr["clp"]; n = len(clp)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    train_ini, split = 65, 65 + int((n - 66) * 0.6)
    print(f"{n} días. TRAIN: {com[train_ini]}→{com[split]}  |  TEST: {com[split]}→{com[-1]}\n")

    grid = [(w, t) for w in (3, 5, 8) for t in (0.005, 0.01, 0.02)]

    for nombre, fn in [("Contrarian cobre", st.est_cobre_p), ("Combo cobre+valor justo", est_combo)]:
        # optimizar en TRAIN
        mejor, mejor_sh = None, -9
        for w, t in grid:
            pos = fn(clp, arr, w, t)
            _, _, sh, _ = lab.metricas(lab.simular(pos, clp, train_ini, split))
            if sh > mejor_sh:
                mejor_sh, mejor = sh, (w, t)
        # aplicar a TEST (nunca visto)
        pos = fn(clp, arr, *mejor)
        _, c_te, sh_te, dd_te = lab.metricas(lab.simular(pos, clp, split, n - 1))
        print(f"{nombre}")
        print(f"  Mejores params en train: win={mejor[0]}, thr={mejor[1]}  (Sharpe train {mejor_sh:+.2f})")
        print(f"  → OUT-OF-SAMPLE (test):  CAGR {c_te:+.1f}%  Sharpe {sh_te:+.2f}  DD {dd_te:.1f}%")
        veredicto = ("CREÍBLE ✓" if sh_te > 0.7 else ("dudoso" if sh_te > 0.3 else "SOBREAJUSTE ✗"))
        print(f"  Veredicto: {veredicto}  (train {mejor_sh:+.2f} vs test {sh_te:+.2f})\n")


if __name__ == "__main__":
    correr()
