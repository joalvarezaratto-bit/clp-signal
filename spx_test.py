"""
¿Agregar el S&P 500 a la estrategia GANADORA (combo v2 + actividad) la mejora?

Lógica: S&P sube = risk-on = capital a emergentes = peso fuerte = corto dólar.
Se prueba como CONFIRMACIÓN (no como ensemble, que ya vimos que empeora). OOS.
"""
import numpy as np
import lab
import oos_test as oos
import experimentos_variables as ev
import rentabilidad as R


def correr():
    com, arr = ev.cargar_todo()
    clp = arr["clp"]; n = len(clp)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    ini = 65; split = ini + int((n - ini) * 0.6)

    # estrategia ganadora: combo v2 × factor actividad
    rango = R._rango(com, clp)
    act = np.ones(n)
    for i in range(20, n):
        med = np.median(rango[i - 20:i]) or 1
        act[i] = np.clip(rango[i] / med, 0.5, 1.5)
    base = oos.est_combo_dxy(clp, arr, 3, 0.01) * act

    # momentum S&P (5d). +1: sube->corto USD (inverso)
    r_spx = np.concatenate([[0], np.diff(np.log(arr["spx"]))])
    spx_mom = np.array([r_spx[max(0, i - 4):i + 1].sum() for i in range(n)])

    def ev_(pos):
        _, _, sh1, _ = lab.metricas(lab.simular(pos, clp, ini, split))
        _, c, sh2, dd = lab.metricas(lab.simular(pos, clp, split, n - 1))
        return sh1, sh2, c, dd

    # variantes de agregar S&P
    exps = {}
    exps["GANADORA (sin S&P)"] = base

    # A) confirmación: no operar si el S&P contradice (S&P subiendo con posición larga USD)
    a = base.copy()
    for i in range(n):
        if base[i] > 0 and spx_mom[i] > 0.005:      # largo USD pero S&P sube (risk-on) -> contradice
            a[i] = 0
        elif base[i] < 0 and spx_mom[i] < -0.005:   # corto USD pero S&P cae (risk-off) -> contradice
            a[i] = 0
    exps["+ confirmación S&P"] = a

    # B) doble confirmación: agrandar si S&P TAMBIÉN confirma, achicar si no
    b = base.copy()
    for i in range(n):
        confirma = (base[i] > 0 and spx_mom[i] < 0) or (base[i] < 0 and spx_mom[i] > 0)
        b[i] = base[i] * (1.3 if confirma else 0.7)
    exps["+ sizing por S&P"] = b

    print(f"{n} días. TEST desde {com[split]}\n")
    print(f"{'Variante':28} {'Sh.train':>9} {'Sh.TEST':>8} {'CAGR':>7} {'DD':>7}  vs base")
    print("-" * 74)
    base_te = ev_(base)[1]
    for nom, pos in exps.items():
        sh1, sh2, c, dd = ev_(pos)
        marca = "🏆 mejora" if sh2 > base_te + 0.05 else ("= igual" if abs(sh2 - base_te) <= 0.05 else "peor")
        print(f"{nom:28} {sh1:>+9.2f} {sh2:>+8.2f} {c:>+6.1f}% {dd:>6.1f}%  {marca}")
    print(f"\nGanadora a superar: Sharpe test {base_te:+.2f}")


if __name__ == "__main__":
    correr()
