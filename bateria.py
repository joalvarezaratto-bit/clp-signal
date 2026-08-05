"""
BATERÍA DE EXPERIMENTOS — probar muchas ideas y ver qué funciona (todo OOS).

Cada idea se mide en TEST (datos nunca vistos). Se ordenan por Sharpe test.
El objetivo: encontrar si algo supera al combo v2 base (Sharpe ~1.79).
"""
import numpy as np
import lab
import oos_test as oos
import experimentos_variables as ev
import analogos


def correr():
    com, arr = ev.cargar_todo()
    clp = arr["clp"]; n = len(clp)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4
    ini = 65; split = ini + int((n - ini) * 0.6)

    def ev_(pos):
        _, _, sh_tr, _ = lab.metricas(lab.simular(pos, clp, ini, split))
        _, c, sh, dd = lab.metricas(lab.simular(pos, clp, split, n - 1))
        return sh_tr, sh, c, dd

    def M(k, w, t, d):
        return ev.mom(arr[k], w, t, d)

    # señales individuales (con su relación conocida al USD/CLP)
    cobre = M("cobre", 5, 0.01, -1)
    dxy = M("dxy", 5, 0.01, +1)
    spx = M("spx", 5, 0.01, -1)
    oro = M("oro", 5, 0.01, -1)
    base = oos.est_combo_dxy(clp, arr, 3, 0.01)

    # señal de análogos (posición por lo que hicieron los días parecidos)
    F, aini = analogos.matriz(clp, arr)
    ana = np.zeros(n)
    for i in range(aini + 60, n):
        p = analogos.pronostico(clp, F, i, aini)
        if p and abs(p["media"]) > 0.15:
            ana[i] = 1 if p["media"] > 0 else -1

    exp = {}
    exp["Combo v2 (base)"] = base
    exp["Ensemble 4 señales (binario)"] = np.sign(cobre + dxy + spx + oro)
    exp["Ensemble 4 señales (convicción)"] = (cobre + dxy + spx + oro) / 4.0
    exp["Ensemble 3 (cobre+dxy+spx) conv."] = (cobre + dxy + spx) / 3.0
    exp["Combo + análogos (ambos concuerdan)"] = np.where(np.sign(base) == np.sign(ana), base, 0)
    # convicción: combo escalado por cuántos confirman (dxy, spx, oro, análogos)
    conf = (np.sign(base) == np.sign(dxy)).astype(float) + (np.sign(base) == np.sign(spx)).astype(float) \
        + (np.sign(base) == np.sign(oro)).astype(float) + (np.sign(base) == np.sign(ana)).astype(float)
    exp["Combo × convicción (0-4 confirman)"] = base * (conf / 4.0)
    exp["Ensemble convicción + análogos"] = ((cobre + dxy + spx + oro) / 4.0 + ana) / 2.0

    print(f"{n} días. TEST desde {com[split]}\n")
    print(f"{'Experimento':38} {'Sh.train':>8} {'Sh.TEST':>8} {'CAGR':>7} {'DD':>7}  vs base")
    print("-" * 82)
    base_te = ev_(base)[1]
    filas = []
    for nombre, pos in exp.items():
        sh_tr, sh_te, c, dd = ev_(pos)
        filas.append((nombre, sh_tr, sh_te, c, dd))
    for nombre, sh_tr, sh_te, c, dd in sorted(filas, key=lambda x: -x[2]):
        marca = "🏆" if sh_te > base_te + 0.05 else ("=" if abs(sh_te - base_te) <= 0.05 else "peor")
        print(f"{nombre:38} {sh_tr:>+8.2f} {sh_te:>+8.2f} {c:>+6.1f}% {dd:>6.1f}%  {marca}")
    print(f"\nBase a superar: Sharpe test {base_te:+.2f}")


if __name__ == "__main__":
    correr()
