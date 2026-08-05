"""
TEST DE ESTRÉS de la estrategia ganadora (contrarian al cobre).

¿Su edge es REAL o sobreajustado? Un edge real:
  - aguanta un RANGO de parámetros (no solo un valor mágico)
  - funciona en VARIOS sub-períodos (no un solo año afortunado)
  - sobrevive a COSTOS más altos
Si solo brilla en un punto exacto, es curva sobreajustada -> no confiable.
"""
import numpy as np
import lab


def est_cobre_p(clp, arr, win, thr):
    """Contrarian al cobre parametrizado: ventana y umbral."""
    n = len(clp); pos = np.zeros(n)
    r = np.concatenate([[0], np.diff(np.log(arr["cobre"]))])
    for i in range(win, n):
        mov = r[i - win + 1:i + 1].sum()
        pos[i] = -1 if mov > thr else (1 if mov < -thr else 0)
    return pos


def correr():
    com, arr = lab.cargar()
    clp = arr["clp"]; n = len(clp)
    print(f"{n} días: {com[0]} → {com[-1]}\n")

    # 1) Sensibilidad a parámetros (Sharpe en cada combinación)
    print("== 1) Sensibilidad a parámetros (Sharpe) ==")
    wins = [3, 5, 8, 13]
    thrs = [0.005, 0.01, 0.02, 0.03]
    print("ventana\\umbral " + "  ".join(f"{t:>7.3f}" for t in thrs))
    sharpes = []
    for w in wins:
        fila = []
        for t in thrs:
            pos = est_cobre_p(clp, arr, w, t)
            _, _, sh, _ = lab.metricas(lab.simular(pos, clp))
            fila.append(sh); sharpes.append(sh)
            print(f"", end="")
        print(f"  win={w:>2}      " + "  ".join(f"{s:>+7.2f}" for s in fila))
    pos_frac = sum(1 for s in sharpes if s > 0.4) / len(sharpes) * 100
    print(f"\n  Celdas con Sharpe>0.4: {pos_frac:.0f}%  (alto = edge robusto, no un punto mágico)\n")

    # 2) Retorno por AÑO (con los parámetros base 5/0.01)
    print("== 2) Retorno por año (params base win=5, thr=0.01) ==")
    pos = est_cobre_p(clp, arr, 5, 0.01)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3
    años = {}
    for d in com:
        años.setdefault(d[:4], None)
    anios = sorted(set(d[:4] for d in com[65:]))
    for y in anios:
        idx = [i for i, d in enumerate(com) if d[:4] == y and i >= 65]
        if len(idx) < 30:
            continue
        cur = lab.simular(pos, clp, ini=idx[0], fin=idx[-1])
        print(f"  {y}: {(cur[-1]-1)*100:+6.1f}%")
    print("  (para confiar: positivo en la MAYORÍA de los años, no 1 solo)\n")

    # 3) Sensibilidad a costos
    print("== 3) Sensibilidad a costos ==")
    for cost in (3, 6, 12, 20):
        lab.COST_BPS = cost
        _, c, sh, dd = lab.metricas(lab.simular(pos, clp))
        print(f"  costo {cost:>2} pb/lado → CAGR {c:+.1f}%  Sharpe {sh:+.2f}  DD {dd:.1f}%")
    lab.COST_BPS = 6
    # turnover
    cambios = np.sum(np.abs(np.diff(pos[65:])) > 0)
    print(f"\n  Cambios de posición: {cambios} en {n-65} días (~{cambios/((n-65)/252):.0f}/año)")


if __name__ == "__main__":
    correr()
