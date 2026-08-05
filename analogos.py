"""
ANÁLOGOS HISTÓRICOS: "hoy se parece a estos días del pasado — ¿qué pasó después?"

Para cada día se arma un 'estado' (momentum cobre, momentum DXY, valor justo z,
RSI, momentum del propio USD/CLP). Dado el estado de HOY, se buscan los K días
más parecidos del PASADO (vecinos cercanos) y se mira qué hizo el USD/CLP en los
días siguientes → una distribución de lo que suele pasar en situaciones así.

Sin mirar el futuro: los vecinos y su resultado deben estar completamente en el
pasado (día j con j+H ≤ i). Se valida como estrategia out-of-sample.
"""
import datetime as dt
import numpy as np
import lab

H = 5   # horizonte: qué hizo el dólar en los siguientes 5 días


def features(clp, arr, i):
    """Estado del día i, en unidades comparables (sin mirar el futuro)."""
    rc = (np.log(arr["cobre"][i] / arr["cobre"][i - 3])) * 100 if i >= 3 else 0
    rd = (np.log(arr["dxy"][i] / arr["dxy"][i - 5])) * 100 if i >= 5 else 0
    z = lab.valor_z(clp, arr, i)
    rsi = lab.rsi(clp, i)
    rp = (np.log(clp[i] / clp[i - 5])) * 100 if i >= 5 else 0
    # escalado a ~unidad (valores típicos) para que ninguna variable domine
    return np.array([rc / 2.0, rd / 1.0, z / 1.0, (rsi - 50) / 12.0, rp / 1.5])


def matriz(clp, arr, ini=65):
    return np.array([features(clp, arr, i) for i in range(len(clp))]), ini


def vecinos(F, i, ini, k=30):
    """Índices de los k días más parecidos a i, todos con resultado ya conocido."""
    cand = np.arange(ini, i - H)          # solo pasado con forward completo
    if len(cand) < k:
        return cand
    d = np.linalg.norm(F[cand] - F[i], axis=1)
    return cand[np.argsort(d)[:k]]


def pronostico(clp, F, i, ini, k=30):
    """Distribución de lo que hizo el dólar tras días parecidos. dict o None."""
    v = vecinos(F, i, ini, k)
    if len(v) < 10:
        return None
    fwd = np.array([np.log(clp[j + H] / clp[j]) * 100 for j in v])
    return {"media": float(fwd.mean()), "prob_sube": float((fwd > 0).mean() * 100),
            "n": len(v), "vecinos": v, "fwd": fwd}


def correr():
    com, arr = lab.cargar()
    clp = arr["clp"]; n = len(clp)
    F, ini = matriz(clp, arr)
    lab.TARGET_VOL = 0.10; lab.MAX_LEV = 3; lab.COST_BPS = 4.4

    # estrategia: posición = signo de lo que hicieron los días parecidos
    pos = np.zeros(n)
    for i in range(ini + 60, n):
        p = pronostico(clp, F, i, ini)
        if p and abs(p["media"]) > 0.15:
            pos[i] = 1 if p["media"] > 0 else -1

    split = ini + int((n - ini) * 0.6)
    _, _, sh_tr, _ = lab.metricas(lab.simular(pos, clp, ini + 60, split))
    _, c_te, sh_te, dd = lab.metricas(lab.simular(pos, clp, split, n - 1))
    print("== ESTRATEGIA por análogos (vecinos cercanos) ==")
    print(f"  Sharpe train {sh_tr:+.2f}  →  Sharpe TEST {sh_te:+.2f}  (CAGR {c_te:+.1f}%, DD {dd:.1f}%)")
    print(f"  Veredicto: {'SIRVE ✓' if sh_te > 0.7 else ('dudoso' if sh_te > 0.3 else 'no sirve ✗')}\n")

    # diagnóstico de HOY
    p = pronostico(clp, F, n - 1, ini)
    if p:
        print(f"== HOY ({com[-1]}) se parece a {p['n']} días del pasado ==")
        print(f"  Tras situaciones así, el dólar en {H} días: media {p['media']:+.2f}%  "
              f"· subió el {p['prob_sube']:.0f}% de las veces")
        top = p["vecinos"][:6]
        print("  Días más parecidos:")
        for j in top:
            print(f"    {com[j]}: luego {np.log(clp[j+H]/clp[j])*100:+.1f}% en {H} días")
    return com, clp, F, ini


if __name__ == "__main__":
    correr()
