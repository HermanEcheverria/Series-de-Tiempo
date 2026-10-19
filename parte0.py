"""Parte 0: procesos simulados (incisos 0.2 a 0.5)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

from salida import guardar_figura, guardar_tabla, num
from semillas import SEMILLAS_AR1, SEMILLAS_SIN_MEMORIA, SEMILLA_GRAFICAS
from series import acf, ljung_box, ols_ar1

N = 300
QUEMA = 200
REPLICAS = 5000
PHIS = [0.0, 0.5, 0.9, 1.0]
BANDA = 1.96 / np.sqrt(N)


def simular_ar1(phi, replicas, rng, c=0.0, sigma=1.0):
    """Matriz (replicas x N) de un AR(1) que arranca en Y_0 = 0.

    Se simulan QUEMA + N pasos y se descartan los primeros QUEMA. Con
    |phi| < 1 el efecto de Y_0 decae como phi^t y a los 200 pasos ya no
    existe; con phi = 1 no decae nunca.
    """
    eps = rng.normal(0.0, sigma, size=(replicas, QUEMA + N))
    y = np.empty_like(eps)
    previo = np.zeros(replicas)
    for t in range(QUEMA + N):
        previo = c + phi * previo + eps[:, t]
        y[:, t] = previo
    return y[:, QUEMA:]


def media_semanal(t):
    return np.where(t % 7 == 6, -1.5, 0.25)


def simular_sin_memoria(tipo, replicas, rng):
    """Y_t = m_t + e_t con e_t iid N(0,1), t = 1..N."""
    t = np.arange(1, N + 1)
    m = 2 * t / N if tipo == "tendencia" else media_semanal(t)
    return m + rng.normal(size=(replicas, N))


def media_y_ee(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def resumir(y, rezagos):
    """Promedios (con su error estandar) de las cantidades de las tablas."""
    r = acf(y, max(rezagos))
    fila = {}
    for k in rezagos:
        fila[f"r{k}"] = media_y_ee(r[:, k - 1])
    fila["rechazo"] = media_y_ee((np.abs(r[:, 0]) > BANDA).astype(float))
    fila["phi_ols"] = media_y_ee(ols_ar1(y))
    return fila


def pm(par, d=3):
    """Formato 'media (ee)' para la tabla."""
    m, ee = par
    return f"{m:.{d}f} ({ee:.{d}f})"


def grafica_realizaciones():
    rng = np.random.default_rng(SEMILLA_GRAFICAS)
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    for ax, phi in zip(axes, PHIS):
        y = simular_ar1(phi, 1, rng)[0]
        ax.plot(np.arange(1, N + 1), y, lw=0.9)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(rf"$\phi = {phi:g}$", loc="left")
    axes[-1].set_xlabel("$t$")
    fig.tight_layout()
    guardar_figura(fig, "p0_realizaciones_ar1")

    rng = np.random.default_rng(SEMILLA_GRAFICAS + 1)
    t = np.arange(1, N + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    for ax, tipo, m in [(axes[0], "tendencia", 2 * t / N),
                        (axes[1], "semanal", media_semanal(t))]:
        ax.plot(t, simular_sin_memoria(tipo, 1, rng)[0], lw=0.8)
        ax.plot(t, m, color="#C44E52", lw=1.5, label="$m_t$")
        ax.set_title(f"(a) tendencia lineal" if tipo == "tendencia"
                     else "(b) patrón semanal", loc="left")
        ax.legend(loc="upper left", frameon=False)
    axes[-1].set_xlabel("$t$")
    fig.tight_layout()
    guardar_figura(fig, "p0_realizaciones_sin_memoria")


def tabla_ar1():
    filas = []
    for phi in PHIS:
        rng = np.random.default_rng(SEMILLAS_AR1[phi])
        s = resumir(simular_ar1(phi, REPLICAS, rng), [1, 5])
        filas.append({"$\\phi$": f"{phi:g}",
                      "$\\bar r_1$": pm(s["r1"]),
                      "$\\bar r_5$": pm(s["r5"]),
                      "$\\phi^5$": f"{phi**5:.3f}",
                      "Rechazo $|r_1|>1.96/\\sqrt{n}$": pm(s["rechazo"]),
                      "$\\bar{\\hat\\phi}_{\\text{OLS}}$": pm(s["phi_ols"])})
        print(phi, {k: tuple(round(x, 4) for x in v) for k, v in s.items()})
    df = pd.DataFrame(filas)
    guardar_tabla(df, "p0_tabla_ar1", alinear="cccccc")
    return df


def tabla_sin_memoria():
    filas = []
    procesos = [("Ruido blanco ($\\phi=0$)", None),
                ("(a) Tendencia", "tendencia"),
                ("(b) Semanal", "semanal")]
    for nombre, tipo in procesos:
        if tipo is None:
            rng = np.random.default_rng(SEMILLAS_AR1[0.0])
            y = simular_ar1(0.0, REPLICAS, rng)
        else:
            rng = np.random.default_rng(SEMILLAS_SIN_MEMORIA[tipo])
            y = simular_sin_memoria(tipo, REPLICAS, rng)
        s = resumir(y, [1, 5, 7])
        q10, p10 = ljung_box(y, 10)
        q6, p6 = ljung_box(y, 6)
        lb10 = media_y_ee((p10 < 0.05).astype(float))
        lb6 = media_y_ee((p6 < 0.05).astype(float))
        filas.append({"Proceso": nombre,
                      "$\\bar r_1$": pm(s["r1"]),
                      "$\\bar r_5$": pm(s["r5"]),
                      "$\\bar r_7$": pm(s["r7"]),
                      "Rechazo $r_1$": pm(s["rechazo"]),
                      "$\\bar{\\hat\\phi}_{\\text{OLS}}$": pm(s["phi_ols"]),
                      "LB $m=10$": pm(lb10),
                      "LB $m=6$": pm(lb6),
                      "_y": y})
        print(nombre, {k: tuple(round(x, 4) for x in v) for k, v in s.items()},
              "LB10", lb10, "LB6", lb6)
    df = pd.DataFrame(filas)
    cols04 = ["Proceso", "$\\bar r_1$", "$\\bar r_5$", "$\\bar r_7$",
              "Rechazo $r_1$", "$\\bar{\\hat\\phi}_{\\text{OLS}}$"]
    guardar_tabla(df[cols04], "p0_tabla_sin_memoria", alinear="lccccc")
    cols05 = ["Proceso", "Rechazo $r_1$", "LB $m=10$", "LB $m=6$"]
    guardar_tabla(df[cols05], "p0_tabla_ljungbox", alinear="lccc")
    return df


def verificar_ljung_box(df):
    """Compara Q(10) propio contra statsmodels en la primera replica."""
    for _, fila in df.iterrows():
        y = fila["_y"][0]
        q_propio, p_propio = ljung_box(y, 10)
        sm = acorr_ljungbox(y, lags=[10])
        q_sm, p_sm = sm["lb_stat"].iloc[0], sm["lb_pvalue"].iloc[0]
        print(f"{fila['Proceso']}: Q propio {q_propio:.6f}  statsmodels "
              f"{q_sm:.6f}  p {p_propio:.6f} vs {p_sm:.6f}")
        assert np.isclose(q_propio, q_sm) and np.isclose(p_propio, p_sm)


if __name__ == "__main__":
    grafica_realizaciones()
    tabla_ar1()
    df = tabla_sin_memoria()
    verificar_ljung_box(df)
