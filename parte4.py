"""Parte 4: del AR(1) al SARIMA (incisos 4.1 a 4.5).

Todo se hace sobre los dias abiertos del entrenamiento, una fila por dia.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from calendario import venta_diaria, entrenamiento
from modelos import ajustar, pronosticar, S
from salida import guardar_figura, guardar_tabla, num
from series import acf, ljung_box, graficar_acf

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
NLAGS = 21


def indicadoras(fechas):
    """Una columna 0/1 por cada dia de la semana que aparece (sin intercepto)."""
    dias = sorted(set(fechas.dayofweek))
    return np.column_stack([(fechas.dayofweek == d).astype(float) for d in dias]), dias


def lb_tabla(nombre, x, gl_menos):
    filas = []
    for m in (6, 12):
        q, p = ljung_box(x, m, gl=m - gl_menos)
        filas.append({"Serie": nombre, "$m$": m, "g.l.": m - gl_menos,
                      "$Q(m)$": q, "Valor-$p$": p})
    return filas


def inciso_4_1(y):
    X, dias = indicadoras(y.index)
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    e = y.values - X @ beta
    print("coeficientes:", dict(zip([DIAS[d] for d in dias], beta.round(0))))
    r_y, r_e = acf(y.values, NLAGS), acf(e, NLAGS)
    print("r_e:", r_e.round(3))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    graficar_acf(r_y, len(y), axes[0], "(a) venta, inciso 3.3")
    graficar_acf(r_e, len(e), axes[1], "(b) residuos $e_t$ de las indicadoras")
    for ax in axes:
        ax.set_xticks(range(1, NLAGS + 1, 2))
    guardar_figura(fig, "p4_acf_indicadoras")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(y.index, e / 1000, lw=0.8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("$e_t$ (miles de £)")
    guardar_figura(fig, "p4_residuos_indicadoras")

    coef = pd.DataFrame({"Día": [DIAS[d] for d in dias], "Coeficiente (\\pounds)": beta})
    guardar_tabla(coef, "p4_coeficientes", formatos={"Coeficiente (\\pounds)": num(0)},
                  alinear="lr")
    return e, r_e


def inciso_4_2(y, e):
    x, z = e[:-1], e[1:]
    A = np.column_stack([np.ones_like(x), x])
    (c, phi), *_ = np.linalg.lstsq(A, z, rcond=None)
    u = z - c - phi * x
    print(f"AR(1) sobre residuos: c = {c:.2f}, phi = {phi:.4f}")
    r_u = acf(u, NLAGS)
    banda = 1.96 / np.sqrt(len(u))
    print("u fuera:", [k + 1 for k in np.where(np.abs(r_u) > banda)[0]], r_u.round(3))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    graficar_acf(r_u, len(u), ax, "")
    guardar_figura(fig, "p4_acf_ar1")
    filas = lb_tabla("$e_t$ (indicadoras)", e, 0) + lb_tabla("$u_t$ (AR(1) sobre $e_t$)", u, 1)
    return phi, u, filas


def inciso_4_3(y):
    resultados = {}
    filas = []
    for nombre in ("S1", "S2"):
        for maxiter in (50, 500):
            res = ajustar(nombre, y.values, maxiter=maxiter)
            ok = res.mle_retvals["converged"]
            print(nombre, maxiter, ok, res.mle_retvals.get("iterations"),
                  dict(zip(res.param_names, res.params.round(4))), round(res.llf, 2))
            fila = {"Modelo": nombre, "maxiter": maxiter,
                    "¿Convergió?": "sí" if ok else "no",
                    "Iteraciones": res.mle_retvals.get("iterations")}
            for p, v in zip(res.param_names, res.params):
                fila[p] = v
            fila["log-verosimilitud"] = res.llf
            fila["AIC"] = res.aic
            filas.append(fila)
            resultados[(nombre, maxiter)] = res
    tabla = pd.DataFrame(filas)
    nombres = {"intercept": "$c$", "ar.L1": "$\\phi$", "ar.S.L6": "$\\Phi$",
               "ma.L1": "$\\theta$", "ma.S.L6": "$\\Theta$", "sigma2": "$\\sigma^2$"}
    tabla = tabla[["Modelo", "maxiter", "¿Convergió?", "Iteraciones"]
                  + list(nombres) + ["log-verosimilitud", "AIC"]]
    tabla = tabla.rename(columns=nombres)
    cols_num = [c for c in tabla.columns if c.startswith("$")] + ["log-verosimilitud", "AIC"]
    fmt = lambda v: "---" if pd.isna(v) else f"{v:.3f}"
    guardar_tabla(tabla, "p4_sarima", formatos={c: fmt for c in cols_num},
                  alinear="lrcr" + "r" * (len(tabla.columns) - 4))
    return resultados[("S1", 500)], resultados[("S2", 500)]


def inciso_4_4(s1, s2):
    r1 = s1.resid
    r2 = s2.resid[S + 1:]   # (1-B)(1-B^6) se come 7 observaciones
    filas = lb_tabla("Residuos de S1", r1, 2) + lb_tabla("Residuos de S2 (sin los 7 primeros)", r2, 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    graficar_acf(acf(r1, NLAGS), len(r1), axes[0], "(a) residuos de S1")
    graficar_acf(acf(r2, NLAGS), len(r2), axes[1], "(b) residuos de S2")
    for ax in axes:
        ax.set_xticks(range(1, NLAGS + 1, 2))
    guardar_figura(fig, "p4_acf_sarima")
    print("primeros residuos S2:", s2.resid[:9].round(2))
    return filas


def inciso_4_5(y, s1, s2, pasos=120):
    f1, f2 = pronosticar(s1, pasos), pronosticar(s2, pasos)
    p = s1.params
    media_log = p[0] / ((1 - p[1]) * (1 - p[2]))
    print("media de largo plazo S1 (libras):", np.exp(media_log))
    print("S1 ultimos pronosticos:", f1[-6:].round(0))
    print("S2 ultimos pronosticos:", f2[-6:].round(0), "primeros:", f2[:6].round(0))
    hist = 72
    t_hist = np.arange(-hist + 1, 1)
    t_f = np.arange(1, pasos + 1)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t_hist, y.values[-hist:] / 1000, color="black", lw=0.9,
            label="venta (últimos 72 días abiertos del entrenamiento)")
    ax.plot(t_f, f1 / 1000, color="#4C72B0", lw=1.3, label="S1 (estacionario)")
    ax.plot(t_f, f2 / 1000, color="#C44E52", lw=1.3, label="S2 (con diferencias)")
    ax.axhline(np.exp(media_log) / 1000, color="#4C72B0", ls=":", lw=1,
               label="media de largo plazo de S1")
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("días abiertos desde el 30 de septiembre de 2011")
    ax.set_ylabel("miles de £")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    guardar_figura(fig, "p4_horizonte")
    return np.exp(media_log)


if __name__ == "__main__":
    y = entrenamiento(venta_diaria()).dropna()
    e, r_e = inciso_4_1(y)
    phi, u, filas_lb = inciso_4_2(y, e)
    s1, s2 = inciso_4_3(y)
    filas_lb += inciso_4_4(s1, s2)
    lb = pd.DataFrame(filas_lb)
    print(lb.to_string())
    guardar_tabla(lb, "p4_ljungbox",
                  formatos={"$Q(m)$": num(1), "Valor-$p$": lambda v: f"{v:.3g}"},
                  alinear="lrrrr")
    inciso_4_5(y, s1, s2)
