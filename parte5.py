"""Parte 5: pronostico y fuga (incisos 5.1 a 5.5).

Cada modelo es una funcion pronostico(serie, origen, objetivos) que solo puede
mirar la serie hasta el origen (un domingo) y devuelve el pronostico de los
dias abiertos 'objetivos' de la semana siguiente. Las variables de M1, M2 y
la media movil se construyen una vez para todo el calendario; las de M1 y M2
solo usan valores anteriores al origen para cualquier objetivo de la semana
siguiente, la media movil no (inciso 5.4).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import KFold

from calendario import (venta_diaria, PRIMER_DIA, FIN_ENTRENAMIENTO,
                        ORIGEN_PRUEBA, FIN_PRUEBA)
from modelos import ajustar, pronosticar
from parte3 import media_movil_centrada
from salida import guardar_figura, guardar_tabla, num
from semillas import SEMILLAS_CV

ORIGENES_B = pd.date_range("2011-05-01", "2011-09-25", freq="7D")  # domingos
ORIGENES_C = pd.date_range(ORIGEN_PRUEBA, "2011-12-04", freq="7D")
COLORES = {"B1": "#8C8C8C", "B2": "#DD8452", "M1": "#55A868", "M2": "#8172B3",
           "S1": "#4C72B0", "S2": "#C44E52", "M2+MA": "#937860"}


# ---------------------------------------------------------------- variables

def rezago_mismo_dia(serie, fecha, k):
    """y_{t-k}; si ese dia esta cerrado, el mismo dia de la semana abierto mas
    reciente antes de el. NaN si no hay ninguno en la serie."""
    d = fecha - pd.Timedelta(days=k)
    while d >= PRIMER_DIA:
        if pd.notna(serie.get(d, np.nan)):
            return serie[d]
        d -= pd.Timedelta(days=7)
    return np.nan


def construir_variables(serie):
    """Una fila por dia abierto con todo lo que usan M1, M2 y M2+MA."""
    abiertos = serie.dropna()
    fechas = abiertos.index
    X = pd.DataFrame(index=fechas)
    X["y"] = abiertos.values
    X["t"] = (fechas - PRIMER_DIA).days
    for d in sorted(set(fechas.dayofweek)):
        X[f"dia{d}"] = (fechas.dayofweek == d).astype(float)
    X["rez7"] = [rezago_mismo_dia(serie, f, 7) for f in fechas]
    X["rez14"] = [rezago_mismo_dia(serie, f, 14) for f in fechas]
    X["ma7"] = media_movil_centrada(serie).reindex(fechas).values
    return X


DIAS_COLS = None  # se llena en main con las columnas dia*


def columnas(modelo):
    if modelo == "M1":
        return ["t"] + DIAS_COLS
    if modelo == "M2":
        return DIAS_COLS + ["rez7", "rez14"]
    if modelo == "M2+MA":
        return DIAS_COLS + ["rez7", "rez14", "ma7"]
    raise ValueError(modelo)


def ols(Xtr, ytr, Xte):
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ beta


# ------------------------------------------------------------------ modelos

def pron_regresion(modelo, V):
    def f(serie, origen, objetivos):
        cols = columnas(modelo)
        tr = V.loc[:origen, cols + ["y"]].dropna()
        return ols(tr[cols].values, tr["y"].values, V.loc[objetivos, cols].values)
    return f


def pron_b1(serie, origen, objetivos):
    return np.full(len(objetivos), serie.loc[:origen].mean())


def pron_b2(serie, origen, objetivos):
    hist = serie.loc[:origen]
    salida = []
    for f in objetivos:
        d = f - pd.Timedelta(days=7)
        while d > origen or pd.isna(hist.get(d, np.nan)):
            d -= pd.Timedelta(days=7)
        salida.append(hist[d])
    return np.array(salida)


CONVERGENCIA = {}


def pron_sarima(nombre):
    def f(serie, origen, objetivos):
        hist = serie.loc[:origen].dropna()
        res = ajustar(nombre, hist.values)
        ok = bool(res.mle_retvals["converged"])
        metodo = "lbfgs"
        if not ok:
            # Si L-BFGS se detiene sin converger, se reintenta con BFGS y se
            # deja registro de los dos ajustes (inciso 5.1).
            previo = res
            res = ajustar(nombre, hist.values, method="bfgs")
            ok, metodo = bool(res.mle_retvals["converged"]), "bfgs"
            print(f"{nombre} {origen.date()}: L-BFGS warnflag "
                  f"{previo.mle_retvals.get('warnflag')}, params "
                  f"{previo.params.round(4)} llf {previo.llf:.4f} -> BFGS "
                  f"{'converge' if ok else 'NO converge'}, params "
                  f"{res.params.round(4)} llf {res.llf:.4f}; pronostico "
                  f"{np.exp(previo.forecast(6)).round(0)} -> "
                  f"{np.exp(res.forecast(6)).round(0)}")
        CONVERGENCIA.setdefault(nombre, []).append((origen, ok, metodo))
        # Horizonte = posicion de cada objetivo entre los dias abiertos
        # posteriores al origen (los cerrados se conocen de antemano).
        abiertos_despues = serie.loc[origen + pd.Timedelta(days=1):].dropna().index
        h = np.array([abiertos_despues.get_loc(o) + 1 for o in objetivos])
        return pronosticar(res, int(h.max()))[h - 1]
    return f


# --------------------------------------------------------------- evaluacion

def objetivos_de(serie, origen, ultimo):
    semana = serie.loc[origen + pd.Timedelta(days=1):
                       min(origen + pd.Timedelta(days=7), ultimo)]
    return semana.dropna().index


def origen_movil(serie, modelos, origenes, ultimo):
    """Devuelve un DataFrame largo: fecha, origen, modelo, real, pronostico."""
    filas = []
    for o in origenes:
        obj = objetivos_de(serie, o, ultimo)
        if len(obj) == 0:
            continue
        for nombre, f in modelos.items():
            p = f(serie, o, obj)
            for fecha, real, pr in zip(obj, serie[obj].values, p):
                filas.append({"fecha": fecha, "origen": o, "modelo": nombre,
                              "real": real, "pron": pr})
    return pd.DataFrame(filas)


def errores(df):
    e = df["real"] - df["pron"]
    return pd.Series({"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
                      "n": len(e)})


def validacion_cruzada(V, modelo):
    """5 folds aleatorios sobre los dias abiertos del entrenamiento, con cada
    una de las 10 semillas. Devuelve media y desviacion de MAE y RMSE."""
    ent = V.loc[:FIN_ENTRENAMIENTO]
    cols = None if modelo == "B1" else columnas(modelo)
    ent = ent.dropna(subset=["y"] + (cols or []))
    res = []
    for semilla in SEMILLAS_CV:
        kf = KFold(n_splits=5, shuffle=True, random_state=semilla)
        e = np.empty(len(ent))
        for itr, ite in kf.split(ent):
            if modelo == "B1":
                p = np.full(len(ite), ent["y"].values[itr].mean())
            else:
                p = ols(ent[cols].values[itr], ent["y"].values[itr],
                        ent[cols].values[ite])
            e[ite] = ent["y"].values[ite] - p
        res.append((np.abs(e).mean(), np.sqrt((e ** 2).mean())))
    res = np.array(res)
    return {"MAE": res[:, 0].mean(), "MAE_sd": res[:, 0].std(ddof=1),
            "RMSE": res[:, 1].mean(), "RMSE_sd": res[:, 1].std(ddof=1),
            "n": len(ent)}


# ----------------------------------------------------------------- incisos

def tabla_principal(serie, V, modelos):
    cv = {m: validacion_cruzada(V, m) for m in ("B1", "M1", "M2")}
    pb = origen_movil(serie, modelos, ORIGENES_B, FIN_ENTRENAMIENTO)
    pc = origen_movil(serie, modelos, ORIGENES_C, FIN_PRUEBA)
    eb = pb.groupby("modelo").apply(errores, include_groups=False)
    ec = pc.groupby("modelo").apply(errores, include_groups=False)
    print("dias evaluados (b):", eb["n"].iloc[0], " (c):", ec["n"].iloc[0])
    filas = []
    for m in modelos:
        fila = {"Modelo": m}
        if m in cv:
            fila["(a) MAE"] = f"{cv[m]['MAE']:,.0f} $\\pm$ {cv[m]['MAE_sd']:,.0f}"
            fila["(a) RMSE"] = f"{cv[m]['RMSE']:,.0f} $\\pm$ {cv[m]['RMSE_sd']:,.0f}"
        else:
            fila["(a) MAE"] = fila["(a) RMSE"] = "---"
        fila["(b) MAE"] = f"{eb.loc[m, 'MAE']:,.0f}"
        fila["(b) RMSE"] = f"{eb.loc[m, 'RMSE']:,.0f}"
        fila["(c) MAE"] = f"{ec.loc[m, 'MAE']:,.0f}"
        fila["(c) RMSE"] = f"{ec.loc[m, 'RMSE']:,.0f}"
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    guardar_tabla(tabla, "p5_errores", alinear="lcccccc")
    print(tabla.to_string())
    print({m: {k: round(v, 1) for k, v in d.items()} for m, d in cv.items()})
    print(eb.round(0)); print(ec.round(0))
    return pb, pc, eb, ec, cv


def grafica_prueba(serie, pc):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    real = serie.loc["2011-10-03":FIN_PRUEBA].dropna()
    for ax, grupo in zip(axes, [("B1", "B2", "M1"), ("M2", "S1", "S2")]):
        ax.plot(real.index, real / 1000, color="black", lw=1.2, marker=".",
                ms=4, label="real")
        for m in grupo:
            d = pc[pc["modelo"] == m].set_index("fecha")["pron"]
            ax.plot(d.index, d / 1000, color=COLORES[m], lw=1.1, marker=".",
                    ms=3, label=m)
        for o in ORIGENES_C:
            ax.axvline(o, color="gray", lw=0.4, ls=":")
        ax.set_ylabel("miles de £")
        ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axes[-1].set_xlabel("domingos de origen (líneas punteadas)")
    guardar_figura(fig, "p5_pronosticos_prueba")


def error_por_semana(pc):
    """MAE por semana de prueba y por modelo, para ver cuando gana cada uno."""
    t = pc.assign(ae=(pc["real"] - pc["pron"]).abs()).pivot_table(
        index="origen", columns="modelo", values="ae", aggfunc="mean")
    print((t / 1000).round(1))
    sesgo = pc.assign(e=pc["real"] - pc["pron"]).groupby("modelo")["e"].mean()
    print("sesgo medio (real - pron):", sesgo.round(0).to_dict())
    orden = ["B1", "B2", "M1", "M2", "S1", "S2"]
    tabla = (t[orden] / 1000).reset_index()
    tabla["origen"] = tabla["origen"].dt.strftime("%d/%m")
    mejor = t[orden].idxmin(axis=1).values
    tabla["Mejor"] = mejor
    tabla = tabla.rename(columns={"origen": "Origen"})
    guardar_tabla(tabla, "p5_semanas",
                  formatos={m: num(1) for m in orden}, alinear="lrrrrrrc")
    sesgo_t = pd.DataFrame([{"": "Sesgo medio (real $-$ pronóstico)",
                             **{m: sesgo[m] for m in orden}}])
    guardar_tabla(sesgo_t, "p5_sesgo", formatos={m: num(0) for m in orden},
                  alinear="lrrrrrr")
    return t, sesgo


def nivel_de_cada_esquema(pb, pc, eb, ec):
    """MAE relativo a la venta promedio de los dias evaluados."""
    nb = pb.drop_duplicates("fecha")["real"].mean()
    nc = pc.drop_duplicates("fecha")["real"].mean()
    print(f"venta media evaluada (b) {nb:,.0f}  (c) {nc:,.0f}")
    print("MAE/nivel (b):", (eb["MAE"] / nb).round(3).to_dict())
    print("MAE/nivel (c):", (ec["MAE"] / nc).round(3).to_dict())


def inciso_5_4(serie, V, pb):
    cv = validacion_cruzada(V, "M2+MA")
    modelos = {"M2+MA": pron_regresion("M2+MA", V)}
    pbm = origen_movil(serie, modelos, ORIGENES_B, FIN_ENTRENAMIENTO)
    eb = errores(pbm)
    base = errores(pb[pb["modelo"] == "M2"])
    cv_m2 = validacion_cruzada(V, "M2")
    tabla = pd.DataFrame([
        {"Modelo": "M2", "(a) MAE": cv_m2["MAE"], "(a) RMSE": cv_m2["RMSE"],
         "(b) MAE": base["MAE"], "(b) RMSE": base["RMSE"]},
        {"Modelo": "M2 + media móvil centrada", "(a) MAE": cv["MAE"],
         "(a) RMSE": cv["RMSE"], "(b) MAE": eb["MAE"], "(b) RMSE": eb["RMSE"]},
    ])
    guardar_tabla(tabla, "p5_trampa", formatos={c: num(0) for c in tabla.columns[1:]},
                  alinear="lrrrr")
    print(tabla.to_string())
    # Coeficiente de la media movil en el ajuste con todo el entrenamiento.
    cols = columnas("M2+MA")
    tr = V.loc[:FIN_ENTRENAMIENTO, cols + ["y"]].dropna()
    beta, *_ = np.linalg.lstsq(tr[cols].values, tr["y"].values, rcond=None)
    print("coeficientes M2+MA:", dict(zip(cols, beta.round(3))))


if __name__ == "__main__":
    serie = venta_diaria()
    V = construir_variables(serie)
    DIAS_COLS = [c for c in V.columns if c.startswith("dia")]
    modelos = {
        "B1": pron_b1, "B2": pron_b2,
        "M1": pron_regresion("M1", V), "M2": pron_regresion("M2", V),
        "S1": pron_sarima("S1"), "S2": pron_sarima("S2"),
    }
    # El ejemplo del enunciado: el 9 de mayo, y_{t-7} es el 18 de abril.
    print("rezago 7 del 9/5:", V.loc["2011-05-09", "rez7"],
          "venta del 18/4:", serie["2011-04-18"])
    assert V.loc["2011-05-09", "rez7"] == serie["2011-04-18"]
    pb, pc, eb, ec, cv = tabla_principal(serie, V, modelos)
    for nombre, lista in CONVERGENCIA.items():
        ok = sum(c for _, c, _ in lista)
        bfgs = [o.date() for o, _, m in lista if m == "bfgs"]
        print(f"{nombre}: convergio en {ok} de {len(lista)} origenes; "
              f"con BFGS: {bfgs}")
    grafica_prueba(serie, pc)
    error_por_semana(pc)
    nivel_de_cada_esquema(pb, pc, eb, ec)
    inciso_5_4(serie, V, pb)
