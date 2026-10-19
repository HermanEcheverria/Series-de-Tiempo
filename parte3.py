"""Parte 3: tendencia, estacionalidad y autocorrelacion (incisos 3.1 a 3.3)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from calendario import venta_diaria, entrenamiento
from salida import guardar_figura, guardar_tabla, num
from series import acf, ljung_box, graficar_acf

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def media_movil_centrada(serie):
    """Media movil centrada de 7 dias con los cerrados rellenos de cero.

    MA_t = (y_{t-3} + ... + y_{t+3}) / 7. Los tres primeros y tres ultimos
    dias no tienen ventana completa y quedan NaN.
    """
    return serie.fillna(0).rolling(7, center=True).mean().rename("MA7")


def inciso_3_1(serie, ma):
    fig, ax = plt.subplots(figsize=(10, 4))
    abiertos = serie.dropna()
    ax.plot(abiertos.index, abiertos / 1000, lw=0.6, color="#9DB4D6",
            marker=".", ms=2.5, label="venta neta diaria (días abiertos)")
    ax.plot(ma.index, ma / 1000, color="#C44E52", lw=1.8,
            label="media móvil centrada de 7 días")
    ax.set_ylabel("miles de £")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.legend(loc="upper left", frameon=False)
    guardar_figura(fig, "p3_media_movil")

    # Resumen por mes para describir la tendencia.
    mensual = ma.resample("MS").mean()
    print((mensual / 1000).round(1).to_string())
    return mensual


def inciso_3_2(serie, ma):
    ent = entrenamiento(pd.DataFrame({"y": serie, "ma": ma})).dropna()
    ent["cociente"] = ent["y"] / ent["ma"]
    ent["dia"] = ent.index.dayofweek
    indices = ent.groupby("dia")["cociente"].agg(["mean", "count", "std"])
    indices = indices.reindex(range(7))
    print(indices)
    media_seis = indices["mean"].dropna().mean()
    print("promedio de los seis indices:", media_seis)

    tabla = pd.DataFrame({
        "Día": DIAS,
        "Índice estacional": indices["mean"],
        "Días": indices["count"].fillna(0).astype(int),
        "Desv. est. del cociente": indices["std"],
    })
    tabla.loc[5, ["Índice estacional", "Desv. est. del cociente"]] = np.nan
    fmt = lambda v: "---" if pd.isna(v) else f"{v:.3f}"
    guardar_tabla(tabla, "p3_indices",
                  formatos={"Índice estacional": fmt,
                            "Desv. est. del cociente": fmt},
                  alinear="lrrr")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    abiertos = indices["mean"].dropna()
    barras = ax.bar([DIAS[i] for i in abiertos.index], abiertos.values,
                    color="#4C72B0")
    ax.bar_label(barras, fmt="%.2f", padding=2, fontsize=9)
    ax.axhline(1, color="gray", ls="--", lw=1, label="índice = 1")
    ax.axhline(media_seis, color="#C44E52", ls=":", lw=1.2,
               label=f"promedio de los seis = {media_seis:.3f}")
    ax.set_ylabel("$y_t / MA_t$ promedio")
    ax.set_title("El sábado no aparece: la tienda no abre", fontsize=9,
                 loc="left", color="gray")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_ylim(0, abiertos.max() * 1.2)
    guardar_figura(fig, "p3_indices")
    return indices, media_seis


def inciso_3_3(serie):
    y = entrenamiento(serie).dropna()
    n = len(y)
    r = acf(y.values, 21)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    fuera = graficar_acf(r, n, ax, "")
    guardar_figura(fig, "p3_acf")

    banda = 1.96 / np.sqrt(n)
    pico = int(np.argmax(r)) + 1
    q6, p6 = ljung_box(y.values, 6)
    q12, p12 = ljung_box(y.values, 12)
    print(f"n = {n}, banda = {banda:.4f}, fuera = {fuera}, pico en k = {pico}"
          f" (r = {r[pico - 1]:.3f}), r1 = {r[0]:.3f}")
    print("rezagos fuera:", [k + 1 for k in np.where(np.abs(r) > banda)[0]])
    print(f"Q(6) = {q6:.1f} (p = {p6:.2e}), Q(12) = {q12:.1f} (p = {p12:.2e})")

    # Cuantos dias calendario separan a dos dias abiertos a distancia 'pico'.
    fechas = y.index
    sep = (fechas[pico:] - fechas[:-pico]).days
    print("dias calendario a distancia", pico, ":",
          pd.Series(sep).value_counts().to_dict())

    tabla = pd.DataFrame({"$k$": np.arange(1, 22), "$r_k$": r,
                          "Fuera": np.where(np.abs(r) > banda, "sí", "")})
    # Dos bloques lado a lado para que la tabla no ocupe una pagina.
    izq, der = tabla.iloc[:11].reset_index(drop=True), tabla.iloc[11:].reset_index(drop=True)
    lado = pd.concat([izq, der.reindex(range(11))], axis=1)
    lado.columns = ["$k$", "$r_k$", "Fuera", "$k$ ", "$r_k$ ", "Fuera "]
    fmt_k = lambda v: "" if pd.isna(v) else f"{int(v)}"
    fmt_r = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    fmt_s = lambda v: "" if pd.isna(v) else v
    guardar_tabla(lado, "p3_acf",
                  formatos={"$k$": fmt_k, "$k$ ": fmt_k, "$r_k$": fmt_r,
                            "$r_k$ ": fmt_r, "Fuera": fmt_s, "Fuera ": fmt_s},
                  alinear="rrc|rrc")
    lb = pd.DataFrame([
        {"Estadístico": "$Q(6)$", "Valor": q6, "g.l.": 6, "Valor-$p$": p6},
        {"Estadístico": "$Q(12)$", "Valor": q12, "g.l.": 12, "Valor-$p$": p12},
    ])
    guardar_tabla(lb, "p3_ljungbox",
                  formatos={"Valor": num(1), "Valor-$p$": lambda v: f"{v:.1e}"},
                  alinear="lrrr")
    return r, n


if __name__ == "__main__":
    serie = venta_diaria()
    ma = media_movil_centrada(serie)
    inciso_3_1(serie, ma)
    inciso_3_2(serie, ma)
    inciso_3_3(serie)
