"""Parte 2: el calendario (incisos 2.1 a 2.4)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import datos
from calendario import (CIERRES, ULTIMO_DIA_DATOS, en_calendario,
                        venta_por_dia, FIN_ENTRENAMIENTO, INICIO_PRUEBA)
from limpiar import limpiar
from salida import guardar_figura, guardar_tabla, num

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def inciso_2_1_y_2_2(por_dia):
    completa = en_calendario(por_dia, hasta=ULTIMO_DIA_DATOS)
    n_cal, n_abiertos = len(completa), int(completa.notna().sum())
    print(f"dias calendario {n_cal}, con transaccion {n_abiertos}")

    cerrados = completa.index[completa.isna()]
    sabados = cerrados[cerrados.dayofweek == 5]
    otros = cerrados[cerrados.dayofweek != 5]
    print(f"cerrados {len(cerrados)}: sabados {len(sabados)}, otros {len(otros)}")
    assert set(otros.strftime("%Y-%m-%d")) == set(CIERRES), "faltan cierres"
    # Ningun sabado tiene ventas en todo el año.
    assert not (completa.index[completa.notna()].dayofweek == 5).any()

    tabla = pd.DataFrame([{
        "Fecha": d.strftime("%d/%m/%Y"),
        "Día": DIAS[d.dayofweek],
        "Tipo": CIERRES[d.strftime("%Y-%m-%d")][0],
        "Motivo": CIERRES[d.strftime("%Y-%m-%d")][1],
    } for d in otros])
    guardar_tabla(tabla, "p2_cierres", alinear="llll")

    resumen = pd.DataFrame([
        {"Concepto": "Días calendario (1/12/2010 al 9/12/2011)", "Días": n_cal},
        {"Concepto": "Días con al menos una transacción", "Días": n_abiertos},
        {"Concepto": "Días sin transacciones", "Días": len(cerrados)},
        {"Concepto": "\\quad sábados", "Días": len(sabados)},
        {"Concepto": "\\quad cierre de Navidad (sin sábados)",
         "Días": int((tabla["Tipo"] == "Cierre de Navidad").sum())},
        {"Concepto": "\\quad feriados de 2011",
         "Días": int((tabla["Tipo"] != "Cierre de Navidad").sum())},
    ])
    guardar_tabla(resumen, "p2_resumen", alinear="lr")
    print(resumen.to_string())
    return completa


def grafica_serie(completa):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    abiertos = completa.dropna()
    ax.plot(abiertos.index, abiertos / 1000, lw=0.8, color="#4C72B0",
            marker=".", ms=3, label="venta neta diaria (días abiertos)")
    otros = [pd.Timestamp(d) for d in CIERRES]
    ax.scatter(otros, np.zeros(len(otros)), marker="x", color="#C44E52",
               zorder=3, label="cierre entre semana")
    ax.axvline(FIN_ENTRENAMIENTO + pd.Timedelta(hours=12), color="gray",
               ls="--", lw=1)
    ax.text(INICIO_PRUEBA, ax.get_ylim()[1] * 0.95, " prueba", color="gray",
            va="top")
    ax.set_ylabel("miles de £")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.legend(loc="upper left", frameon=False)
    guardar_figura(fig, "p2_serie_diaria")


def inciso_2_4(limpio, por_dia):
    """Que tan completo esta el 9 de diciembre: compara la hora de corte."""
    ultimo = limpio[limpio["InvoiceDate"].dt.normalize() == ULTIMO_DIA_DATOS]
    corte = ultimo["InvoiceDate"].max()
    print("ultima factura del 9/12:", corte, "facturas", ultimo["InvoiceNo"].nunique())

    # Para cada dia abierto anterior, fraccion de su venta (bruta, sin
    # cancelaciones, para que sea una fraccion entre 0 y 1) antes de la hora
    # de corte, y la hora de su ultima factura.
    previos = limpio[limpio["InvoiceDate"].dt.normalize() < ULTIMO_DIA_DATOS]
    previos = previos[previos["Monto"] > 0]
    dia = previos["InvoiceDate"].dt.normalize()
    minuto = previos["InvoiceDate"].dt.hour * 60 + previos["InvoiceDate"].dt.minute
    corte_min = corte.hour * 60 + corte.minute
    frac = (previos["Monto"].where(minuto <= corte_min, 0).groupby(dia).sum()
            / previos["Monto"].groupby(dia).sum())
    ultima_hora = previos.groupby(dia)["InvoiceDate"].max()
    ult_min = ultima_hora.dt.hour * 60 + ultima_hora.dt.minute
    viernes = frac.index.dayofweek == 4
    f_med = frac[viernes].median()
    estimado = por_dia[ULTIMO_DIA_DATOS] / f_med
    resumen = {
        "corte": corte.strftime("%H:%M"),
        "frac_mediana_viernes": f_med,
        "frac_mediana_todos": frac.median(),
        "dias_que_cierran_antes_corte": int((ult_min <= corte_min).sum()),
        "venta_9_dic": por_dia[ULTIMO_DIA_DATOS],
        "mediana_viernes_nov_dic": por_dia[(por_dia.index >= "2011-11-01")
                                           & (por_dia.index.dayofweek == 4)
                                           & (por_dia.index < ULTIMO_DIA_DATOS)].median(),
        "estimado_dia_completo": estimado,
    }
    print(resumen)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(frac[viernes], bins=20, color="#4C72B0", alpha=0.85)
    ax.axvline(f_med, color="#C44E52", lw=1.5,
               label=f"mediana = {f_med:.0%}")
    ax.set_xlabel(f"fracción de la venta del día registrada antes de las {resumen['corte']}")
    ax.set_ylabel("viernes")
    ax.legend(frameon=False)
    guardar_figura(fig, "p2_fraccion_corte")
    return resumen


if __name__ == "__main__":
    limpio = limpiar(datos.cargar_crudo())
    por_dia = venta_por_dia(limpio)
    completa = inciso_2_1_y_2_2(por_dia)
    grafica_serie(completa.loc[:"2011-12-08"])
    inciso_2_4(limpio, por_dia)
