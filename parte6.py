"""Parte 6: analisis RFM (incisos 6.1 a 6.5)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import datos
from calendario import PRIMER_DIA, ULTIMO_DIA
from limpiar import limpiar
from salida import guardar_figura, guardar_tabla, num
from semillas import SEMILLA_RFM

CORTE = pd.Timestamp("2011-10-01")
FIN_VENTANA = ULTIMO_DIA  # 8 de diciembre; el 9 esta incompleto (inciso 2.4)
# Recencia de quien no tiene ninguna factura de venta antes del corte: su
# ultima compra fue antes del 1 de diciembre de 2010, asi que se le asigna un
# dia mas que a quien compro ese primer dia.
R_SIN_COMPRA = (CORTE - PRIMER_DIA).days + 1


def es_venta(df):
    return ~df["InvoiceNo"].str.startswith("C")


def calcular_rfm(limpio):
    antes = limpio[limpio["InvoiceDate"] < CORTE]
    ventas = antes[es_venta(antes)]
    clientes = pd.Index(sorted(antes["CustomerID"].unique()), name="CustomerID")
    ultima = ventas.groupby("CustomerID")["InvoiceDate"].max().dt.normalize()
    rfm = pd.DataFrame(index=clientes)
    rfm["R"] = (CORTE - ultima).dt.days.reindex(clientes).fillna(R_SIN_COMPRA).astype(int)
    rfm["F"] = ventas.groupby("CustomerID")["InvoiceNo"].nunique().reindex(clientes).fillna(0).astype(int)
    rfm["M"] = antes.groupby("CustomerID")["Monto"].sum().reindex(clientes)
    return rfm


def puntuar(rfm):
    """Quintiles 1-5 con el desempate aleatorio que fija el enunciado."""
    rng = np.random.default_rng(SEMILLA_RFM)
    barajado = rfm.sort_index().iloc[rng.permutation(len(rfm))]
    etiquetas = [1, 2, 3, 4, 5]
    p = pd.DataFrame(index=barajado.index)
    # En recencia, 5 es el mas reciente: se ordena -R.
    p["pR"] = pd.qcut((-barajado["R"]).rank(method="first"), 5, labels=etiquetas).astype(int)
    p["pF"] = pd.qcut(barajado["F"].rank(method="first"), 5, labels=etiquetas).astype(int)
    p["pM"] = pd.qcut(barajado["M"].rank(method="first"), 5, labels=etiquetas).astype(int)
    return rfm.join(p)


def inciso_6_1(rfm):
    m_no_pos = rfm["M"] <= 0
    resumen = {
        "clientes": len(rfm),
        "M<=0": int(m_no_pos.sum()),
        "M<0": int((rfm["M"] < 0).sum()),
        "M==0": int((rfm["M"] == 0).sum()),
        "M<=0 y F==0": int((m_no_pos & (rfm["F"] == 0)).sum()),
        "F==0": int((rfm["F"] == 0).sum()),
    }
    print(resumen)
    print(rfm.describe().round(1))
    desc = rfm[["R", "F", "M"]].describe(percentiles=[0.2, 0.4, 0.5, 0.6, 0.8]).T
    desc = desc[["mean", "min", "20%", "40%", "50%", "60%", "80%", "max"]]
    desc.insert(0, "Variable", ["R (días)", "F (facturas)", "M (\\pounds)"])
    desc.columns = ["Variable", "Media", "Mín.", "P20", "P40", "Mediana", "P60",
                    "P80", "Máx."]
    guardar_tabla(desc, "p6_rfm_resumen",
                  formatos={c: num(1) for c in desc.columns[1:]}, alinear="lrrrrrrrr")
    return resumen


def inciso_6_2(rfm):
    f = rfm["F"]
    frac = f.value_counts(normalize=True).sort_index()
    print("fraccion con F = 1, 2, 3:", frac.loc[[1, 2, 3]].round(3).to_dict(),
          "F = 0:", round(frac.get(0, 0), 4))
    try:
        pd.qcut(f, 5)
    except ValueError as e:
        print("pd.qcut(F, 5) sin desempatar falla:", e)
    # Puntaje de F por cada valor de F: cuantos puntajes distintos recibe.
    pt = rfm.groupby("F")["pF"].agg(lambda s: sorted(s.unique()))
    print(pt.head(6).to_dict())
    tabla = (rfm.groupby(["F", "pF"]).size().unstack(fill_value=0)
             .loc[:6])
    tabla.columns = [f"Puntaje {c}" for c in tabla.columns]
    tabla = tabla.reset_index().rename(columns={"F": "$F$"})
    tabla["$F$"] = tabla["$F$"].astype(str)
    guardar_tabla(tabla, "p6_puntaje_f", alinear="lrrrrr")
    print(tabla.to_string())


def compra_en_ventana(limpio, clientes):
    v = limpio[(limpio["InvoiceDate"] >= CORTE)
               & (limpio["InvoiceDate"] < FIN_VENTANA + pd.Timedelta(days=1))]
    compradores = set(v.loc[es_venta(v), "CustomerID"])
    return pd.Series([c in compradores for c in clientes], index=clientes).astype(int), v


def inciso_6_3(rfm, compra):
    rfm = rfm.assign(compra=compra)
    filas = []
    for p in range(1, 6):
        filas.append({"Puntaje": p,
                      "R": rfm.loc[rfm["pR"] == p, "compra"].mean(),
                      "F": rfm.loc[rfm["pF"] == p, "compra"].mean(),
                      "M": rfm.loc[rfm["pM"] == p, "compra"].mean()})
    t = pd.DataFrame(filas)
    print(t.round(3)); print("tasa global de compra:", rfm["compra"].mean())
    pct = lambda v: f"{v:.1%}".replace("%", "\\%")
    guardar_tabla(t, "p6_tasas", formatos={"R": pct, "F": pct, "M": pct},
                  alinear="crrr")

    suma = rfm["pR"] + rfm["pF"] + rfm["pM"]
    aucs = {
        "R (con signo invertido)": roc_auc_score(rfm["compra"], -rfm["R"]),
        "F": roc_auc_score(rfm["compra"], rfm["F"]),
        "M": roc_auc_score(rfm["compra"], rfm["M"]),
        "Suma de los tres puntajes": roc_auc_score(rfm["compra"], suma),
    }
    print({k: round(v, 4) for k, v in aucs.items()})
    ta = pd.DataFrame({"Clasificador": list(aucs), "AUC": list(aucs.values())})
    guardar_tabla(ta, "p6_auc", formatos={"AUC": num(3)}, alinear="lr")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(1, 6)
    for i, (dim, color) in enumerate([("R", "#4C72B0"), ("F", "#55A868"),
                                      ("M", "#C44E52")]):
        ax.bar(x + (i - 1) * 0.27, t[dim], width=0.27, color=color, label=dim)
    ax.axhline(rfm["compra"].mean(), color="gray", ls="--", lw=1,
               label=f"tasa global = {rfm['compra'].mean():.1%}")
    ax.set_xticks(x)
    ax.set_xlabel("puntaje")
    ax.set_ylabel("fracción que compró del 1/10 al 8/12")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    guardar_figura(fig, "p6_tasas")
    return aucs


def ventas_ventana_por_cliente(v):
    """Venta neta de cada cliente con CustomerID en la ventana de prueba."""
    return v.groupby("CustomerID")["Monto"].sum()


def inciso_6_4(rfm, v):
    campeones = (rfm["pR"] >= 4) & (rfm["pF"] >= 4) & (rfm["pM"] >= 4)
    venta = ventas_ventana_por_cliente(v)
    total = venta.sum()
    de_campeones = venta.reindex(rfm.index[campeones]).fillna(0).sum()
    res = {"campeones": int(campeones.sum()),
           "fraccion_clientes": campeones.mean(),
           "venta_total_ventana": total,
           "venta_campeones": de_campeones,
           "fraccion_venta": de_campeones / total,
           "tasa_compra_campeones": None}
    print(res)
    return campeones, res


def inciso_6_5(limpio, rfm, v, campeones):
    antes = limpio[limpio["InvoiceDate"] < CORTE]
    con_venta_previa = set(antes.loc[es_venta(antes), "CustomerID"])
    venta = ventas_ventana_por_cliente(v)
    nuevos = ~venta.index.isin(list(con_venta_previa))
    total = venta.sum()
    # Desglose de toda la venta de la ventana entre los grupos.
    camp = venta.index.isin(list(rfm.index[campeones]))
    grupos = pd.DataFrame([
        {"Grupo": "Campeones (6.4)", "Clientes con movimiento": int(camp.sum()),
         "Venta (\\pounds)": venta[camp].sum()},
        {"Grupo": "Otros clientes con historia", "Clientes con movimiento": int((~camp & ~nuevos).sum()),
         "Venta (\\pounds)": venta[~camp & ~nuevos].sum()},
        {"Grupo": "Clientes nuevos", "Clientes con movimiento": int(nuevos.sum()),
         "Venta (\\pounds)": venta[nuevos].sum()},
    ])
    grupos["\\% de la venta"] = grupos["Venta (\\pounds)"] / total
    grupos.loc[len(grupos)] = {"Grupo": "Total con CustomerID",
                               "Clientes con movimiento": len(venta),
                               "Venta (\\pounds)": total, "\\% de la venta": 1.0}
    pct = lambda x: f"{x:.1%}".replace("%", "\\%")
    guardar_tabla(grupos, "p6_grupos",
                  formatos={"Clientes con movimiento": num(0), "Venta (\\pounds)": num(0),
                            "\\% de la venta": pct}, alinear="lrrr")
    print(grupos.to_string())
    # Que parte de la venta total de la tienda (con y sin cliente) es eso.
    return grupos


if __name__ == "__main__":
    limpio = limpiar(datos.cargar_crudo(), requiere_cliente=True)
    rfm = puntuar(calcular_rfm(limpio))
    inciso_6_1(rfm)
    inciso_6_2(rfm)
    compra, v = compra_en_ventana(limpio, rfm.index)
    inciso_6_3(rfm, compra)
    campeones, _ = inciso_6_4(rfm, v)
    print("tasa de compra de campeones:", compra[campeones].mean())
    inciso_6_5(limpio, rfm, v, campeones)
