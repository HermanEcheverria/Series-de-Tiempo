"""Parte 1: limpieza (incisos 1.1 a 1.4). La funcion esta en limpiar.py."""

import pandas as pd

import datos
from limpiar import (CODIGOS_NO_PRODUCTO, PREFIJO_VALES, REGLAS, REGLA_CLIENTE,
                     limpiar_con_bitacora)
from salida import guardar_tabla, num

entero = num(0)
libras = num(2)


def tex(s):
    """Escapa los caracteres de LaTeX que aparecen en descripciones."""
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("£", r"\pounds{}")]:
        s = s.replace(a, b)
    return s


def inciso_1_1(df):
    sin_cliente = df["CustomerID"].isna()
    resumen = pd.DataFrame([
        {"Concepto": "Filas", "Valor": f"{len(df):,}"},
        {"Concepto": "Facturas distintas", "Valor": f"{df['InvoiceNo'].nunique():,}"},
        {"Concepto": "Productos distintos (StockCode)",
         "Valor": f"{df['StockCode'].nunique():,}"},
        {"Concepto": "Productos distintos sin distinguir mayúsculas",
         "Valor": f"{df['StockCode'].str.upper().nunique():,}"},
        {"Concepto": "Clientes distintos (CustomerID)",
         "Valor": f"{df['CustomerID'].nunique():,}"},
        {"Concepto": "Filas sin CustomerID",
         "Valor": f"{sin_cliente.sum():,} ({sin_cliente.mean():.1%})".replace("%", r"\%")},
        {"Concepto": "Primera factura", "Valor": str(df["InvoiceDate"].min())},
        {"Concepto": "Última factura", "Valor": str(df["InvoiceDate"].max())},
    ])
    guardar_tabla(resumen, "p1_resumen", alinear="lr")
    print(resumen.to_string())


def inciso_1_2(df):
    """Conteo de cada caso sobre los datos crudos (sin aplicar otras reglas)."""
    df = df.assign(Monto=df["Quantity"] * df["UnitPrice"])
    canc = df["InvoiceNo"].str.startswith("C")
    casos = [
        ("(a) Duplicados exactos", df.duplicated(keep="first"), "Se va"),
        ("(b) Facturas con prefijo C", canc, "Se queda (resta)"),
        ("(b) Facturas con prefijo A", df["InvoiceNo"].str.startswith("A"), "Se va"),
        ("(c) Códigos que no son productos",
         df["StockCode"].isin(CODIGOS_NO_PRODUCTO)
         | df["StockCode"].str.startswith(PREFIJO_VALES), "Se va"),
        ("(c) Códigos en minúscula", df["StockCode"].str.contains("[a-z]")
         & ~df["StockCode"].str.startswith(PREFIJO_VALES), "Se transforma"),
        ("(d) Precio negativo", df["UnitPrice"] < 0, "Se va"),
        ("(d) Precio cero", df["UnitPrice"] == 0, "Se va"),
        ("(e) Cantidad negativa sin ser cancelación",
         (df["Quantity"] < 0) & ~canc, "Se va"),
        ("(f) Sin CustomerID", df["CustomerID"].isna(),
         "Serie: se queda; RFM: se va"),
    ]
    tabla = pd.DataFrame([{
        "Caso": c, "Filas": int(m.sum()),
        "Monto neto (\\pounds)": df.loc[m, "Monto"].sum(), "Decisión": d}
        for c, m, d in casos])
    guardar_tabla(tabla, "p1_casos",
                  formatos={"Filas": entero, "Monto neto (\\pounds)": libras},
                  alinear="lrrl")
    print(tabla.to_string())

    # (c) Detalle de los codigos que no empiezan con cinco digitos.
    otros = df[~df["StockCode"].str.match(r"^\d{5}")]
    otros = otros.assign(Grupo=otros["StockCode"].where(
        ~otros["StockCode"].str.startswith("DCGS"), "DCGS...").where(
        ~otros["StockCode"].str.startswith(PREFIJO_VALES), "gift\\_0001\\_..."))
    det = otros.groupby("Grupo").agg(
        Filas=("Monto", "size"), Monto=("Monto", "sum"),
        Descripcion=("Description", lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
        Canc=("InvoiceNo", lambda s: s.str.startswith("C").mean()),
    ).sort_values("Filas", ascending=False).reset_index()
    no_prod = CODIGOS_NO_PRODUCTO | {"gift\\_0001\\_..."}
    det["Decisión"] = det["Grupo"].map(
        lambda g: "Se va" if g in no_prod else "Se queda")
    det["Descripcion"] = det["Descripcion"].map(lambda s: tex(s[:32]))
    det = det.rename(columns={"Grupo": "StockCode", "Descripcion": "Descripción",
                              "Monto": "Monto neto (\\pounds)",
                              "Canc": "\\% cancelación"})
    guardar_tabla(det, "p1_codigos",
                  formatos={"Filas": entero, "Monto neto (\\pounds)": libras,
                            "\\% cancelación": lambda v: f"{v:.0%}".replace("%", "")},
                  alinear="lrrlrl")
    print(det.to_string())

    # (e) Que dicen las descripciones de las cantidades negativas.
    neg = df[(df["Quantity"] < 0) & ~canc]
    print("(e) precio max", neg["UnitPrice"].max(), "con cliente",
          neg["CustomerID"].notna().sum())
    desc = neg["Description"].fillna("(vacía)").str.lower().value_counts()
    print(desc.head(15).to_string())
    return neg


def inciso_1_3(df):
    """Bitacora en orden, y la misma regla aplicada sola sobre los crudos."""
    limpio, bit = limpiar_con_bitacora(df, requiere_cliente=False)
    _, bit_rfm = limpiar_con_bitacora(df, requiere_cliente=True)
    crudo = df.assign(Monto=df["Quantity"] * df["UnitPrice"])
    aisladas = [int(m(crudo).sum()) for _, m in REGLAS + [REGLA_CLIENTE]]
    bit = bit_rfm.copy()
    bit["Filas si se aplica sola"] = aisladas
    bit = bit[["Orden", "Regla", "Filas si se aplica sola", "Filas eliminadas",
               "Monto neto (£)", "Monto absoluto (£)"]]
    bit.columns = ["Orden", "Regla", "Sola", "En orden",
                   "Monto neto (\\pounds)", "Monto absoluto (\\pounds)"]
    total = {"Orden": "", "Regla": "Total", "Sola": None,
             "En orden": bit["En orden"].sum(),
             "Monto neto (\\pounds)": bit["Monto neto (\\pounds)"].sum(),
             "Monto absoluto (\\pounds)": bit["Monto absoluto (\\pounds)"].sum()}
    bit = pd.concat([bit, pd.DataFrame([total])], ignore_index=True)
    guardar_tabla(bit, "p1_bitacora",
                  formatos={"Sola": lambda v: "" if pd.isna(v) else f"{int(v):,}",
                            "En orden": entero,
                            "Monto neto (\\pounds)": libras,
                            "Monto absoluto (\\pounds)": libras},
                  alinear="clrrrr")
    print(bit.to_string())
    print("serie: filas", len(limpio), "monto", limpio["Monto"].sum())
    return limpio


def inciso_1_4(df, limpio):
    crudo = df.assign(Monto=df["Quantity"] * df["UnitPrice"])
    top = crudo.reindex(crudo["Monto"].abs().sort_values(ascending=False).index).head(6)
    claves = set(zip(limpio["InvoiceNo"], limpio["StockCode"]))
    top["Sobrevive"] = [
        "Sí" if (f, c.upper()) in claves else "No"
        for f, c in zip(top["InvoiceNo"], top["StockCode"])]
    tabla = pd.DataFrame({
        "Factura": top["InvoiceNo"],
        "Fecha": top["InvoiceDate"].dt.strftime("%d/%m/%Y %H:%M"),
        "Cliente": top["CustomerID"],
        "Producto": top["Description"].map(lambda s: tex(s.title()[:30])),
        "Cantidad": top["Quantity"],
        "Precio": top["UnitPrice"],
        "Monto (\\pounds)": top["Monto"],
        "¿Sobrevive?": top["Sobrevive"],
    })
    guardar_tabla(tabla, "p1_top6",
                  formatos={"Cantidad": entero, "Precio": libras,
                            "Monto (\\pounds)": libras},
                  alinear="llllrrrc")
    print(tabla.to_string())

    # Neto por dia del cliente 15098 el 10 de junio: la serie lo veria asi.
    c = limpio[(limpio["CustomerID"] == "15098")
               & (limpio["InvoiceDate"].dt.date == pd.Timestamp("2011-06-10").date())]
    print(c[["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "Monto"]].to_string())
    print("neto 15098 el 10/06 tras limpieza:", c["Monto"].sum())


if __name__ == "__main__":
    df = datos.cargar_crudo()
    inciso_1_1(df)
    inciso_1_2(df)
    limpio = inciso_1_3(df)
    inciso_1_4(df, limpio)
