"""Limpieza de Online Retail.

limpiar(df) aplica las reglas en el orden de REGLAS y devuelve el DataFrame
limpio. limpiar_con_bitacora(df) hace lo mismo y ademas devuelve la bitacora:
una fila por regla con las filas que elimino y el monto que representaban.

Que se queda:
  - ventas normales de mercancia;
  - cancelaciones (InvoiceNo con C), como lineas negativas que restan el dia
    en que ocurren, tal como define el enunciado la venta neta diaria;
  - descuentos (D) y ajustes manuales (M): corrigen el precio de mercancia
    vendida. Quitarlos deja ventas fantasma; por ejemplo, la factura 556444
    cobro 60 canastas a 649.50 y la correccion C556445 es una linea M.
"""

import pandas as pd

# Codigos que no son mercancia: envio, comisiones, cargos bancarios,
# muestras y ajustes contables. Los DCGS... (dotcom gift shop) y PADS si son
# productos, aunque no empiecen con cinco digitos.
CODIGOS_NO_PRODUCTO = {
    "POST",          # envio
    "DOT",           # envio de pedidos dotcom
    "C2",            # transporte (carriage)
    "AMAZONFEE",     # comision de Amazon
    "BANK CHARGES",  # cargos bancarios
    "CRUK",          # comision de Cancer Research UK
    "S",             # muestras
    "B",             # ajuste de deuda incobrable
}
PREFIJO_VALES = "gift_0001"  # vales de regalo: dinero adelantado, no mercancia


def _es_cancelacion(df):
    return df["InvoiceNo"].str.startswith("C")


# Cada regla es (nombre, funcion que marca las filas a eliminar). El orden
# importa: una fila que cumple dos reglas se cuenta en la primera.
REGLAS = [
    ("Duplicados exactos",
     lambda df: df.duplicated(keep="first")),
    ("Facturas con prefijo A (ajuste de deuda)",
     lambda df: df["InvoiceNo"].str.startswith("A")),
    ("Códigos que no son productos",
     lambda df: df["StockCode"].isin(CODIGOS_NO_PRODUCTO)
     | df["StockCode"].str.startswith(PREFIJO_VALES)),
    ("Precio igual a cero o negativo",
     lambda df: df["UnitPrice"] <= 0),
    ("Cantidad negativa sin ser cancelación",
     lambda df: (df["Quantity"] < 0) & ~_es_cancelacion(df)),
]
REGLA_CLIENTE = ("Sin CustomerID", lambda df: df["CustomerID"].isna())


def _agregar_monto(df):
    df = df.copy()
    df["Monto"] = df["Quantity"] * df["UnitPrice"]
    return df


def _normalizar_codigos(df):
    """117 codigos aparecen tambien en minuscula (85123a junto a 85123A): son
    el mismo producto capturado distinto. Se unifican al final, para que la
    regla de duplicados exactos compare las filas tal como vienen."""
    df = df.copy()
    df["StockCode"] = df["StockCode"].str.upper()
    return df


def limpiar_con_bitacora(df, requiere_cliente=False):
    """Devuelve (df limpio, bitacora).

    requiere_cliente=False es la limpieza para la serie de ventas: una venta
    sin CustomerID sigue siendo dinero que entro a la tienda.
    requiere_cliente=True agrega una ultima regla que quita las filas sin
    CustomerID, para el analisis RFM (Parte 6), donde hay que saber quien
    compro.
    """
    df = _agregar_monto(df)
    reglas = REGLAS + ([REGLA_CLIENTE] if requiere_cliente else [])
    filas = []
    for orden, (nombre, marcar) in enumerate(reglas, start=1):
        mascara = marcar(df)
        filas.append({
            "Orden": orden,
            "Regla": nombre,
            "Filas eliminadas": int(mascara.sum()),
            "Monto neto (£)": df.loc[mascara, "Monto"].sum(),
            "Monto absoluto (£)": df.loc[mascara, "Monto"].abs().sum(),
        })
        df = df.loc[~mascara]
    return _normalizar_codigos(df).reset_index(drop=True), pd.DataFrame(filas)


def limpiar(df, requiere_cliente=False):
    """DataFrame limpio; ver limpiar_con_bitacora."""
    return limpiar_con_bitacora(df, requiere_cliente)[0]
