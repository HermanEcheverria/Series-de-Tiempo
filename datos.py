"""Carga del archivo original Online Retail.xlsx.

InvoiceNo, StockCode y CustomerID se leen como texto. Si se dejan a pandas:
  - InvoiceNo mezcla numeros (536365) con texto (C536379): la columna queda
    con 532,618 int y 9,291 str revueltos, y str.startswith('C') devuelve NaN
    en las 532,618 filas numericas en vez de False.
  - StockCode mezcla 487,036 codigos numericos (71053) con 54,873
    alfanumericos (85123A, POST): cualquier regla sobre el texto del codigo,
    como "empieza con cinco digitos", se salta las filas que son int.
  - CustomerID tiene faltantes, asi que pandas la convierte a float (17850.0):
    un identificador no es una cantidad y no debe tener decimales.
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
RUTA_XLSX = RAIZ / "datos" / "Online Retail.xlsx"
# Leer el .xlsx tarda cerca de un minuto; la primera lectura se guarda en
# pickle para que los scripts siguientes arranquen rapido.
RUTA_CACHE = RAIZ / "datos" / "online_retail.pkl"

COLUMNAS_TEXTO = {"InvoiceNo": str, "StockCode": str, "CustomerID": str}


def cargar_crudo(usar_cache=True):
    """Devuelve el DataFrame original, sin ninguna limpieza."""
    if usar_cache and RUTA_CACHE.exists():
        return pd.read_pickle(RUTA_CACHE)
    if not RUTA_XLSX.exists():
        raise FileNotFoundError(
            f"No se encontro {RUTA_XLSX}. Descarguen el archivo de "
            "https://archive.ics.uci.edu/dataset/352/online+retail "
            "y descomprimanlo en la carpeta datos/."
        )
    df = pd.read_excel(RUTA_XLSX, dtype=COLUMNAS_TEXTO)
    df["InvoiceNo"] = df["InvoiceNo"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip()
    df["CustomerID"] = df["CustomerID"].str.strip()
    df.to_pickle(RUTA_CACHE)
    return df
