"""Serie de venta neta diaria sobre el calendario completo.

venta_diaria() devuelve la serie de trabajo de las Partes 3 a 5: un valor por
dia calendario del 1 de diciembre de 2010 al 8 de diciembre de 2011, con los
dias cerrados como NaN. El 9 de diciembre se descarta porque los datos se
cortan a las 12:50 (inciso 2.4).
"""

import pandas as pd

import datos
from limpiar import limpiar

PRIMER_DIA = pd.Timestamp("2010-12-01")
ULTIMO_DIA_DATOS = pd.Timestamp("2011-12-09")   # incompleto, se descarta
ULTIMO_DIA = pd.Timestamp("2011-12-08")
FIN_ENTRENAMIENTO = pd.Timestamp("2011-09-30")
ORIGEN_PRUEBA = pd.Timestamp("2011-10-02")      # domingo: historia, no se evalua
INICIO_PRUEBA = pd.Timestamp("2011-10-03")
FIN_PRUEBA = ULTIMO_DIA

# Dias entre semana sin transacciones y su explicacion. Los feriados oficiales
# de Inglaterra y Gales en 2011 son los de gov.uk; Navidad 2010 cae en sabado
# y el 26 en domingo, asi que se trasladan al lunes 27 y martes 28; Año Nuevo
# 2011 cae en sabado y se traslada al lunes 3 de enero.
CIERRES = {
    "2010-12-24": ("Cierre de Navidad", "Nochebuena"),
    "2010-12-26": ("Cierre de Navidad", "Boxing Day (domingo)"),
    "2010-12-27": ("Cierre de Navidad", "Feriado: traslado de Navidad"),
    "2010-12-28": ("Cierre de Navidad", "Feriado: traslado de Boxing Day"),
    "2010-12-29": ("Cierre de Navidad", "Entre Navidad y Año Nuevo"),
    "2010-12-30": ("Cierre de Navidad", "Entre Navidad y Año Nuevo"),
    "2010-12-31": ("Cierre de Navidad", "Fin de año"),
    "2011-01-02": ("Cierre de Navidad", "Domingo antes del traslado"),
    "2011-01-03": ("Cierre de Navidad", "Feriado: traslado de Año Nuevo"),
    "2011-04-22": ("Feriado móvil (Pascua)", "Viernes Santo"),
    "2011-04-24": ("Feriado móvil (Pascua)", "Domingo de Pascua"),
    "2011-04-25": ("Feriado móvil (Pascua)", "Lunes de Pascua"),
    "2011-04-29": ("Feriado único", "Boda real (Guillermo y Catalina)"),
    "2011-05-02": ("Feriado de lunes", "Early May bank holiday"),
    "2011-05-30": ("Feriado de lunes", "Spring bank holiday"),
    "2011-08-29": ("Feriado de lunes", "Summer bank holiday"),
}


def venta_por_dia(limpio):
    """Suma de Monto por dia, solo en los dias con al menos una transaccion."""
    return limpio.groupby(limpio["InvoiceDate"].dt.normalize())["Monto"].sum()


def en_calendario(por_dia, hasta=ULTIMO_DIA):
    """Lleva la venta al calendario completo; los dias sin datos quedan NaN."""
    dias = pd.date_range(PRIMER_DIA, hasta, freq="D", name="Fecha")
    return por_dia.reindex(dias).rename("Venta")


def venta_diaria(df_crudo=None):
    """Serie de trabajo: calendario completo, cerrados como NaN, sin el 9/12."""
    df_crudo = datos.cargar_crudo() if df_crudo is None else df_crudo
    return en_calendario(venta_por_dia(limpiar(df_crudo)))


def entrenamiento(serie):
    return serie.loc[:FIN_ENTRENAMIENTO]


def prueba(serie):
    return serie.loc[INICIO_PRUEBA:FIN_PRUEBA]
