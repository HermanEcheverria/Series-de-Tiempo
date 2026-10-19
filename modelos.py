"""Configuracion de los modelos SARIMA S1 y S2 (Parte 4), reusada en la Parte 5.

Los dos se ajustan sobre el logaritmo de la venta en los dias abiertos, una
fila por dia, con estacionalidad s = 6 (seis dias abiertos por semana).
"""

import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

S = 6
CONFIG = {
    "S1": dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, S), trend="c"),
    "S2": dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, S), trend="n"),
}
# Con el maxiter por defecto (50) S1 no converge (inciso 4.3); con 500 si.
MAXITER = 500


def ajustar(nombre, valores, maxiter=MAXITER):
    """Ajusta S1 o S2 sobre log(valores). valores es un arreglo, sin fechas."""
    modelo = SARIMAX(np.log(np.asarray(valores, dtype=float)), **CONFIG[nombre])
    with warnings.catch_warnings():
        # La convergencia se revisa explicitamente con mle_retvals.
        warnings.simplefilter("ignore", ConvergenceWarning)
        return modelo.fit(disp=False, maxiter=maxiter)


def pronosticar(resultado, pasos):
    """Pronostico en libras: exp() del pronostico en logaritmo, sin
    correccion de sesgo, como pide el enunciado."""
    return np.exp(resultado.forecast(pasos))
