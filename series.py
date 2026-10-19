"""Autocorrelacion y Ljung-Box implementadas a mano con numpy.

Son las mismas funciones en la Parte 0 (procesos simulados) y en las Partes 3
y 4 (datos reales). Ambas aceptan una serie (1-D) o una matriz con una serie
por fila (2-D), para que las 5,000 replicas de la Parte 0 se calculen de una
vez en lugar de con un ciclo.
"""

import numpy as np
from scipy import stats


def acf(y, nlags):
    """r_k para k = 1..nlags, sobre el ultimo eje.

    r_k = sum_{t=k+1}^{n} (y_t - ybar)(y_{t-k} - ybar) / sum_{t=1}^{n} (y_t - ybar)^2

    Se usa la media y el denominador de toda la serie (no los del tramo
    traslapado), igual que FPP2 y statsmodels. Por eso r_k esta sesgado hacia
    cero en muestras finitas.
    """
    y = np.asarray(y, dtype=float)
    d = y - y.mean(axis=-1, keepdims=True)
    den = np.sum(d * d, axis=-1)
    r = np.empty(y.shape[:-1] + (nlags,))
    for k in range(1, nlags + 1):
        r[..., k - 1] = np.sum(d[..., k:] * d[..., :-k], axis=-1) / den
    return r


def ljung_box(y, m, gl=None):
    """Estadistico Q(m) de Ljung-Box y su valor-p.

    Q(m) = n(n+2) sum_{k=1}^{m} r_k^2 / (n - k), aproximadamente chi^2 con
    m grados de libertad bajo ruido blanco. Cuando se aplica a residuos de un
    modelo con K parametros, gl = m - K.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[-1]
    r = acf(y, m)
    k = np.arange(1, m + 1)
    q = n * (n + 2) * np.sum(r**2 / (n - k), axis=-1)
    gl = m if gl is None else gl
    return q, stats.chi2.sf(q, gl)


def ols_ar1(y):
    """Estimado OLS de phi en y_t = c + phi y_{t-1} + e_t, sobre el ultimo eje."""
    y = np.asarray(y, dtype=float)
    x, z = y[..., :-1], y[..., 1:]
    xc = x - x.mean(axis=-1, keepdims=True)
    zc = z - z.mean(axis=-1, keepdims=True)
    return np.sum(xc * zc, axis=-1) / np.sum(xc * xc, axis=-1)


def graficar_acf(r, n, ax, titulo=""):
    """Grafica r_k con las bandas +-1.96/sqrt(n) del ruido blanco."""
    k = np.arange(1, len(r) + 1)
    banda = 1.96 / np.sqrt(n)
    fuera = np.abs(r) > banda
    ax.vlines(k, 0, r, color="#4C72B0", lw=2)
    ax.scatter(k[fuera], r[fuera], color="#C44E52", zorder=3, s=18,
               label="fuera de las bandas")
    ax.scatter(k[~fuera], r[~fuera], color="#4C72B0", zorder=3, s=18)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(banda, color="gray", ls="--", lw=1)
    ax.axhline(-banda, color="gray", ls="--", lw=1)
    ax.set_xticks(k)
    ax.set_xlabel("rezago $k$")
    ax.set_ylabel("$r_k$")
    ax.set_title(titulo)
    return fuera.sum()
