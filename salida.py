"""Rutas de salida y escritura de tablas LaTeX.

Los scripts escriben cada tabla en resultados/ como un fragmento .tex que el
informe incluye con \\input, para que ningun numero de una tabla se copie a
mano.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def guardar_figura(fig, nombre):
    fig.savefig(FIGURAS / f"{nombre}.png")
    plt.close(fig)


def guardar_tabla(df, nombre, formatos=None, alinear=None):
    """Escribe df como un tabular de booktabs en resultados/<nombre>.tex.

    formatos: dict columna -> funcion que convierte el valor a texto LaTeX.
    El indice no se escribe; si importa, se pasa como columna.
    """
    formatos = formatos or {}
    cols = list(df.columns)
    alinear = alinear or "l" + "r" * (len(cols) - 1)
    lineas = [rf"\begin{{tabular}}{{{alinear}}}", r"\toprule",
              " & ".join(str(c) for c in cols) + r" \\", r"\midrule"]
    for _, fila in df.iterrows():
        celdas = []
        for c in cols:
            v = fila[c]
            celdas.append(formatos[c](v) if c in formatos else str(v))
        lineas.append(" & ".join(celdas) + r" \\")
    lineas += [r"\bottomrule", r"\end{tabular}"]
    (RESULTADOS / f"{nombre}.tex").write_text("\n".join(lineas) + "\n",
                                               encoding="utf-8")


def num(decimales=3):
    """Formateador: numero con separador de miles y decimales fijos."""
    def f(v):
        return f"{v:,.{decimales}f}"
    return f
