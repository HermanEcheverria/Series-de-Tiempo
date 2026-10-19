"""Corre las siete partes en orden y regenera todas las figuras y tablas.

    py ejecutar_todo.py

Necesita datos/Online Retail.xlsx (ver README.md). La primera lectura del
.xlsx tarda cerca de un minuto; despues se usa la copia en datos/.
"""

import runpy
import time
import warnings

warnings.filterwarnings("ignore")

PARTES = ["parte0", "parte1", "parte2", "parte3", "parte4", "parte5", "parte6"]

if __name__ == "__main__":
    for parte in PARTES:
        inicio = time.time()
        print(f"\n===== {parte} =====")
        runpy.run_module(parte, run_name="__main__")
        print(f"----- {parte}: {time.time() - inicio:.0f} s")
