# Tarea 3: Introducción a Series de Tiempo

**Análisis y Ciencia de Datos** · Facultad de Ingeniería, Universidad del Istmo de Guatemala

**Herman Andrés Echeverría Rojas** · `hecheverria@unis.edu.gt`

Venta neta diaria de la tienda en línea *Online Retail* (UCI): limpieza,
calendario, descomposición, modelos AR(1) y SARIMA, pronóstico con origen
móvil y análisis RFM de clientes.

📄 **[`informe/main.pdf`](informe/main.pdf)**: las derivaciones de 0.1, el
análisis de todas las partes, la bitácora de limpieza y todas las figuras y
tablas. El código fuente está en `informe/main.tex` y `informe/secciones/`.

## Requisitos

- Python 3.10 o superior
- Una distribución de LaTeX (MiKTeX o TeX Live) para compilar el informe

```bash
py -m pip install -r requirements.txt
```

## Datos

El archivo original no se versiona. Descarguen `Online Retail.xlsx` de
<https://archive.ics.uci.edu/dataset/352/online+retail>, descompriman el zip
y dejen el archivo en `datos/Online Retail.xlsx`.

## Cómo ejecutar

Todo, en orden, desde el `.xlsx` original:

```bash
py ejecutar_todo.py
```

Tarda alrededor de un minuto. La primera lectura del `.xlsx` es la parte
lenta; después se guarda una copia en `datos/online_retail.pkl` y las
siguientes corridas arrancan al instante. Cada parte también se puede correr
sola, por ejemplo `py parte4.py`.

Los scripts escriben las figuras en `figuras/` y las tablas en `resultados/`
(como fragmentos `.tex`). El informe las incluye directamente, así que ningún
número de una tabla está copiado a mano. Para compilarlo:

```bash
cd informe
latexmk -pdf main.tex
```

## Qué hay en cada archivo

| Archivo | Contenido |
|---|---|
| `datos.py` | Lee el `.xlsx` con `InvoiceNo`, `StockCode` y `CustomerID` como texto |
| `limpiar.py` | **`limpiar(df)`**: las reglas de limpieza y su bitácora |
| `calendario.py` | Serie de venta neta diaria sobre el calendario, cierres y fechas de corte |
| `series.py` | ACF, Ljung–Box y OLS del AR(1), implementados a mano con numpy |
| `modelos.py` | Configuración de los SARIMA S1 y S2 |
| `semillas.py` | Todas las semillas aleatorias |
| `salida.py` | Rutas de salida y escritura de tablas LaTeX |
| `parte0.py` … `parte6.py` | Un script por parte del enunciado |
| `ejecutar_todo.py` | Corre las siete partes en orden |

## Semillas

Todas están en `semillas.py`:

- **Parte 0:** una semilla distinta por escenario de simulación (25062003 a 25062009).
- **Parte 5:** semillas 1 a 10 para las repeticiones de la validación cruzada de 5 *folds*.
- **Parte 6:** semilla 2026 para el desempate de los puntajes RFM, como fija el enunciado.

Con estas semillas, dos corridas producen exactamente las mismas figuras y
tablas.
