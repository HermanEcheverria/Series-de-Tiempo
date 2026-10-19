"""Registro central de semillas. Todo numero aleatorio del repo sale de aqui."""

# Parte 0: una semilla distinta por escenario, para que las replicas de un
# escenario no compartan los mismos choques que las de otro.
SEMILLAS_AR1 = {0.0: 25062003, 0.5: 25062004, 0.9: 25062005, 1.0: 25062006}
SEMILLAS_SIN_MEMORIA = {"tendencia": 25062007, "semanal": 25062008}
# Una realizacion de cada proceso para las graficas.
SEMILLA_GRAFICAS = 25062009

# Parte 5.2(a): validacion cruzada aleatoria de 5 folds, repetida con 10
# semillas para medir cuanto depende el resultado de la particion.
SEMILLAS_CV = list(range(1, 11))

# Parte 6.2: semilla fijada por el enunciado para desempatar los puntajes.
SEMILLA_RFM = 2026
