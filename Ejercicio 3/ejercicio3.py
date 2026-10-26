"""Ejercicio 3. Búsqueda local 2-opt para el TSP.

Ejecutar: python ejercicio3.py
"""
from __future__ import annotations


import csv
import math
import random
import statistics as st
from time import perf_counter

SEMILLA = 20261025


def puntos_aleatorios(semilla, n):
    rng = random.Random(semilla)
    return [(100 * rng.random(), 100 * rng.random()) for _ in range(n)]


def guardar_csv(nombre, filas):
    with open(nombre, 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def medir(funcion, *args, **kwargs):
    inicio = perf_counter()
    resultado = funcion(*args, **kwargs)
    return resultado, perf_counter() - inicio


def euclidean_matrix(points):
    return [[math.dist(a, b) for b in points] for a in points]


def tour_cost(tour, d):
    return sum((d[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour))))


def nearest_neighbor(d, start=0):
    remaining = set(range(len(d))) - {start}
    tour = [start]
    while remaining:
        nxt = min(remaining, key=lambda j: (d[tour[-1]][j], j))
        tour.append(nxt)
        remaining.remove(nxt)
    return tour


def two_opt_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1)]


def two_opt_delta(tour, d, i, j):
    a, b = (tour[i], tour[(i + 1) % len(tour)])
    c, e = (tour[j], tour[(j + 1) % len(tour)])
    return d[a][c] + d[b][e] - d[a][b] - d[c][e]


def two_opt(initial, d, tolerance=1e-10):
    tour = list(initial)
    pairs = two_opt_pairs(len(tour))
    history = [tour_cost(tour, d)]
    improvements = 0
    while True:
        best_delta, best_pair = (-tolerance, None)
        for i, j in pairs:
            delta = two_opt_delta(tour, d, i, j)
            if delta < best_delta:
                best_delta, best_pair = (delta, (i, j))
        if best_pair is None:
            break
        i, j = best_pair
        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
        improvements += 1
        history.append(tour_cost(tour, d))
    return {'tour': tour, 'cost': tour_cost(tour, d), 'improvements': improvements, 'history': history}


def main():
    resultados = []
    for prueba in range(30):
        puntos = puntos_aleatorios(SEMILLA + 1000 + prueba, 60)
        distancias = euclidean_matrix(puntos)
        rng = random.Random(SEMILLA + 2000 + prueba)
        aleatoria = rng.sample(range(60), 60)
        inicio = rng.randrange(60)
        vecino = nearest_neighbor(distancias, inicio)
        for nombre, ruta in [('Aleatorio', aleatoria),
                             ('Vecino cercano', vecino)]:
            r, segundos = medir(two_opt, ruta, distancias)
            resultados.append({'trial': prueba, 'method': nombre,
                               'improvements': r['improvements'],
                               'cost': r['cost'], 'seconds': segundos})
        a, v = resultados[-2:]
        print(f'Instancia {prueba + 1:2}: mejoras aleatorio = '
              f'{a["improvements"]:2}, vecino = {v["improvements"]:2}')
    nombres = ['Aleatorio', 'Vecino cercano']
    print('\nInicio              Mejoras medias     Costo medio')
    for nombre in nombres:
        datos = [r for r in resultados if r['method'] == nombre]
        print(f'{nombre:19} {st.mean(r["improvements"] for r in datos):14.2f}'
              f'{st.mean(r["cost"] for r in datos):16.2f}')
    menos = sum(resultados[2*i+1]['improvements'] < resultados[2*i]['improvements']
                for i in range(30))
    print(f'Vecino Más Cercano necesita menos mejoras en {menos}/30 instancias.')
    print('Una mejora significa una inversión aceptada, no un barrido.')
    guardar_csv('busqueda_local.csv', resultados)

    import matplotlib.pyplot as plt
    plt.style.use('grayscale')
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    for eje, campo, titulo in [(ejes[0], 'improvements', 'Mejoras hasta converger'),
                               (ejes[1], 'cost', 'Costo del óptimo local')]:
        datos = [[r[campo] for r in resultados if r['method'] == nombre]
                 for nombre in nombres]
        eje.boxplot(datos)
        eje.set_xticks([1, 2], nombres)
        eje.set_title(titulo)
        eje.set_ylabel('Mejoras aceptadas' if campo == 'improvements' else 'Longitud del ciclo')
        eje.grid(axis='y', alpha=0.3)
    fig.suptitle('2-opt en 30 instancias de 60 ciudades')
    fig.tight_layout()
    fig.savefig('grafica_ejercicio3.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
