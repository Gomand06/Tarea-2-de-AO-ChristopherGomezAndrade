"""Ejercicio 4. Recocido simulado con movimientos 2-opt.

Ejecutar: python ejercicio4.py
"""
from __future__ import annotations

from statistics import median
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


def two_opt_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1)]


def two_opt_delta(tour, d, i, j):
    a, b = (tour[i], tour[(i + 1) % len(tour)])
    c, e = (tour[j], tour[(j + 1) % len(tour)])
    return d[a][c] + d[b][e] - d[a][b] - d[c][e]


def temperature(k, steps, t0, tf, schedule):
    x = k / max(1, steps - 1)
    if schedule == 'lineal':
        return t0 + (tf - t0) * x
    if schedule == 'exponencial':
        return t0 * (tf / t0) ** x
    if schedule == 'cuadratico':
        return tf + (t0 - tf) * (1 - x) ** 2
    raise ValueError('Esquema desconocido')


def calibrate_temperature(initial, d, seed):
    rng = random.Random(seed)
    pairs = two_opt_pairs(len(initial))
    positives = [delta for i, j in rng.choices(pairs, k=300) if (delta := two_opt_delta(initial, d, i, j)) > 0]
    return (median(positives) if positives else 1.0) / -math.log(0.8)


def simulated_annealing(initial, d, schedule='exponencial', steps=15000, seed=2026, t0=None, tf=None, trace_every=100):
    if len(initial) < 4 or steps < 2:
        raise ValueError('Se requieren al menos 4 ciudades y 2 iteraciones')
    t0 = calibrate_temperature(initial, d, seed + 17) if t0 is None else t0
    tf = t0 * 0.001 if tf is None else tf
    if not 0 < tf < t0:
        raise ValueError('Se requiere 0 < Tf < T0')
    moves, accept = (random.Random(seed), random.Random(seed + 1))
    pairs = two_opt_pairs(len(initial))
    tour, best = (list(initial), list(initial))
    current = best_cost = tour_cost(tour, d)
    accepted = uphill = 0
    trace = [{'iteration': 0, 'temperature': t0, 'current': current, 'best': best_cost}]
    for k in range(steps):
        i, j = pairs[moves.randrange(len(pairs))]
        delta = two_opt_delta(tour, d, i, j)
        temp = temperature(k, steps, t0, tf, schedule)
        if delta <= 0 or accept.random() < math.exp(-delta / temp):
            tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
            current += delta
            accepted += 1
            uphill += delta > 0
            if current < best_cost - 1e-10:
                best, best_cost = (list(tour), current)
        if (k + 1) % trace_every == 0 or k + 1 == steps:
            trace.append({'iteration': k + 1, 'temperature': temp, 'current': current, 'best': best_cost})
    return {'tour': best, 'cost': tour_cost(best, d), 'last_cost': tour_cost(tour, d), 'accepted': accepted, 'uphill': uphill, 't0': t0, 'tf': tf, 'trace': trace}


def main():
    esquemas = ['lineal', 'exponencial', 'cuadratico']
    resultados, curvas = [], {e: [] for e in esquemas}
    iteraciones = 15_000
    for prueba in range(30):
        puntos = puntos_aleatorios(SEMILLA + 1000 + prueba, 60)
        distancias = euclidean_matrix(puntos)
        inicial = random.Random(SEMILLA + 2000 + prueba).sample(range(60), 60)
        t0 = calibrate_temperature(inicial, distancias, SEMILLA + 3000 + prueba)
        orden = esquemas.copy()
        random.Random(SEMILLA + 4000 + prueba).shuffle(orden)
        for esquema in orden:
            r, segundos = medir(simulated_annealing, inicial, distancias,
                                 esquema, iteraciones, SEMILLA + 5000 + prueba,
                                 t0=t0)
            resultados.append({'trial': prueba, 'schedule': esquema,
                               'cost': r['cost'], 'seconds': segundos,
                               'accepted': r['accepted'], 't0': t0, 'tf': r['tf']})
            curvas[esquema].append(r['trace'])
        print(f'Instancia {prueba + 1}/30 completada')
    guardar_csv('recocido.csv', resultados)
    filas = [{'trial': i, 'schedule': e, **paso}
             for e in esquemas for i, curva in enumerate(curvas[e]) for paso in curva]
    guardar_csv('convergencia.csv', filas)
    print('\nEnfriamiento     Costo medio    Mediana tiempo (ms)')
    for esquema in esquemas:
        datos = [r for r in resultados if r['schedule'] == esquema]
        print(f'{esquema:14} {st.mean(r["cost"] for r in datos):12.2f}'
              f'{1000*st.median(r["seconds"] for r in datos):22.2f}')
    print('Mismos inicios, temperaturas extremas y 15,000 propuestas por esquema.')
    print('Se reporta el mejor costo visitado; no se aplica 2-opt al final.')

    import matplotlib.pyplot as plt
    plt.style.use('grayscale')
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.5))
    estilos = ['-', '--', ':']
    pasos = [r['iteration'] for r in curvas['lineal'][0]]
    for esquema, estilo in zip(esquemas, estilos):
        mediana = [st.median(curva[j]['best'] for curva in curvas[esquema])
                   for j in range(len(pasos))]
        ejes[0].plot(pasos, mediana, estilo, label=esquema)
        curva = curvas[esquema][0]
        ejes[1].plot(pasos, [r['temperature']/curva[0]['temperature'] for r in curva],
                     estilo, label=esquema)
        datos = [r for r in resultados if r['schedule'] == esquema]
        marca = {'lineal': 'o', 'exponencial': 's', 'cuadratico': '^'}[esquema]
        ejes[2].scatter([1000*r['seconds'] for r in datos], [r['cost'] for r in datos],
                        marker=marca, label=esquema, alpha=0.7)
    ejes[0].set(title='Convergencia', xlabel='Iteraciones', ylabel='Mediana del mejor costo')
    ejes[1].set(title='Enfriamiento', xlabel='Iteraciones', ylabel='T / T0', yscale='log')
    ejes[2].set(title='Tiempo frente a calidad', xlabel='Tiempo de búsqueda (ms)',
                ylabel='Mejor costo visitado')
    for eje in ejes:
        eje.legend()
        eje.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('grafica_ejercicio4.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
