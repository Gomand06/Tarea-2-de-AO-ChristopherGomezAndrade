"""Ejercicio 5. Aproximación MST y comparación con el óptimo exacto.

Ejecutar: python ejercicio5.py
"""
from __future__ import annotations

from statistics import median
import csv
import math
import random
import statistics as st

SEMILLA = 20261025


def puntos_aleatorios(semilla, n):
    rng = random.Random(semilla)
    return [(100 * rng.random(), 100 * rng.random()) for _ in range(n)]


def guardar_csv(nombre, filas):
    with open(nombre, 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)

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


def mst_approximation(d, root=0):
    n = len(d)
    used, key, parent = ([False] * n, [math.inf] * n, [-1] * n)
    key[root] = 0
    for _ in range(n):
        u = min((i for i in range(n) if not used[i]), key=lambda i: (key[i], i))
        used[u] = True
        for v in range(n):
            if not used[v] and d[u][v] < key[v]:
                key[v], parent[v] = (d[u][v], u)
    adj = [[] for _ in range(n)]
    for v, u in enumerate(parent):
        if u >= 0:
            adj[u].append(v)
            adj[v].append(u)
    tour = []

    def dfs(u, p):
        tour.append(u)
        for v in sorted(adj[u]):
            if v != p:
                dfs(v, u)
    dfs(root, -1)
    return {'tour': tour, 'cost': tour_cost(tour, d), 'tree_weight': sum(key)}


def held_karp(d):
    n = len(d)
    dp = {(1 << j, j): d[0][j] for j in range(1, n)}
    for mask in range(1, 1 << n):
        if mask & 1:
            continue
        for j in range(1, n):
            if not mask & 1 << j:
                continue
            prev = mask ^ 1 << j
            if not prev:
                continue
            dp[mask, j] = min((dp[prev, k] + d[k][j] for k in range(1, n) if prev & 1 << k))
    full = (1 << n) - 1 ^ 1
    return min((dp[full, j] + d[j][0] for j in range(1, n)))


def main():
    resultados = []
    for prueba in range(100):
        puntos = puntos_aleatorios(SEMILLA + 6000 + prueba, 9)
        distancias = euclidean_matrix(puntos)
        optimo = held_karp(distancias)
        rng = random.Random(SEMILLA + 7000 + prueba)
        inicial = rng.sample(range(9), 9)
        inicio = rng.randrange(9)
        mst = mst_approximation(distancias)
        metodos = [('MST puro', mst),
                   ('MST + 2-opt', two_opt(mst['tour'], distancias)),
                   ('2-opt aleatorio', two_opt(inicial, distancias)),
                   ('2-opt vecino', two_opt(nearest_neighbor(distancias, inicio), distancias))]
        t0 = calibrate_temperature(inicial, distancias, SEMILLA + 8000 + prueba)
        for esquema in ['lineal', 'exponencial']:
            r = simulated_annealing(inicial, distancias, esquema, 3000,
                                    SEMILLA + 9000 + prueba, t0=t0)
            metodos.append(('SA ' + esquema, r))
        for nombre, r in metodos:
            razon = r['cost']/optimo
            assert razon >= 1 - 1e-8
            if nombre.startswith('MST'):
                assert razon <= 2 + 1e-8
            resultados.append({'instance': prueba, 'method': nombre,
                               'cost': r['cost'], 'optimum': optimo, 'ratio': razon})
        if (prueba + 1) % 10 == 0:
            print(f'{prueba + 1}/100 instancias completadas')
    guardar_csv('mst.csv', resultados)
    nombres = [nombre for nombre, _ in metodos]
    print('\nMétodo              Costo medio    Razón media    Óptimos de 100')
    for nombre in nombres:
        datos = [r for r in resultados if r['method'] == nombre]
        exactos = sum(abs(r['ratio'] - 1) <= 1e-8 for r in datos)
        print(f'{nombre:19} {st.mean(r["cost"] for r in datos):11.3f}'
              f'{st.mean(r["ratio"] for r in datos):15.5f} {exactos:15}')
    razones = [r['ratio'] for r in resultados if r['method'] == 'MST puro']
    print(f'Razón máxima MST: {max(razones):.4f}; límite teórico: 2.')
    print('MST puro significa el ciclo de preorden con atajos, no el peso del árbol.')

    import matplotlib.pyplot as plt
    plt.style.use('grayscale')
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.5))
    ejes[0].hist(razones, bins=[1 + i/16 for i in range(17)], edgecolor='white')
    ejes[0].axvline(2, linestyle='--', label='Límite 2')
    ejes[0].set(xlim=(0.98, 2.04), title='Razón de aproximación MST',
                xlabel='Costo / óptimo', ylabel='Frecuencia')
    ejes[0].legend()
    etiquetas = ['MST', 'MST +\n2-opt', '2-opt\naleat.', '2-opt\nvecino',
                 'SA\nlineal', 'SA\nexp.']
    for eje, campo, titulo in [(ejes[1], 'cost', 'Costos obtenidos'),
                               (ejes[2], 'ratio', 'Razones respecto al óptimo')]:
        datos = [[r[campo] for r in resultados if r['method'] == nombre]
                 for nombre in nombres]
        eje.boxplot(datos)
        eje.set_xticks(range(1, 7), etiquetas)
        eje.set_title(titulo)
        eje.set_ylabel('Longitud del ciclo' if campo == 'cost' else 'Costo / óptimo')
        eje.grid(axis='y', alpha=0.3)
    fig.suptitle('Los seis métodos usan las mismas 100 instancias de 9 ciudades')
    fig.tight_layout()
    fig.savefig('grafica_ejercicio5.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
