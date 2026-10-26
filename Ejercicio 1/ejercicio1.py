"""Ejercicio 1. Estimación de pi y aceleración paralela.

Ejecutar: python ejercicio1.py
"""
from __future__ import annotations

import multiprocessing as mp
import csv
import random
import statistics as st
from time import perf_counter

SEMILLA = 20261025


def guardar_csv(nombre, filas):
    with open(nombre, 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def medir(funcion, *args, **kwargs):
    inicio = perf_counter()
    resultado = funcion(*args, **kwargs)
    return resultado, perf_counter() - inicio


def _pi_chunk(task: tuple[int, int]) -> int:
    samples, seed = task
    rng = random.Random(seed)
    inside = 0
    for _ in range(samples):
        x, y = (rng.random(), rng.random())
        inside += x * x + y * y <= 1.0
    return inside


def estimate_pi(samples=4000000, processes=1, seed=2026, chunks=32):
    if min(samples, processes, chunks) < 1:
        raise ValueError('samples, processes y chunks deben ser positivos')
    q, r = divmod(samples, chunks)
    tasks = [(q + (i < r), seed + 104729 * i) for i in range(chunks)]
    with mp.get_context('spawn').Pool(processes) as pool:
        hits = sum(pool.map(_pi_chunk, tasks, chunksize=1))
    return {'pi': 4 * hits / samples, 'hits': hits, 'samples': samples}


def main():
    muestras = 4_000_000
    repeticiones = 5
    mediciones = []
    print('4,000,000 muestras, 5 repeticiones y 32 bloques fijos.')
    print('El tiempo incluye crear y cerrar los procesos.\n')
    estimate_pi(10_000, 1, SEMILLA)  # Calentamiento que no se cuenta.
    for repeticion in range(repeticiones):
        orden = [1, 2, 4, 8]
        random.Random(SEMILLA + repeticion).shuffle(orden)
        for procesos in orden:
            resultado, segundos = medir(estimate_pi, muestras, procesos,
                                        SEMILLA + repeticion)
            mediciones.append({'repeat': repeticion, 'processes': procesos,
                               'seconds': segundos, **resultado})
            print(f'Repetición {repeticion + 1} | procesos {procesos} | '
                  f'{segundos:.4f} s | pi = {resultado["pi"]:.6f}')
        # La partición no cambia con el número de procesos.
        assert len({r['hits'] for r in mediciones
                    if r['repeat'] == repeticion}) == 1
    tiempos = {p: st.median(r['seconds'] for r in mediciones
               if r['processes'] == p) for p in [1, 2, 4, 8]}
    resumen = [{'processes': p, 'median_seconds': tiempos[p],
                'speedup': tiempos[1] / tiempos[p],
                'efficiency': tiempos[1] / tiempos[p] / p} for p in tiempos]
    print('\nProcesos     Mediana (s)     Speedup     Eficiencia')
    for r in resumen:
        print(f'{r["processes"]:8d} {r["median_seconds"]:15.4f} '
              f'{r["speedup"]:11.3f} {100*r["efficiency"]:12.1f}%')
    guardar_csv('pi.csv', mediciones)
    guardar_csv('pi_resumen.csv', resumen)

    # Se importa después de medir, para no cargar gráficas en cada proceso.
    import matplotlib.pyplot as plt
    plt.style.use('grayscale')
    procesos = [1, 2, 4, 8]
    plt.plot(procesos, procesos, '--', label='Ideal')
    plt.plot(procesos, [r['speedup'] for r in resumen], 'o-', label='Medido')
    plt.xticks(procesos)
    plt.xlabel('Número de procesos')
    plt.ylabel('Aceleración')
    plt.title('Speedup de estimate_pi')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('grafica_ejercicio1.png', dpi=150)
    print('\nGráfica y CSV guardados en la carpeta de ejecución.')
    plt.show()


if __name__ == '__main__':
    mp.freeze_support()
    main()
