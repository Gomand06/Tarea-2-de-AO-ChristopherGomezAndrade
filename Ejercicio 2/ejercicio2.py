"""Ejercicio 2. Mochila 0/1 con dos criterios voraces.

Ejecutar: python ejercicio2.py
"""
from __future__ import annotations

from dataclasses import dataclass
import csv



def guardar_csv(nombre, filas):
    with open(nombre, 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)

@dataclass(frozen=True)
class Item:
    name: str
    weight: int
    value: int


def greedy_knapsack(items, capacity, criterion='ratio'):
    if capacity < 0 or any((i.weight <= 0 or i.value < 0 for i in items)):
        raise ValueError('Capacidad/valores no negativos y pesos positivos')
    if criterion == 'ratio':
        ordered = sorted(items, key=lambda i: (-i.value / i.weight, i.weight, i.name))
    elif criterion == 'weight':
        ordered = sorted(items, key=lambda i: (i.weight, -i.value, i.name))
    else:
        raise ValueError('Criterio desconocido')
    selected, used, value = ([], 0, 0)
    for item in ordered:
        if used + item.weight <= capacity:
            selected.append(item.name)
            used += item.weight
            value += item.value
    return {'selected': selected, 'weight': used, 'value': value}


def main():
    objetos = [Item('A', 1, 1), Item('B', 2, 2), Item('C', 10, 100)]
    capacidad = 10
    peso = greedy_knapsack(objetos, capacidad, 'weight')
    razon = greedy_knapsack(objetos, capacidad, 'ratio')
    resultados = [{'method': 'Menor peso', **peso},
                  {'method': 'Valor/peso', **razon}]
    print('Capacidad: 10. Cada objeto se toma completo, una sola vez.')
    print('\nObjeto    Peso    Valor    Valor/peso')
    for o in objetos:
        print(f'{o.name:6} {o.weight:7} {o.value:8} {o.value/o.weight:13.1f}')
    print('\nMétodo         Objetos    Peso usado    Valor total')
    for r in resultados:
        print(f'{r["method"]:14} {",".join(r["selected"]):10} '
              f'{r["weight"]:10} {r["value"]:14}')
    print('\nMenor peso obtiene 3; valor/peso obtiene 100.')
    print('El óptimo de esta instancia es C, con valor 100.')
    guardar_csv('mochila.csv', resultados)

    import matplotlib.pyplot as plt
    plt.style.use('grayscale')
    nombres = ['Menor peso', 'Valor/peso']
    fig, ejes = plt.subplots(1, 3, figsize=(11, 4))
    ejes[0].table(cellText=[['Menor peso', 'A, B'], ['Valor/peso', 'C']],
                  colLabels=['Criterio', 'Objetos'], loc='center',
                  cellLoc='center')
    ejes[0].axis('off')
    for eje, campo, titulo in [(ejes[1], 'weight', 'Capacidad usada de 10'),
                               (ejes[2], 'value', 'Valor total')]:
        valores = [r[campo] for r in resultados]
        eje.bar(nombres, valores, color=['0.35', '0.7'])
        eje.set_title(titulo)
        eje.set_ylim(0, max(valores) * 1.2)
        for i, valor in enumerate(valores):
            eje.text(i, valor, str(valor), ha='center', va='bottom')
    fig.suptitle('Comparación de los criterios de mochila')
    fig.tight_layout()
    fig.savefig('grafica_ejercicio2.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
