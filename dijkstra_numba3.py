'''
um exemplo didático completo: um “mapa de cidade” onde queremos achar a rota mais rápida (em minutos) da Base de Emergência (fonte = 0) até o Parque Municipal usando Dijkstra.
O grafo é exatamente o do seu exemplo em C; além das distâncias mínimas, o código reconstrói a rota usando um vetor de predecessores.

'''


# dijkstra_exemplo_didatico.py
# Exemplo didático: rota mais rápida da Base de Emergência até o Parque Municipal
# Dijkstra (fonte=0) com relaxamento paralelo via Numba (CPU)

import numpy as np
from numba import njit, prange, set_num_threads

INF = np.int64(2**31 - 1)

# Nome dos pontos da cidade (índices 0..5)
NOMES = [
    "Base de Emergência",  # 0
    "Hospital Central",    # 1
    "Universidade",        # 2
    "Shopping",            # 3
    "Parque Municipal",    # 4 (destino do exemplo)
    "Estação de Trem"      # 5
]

def build_graph_nv6():
    """
    Mesmo grafo do exemplo em C (bidirecional, pesos = minutos).
    """
    NV = 6
    ohd = np.full((NV, NV), INF, dtype=np.int64)
    for i in range(NV):
        ohd[i, i] = 0

    # Ruas (minutos)
    ohd[0,1] = ohd[1,0] = 40   # Base <-> Hospital
    ohd[0,2] = ohd[2,0] = 15   # Base <-> Universidade
    ohd[1,2] = ohd[2,1] = 20   # Hospital <-> Universidade
    ohd[1,3] = ohd[3,1] = 10   # Hospital <-> Shopping
    ohd[1,4] = ohd[4,1] = 25   # Hospital <-> Parque
    ohd[2,3] = ohd[3,2] = 100  # Universidade <-> Shopping
    ohd[1,5] = ohd[5,1] = 6    # Hospital <-> Estação
    ohd[4,5] = ohd[5,4] = 8    # Parque <-> Estação
    return ohd

@njit(parallel=True)
def dijkstra_from0_parallel_with_pred(ohd):
    """
    Dijkstra (fonte=0). Seleção do vértice mínimo é sequencial; relaxamento é paralelo (prange).
    Retorna:
      - mind: vetor de distâncias mínimas a partir de 0
      - pred: predecessor de cada vértice no caminho mínimo (pred[0] = -1)
    """
    n = ohd.shape[0]
    mind = np.empty(n, dtype=np.int64)
    pred = np.empty(n, dtype=np.int64)
    notdone = np.ones(n, dtype=np.uint8)

    # inicia a partir da fonte 0
    mind[0] = 0
    pred[0] = -1
    notdone[0] = 0
    for i in range(1, n):
        w = ohd[0, i]
        mind[i] = w
        pred[i] = 0 if w < INF else -1

    # n-1 iterações
    for _ in range(n - 1):
        # encontra vértice aberto com menor distância
        md = INF
        mv = 0
        for i in range(1, n):
            if notdone[i] == 1 and mind[i] < md:
                md = mind[i]
                mv = i

        # fecha mv
        notdone[mv] = 0

        # relaxamento em paralelo
        for i in prange(n):
            if notdone[i] == 1:
                w = ohd[mv, i]
                if w < INF:
                    alt = md + w
                    if alt < mind[i]:
                        mind[i] = alt
                        pred[i] = mv

    return mind, pred

def reconstruir_caminho(pred, destino):
    """
    Reconstrói caminho 0 -> destino usando o vetor de predecessores.
    """
    caminho = []
    v = destino
    while v != -1 and v >= 0:
        caminho.append(v)
        v = int(pred[v])
    caminho.reverse()
    return caminho

def main():
    # Opcional: fixe nº de threads do Numba
    # set_num_threads(4)

    ohd = build_graph_nv6()
    mind, pred = dijkstra_from0_parallel_with_pred(ohd)

    # Problema didático:
    # "Qual a rota mais rápida da Base de Emergência (0) até o Parque Municipal (4)?"
    origem = 0
    destino = 4
    caminho = reconstruir_caminho(pred, destino)

    print("Distâncias mínimas (a partir de '%s'):" % NOMES[origem])
    for i in range(1, ohd.shape[0]):
        print("%s: %d" % (NOMES[i], int(mind[i])))

    print("\nRota mais rápida até '%s':" % NOMES[destino])
    nomes = [NOMES[v] for v in caminho]
    print(" -> ".join(nomes))
    print("Custo total (min): %d" % int(mind[destino]))

    # (opcional) outro destino para comparação
    outro = 3  # Shopping
    cam2 = reconstruir_caminho(pred, outro)
    print("\nRota até '%s':" % NOMES[outro])
    print(" -> ".join(NOMES[v] for v in cam2))
    print("Custo total (min): %d" % int(mind[outro]))

if __name__ == "__main__":
    main()
