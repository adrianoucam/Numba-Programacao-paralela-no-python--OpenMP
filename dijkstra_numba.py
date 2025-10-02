# dijkstra_numba.py
import numpy as np
from numba import njit, prange, set_num_threads

INF = np.int64(2**31 - 1)

def build_graph_nv6():
    """
    Constrói o mesmo grafo NV=6 do exemplo em C (bidirecional).
    Retorna: ohd (matriz de adjacência com pesos int64)
    """
    NV = 6
    ohd = np.full((NV, NV), INF, dtype=np.int64)
    for i in range(NV):
        ohd[i, i] = 0

    # arestas do exemplo
    ohd[0,1] = ohd[1,0] = 40
    ohd[0,2] = ohd[2,0] = 15
    ohd[1,2] = ohd[2,1] = 20
    ohd[1,3] = ohd[3,1] = 10
    ohd[1,4] = ohd[4,1] = 25
    ohd[2,3] = ohd[3,2] = 100
    ohd[1,5] = ohd[5,1] = 6
    ohd[4,5] = ohd[5,4] = 8
    return ohd

@njit(parallel=True)
def dijkstra_from0_parallel(ohd):
    """
    Dijkstra a partir do vértice 0.
    Paraleliza a etapa de relaxamento (update dos vizinhos) com prange,
    em analogia ao updateohd do OpenMP.
    """
    n = ohd.shape[0]
    mind = np.empty(n, dtype=np.int64)
    notdone = np.ones(n, dtype=np.uint8)  # 1 = ainda não fechado

    # inicialização como no C:
    mind[0] = 0
    notdone[0] = 0
    for i in range(1, n):
        mind[i] = ohd[0, i]  # distâncias 1-hop a partir de 0

    # NV-1 passos
    for _ in range(n - 1):
        # --- "findmymin" global (sequencial): menor mind[i] entre notdone ---
        md = INF
        mv = 0
        for i in range(1, n):
            if notdone[i] == 1 and mind[i] < md:
                md = mind[i]
                mv = i

        # marca mv como fechado (equivalente ao "notdone[mv]=0" em single)
        notdone[mv] = 0

        # --- updateohd: relaxamento paralelo dos vizinhos de mv ---
        # Cada i é independente -> pode ir em prange
        for i in prange(n):
            if notdone[i] == 1:
                w = ohd[mv, i]
                if w < INF:  # existe aresta
                    alt = md + w
                    if alt < mind[i]:
                        mind[i] = alt

    return mind

def main():
    # Opcional: fixe nº de threads (análogas a num_threads(…))
    # set_num_threads(8)

    ohd = build_graph_nv6()
    mind = dijkstra_from0_parallel(ohd)

    print("minimum distances:")
    # imprime como o C: i = 1..NV-1
    for i in range(1, ohd.shape[0]):
        print(int(mind[i]))

if __name__ == "__main__":
    main()
