# dijkstra_numba_bench.py
import time
import numpy as np
from numba import njit, prange, set_num_threads

INF = np.int64(2**31 - 1)

# --------------------- Construção de grafos ---------------------

def build_graph_nv6():
    """Grafo do exemplo em C (NV=6, bidirecional)."""
    NV = 6
    ohd = np.full((NV, NV), INF, dtype=np.int64)
    for i in range(NV):
        ohd[i, i] = 0
    ohd[0,1] = ohd[1,0] = 40
    ohd[0,2] = ohd[2,0] = 15
    ohd[1,2] = ohd[2,1] = 20
    ohd[1,3] = ohd[3,1] = 10
    ohd[1,4] = ohd[4,1] = 25
    ohd[2,3] = ohd[3,2] = 100
    ohd[1,5] = ohd[5,1] = 6
    ohd[4,5] = ohd[5,4] = 8
    return ohd

def build_random_graph(N, density=0.01, wmax=100, seed=0):
    """
    Grafo aleatório não-direcionado, pesos positivos.
    - N: nº de vértices
    - density: probabilidade de aresta (em todo par i<j)
    """
    rng = np.random.default_rng(seed)
    ohd = np.full((N, N), INF, dtype=np.int64)
    np.fill_diagonal(ohd, 0)

    # Máscara superior (i<j)
    mask = rng.random((N, N)) < density
    mask = np.triu(mask, 1)
    idx_i, idx_j = np.where(mask)
    if idx_i.size > 0:
        w = rng.integers(1, wmax + 1, size=idx_i.size, dtype=np.int64)
        ohd[idx_i, idx_j] = w
        ohd[idx_j, idx_i] = w  # simetria

    return ohd

# --------------------- Dijkstra (Seq / Paralelo) ---------------------

@njit
def dijkstra_seq(ohd):
    """
    Dijkstra fonte=0. Seleção do vértice mínimo e relaxamentos são sequenciais.
    ohd: matriz de adjacência com INF para ausência de aresta.
    """
    n = ohd.shape[0]
    mind = np.empty(n, dtype=np.int64)
    notdone = np.ones(n, dtype=np.uint8)

    mind[0] = 0
    notdone[0] = 0
    for i in range(1, n):
        mind[i] = ohd[0, i]

    for _ in range(n - 1):
        md = INF
        mv = 0
        for i in range(1, n):
            if notdone[i] == 1 and mind[i] < md:
                md = mind[i]
                mv = i
        notdone[mv] = 0

        # relaxamento sequencial
        for i in range(n):
            if notdone[i] == 1:
                w = ohd[mv, i]
                if w < INF:
                    alt = md + w
                    if alt < mind[i]:
                        mind[i] = alt
    return mind

@njit(parallel=True)
def dijkstra_par(ohd):
    """
    Dijkstra fonte=0. Seleção do vértice mínimo é sequencial,
    mas o relaxamento dos vizinhos é paralelizado com prange.
    """
    n = ohd.shape[0]
    mind = np.empty(n, dtype=np.int64)
    notdone = np.ones(n, dtype=np.uint8)

    mind[0] = 0
    notdone[0] = 0
    for i in range(1, n):
        mind[i] = ohd[0, i]

    for _ in range(n - 1):
        # menor vértice ainda não fechado
        md = INF
        mv = 0
        for i in range(1, n):
            if notdone[i] == 1 and mind[i] < md:
                md = mind[i]
                mv = i
        notdone[mv] = 0

        # relaxamento paralelo
        for i in prange(n):
            if notdone[i] == 1:
                w = ohd[mv, i]
                if w < INF:
                    alt = md + w
                    if alt < mind[i]:
                        mind[i] = alt
    return mind

# --------------------- Benchmark ---------------------

def bench_once(ohd, n_threads):
    """
    Mede tempos seq/par (com n_threads), usando o mesmo grafo.
    Devolve (t_seq, t_par, ok)
    """
    # Warm-up JIT com o grafo real (evita tempo de compilação)
    _ = dijkstra_seq(ohd)
    _ = dijkstra_par(ohd)

    t0 = time.perf_counter()
    mind_seq = dijkstra_seq(ohd)
    t1 = time.perf_counter()
    t_seq = t1 - t0

    set_num_threads(n_threads)
    t0 = time.perf_counter()
    mind_par = dijkstra_par(ohd)
    t1 = time.perf_counter()
    t_par = t1 - t0

    ok = np.array_equal(mind_seq, mind_par)
    return t_seq, t_par, ok

def main():
    # -------- Demo NV=6 (igual ao C) --------
    ohd6 = build_graph_nv6()
    mind_demo = dijkstra_par(ohd6)  # compila também
    print("Distâncias mínimas (NV=6, fonte=0):")
    for i in range(1, ohd6.shape[0]):
        print(int(mind_demo[i]))
    print("----\n")

    # -------- Benchmark em grafo maior --------
    N = 2000          # nº de vértices (ajuste conforme sua máquina)
    density = 0.01    # probabilidade de aresta
    threads_list = [1, 2, 4, 8]  # ajuste conforme sua CPU

    print("Construindo grafo aleatório: N=%d, density=%.4f ..." % (N, density))
    ohd_big = build_random_graph(N, density=density, wmax=100, seed=42)

    # tempo sequencial de referência (usaremos o de threads=1 como base)
    t_seq_ref, _, ok = bench_once(ohd_big, n_threads=1)
    if not ok:
        print("Aviso: divergência entre versões seq/par com 1 thread.")
    print("Referência sequencial: t_seq = %.6f s\n" % t_seq_ref)

    print("Threads |  t_par (s)  | Speedup (t_seq/t_par) | Eficiência (Speedup/threads)")
    print("--------+-------------+------------------------+-----------------------------")
    for nt in threads_list:
        t_seq, t_par, ok = bench_once(ohd_big, n_threads=nt)
        speedup = t_seq_ref / t_par if t_par > 0 else float("inf")
        eff = speedup / nt
        print("%7d | %11.6f | %22.3f | %27.3f" % (nt, t_par, speedup, eff))
        if not ok:
            print("  * Aviso: divergência detectada com %d threads!" % nt)

    print("\nObservações:")
    print("- O paralelismo atua no RELAXAMENTO (laço sobre vértices i).")
    print("- Speedup = t_seq_ref / t_par; Eficiência = Speedup / #threads.")
    print("- A seleção do vértice mínimo é sequencial (custo O(N) por passo).")
    print("- Para grafos muito grandes, considere estruturas esparsas (listas de adjacência) ou heurísticas.")

if __name__ == "__main__":
    main()
