# omp_collapse_numba.py
'''

Problema proposto — Análise de matriz bidimensional com paralelismo colapsado
Contexto

Suponha que você esteja desenvolvendo um programa de processamento de imagens em que cada pixel da imagem é representado por uma matriz M[k][j].
Você deseja percorrer essa matriz linha por linha (índice k) e coluna por coluna (índice j) para calcular um valor auxiliar e registrar, ao final, qual foi o último pixel processado (índices klast e jlast).

No entanto, como o tamanho da imagem é grande, você decide paralelizar o duplo loop que percorre k e j.
Para garantir que todos os pares (k, j) sejam divididos entre as threads, e que a ordem lexicográfica (linha por linha) seja respeitada, você usa colapso de loops (collapse(2)).
Por fim, você usa lastprivate para identificar, ao término da execução, qual elemento foi o último processado no espaço 2D.

Enunciado

Considere uma matriz de dimensão K × J, onde:

K representa o número de linhas,

J representa o número de colunas.

Cada célula da matriz deve armazenar a soma M[k][j] = k + j.

Paralelize o duplo loop (k, j) de forma que:

As iterações sejam distribuídas entre as threads;

O loop seja colapsado em uma única dimensão paralela (como collapse(2) do OpenMP);

No final, sejam exibidos os valores de klast e jlast, representando a última posição visitada do loop colapsado.

Compare:

O resultado sequencial (sem paralelismo);

O resultado paralelo colapsado (com collapse(2)).

Mostre que o lastprivate (ou seu equivalente no Python) retorna sempre o último elemento lexicográfico (por exemplo, (K, J) = (3, 7)).


Objetivo didático

Entender como o colapso de loops distribui pares (k, j) entre as threads;

Compreender o conceito de lastprivate e sua relação com o último elemento lexicográfico;

Comparar o comportamento sequencial vs paralelo;

Introduzir o aluno ao raciocínio de transformar laços 2D em índices lineares (flattening).


'''
# collapse_benchmark_numba.py
# Comparação sequencial vs. paralelo (Numba) com speedup e eficiência

import time
import numpy as np
from numba import njit, prange, set_num_threads

# -------- Implementações --------

@njit
def collapsed_seq(K, J):
    """
    Versão sequencial: percorre (k,j) e preenche M[k,j] = (k+1)+(j+1).
    Retorna também (klast, jlast) como no 'lastprivate' (laços crescentes -> (K,J)).
    """
    M = np.zeros((K, J), dtype=np.int32)
    for k in range(K):
        for j in range(J):
            M[k, j] = (k + 1) + (j + 1)
    klast, jlast = K, J
    return M, klast, jlast

@njit(parallel=True)
def collapsed_par(K, J):
    """
    Versão paralela: lineariza (k,j) -> idx em [0, K*J) e usa prange.
    Equivalente a 'collapse(2)'.
    """
    total = K * J
    M = np.zeros((K, J), dtype=np.int32)
    for idx in prange(total):
        k = idx // J
        j = idx % J
        M[k, j] = (k + 1) + (j + 1)
    klast, jlast = K, J
    return M, klast, jlast

# -------- Utilitário de benchmark --------

def bench_once(K, J, n_threads):
    """
    Roda 1 medição sequencial e 1 paralela com n_threads.
    Faz warm-up para evitar incluir tempo de compilação.
    Retorna: (t_seq, t_par, ok_correcao, klast, jlast)
    """
    # Warm-up JIT (pequeno)
    _ = collapsed_seq(1, 1)
    _ = collapsed_par(1, 1)

    # Sequencial
    t0 = time.perf_counter()
    M_seq, klast_s, jlast_s = collapsed_seq(K, J)
    t1 = time.perf_counter()
    t_seq = t1 - t0

    # Paralelo
    set_num_threads(n_threads)
    t0 = time.perf_counter()
    M_par, klast_p, jlast_p = collapsed_par(K, J)
    t1 = time.perf_counter()
    t_par = t1 - t0

    # Verificação de correção
    ok = np.array_equal(M_seq, M_par) and (klast_s == klast_p == K) and (jlast_s == jlast_p == J)

    return t_seq, t_par, ok, klast_p, jlast_p

def main():
    # Tamanhos para teste (ajuste se quiser algo maior)
    K, J = 3000, 3000  # 4 milhões de elementos (ajuste conforme sua máquina)
    thread_list = [1, 2, 4, 8]

    print("Benchmark colapsado (Numba) — Matriz %d x %d" % (K, J))
    print("i) M[k,j] = (k+1)+(j+1)")
    print("ii) lastprivate -> (K,J) = (%d,%d)\n" % (K, J))

    # Tempo sequencial de referência (usa 1ª execução com threads=1 para coerência)
    t_seq_ref, _, ok, klast, jlast = bench_once(K, J, n_threads=1)
    if not ok:
        print("Falha de correção na referência (threads=1). Abortando.")
        return
    print("Referência sequencial: t_seq = %.6f s  | last=(%d,%d)\n" % (t_seq_ref, klast, jlast))

    print("Threads | t_par (s) | Speedup (t_seq/t_par) | Eficiência (Speedup/threads)")
    print("--------+-----------+------------------------+-----------------------------")
    for nt in thread_list:
        t_seq, t_par, ok, klast, jlast = bench_once(K, J, n_threads=nt)
        # Para speedup, use o t_seq_ref para ser uma única referência
        speedup = t_seq_ref / t_par if t_par > 0 else float("inf")
        eff = speedup / nt
        print("%7d | %9.6f | %22.3f | %27.3f" % (nt, t_par, speedup, eff))

        if not ok:
            print("  Aviso: divergência de resultado detectada com %d threads!" % nt)

    print("\nObservações:")
    print("- Speedup = t_seq_ref / t_par")
    print("- Eficiência = Speedup / #threads")
    print("- Para ver ganhos maiores, aumente K e J (atenção à memória).")

if __name__ == "__main__":
    main()

