# off_map_numba.py
import numpy as np
from numba import njit, prange, set_num_threads

# --------- Regiões "target" modeladas em funções paralelas ---------

@njit(parallel=True)
def region1_add(a, b):
    """
    map(to: a, b) / map(from: c)
    c[i] = a[i] + b[i]
    """
    n = a.size
    c = np.empty_like(a)
    for i in prange(n):
        c[i] = a[i] + b[i]
    return c  # "map(from: c)"

@njit(parallel=True)
def region2_alloc_x_from_a(a):
    """
    map(to: a) / map(alloc: x)
    x é alocado e usado apenas dentro da região (não retorna ao host).
    Para evitar eliminação por otimização, computamos um checksum.
    """
    n = a.size
    x = np.empty_like(a)
    acc = 0
    for i in prange(n):
        x[i] = a[i] * 2
        acc += x[i]                 # só para evitar DCE
    return acc                      # x não retorna (simula map(alloc: x))

@njit(parallel=True)
def region3_double_first_half(a, c):
    """
    map(to: a[0:N/2]) / map(from: c[0:N/2])
    c[i] = 2 * a[i], para i em [0, N/2)
    Apenas a metade inicial de c é modificada (resto fica como estava).
    """
    n2 = a.size // 2
    for i in prange(n2):
        c[i] = a[i] * 2
    # retorno de c é opcional; a modificação é in-place

def main():
    # Opcional: fixe número de threads (similar ao num_threads do OpenMP)
    # set_num_threads(4)

    N = 10
    a = np.empty(N, dtype=np.int32)
    b = np.empty(N, dtype=np.int32)
    c = np.zeros(N, dtype=np.int32)
    x_checksum = 0  # só para ilustrar que x não "sai" da região 2

    # Inicialização (host)
    for i in range(N):
        a[i] = i + 1
        b[i] = 2 * (i + 1)

    # --- Região 1: map(to: a, b) map(from: c)  -> c = a + b
    # Warm-up JIT (opcional, compila antes de medir/usar)
    _ = region1_add(a[:1], b[:1])

    c = region1_add(a, b)
    print("1. c[0] = %d no hospedeiro" % c[0])

    # --- Região 2: map(to: a) map(alloc: x) -> x alocado e usado só no "device"
    _ = region2_alloc_x_from_a(a[:1])  # warm-up
    x_checksum = region2_alloc_x_from_a(a)
    # (x não é devolvido ao host; apenas ilustramos que houve trabalho)
    # print("checksum(x) = %d (apenas para evitar DCE)" % x_checksum)

    # --- Região 3: map(to: a[0:N/2]) map(from: c[0:N/2]) -> c[0:N/2] = 2*a[0:N/2]
    _ = region3_double_first_half(a[:2], c[:2])  # warm-up
    region3_double_first_half(a, c)

    print("2. c[0] = %d no hospedeiro" % c[0])
    # (Opcional) mostrar vetores para conferir
    # print("a =", a.tolist())
    # print("b =", b.tolist())
    # print("c =", c.tolist())

if __name__ == "__main__":
    main()
