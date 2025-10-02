# off_depend_nowait_numba.py
import numpy as np
from numba import njit, prange, set_num_threads

SIZE = 5

@njit(parallel=True)
def stage_b_from_a(a):
    """Equivalente à região target + parallel for: b[i] = 2 * a[i]"""
    n = a.size
    b = np.empty_like(a)
    for i in prange(n):
        b[i] = a[i] * 2
    return b

@njit(parallel=True)
def stage_c_from_b(b):
    """Equivalente à região target + parallel for depend(in: b): c[i] = b[i] + 5"""
    n = b.size
    c = np.empty_like(b)
    for i in prange(n):
        c[i] = b[i] + 5
    return c

def main():
    # Opcional: fixe o número de threads como no OpenMP (ex.: 4)
    # set_num_threads(8)

    a = np.arange(SIZE, dtype=np.int32)
    b = np.zeros(SIZE, dtype=np.int32)
    c = np.zeros(SIZE, dtype=np.int32)

    print("--- Antes ---")
    for i in range(SIZE):
        print("a[%d]=%d b[%d]=%d c[%d]=%d" % (i, a[i], i, b[i], i, c[i]))

    # Warm-up (compilação JIT fora da demonstração)
    _ = stage_b_from_a(a[:1])
    _ = stage_c_from_b(b[:1])

    # "Região target" 1: b = 2*a  (paralelo)
    b = stage_b_from_a(a)

    # "Região target" 2: c = b + 5 (paralelo; depende de b pronto)
    c = stage_c_from_b(b)

    print("--- Depois ---")
    for i in range(SIZE):
        print("a[%d]=%d b[%d]=%d c[%d]=%d" % (i, a[i], i, b[i], i, c[i]))

if __name__ == "__main__":
    main()
