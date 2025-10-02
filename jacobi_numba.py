# jacobi_numba.py
import time
import numpy as np
from numba import njit, prange, set_num_threads

# Parâmetros (iguais ao código C)
ROWS = 1000
COLUMNS = 1000
MAX_TEMP_ERROR = 0.01
MAX_ITER = 3000

# ------------------------ Inicialização ------------------------

@njit
def iniciar(A):
    """
    Inicializa a matriz A com fronteiras:
      - esquerda e topo = 0
      - direita = gradiente linear por linha: (100/ROWS)*i
      - base   = gradiente linear por coluna: (100/COLUMNS)*j
    Interior começa em 0.
    """
    rows = A.shape[0] - 2
    cols = A.shape[1] - 2

    # zera tudo
    for i in range(rows + 2):
        for j in range(cols + 2):
            A[i, j] = 0.0

    # fronteiras esquerda/direita
    for i in range(rows + 2):
        A[i, 0] = 0.0
        A[i, cols + 1] = (100.0 / rows) * i  # gradiente vertical

    # fronteiras topo/base
    for j in range(cols + 2):
        A[0, j] = 0.0
        A[rows + 1, j] = (100.0 / cols) * j  # gradiente horizontal

# ------------------------ Núcleos paralelos ------------------------

@njit(parallel=True)
def jacobi_update(A, Anew):
    """
    Atualiza o interior de Anew usando A:
      Anew[i,j] = 0.25 * (A[i+1,j] + A[i-1,j] + A[i,j+1] + A[i,j-1])
    Não toca nas fronteiras.
    """
    rows = A.shape[0] - 2
    cols = A.shape[1] - 2

    for i in prange(1, rows + 1):
        for j in range(1, cols + 1):
            Anew[i, j] = 0.25 * (
                A[i + 1, j] + A[i - 1, j] + A[i, j + 1] + A[i, j - 1]
            )

@njit(parallel=True)
def copy_and_maxdiff(A, Anew):
    """
    Copia o interior de Anew -> A e calcula dt = max |Anew - A (antigo)|.
    Implementa redução de máximo em duas fases:
      - cada linha calcula seu máximo local (em paralelo)
      - redução final sequencial sobre os máximos por linha
    Retorna dt.
    """
    rows = A.shape[0] - 2
    cols = A.shape[1] - 2
    row_max = np.zeros(rows, dtype=np.float64)  # máximos por linha

    for i in prange(1, rows + 1):
        local_max = 0.0
        for j in range(1, cols + 1):
            diff = Anew[i, j] - A[i, j]
            if diff < 0:
                diff = -diff
            if diff > local_max:
                local_max = diff
            A[i, j] = Anew[i, j]
        row_max[i - 1] = local_max

    # redução final
    dt = 0.0
    for i in range(rows):
        if row_max[i] > dt:
            dt = row_max[i]
    return dt

# ------------------------ Driver ------------------------

def main():
    # Opcional: fixe nº de threads (ex.: 8)
    # set_num_threads(8)

    # Aloca matrizes (ROWS+2) x (COLUMNS+2)
    A    = np.empty((ROWS + 2, COLUMNS + 2), dtype=np.float64)
    Anew = np.empty_like(A)

    # Inicializa (fronteiras em A, interior zero)
    iniciar(A)
    # Inicializa Anew coerente (não estritamente necessário, mas evita lixo)
    iniciar(Anew)

    # Warm-up do JIT (compila funções antes de cronometrar)
    jacobi_update(A, Anew)
    _ = copy_and_maxdiff(A, Anew)

    iteration = 1
    dt = 100.0  # como no C

    inicio = time.perf_counter()
    while dt > MAX_TEMP_ERROR and iteration <= MAX_ITER:
        jacobi_update(A, Anew)
        dt = copy_and_maxdiff(A, Anew)
        iteration += 1
    fim = time.perf_counter()

    print("\n Erro maximo na iteracao %d era %.6f. O tempo de execucao foi de %.6f segundos"
          % (iteration - 1, dt, fim - inicio))

if __name__ == "__main__":
    main()
