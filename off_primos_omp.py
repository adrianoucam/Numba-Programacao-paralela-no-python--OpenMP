# primos_numba.py
'''
O trecho set_num_threads(4) (comentado) permite fixar o número de threads como no seu num_threads(4) do OpenMP; você também pode usar a variável de ambiente NUMBA_NUM_THREADS.

O Numba aqui paraleliza o loop com prange, semelhante a parallel for do OpenMP.

O target offload para GPU (#pragma omp target ... device(1)) não tem equivalente direto no Numba CPU. Se quiser GPU, o caminho é reescrever o kernel com Numba CUDA (ou usar CuPy), mas o teste de primalidade com controle de fluxo não costuma escalar bem em GPU.
'''
# off_primos_omp.py
import sys, time, math
from numba import njit, prange, set_num_threads, get_num_threads

@njit
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    r = int(math.sqrt(n))
    i = 3
    while i <= r:
        if n % i == 0:
            return False
        i += 2
    return True

@njit(parallel=True)
def count_primes_upto(n):
    if n < 2:
        return 0

    # Contaremos apenas os ímpares a partir de 3.
    # Número de candidatos ímpares no intervalo [3, n]:
    #   3,5,7,... -> m elementos, onde m = floor((n-3)/2) + 1  (se n >= 3)
    m = 0
    if n >= 3:
        m = ((n - 3) // 2) + 1

    total = 0
    # prange com passo 1; mapeamos k -> candidato i = 2*k + 3
    for k in prange(m):
        i = 2 * k + 3
        if is_prime(i):
            total += 1

    # soma o 2
    return total + 1  # (inclui o primo 2)

def main():
    if len(sys.argv) < 2:
        print("Valor inválido! Entre com o valor do maior inteiro")
        return

    n = int(sys.argv[1])

    # Opcional: defina o nº de threads
    # set_num_threads(4)

    # Warm-up para não contar tempo de compilação JIT
    _ = count_primes_upto(3)

    t0 = time.perf_counter()
    total = count_primes_upto(n)
    t1 = time.perf_counter()

    print(f"Quant. de primos entre 1 e {n}: {total}")
    print(f"Tempo de execução: {t1 - t0:.6f} s")
    # print(f"Threads usadas: {get_num_threads()}")

if __name__ == "__main__":
    main()
