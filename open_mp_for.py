# open_mp_for_numba.py
'''
Nota: o Numba não expõe o omp_get_thread_num() (não há ID de thread “real” no @njit(parallel=True)). Abaixo mostro:

uma versão paralela direta (sem ID de thread),

uma versão didática que divide o range em 4 “fatias” e imprime um ID lógico de worker (não é o ID real da thread do SO, mas ajuda a visualizar o particionamento).

'''
import time
from numba import njit, prange, set_num_threads

@njit(parallel=True)
def paralelo_sem_tid(n):
    for i in prange(n):
        # não use f"{i:2d}", pois Numba não suporta f-string formatadas
        print("Iteração " + str(i) + " executada em paralelo")

@njit(parallel=True)
def paralelo_workers_logicos(n, nworkers=4):
    bloco = (n + nworkers - 1) // nworkers
    for w in prange(nworkers):
        ini = w * bloco
        fim = min(n, ini + bloco)
        for i in range(ini, fim):
            # também substituímos por concatenação simples
            print("Iteração " + str(i) + " executada pelo worker lógico " + str(w))

def main():
    N = 17
    set_num_threads(4)  # equivalente ao num_threads(4)

    # warm-up para evitar medir tempo de compilação
    paralelo_sem_tid(1)
    paralelo_workers_logicos(1, 4)

    print("=== Versão paralela direta ===")
    t0 = time.perf_counter()
    paralelo_sem_tid(N)
    t1 = time.perf_counter()
    print("(tempo: %.6f s)\n" % (t1 - t0))

    print("=== Versão com workers lógicos (4) ===")
    t0 = time.perf_counter()
    paralelo_workers_logicos(N, 4)
    t1 = time.perf_counter()
    print("(tempo: %.6f s)" % (t1 - t0))

if __name__ == "__main__":
    main()
