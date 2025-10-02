# numba_barreira.py
from numba import njit, prange, set_num_threads, get_num_threads

# Região paralela: cada iteração de prange atua como uma "thread lógica".
@njit(parallel=True)
def apos_barreira():
    nthreads = get_num_threads()
    # Distribui as iterações entre as threads do runtime do Numba
    for tid in prange(nthreads):
        # Impressões em paralelo (a ordem pode intercalar)
        print(f"Passei da barreira. Eu sou o {tid} de {nthreads} processos")

def main():
    # Configura explicitamente 4 threads (como no exemplo do OpenMP)
    set_num_threads(4)

    # "Thread 0" atrasada (simulada no lado Python)
    # Em OpenMP, o tid==0 faria getchar() e depois haveria um barrier.
    # Aqui, simulamos esse atraso ANTES de iniciar a região paralela.
    print("Estou atrasado para a barreira! Tecle enter")
    input()

    # A partir daqui, todos "passaram da barreira" e entram na região paralela.
    apos_barreira()

if __name__ == "__main__":
    main()
