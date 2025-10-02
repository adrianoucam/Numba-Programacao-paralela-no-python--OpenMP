# off_codigo1_numba.py
'''


Detecção de GPU: uso numba.cuda.is_available(); se não houver GPU ou a seleção falhar, cai para CPU.

firstprivate: o escalar var_firstprivate é passado por valor ao kernel (CUDA/CPU), como no OpenMP.

private: var_private_local = 50 é uma variável local por thread (CUDA) ou por iteração (CPU), não afeta var_private do hospedeiro.

map(to/from): no CUDA, to_device/copy_to_host reproduz o comportamento; no CPU, apenas retornamos a result.

Evitei prints dentro de funções @njit e kernels CUDA (podem funcionar, mas são lentos/verbosos). As mensagens didáticas são emitidas no host.

Se você quiser forçar o uso de 4 threads no caminho CPU (como no seu num_threads(4)), basta adicionar from numba import set_num_threads; set_num_threads(4) antes de chamar kernel_cpu.


'''
# off_codigo1_numba.py
# Equivalente ao off_codigo1.c usando Numba (CUDA se disponível; CPU paralelo como fallback)

# off_codigo1_numba_cpu_first.py
import numpy as np
from numba import njit, prange

def _cuda_ok():
    try:
        from numba import cuda
        if not cuda.is_available():
            return None
        # toolkit version check (ex.: (12, 4, 0))
        try:
            tk = cuda.runtime.get_version()
            if tk and isinstance(tk, tuple) and tk[0] < 11:
                return None
        except Exception:
            pass
        return cuda
    except Exception:
        return None

CUDA = _cuda_ok()  # será None se toolkit < 11.2, sem GPU, etc.

@njit(parallel=True)
def kernel_cpu(vetor, var_firstprivate):
    n = vetor.shape[0]
    result = np.empty_like(vetor)
    for i in prange(n):
        var_private_local = 50
        result[i] = vetor[i] * var_firstprivate + var_private_local
    return result

if CUDA:
    @CUDA.jit
    def kernel_cuda(vetor, result, var_firstprivate):
        i = CUDA.grid(1)
        if i < vetor.size:
            var_private_local = 50
            result[i] = vetor[i] * var_firstprivate + var_private_local

def main():
    var_firstprivate = 10
    var_private = 20
    vetor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = np.empty_like(vetor)

    if CUDA is None:
        print("Executando no Hospedeiro (CPU) — CUDA indisponível ou toolkit < 11.2.")
        result = kernel_cpu(vetor, var_firstprivate)
    else:
        try:
            device_num = 0
            CUDA.select_device(device_num)
            print("Executando no Dispositivo", device_num)
            d_vetor = CUDA.to_device(vetor)
            d_result = CUDA.device_array_like(vetor)
            tpb = 128
            blocks = (vetor.size + tpb - 1) // tpb or 1
            kernel_cuda[blocks, tpb](d_vetor, d_result, var_firstprivate)
            result = d_result.copy_to_host()
        except Exception as e:
            print("Falha no CUDA; usando CPU:", e)
            result = kernel_cpu(vetor, var_firstprivate)
            print("Executando no Hospedeiro (fallback)")

    print("\nValores finais no hospedeiro:")
    print("var_firstprivate =", var_firstprivate)
    print("var_private =", var_private)
    print("Vetor result =", result.tolist())

if __name__ == "__main__":
    main()

