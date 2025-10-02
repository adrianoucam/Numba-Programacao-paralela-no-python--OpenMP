'''
# (opcional) isole ambiente: conda create -n mpi-numba python=3.11 numba mpi4py numpy -c conda-forge
# controle de threads por processo:
set NUMBA_NUM_THREADS=4     # Windows (cmd)
export NUMBA_NUM_THREADS=4  # Linux/macOS

mpiexec -np 4 python dpois_mpi_numba.py --nx 1024 --ny 1024 --tol 1e-4 --max-it 8000 --omega 0.8 --threads 4 --report-every 200

'''
# dpois_mpi_numba.py
# Poisson 2D (fonte conhecida) no quadrado [0,1]x[0,1]
# Distribuição por MPI (faixas de linhas) + paralelismo intra-processo com Numba.
#
# - MPI: mpi4py (OpenMPI/MPICH)
# - Numba: @njit(parallel=True) para atualizar interior e calcular resíduos/erros
#
# Execução:
#   mpirun -np 4 python dpois_mpi_numba.py --nx 1024 --ny 1024 --tol 1e-4 --max-it 8000 --omega 0.8 --threads 4
#
# Observação:
#   Speedup/Eficiência: compare o tempo total (rank0) desta execução com o tempo de
#   uma execução de referência (-np 1, threads=1). Eficiência ~= (T1 / (P * TP)).

import os
import time
import math
import argparse
import numpy as np
from mpi4py import MPI
from numba import njit, prange, set_num_threads

# ---------------------- Utilidades numéricas (Numba) ----------------------

@njit
def u_exact(x, y):
    # solução manufaturada: u* = sin(pi x) sin(pi y)
    return math.sin(math.pi * x) * math.sin(math.pi * y)

@njit
def f_source(x, y):
    # -Laplace(u*) = f  =>  f = 2*pi^2 sin(pi x) sin(pi y)
    return 2.0 * math.pi * math.pi * math.sin(math.pi * x) * math.sin(math.pi * y)

@njit(parallel=True)
def init_dirichlet_local(A, nx, local_ny, hx, hy, g0_row, is_first, is_last):
    """
    Define condições de Dirichlet:
      - bordas x=0 e x=1: sempre setadas (todas as linhas locais, incluindo fantasmas)
      - borda y=0: setada apenas no primeiro rank (linha fantasma superior)
      - borda y=1: setada apenas no último rank (linha fantasma inferior)
    A é (local_ny+2) x (nx+2). g0_row é o índice global da 1ª linha interna deste rank (1..NY).
    """
    # esquerda/direita para TODAS as linhas locais
    for li in prange(0, local_ny + 2):   # inclui fantasmas 0 e local_ny+1
        gy = (g0_row + (li - 1)) * hy    # y global da linha li
        if li == 0:
            gy = (g0_row - 1) * hy
        if li == local_ny + 1:
            gy = (g0_row + local_ny) * hy
        A[li, 0]      = u_exact(0.0, gy)
        A[li, nx + 1] = u_exact(1.0, gy)

    # topo (y=0) só no primeiro rank
    if is_first:
        for j in prange(0, nx + 2):
            x = j * hx
            A[0, j] = u_exact(x, 0.0)

    # base (y=1) só no último rank
    if is_last:
        for j in prange(0, nx + 2):
            x = j * hx
            A[local_ny + 1, j] = u_exact(x, 1.0)

@njit(parallel=True)
def weighted_jacobi_step(A, Anew, nx, local_ny, hx, hy, omega):
    """
    Um passo de Jacobi ponderado no interior (linha 1..local_ny, col 1..nx).
    Anew <- (1-omega)*A + omega*Jacobi(A)  , com RHS f(x,y).
    """
    invhx2 = 1.0 / (hx * hx)
    invhy2 = 1.0 / (hy * hy)
    denom  = 2.0 * (invhx2 + invhy2)

    for li in prange(1, local_ny + 1):
        y = li * hy  # y localizado corretamente pois (g0_row*hy) será somado fora
        for j in range(1, nx + 1):
            # coordenadas globais (x,y)
            x = j * hx
            # laplaciano discreto + fonte
            rhs = f_source(x, (0.0))  # placeholder; será ajustado pelo deslocamento externo
            rhs = rhs  # evita advertência; de fato ajustaremos y correto no chamador
    # (OBS: para evitar recomputar (g0_row) em cada iteração acima, passaremos 'y_offset' no kernel de fato.)

@njit(parallel=True)
def weighted_jacobi_step_with_offset(A, Anew, nx, local_ny, hx, hy, omega, y_offset):
    """
    Versão com offset de y (y = (g0_row + li-0)*hy), para não recalcular dentro do laço Python.
    """
    invhx2 = 1.0 / (hx * hx)
    invhy2 = 1.0 / (hy * hy)
    denom  = 2.0 * (invhx2 + invhy2)

    for li in prange(1, local_ny + 1):
        y = (y_offset + li) * hy
        for j in range(1, nx + 1):
            x = j * hx
            rhs = f_source(x, y)
            jval = ((A[li + 1, j] + A[li - 1, j]) * invhx2 +
                    (A[li, j + 1] + A[li, j - 1]) * invhy2 -
                    rhs) / denom
            Anew[li, j] = (1.0 - omega) * A[li, j] + omega * jval

@njit(parallel=True)
def copy_and_maxdiff_interior(A, Anew, nx, local_ny):
    """
    Copia interior Anew -> A e retorna dt_local = max |Anew - A_old| (apenas interior).
    Redução em duas fases (máximo por linha -> máximo global local).
    """
    row_max = np.zeros(local_ny, dtype=np.float64)
    for li in prange(1, local_ny + 1):
        local_m = 0.0
        for j in range(1, nx + 1):
            diff = Anew[li, j] - A[li, j]
            if diff < 0.0:
                diff = -diff
            if diff > local_m:
                local_m = diff
            A[li, j] = Anew[li, j]
        row_max[li - 1] = local_m
    dt = 0.0
    for i in range(local_ny):
        if row_max[i] > dt:
            dt = row_max[i]
    return dt

@njit(parallel=True)
def errors_local(A, nx, local_ny, hx, hy, y_offset):
    """
    Erros locais (apenas interior) contra u_exact: retorna (l2sq_local, linf_local).
    l2sq_local = soma dos e^2 (sem multiplicar por hx*hy; multiplicaremos fora).
    """
    l2sq = 0.0
    linf = 0.0
    for li in prange(1, local_ny + 1):
        y = (y_offset + li) * hy
        local_linf = 0.0
        local_l2   = 0.0
        for j in range(1, nx + 1):
            x = j * hx
            e = A[li, j] - u_exact(x, y)
            ae = e if e >= 0.0 else -e
            if ae > local_linf:
                local_linf = ae
            local_l2 += e * e
        if local_linf > linf:
            linf = local_linf
        l2sq += local_l2
    return l2sq, linf

# ---------------------- Decomposição e Halo (MPI) ----------------------

def decompose_rows(ny, size, rank):
    """
    Particiona 'ny' linhas internas entre 'size' ranks.
    Retorna (local_ny, g0_row), onde g0_row é o índice global (1..ny) da 1ª linha interna do rank.
    """
    base = ny // size
    rem  = ny % size
    if rank < rem:
        local_ny = base + 1
        start = rank * (base + 1)
    else:
        local_ny = base
        start = rem * (base + 1) + (rank - rem) * base
    g0_row = start  # deslocamento 0-based para interior; y_offset = g0_row
    return local_ny, g0_row

def halo_exchange(comm, A, nx, local_ny, rank, size):
    """
    Troca halos verticais (linhas fantasmas) com vizinhos:
      - Envia linha 1 ao vizinho de cima; recebe fantasma inferior (local_ny+1) do vizinho de baixo.
      - Envia linha local_ny ao vizinho de baixo; recebe fantasma superior (0) do vizinho de cima.
    """
    up   = rank - 1 if rank > 0 else MPI.PROC_NULL
    down = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    # buffer são *views* de linhas contíguas
    send_up    = A[1, :].copy()
    recv_down  = np.empty_like(send_up)
    send_down  = A[local_ny, :].copy()
    recv_up    = np.empty_like(send_up)

    comm.Sendrecv(sendbuf=send_up,   dest=up,   sendtag=10,
                  recvbuf=recv_down, source=down, recvtag=10)
    comm.Sendrecv(sendbuf=send_down, dest=down, sendtag=20,
                  recvbuf=recv_up,   source=up,   recvtag=20)

    if down != MPI.PROC_NULL:
        A[local_ny + 1, :] = recv_down
    if up != MPI.PROC_NULL:
        A[0, :]            = recv_up

# ---------------------- Programa principal ----------------------

def main():
    parser = argparse.ArgumentParser(description="Poisson 2D com Jacobi ponderado (MPI + Numba).")
    parser.add_argument("--nx", type=int, default=1024, help="pontos internos em x")
    parser.add_argument("--ny", type=int, default=1024, help="pontos internos em y")
    parser.add_argument("--tol", type=float, default=1e-4, help="tolerância (norma inf do delta)")
    parser.add_argument("--max-it", type=int, default=8000, help="máx. iterações")
    parser.add_argument("--omega", type=float, default=0.8, help="fator do Jacobi ponderado (0<ω<=1)")
    parser.add_argument("--threads", type=int, default=0, help="threads Numba por processo (0 = usar NUMBA_NUM_THREADS/env)")
    parser.add_argument("--report-every", type=int, default=200, help="intervalo de relatório de dt")
    args = parser.parse_args()

    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    # Threads por processo (Numba)
    if args.threads > 0:
        set_num_threads(args.threads)
    # (opcional) respeitar env NUMBA_NUM_THREADS se threads==0

    nx, ny = args.nx, args.ny
    hx = 1.0 / (nx + 1)
    hy = 1.0 / (ny + 1)

    # Decomposição por faixas de linhas
    local_ny, g0_row = decompose_rows(ny, size, rank)
    is_first = (rank == 0)
    is_last  = (rank == size - 1)

    # Alocação com fantasmas: (local_ny+2) x (nx+2)
    A    = np.empty((local_ny + 2, nx + 2), dtype=np.float64)
    Anew = np.empty_like(A)

    # Inicialização de fronteiras (Dirichlet) e interior=0
    init_dirichlet_local(A,    nx, local_ny, hx, hy, g0_row, is_first, is_last)
    init_dirichlet_local(Anew, nx, local_ny, hx, hy, g0_row, is_first, is_last)

    # Warm-up JIT: compilar kernels nas formas locais
    init_t0 = time.perf_counter()
    _ = errors_local(A, nx, local_ny, hx, hy, g0_row)
    _ = copy_and_maxdiff_interior(A, Anew, nx, local_ny)
    _ = weighted_jacobi_step_with_offset(A, Anew, nx, local_ny, hx, hy, args.omega, g0_row)
    comm.Barrier()
    init_t1 = time.perf_counter()

    # Loop de Jacobi
    dt = 1.0
    it = 0
    comm.Barrier()
    t0 = MPI.Wtime()

    while True:
        # 1) halo: troca de linhas com vizinhos
        halo_exchange(comm, A, nx, local_ny, rank, size)

        # 2) passo Jacobi ponderado (usando y_offset = g0_row)
        weighted_jacobi_step_with_offset(A, Anew, nx, local_ny, hx, hy, args.omega, g0_row)

        # 3) copia interior + delta local
        dt_local = copy_and_maxdiff_interior(A, Anew, nx, local_ny)

        # 4) redução global do delta
        dt = comm.allreduce(dt_local, op=MPI.MAX)

        it += 1
        if (rank == 0) and (it % args.report_every == 0):
            print(f"[iter {it:6d}] dt = {dt:.3e}")

        if (dt <= args.tol) or (it >= args.max_it):
            break

    comm.Barrier()
    t1 = MPI.Wtime()

    # Erros globais vs. solução exata
    l2sq_local, linf_local = errors_local(A, nx, local_ny, hx, hy, g0_row)
    l2sq_glob = comm.allreduce(l2sq_local, op=MPI.SUM)
    linf_glob = comm.allreduce(linf_local, op=MPI.MAX)
    l2_glob   = math.sqrt(hx * hy * l2sq_glob)

    # Tempo global = máximo entre ranks
    elapsed = comm.allreduce(t1 - t0, op=MPI.MAX)
    warmup  = comm.allreduce(init_t1 - init_t0, op=MPI.MAX)

    if rank == 0:
        th_env = os.getenv("NUMBA_NUM_THREADS", "env_not_set")
        print("\n== RESUMO ==")
        print(f"ranks (MPI): {size} | threads/Rank (Numba): {args.threads or th_env}")
        print(f"grid: nx={nx}, ny={ny} | hx={hx:.3e}, hy={hy:.3e}")
        print(f"omega={args.omega:.2f} | tol={args.tol:.2e} | max_it={args.max_it}")
        print(f"iters={it} | dt_final={dt:.3e}")
        print(f"tempo_total={elapsed:.3f}s | warmup={warmup:.3f}s")
        print(f"erro L2 = {l2_glob:.6e} | erro Linf = {linf_glob:.6e}")
        print("\nDica: Para medir speedup/eficiência, compare este tempo com:")
        print("  1) mpirun -np 1 --bind-to none python dpois_mpi_numba.py --nx ... --ny ... --threads 1")
        print("Speedup ~= T(1proc,1thr) / T(P,thr) ;  Eficiência ~= Speedup / P")

if __name__ == "__main__":
    main()
