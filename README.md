# Numba-Programacao-paralela-no-python--OpenMP
Numba Programacao paralela no python- OpenMP

Numba + (OpenMP → Python) — Guia e Exemplos

Este repositório reúne traduções de padrões do OpenMP para Python com Numba e alguns exercícios/benchmarks. A ideia é mostrar, na prática, como mapear parallel for, collapse, private/firstprivate, map(to/from/alloc), dependências entre regiões, e até MPI + Numba para execução híbrida.

Todos os códigos estão em Python 3.10+ com Numba e NumPy. Onde indicado, há variações com mpi4py e (opcionalmente) CUDA.

Requisitos

Python 3.10 ou 3.11

numpy

numba

(opcional) mpi4py + OpenMPI/MPICH para a versão distribuída

(opcional) CUDA Toolkit ≥ 11.2 para usar Numba-CUDA (se aplicável)

Instalação rápida (via conda)
conda create -n numba-playground python=3.11 numpy numba -c conda-forge -y
conda activate numba-playground

# Para MPI:
conda install mpi4py -c conda-forge -y
# (instale OpenMPI/MPICH do seu sistema, se necessário)


Dicas (Numba vs OpenMP)

#pragma omp parallel for → @njit(parallel=True) + prange(...).

collapse(2) → linearizar (k,j) em um único índice idx com prange(total).

private → variável local dentro do loop; firstprivate → passar por valor como argumento.

map(to/from/alloc) → modelar com assinatura/retorno de função e alocação local.

reduction(max: dt) → usar redução em duas fases (máximos por linha → máximo global).

single/critical → organizar a lógica no host; seleções globais baratas podem ficar sequenciais.

prange só aceita passo 1.

Evite f"{i:2d}" em @njit: use concatenação ("texto "+str(i)) ou "%d".

Controle de threads (por processo):

# Windows (cmd)
set NUMBA_NUM_THREADS=8

# Linux/macOS
export NUMBA_NUM_THREADS=8


Exemplos & Scripts
1) Barreira e região paralela

Arquivo: numba_barreira.py
Mostra: simulação de omp barrier (a “espera” ocorre antes da região paralela).
Rodar:

python numba_barreira.py

2) omp for básico (sem ID real de thread)

Arquivo: omp_for_numba.py
Mostra: prange e “workers lógicos” para ilustrar particionamento.
Nota: evite f-strings formatadas dentro de @njit.
Rodar:

python omp_for_numba.py

3) Contagem de primos paralela

Arquivo: off_primos_omp.py (versão Numba)
Mostra: paralelismo com prange, correção do passo 2 (usar k → i=2*k+3).
Rodar:

python off_primos_omp.py 1000

4) firstprivate/private e map(to/from/alloc)

Arquivo: off_codigo1_numba.py
Mostra: escalar passado por valor (firstprivate), variável local por thread (private), cópias de entrada/saída (to/from) e alocação temporária (alloc).
Observação: há fallback automático para CPU; se quiser forçar CPU:

# Windows cmd
set NUMBA_DISABLE_CUDA=1
# Linux/macOS
export NUMBA_DISABLE_CUDA=1


Rodar:

python off_codigo1_numba.py

5) Dependências entre regiões (depend/nowait)

Arquivo: off_depend_nowait_numba.py
Mostra: pipeline em duas fases independentes, garantindo ordem via chamadas sequenciais:

b = 2*a

c = b + 5
Rodar:

python off_depend_nowait_numba.py

6) collapse(2) + lastprivate

Arquivo: omp_collapse_numba.py
Mostra: linearização do par (k,j) e recuperação do “último” (kmax, jmax).
Rodar:

python omp_collapse_numba.py

6.1) Benchmark de collapse: speedup e eficiência

Arquivo: collapse_benchmark_numba.py
Rodar:

python collapse_benchmark_numba.py

7) Dijkstra (menor caminho a partir de 0)

Relaxamento paralelo (prange) — seleção do mínimo sequencial:

Core: dijkstra_numba.py

Benchmark (speedup/eficiência):

dijkstra_numba_bench.py

Exemplo didático com reconstrução de rota:

dijkstra_exemplo_didatico.py

Rodar:

python dijkstra_numba.py
python dijkstra_numba_bench.py
python dijkstra_exemplo_didatico.py

8) map(to/from/alloc) em pipeline vetorial

Três estágios (c=a+b; x=2*a alocado e usado localmente; ROI):
Arquivo: off_map_numba.py

Problema acadêmico + benchmark de estágios:
Arquivo: pipeline_map_numba_bench.py

Rodar:

python off_map_numba.py
python pipeline_map_numba_bench.py

9) Jacobi 2D (Laplace) com fronteiras fixas

Arquivo: jacobi_numba.py
Mostra: atualização interior paralela e redução max em duas fases.
Rodar:

python jacobi_numba.py

10) Poisson 2D (solução manufaturada) — validação e estudo de malha

Arquivo: jacobi_poisson_numba.py
Mostra: ajuste do Jacobi para Poisson (inclui RHS), cálculo de erros L2/L∞ e logs de convergência.
Rodar:

python jacobi_poisson_numba.py

11) MPI + Numba (híbrido) — Poisson 2D distribuído

Arquivo: dpois_mpi_numba.py
Mostra: decomposição 1D em faixas de linhas, troca de halos com Sendrecv, reduções globais (allreduce), e paralelismo interno por processo com Numba.

Exemplo (4 processos, 4 threads/proc):

# Windows (PowerShell)
$env:NUMBA_NUM_THREADS="4"
mpirun -np 4 python dpois_mpi_numba.py --nx 1024 --ny 1024 --tol 1e-4 --max-it 8000 --omega 0.8 --threads 4 --report-every 200

# Linux/macOS
export NUMBA_NUM_THREADS=4
mpirun -np 4 python dpois_mpi_numba.py --nx 1024 --ny 1024 --tol 1e-4 --max-it 8000 --omega 0.8 --threads 4 --report-every 200

🧪 Medição: speedup & eficiência

Speedup: 
𝑆
𝑝
=
𝑇
1
/
𝑇
𝑝
S
p
	​

=T
1
	​

/T
p
	​


Eficiência: 
𝐸
𝑝
=
𝑆
𝑝
/
𝑝
E
p
	​

=S
p
	​

/p

Dicas:

Faça warm-up (chame o kernel uma vez) antes de cronometrar.

Meça várias vezes e use mediana.

Fixe NUMBA_NUM_THREADS para comparações justas.

Armadilhas & Soluções

prange passo ≠ 1 → erro: “Only constant step size of 1 is supported”.
✅ Use índice auxiliar: k in prange(m) e mapeie i = 2*k+3, etc.

f-strings formatadas em @njit → UnsupportedBytecodeError.
✅ Use concatenação ("txt "+str(i)) ou "%d".

print em funções jitted → é suportado mas lento/limitado; prefira imprimir no host.

CUDA (Numba-CUDA) → cuidado com versões. Numba requer CUDA Toolkit ≥ 11.2.
✅ Se não quiser GPU: NUMBA_DISABLE_CUDA=1.

Reduções (máximo/soma) → para máximo, use duas fases (por linha → global); soma simples pode ser direta em prange em muitos casos.
