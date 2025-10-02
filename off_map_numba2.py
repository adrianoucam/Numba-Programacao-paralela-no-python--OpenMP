
'''
problema acadêmico desenhado para ser resolvido com o algoritmo (as três “regiões target” com map(to:), map(from:) e map(alloc:) — na sua versão em Numba/CPU paralelo):

Problema — Fusão de sinais com buffer temporário e região de interesse (ROI) em pipeline heterogêneo
Contexto

Você tem dois sinais 1-D sincronizados (ex.: dois sensores distintos de um sistema ciber-físico). Deseja:

Fundir os sinais por soma ponto-a-ponto (estágio 1);

Preparar um buffer temporário interno ao “dispositivo” para normalização/uso posterior (estágio 2);

Refinar uma região de interesse (ROI) (primeira metade do vetor) sobrescrevendo a fusão por um realce simples (estágio 3).

O objetivo é praticar políticas de mapeamento de dados (to/from/alloc) e paralelismo em um pipeline de três estágios.

Dados (entrada)

Inteiro 
𝑁
≥
10
N≥10 (tamanho do vetor).

Vetores inteiros 
𝑎
,
𝑏
∈
𝑍
𝑁
a,b∈Z
N
.

ROI fixa: os primeiros 
𝑁
/
2
N/2 elementos.

Para validação didática, use inicialmente:

𝑁
=
10
N=10, 
𝑎
[
𝑖
]
=
𝑖
+
1
a[i]=i+1, 
𝑏
[
𝑖
]
=
2
(
𝑖
+
1
)
b[i]=2(i+1), 
𝑖
=
0
,
…
,
9
i=0,…,9.

Pipeline a implementar

Estágio 1 (fusão): map(to: a,b) e map(from: c)

𝑐
[
𝑖
]
←
𝑎
[
𝑖
]
+
𝑏
[
𝑖
]
c[i]←a[i]+b[i], 
𝑖
=
0
,
…
,
𝑁
−
1
i=0,…,N−1.

Estágio 2 (buffer temporário): map(to: a) e map(alloc: x)

𝑥
[
𝑖
]
←
2
 
𝑎
[
𝑖
]
x[i]←2a[i], usar 
𝑥
x apenas dentro da região para computar um escalar (ex.: soma ou média) e não devolver 
𝑥
x ao hospedeiro. Retorne apenas esse escalar (ex.: 
s
u
m
(
𝑥
)
sum(x)).

Estágio 3 (refino na ROI): map(to: a[0:N/2]) e map(from: c[0:N/2])
Para 
𝑖
=
0
,
…
,
𝑁
/
2
−
1
i=0,…,N/2−1: 
𝑐
[
𝑖
]
←
2
 
𝑎
[
𝑖
]
c[i]←2a[i].
(Os elementos 
𝑐
[
𝑁
/
2
]
,
…
,
𝑐
[
𝑁
−
1
]
c[N/2],…,c[N−1] permanecem como saíram do Estágio 1.)

Observação: Em Numba/CPU, “map” é modelado por assinatura de função / retorno (passagem de dados) e alocação local; o conceito didático é o mesmo: o que entra, o que sai e o que existe só dentro.

Saída esperada (caso de teste 
𝑁
=
10
N=10)

Após Estágio 1: 
𝑐
=
[
3
,
6
,
9
,
12
,
15
,
18
,
21
,
24
,
27
,
30
]
c=[3,6,9,12,15,18,21,24,27,30].

Após Estágio 3 (ROI = 5 primeiros):

𝑐
=
[
2
,
4
,
6
,
8
,
10
,
18
,
21
,
24
,
27
,
30
]
c=[2,4,6,8,10,18,21,24,27,30].

Estágio 2: retorne um escalar (ex.: 
s
u
m
(
𝑥
)
=
∑
2
𝑎
[
𝑖
]
=
2
∑
(
𝑖
+
1
)
=
2
⋅
55
=
110
sum(x)=∑2a[i]=2∑(i+1)=2⋅55=110) sem materializar 
𝑥
x no hospedeiro.

Tarefas

Implemente os três estágios em funções separadas com Numba:

Estágios 1 e 3 com @njit(parallel=True) e prange.

Estágio 2 com @njit(parallel=True), alocando x localmente e retornando só um escalar.

Verifique a correção no caso 
𝑁
=
10
N=10 e imprima o vetor 
𝑐
c final e o escalar do Estágio 2.

Experimentos de desempenho:

Varie 
𝑁
∈
{
10
6
,
2
⋅
10
6
,
5
⋅
10
6
}
N∈{10
6
,2⋅10
6
,5⋅10
6
} (ajuste à sua RAM).

Meça o tempo de cada estágio separadamente.

Calcule speedup e eficiência do Estágio 1 e do Estágio 3 (paralelos) versus uma versão sequencial (mesma lógica sem prange).

Análise de tráfego de memória (modelagem):

Estime, em bytes, o volume lógico de leitura/escrita por estágio:

E1: lê 
𝑎
,
𝑏
a,b (2N elementos), escreve 
𝑐
c (N).

E2: lê 
𝑎
a (N), escreve 
𝑥
x (N, não retorna), lê 
𝑥
x para reduzir (N).

E3: lê 
𝑎
[
0
:
𝑁
/
2
]
a[0:N/2], escreve 
𝑐
[
0
:
𝑁
/
2
]
c[0:N/2].

Discuta como map(alloc:) reduz tráfego de ida/volta (na analogia com um device real).

Discussão técnica:

Quando e por que preferir alloc em vez de to/from para buffers temporários?

Como a escolha do tamanho da ROI impacta o tempo do Estágio 3?

Identifique gargalos de memória (bound) vs CPU (compute-bound).

Critérios de avaliação

Correção funcional (valores esperados no caso 
𝑁
=
10
N=10).

Qualidade do paralelismo (uso adequado de prange, ausência de race conditions).

Metodologia de medição (warm-up do JIT, múltiplas repetições, média/mediana).

Análise de tráfego e discussão sobre map(alloc:).

Clareza do relatório (1–2 páginas) com tabelas de tempo, speedup e eficiência.

Extensões (opcional)

Generalize o Estágio 3 para uma ROI dinâmica 
[
𝐿
,
𝑈
)
[L,U) e compare tempos.

Substitua o Estágio 2 por uma normalização z-score usando apenas o escalar retornado (média) e um segundo passe para o desvio-padrão — ainda sem retornar x.

Compare com uma implementação equivalente em OpenMP/C com target real e discuta diferenças práticas (latência PCIe, pinning, pageable vs pinned).


'''



# pipeline_map_numba_bench.py
# Estágio 1: c = a + b            (map(to: a,b) map(from: c))
# Estágio 2: x = 2*a (alloc)      (map(to: a)   map(alloc: x))  -> retorna apenas soma(x)
# Estágio 3: c[0:N/2] = 2*a[...]  (map(to: a[0:N/2]) map(from: c[0:N/2]))

import time
import numpy as np
from numba import njit, prange, set_num_threads

# ---------------------------
# Implementações SEQ e PAR
# ---------------------------

@njit
def stage1_seq(a, b):
    n = a.size
    c = np.empty_like(a)
    for i in range(n):
        c[i] = a[i] + b[i]
    return c

@njit(parallel=True)
def stage1_par(a, b):
    n = a.size
    c = np.empty_like(a)
    for i in prange(n):
        c[i] = a[i] + b[i]
    return c

@njit
def stage2_seq_sumx(a):
    # x é "alloc" (temporário); retornamos somente a soma(x)
    n = a.size
    acc = 0
    for i in range(n):
        acc += 2 * a[i]
    return acc

@njit(parallel=True)
def stage2_par_sumx(a):
    # redução paralela suportada pelo Numba para += em escalar simples
    n = a.size
    acc = 0
    for i in prange(n):
        acc += 2 * a[i]
    return acc

@njit
def stage3_seq(a, c):
    # modifica apenas a metade inicial de c
    n2 = a.size // 2
    for i in range(n2):
        c[i] = 2 * a[i]

@njit(parallel=True)
def stage3_par(a, c):
    n2 = a.size // 2
    for i in prange(n2):
        c[i] = 2 * a[i]

# ---------------------------
# Utilidades de benchmark
# ---------------------------

def _median(x):
    y = sorted(x)
    m = len(y) // 2
    return (y[m] if len(y) % 2 else 0.5 * (y[m - 1] + y[m]))

def bench_stage1(N, n_threads, reps=3):
    # dados
    a = np.arange(1, N + 1, dtype=np.int32)
    b = 2 * a

    # warm-up JIT
    _ = stage1_seq(a[:1], b[:1])
    _ = stage1_par(a[:1], b[:1])

    # seq
    t_seq = []
    for _r in range(reps):
        t0 = time.perf_counter()
        c1 = stage1_seq(a, b)
        t1 = time.perf_counter()
        t_seq.append(t1 - t0)

    # par
    set_num_threads(n_threads)
    t_par = []
    for _r in range(reps):
        t0 = time.perf_counter()
        c2 = stage1_par(a, b)
        t1 = time.perf_counter()
        t_par.append(t1 - t0)

    ok = np.array_equal(c1, c2)
    return _median(t_seq), _median(t_par), ok

def bench_stage3(N, n_threads, reps=3):
    a = np.arange(1, N + 1, dtype=np.int32)
    # c inicial vem do estágio 1 (por exemplo); aqui simulamos qualquer estado
    c_seq = np.zeros(N, dtype=np.int32)
    c_par = np.zeros(N, dtype=np.int32)

    # warm-up JIT
    _ = stage3_seq(a[:2], c_seq[:2])
    _ = stage3_par(a[:2], c_par[:2])

    # seq
    t_seq = []
    for _r in range(reps):
        c_seq[:] = 0
        t0 = time.perf_counter()
        stage3_seq(a, c_seq)
        t1 = time.perf_counter()
        t_seq.append(t1 - t0)

    # par
    set_num_threads(n_threads)
    t_par = []
    for _r in range(reps):
        c_par[:] = 0
        t0 = time.perf_counter()
        stage3_par(a, c_par)
        t1 = time.perf_counter()
        t_par.append(t1 - t0)

    # conferir apenas metade alterada; resto igual (zero)
    ok = np.array_equal(c_seq, c_par)
    return _median(t_seq), _median(t_par), ok

# ---------------------------
# Demonstração didática N=10
# ---------------------------

def demo_N10():
    N = 10
    a = np.arange(1, N + 1, dtype=np.int32)
    b = 2 * a

    # warm-ups mínimos
    _ = stage1_seq(a[:1], b[:1]); _ = stage1_par(a[:1], b[:1])
    _ = stage2_seq_sumx(a[:1]);   _ = stage2_par_sumx(a[:1])
    c_demo = stage1_par(a, b)  # estágio 1 (paralelo)
    sumx   = stage2_par_sumx(a) # estágio 2 (paralelo)
    stage3_par(a, c_demo)       # estágio 3 (paralelo)

    print("Demonstração N=10")
    print("a =", a.tolist())
    print("b =", b.tolist())
    # Após estágio 1: soma ponto-a-ponto
    c_stage1 = stage1_seq(a, b)
    print("Após Estágio 1 (c = a + b):", c_stage1.tolist())
    # Estágio 2: soma(x) com x=2*a (alloc)
    print("Estágio 2 (soma de x=2*a):", int(sumx))
    # Estágio 3: metade inicial substituída por 2*a
    print("Após Estágio 3 (ROI 0..N/2-1):", c_demo.tolist())
    # valores esperados:
    # c_stage1 = [3,6,9,12,15,18,21,24,27,30]
    # c_final  = [2,4,6,8,10,18,21,24,27,30]
    # sumx     = 2 * sum(1..10) = 2 * 55 = 110

# ---------------------------
# Main: benchmark + demo
# ---------------------------

def main():
    demo_N10()
    print("\n==== Benchmark ====")
    # escolha um N que caiba na sua RAM
    N_BENCH = 2_000_000
    threads_list = [1, 2, 4, 8]

    # Stage 1
    print("\n[Estágio 1] c = a + b")
    # referência sequencial (com 1 thread)
    t_seq_ref, _, ok = bench_stage1(N_BENCH, n_threads=1, reps=3)
    if not ok:
        print("Aviso: divergência Stage 1 com 1 thread.")
    print("t_seq_ref = %.6f s" % t_seq_ref)
    print("Threads | t_par (s) | Speedup | Eficiência")
    print("--------+-----------+---------+-----------")
    for nt in threads_list:
        t_seq, t_par, ok = bench_stage1(N_BENCH, n_threads=nt, reps=3)
        speed = (t_seq_ref / t_par) if t_par > 0 else float("inf")
        eff   = speed / nt
        print("%7d | %9.6f | %7.3f | %9.3f" % (nt, t_par, speed, eff))
        if not ok:
            print("  * divergência detectada (Stage 1)")

    # Stage 3
    print("\n[Estágio 3] c[0:N/2] = 2*a[0:N/2]")
    # referência sequencial (com 1 thread)
    t_seq_ref, _, ok = bench_stage3(N_BENCH, n_threads=1, reps=3)
    if not ok:
        print("Aviso: divergência Stage 3 com 1 thread.")
    print("t_seq_ref = %.6f s" % t_seq_ref)
    print("Threads | t_par (s) | Speedup | Eficiência")
    print("--------+-----------+---------+-----------")
    for nt in threads_list:
        t_seq, t_par, ok = bench_stage3(N_BENCH, n_threads=nt, reps=3)
        speed = (t_seq_ref / t_par) if t_par > 0 else float("inf")
        eff   = speed / nt
        print("%7d | %9.6f | %7.3f | %9.3f" % (nt, t_par, speed, eff))
        if not ok:
            print("  * divergência detectada (Stage 3)")

    print("\nObservações:")
    print("- Use set_num_threads(k) se quiser fixar manualmente os threads.")
    print("- A paralelização aqui é *data-parallel* e bound por memória; ganhos saturam.")
    print("- A etapa 2 demonstra 'map(alloc: x)': x não retorna ao host (retornamos só soma(x)).")
    print("- Aumente N_BENCH se quiser estressar mais a banda de memória, respeitando sua RAM.")

if __name__ == "__main__":
    main()
