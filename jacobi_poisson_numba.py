'''

um problema acadêmico completo (com validação e análise) para ser resolvido com o seu Jacobi 2D — com um ajuste no algoritmo para resolver Poisson em vez de apenas Laplace. Incluo também um script pronto (Numba/CPU) que calcula erros, imprime iterações e permite estudos de convergência e desempenho.

Problema acadêmico — Poisson 2D com solução manufaturada (validação do Jacobi)
Enunciado

Considere o problema de Poisson no quadrado unitário 
Ω
=
(
0
,
1
)
×
(
0
,
1
)
Ω=(0,1)×(0,1):

−
∇
2
𝑢
(
𝑥
,
𝑦
)
=
𝑓
(
𝑥
,
𝑦
)
em 
Ω
,
𝑢
(
𝑥
,
𝑦
)
=
𝑔
(
𝑥
,
𝑦
)
em 
∂
Ω
.
−∇
2
u(x,y)=f(x,y)em Ω,u(x,y)=g(x,y)em ∂Ω.

Use a técnica de solução manufaturada:

𝑢
⋆
(
𝑥
,
𝑦
)
=
sin
⁡
(
𝜋
𝑥
)
sin
⁡
(
𝜋
𝑦
)
.
u
⋆
(x,y)=sin(πx)sin(πy).

Então

∇
2
𝑢
⋆
=
−
2
𝜋
2
sin
⁡
(
𝜋
𝑥
)
sin
⁡
(
𝜋
𝑦
)
  
  
⇒
  
  
𝑓
(
𝑥
,
𝑦
)
=
2
𝜋
2
sin
⁡
(
𝜋
𝑥
)
sin
⁡
(
𝜋
𝑦
)
.
∇
2
u
⋆
=−2π
2
sin(πx)sin(πy)⇒f(x,y)=2π
2
sin(πx)sin(πy).

Imponha fronteiras de Dirichlet com 
𝑔
=
𝑢
⋆
g=u
⋆
 em todo o contorno.

Tarefas

Resolver Poisson com o método de Jacobi em malha 
𝑁
𝑥
×
𝑁
𝑦
N
x
	​

×N
y
	​

 (nós internos), usando o esquema de 5 pontos.

Usar critério de parada 
∥
𝑢
(
𝑘
+
1
)
−
𝑢
(
𝑘
)
∥
∞
<
MAX_TEMP_ERROR
∥u
(k+1)
−u
(k)
∥
∞
	​

<MAX_TEMP_ERROR ou 
MAX_ITER
MAX_ITER iterações.

Validar comparando 
𝑢
u numérico com 
𝑢
⋆
u
⋆
: relatar 
∥
𝑒
∥
∞
∥e∥
∞
	​

 e 
∥
𝑒
∥
2
∥e∥
2
	​

.

Estudo de malha: rodar para 
𝑁
=
64
,
128
,
256
N=64,128,256 (com 
𝑁
𝑥
=
𝑁
𝑦
=
𝑁
N
x
	​

=N
y
	​

=N) e estimar a ordem assintótica de convergência.

Desempenho: variar nº de threads (1,2,4,8) e relatar tempo/iter e total.

Ajuste no algoritmo (Jacobi para Poisson)

Para passo de malha 
ℎ
𝑥
=
1
/
(
𝑁
𝑥
+
1
)
h
x
	​

=1/(N
x
	​

+1), 
ℎ
𝑦
=
1
/
(
𝑁
𝑦
+
1
)
h
y
	​

=1/(N
y
	​

+1), o update interno é:

𝑢
𝑖
,
𝑗
(
𝑘
+
1
)
=
𝑢
𝑖
+
1
,
𝑗
(
𝑘
)
+
𝑢
𝑖
−
1
,
𝑗
(
𝑘
)
ℎ
𝑥
2
+
𝑢
𝑖
,
𝑗
+
1
(
𝑘
)
+
𝑢
𝑖
,
𝑗
−
1
(
𝑘
)
ℎ
𝑦
2
−
𝑓
𝑖
,
𝑗
2
(
1
ℎ
𝑥
2
+
1
ℎ
𝑦
2
)
.
u
i,j
(k+1)
	​

=
2(
h
x
2
	​

1
	​

+
h
y
2
	​

1
	​

)
h
x
2
	​

u
i+1,j
(k)
	​

+u
i−1,j
(k)
	​

	​

+
h
y
2
	​

u
i,j+1
(k)
	​

+u
i,j−1
(k)
	​

	​

−f
i,j
	​

	​

.

Para malha uniforme 
ℎ
𝑥
=
ℎ
𝑦
=
ℎ
h
x
	​

=h
y
	​

=h, reduz a

𝑢
𝑖
,
𝑗
(
𝑘
+
1
)
=
1
4
(
𝑢
𝑖
+
1
,
𝑗
(
𝑘
)
+
𝑢
𝑖
−
1
,
𝑗
(
𝑘
)
+
𝑢
𝑖
,
𝑗
+
1
(
𝑘
)
+
𝑢
𝑖
,
𝑗
−
1
(
𝑘
)
−
ℎ
2
𝑓
𝑖
,
𝑗
)
.
u
i,j
(k+1)
	​

=
4
1
	​

(u
i+1,j
(k)
	​

+u
i−1,j
(k)
	​

+u
i,j+1
(k)
	​

+u
i,j−1
(k)
	​

−h
2
f
i,j
	​

).


'''
# jacobi_poisson_numba.py
import time
import math
import numpy as np
from numba import njit, prange, set_num_threads

# Parâmetros da malha (nós internos)
NX, NY = 256, 256
MAX_TEMP_ERROR = 1e-4
MAX_ITER = 10000

# ---------------------------------------------
# Utilidades de coordenadas e manufatura
# ---------------------------------------------
@njit
def hx_hy(nx, ny):
    return 1.0 / (nx + 1), 1.0 / (ny + 1)

@njit
def f_source(x, y):
    # f(x,y) = 2*pi^2 sin(pi x) sin(pi y)
    return 2.0 * math.pi * math.pi * math.sin(math.pi * x) * math.sin(math.pi * y)

@njit
def u_exact(x, y):
    return math.sin(math.pi * x) * math.sin(math.pi * y)

# ---------------------------------------------
# Inicialização: fronteiras g = u_exact
# Matrizes com "ghost" layers: (NY+2) x (NX+2) => A[y, x]
# ---------------------------------------------
@njit
def init_dirichlet_from_exact(A, NX, NY):
    hx, hy = hx_hy(NX, NY)
    # zera interior
    for i in range(1, NY + 1):
        for j in range(1, NX + 1):
            A[i, j] = 0.0
    # contorno: y=0, y=1, x=0, x=1
    for j in range(0, NX + 2):
        x = j * hx
        A[0, j]      = u_exact(x, 0.0)     # y=0
        A[NY + 1, j] = u_exact(x, 1.0)     # y=1
    for i in range(0, NY + 2):
        y = i * hy
        A[i, 0]      = u_exact(0.0, y)     # x=0
        A[i, NX + 1] = u_exact(1.0, y)     # x=1

# ---------------------------------------------
# Passo de Jacobi (Poisson 2D, malha poss. não uniforme)
# ---------------------------------------------
@njit(parallel=True)
def jacobi_poisson_update(A, Anew, NX, NY):
    hx, hy = hx_hy(NX, NY)
    invhx2 = 1.0 / (hx * hx)
    invhy2 = 1.0 / (hy * hy)
    denom = 2.0 * (invhx2 + invhy2)

    for i in prange(1, NY + 1):
        y = i * hy
        for j in range(1, NX + 1):
            x = j * hx
            rhs = f_source(x, y)
            Anew[i, j] = ((A[i + 1, j] + A[i - 1, j]) * invhx2 +
                          (A[i, j + 1] + A[i, j - 1]) * invhy2 -
                          rhs) / denom
    # fronteiras permanecem em Anew como já impostas antes do loop (não tocar aqui)

@njit(parallel=True)
def copy_and_maxdiff(A, Anew, NX, NY):
    # Atualiza interior e retorna norma-infinito da diferença
    row_max = np.zeros(NY, dtype=np.float64)
    for i in prange(1, NY + 1):
        local_max = 0.0
        for j in range(1, NX + 1):
            diff = Anew[i, j] - A[i, j]
            if diff < 0.0:
                diff = -diff
            if diff > local_max:
                local_max = diff
            A[i, j] = Anew[i, j]
        row_max[i - 1] = local_max
    dt = 0.0
    for i in range(NY):
        if row_max[i] > dt:
            dt = row_max[i]
    return dt

@njit(parallel=True)
def boundary_copy(A_src, A_dst, NX, NY):
    # copia apenas as bordas de A_src -> A_dst (mantendo Dirichlet consistente)
    for j in prange(0, NX + 2):
        A_dst[0, j]      = A_src[0, j]
        A_dst[NY + 1, j] = A_src[NY + 1, j]
    for i in prange(0, NY + 2):
        A_dst[i, 0]      = A_src[i, 0]
        A_dst[i, NX + 1] = A_src[i, NX + 1]

# ---------------------------------------------
# Métricas de erro vs solução exata
# ---------------------------------------------
@njit(parallel=True)
def errors_against_exact(A, NX, NY):
    hx, hy = hx_hy(NX, NY)
    # L2 e Linf no interior
    l2_acc = 0.0
    linf = 0.0
    for i in prange(1, NY + 1):
        y = i * hy
        local_linf = 0.0
        local_l2 = 0.0
        for j in range(1, NX + 1):
            x = j * hx
            e = A[i, j] - u_exact(x, y)
            ae = e if e >= 0.0 else -e
            if ae > local_linf:
                local_linf = ae
            local_l2 += e * e
        # redução manual parcial
        if local_linf > linf:
            linf = local_linf
        l2_acc += local_l2
    area = 1.0  # domínio unitário; L2 discreto ~ sqrt(hx*hy*sum e^2)
    l2 = math.sqrt((hx * hy) * l2_acc / area)
    return l2, linf

# ---------------------------------------------
# Driver
# ---------------------------------------------
def main():
    # Ajuste de threads (opcional)
    # set_num_threads(8)

    # Alocação com ghost layers
    A    = np.empty((NY + 2, NX + 2), dtype=np.float64)
    Anew = np.empty_like(A)

    # Inicializa fronteiras com solução exata e interior zero
    init_dirichlet_from_exact(A, NX, NY)
    init_dirichlet_from_exact(Anew, NX, NY)  # garante mesmas bordas em Anew
    # Warm-up JIT
    jacobi_poisson_update(A, Anew, NX, NY)
    _ = copy_and_maxdiff(A, Anew, NX, NY)

    iteration = 1
    dt = 1.0

    t0 = time.perf_counter()
    while dt > MAX_TEMP_ERROR and iteration <= MAX_ITER:
        # garante bordas fixas também em Anew antes do passo
        boundary_copy(A, Anew, NX, NY)
        jacobi_poisson_update(A, Anew, NX, NY)
        dt = copy_and_maxdiff(A, Anew, NX, NY)
        if iteration % 200 == 0:
            print("iter %d  dt=%.3e" % (iteration, dt))
        iteration += 1
    t1 = time.perf_counter()

    l2, linf = errors_against_exact(A, NX, NY)
    print("\nConvergiu em %d iterações (dt=%.3e). Tempo: %.3fs"
          % (iteration - 1, dt, t1 - t0))
    print("Erro L2  = %.6e" % l2)
    print("Erro Linf= %.6e" % linf)

if __name__ == "__main__":
    main()
