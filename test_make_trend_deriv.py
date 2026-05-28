"""
test_make_trend_deriv.py
Teste make_trend_handle_deriv — Hermite (der=0, der=1) et Legendre (der=0, der=1).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from branche5 import (
    uq_PCK_eval_unipoly, uq_PCE_create_Psi,
    uq_eval_hermite_deriv, uq_eval_legendre_deriv,
)

# ---------------------------------------------------------------------------
# make_trend_handle  (copie de branche3)
# ---------------------------------------------------------------------------
def make_trend_handle(selected_idx, Indices, poly_types):
    Idx_sel = Indices[np.array(selected_idx), :]
    p_types = poly_types
    def F_handle(U):
        uv = uq_PCK_eval_unipoly(U, Idx_sel, p_types)
        return uq_PCE_create_Psi(Idx_sel, uv)
    return F_handle


# ---------------------------------------------------------------------------
# make_trend_handle_deriv  (prototype)
# ---------------------------------------------------------------------------
def make_trend_handle_deriv(selected_idx, Indices, poly_types, der):
    Idx_sel = Indices[np.array(selected_idx), :]
    p_types = poly_types
    def F_der_handle(U):
        uv  = uq_PCK_eval_unipoly(U, Idx_sel, p_types)
        P   = uv.shape[2] - 1
        pt  = p_types[der].lower()
        if pt == 'hermite':
            uv[:, der, :] = uq_eval_hermite_deriv(P, U[:, der])
        elif pt == 'legendre':
            uv[:, der, :] = uq_eval_legendre_deriv(P, U[:, der])
        Psi = uq_PCE_create_Psi(Idx_sel, uv)
        Psi[:, Idx_sel[:, der] == 0] = 0.0
        return Psi
    return F_der_handle


# ---------------------------------------------------------------------------
# Helper FD
# ---------------------------------------------------------------------------
EPS = 1e-5

def fd(F_h, U, dim):
    N   = U.shape[0]
    res = np.zeros((N, F_h(U).shape[1]))
    for n in range(N):
        Up, Um = U.copy(), U.copy()
        Up[n, dim] += EPS
        Um[n, dim] -= EPS
        res[n, :] = (F_h(Up)[n, :] - F_h(Um)[n, :]) / (2 * EPS)
    return res


def run_test(label, F_der, F_h, U, dim, expected):
    ok_ana = np.allclose(F_der, expected,       atol=1e-12)
    ok_fd  = np.allclose(F_der, fd(F_h, U, dim), atol=1e-7)
    status = "PASS" if (ok_ana and ok_fd) else "FAIL"
    print(f"  {label} — analytique:{ok_ana}  FD:{ok_fd}  [{status}]")
    if not ok_ana: print("    err max ana:", np.abs(F_der - expected).max())
    if not ok_fd:  print("    err max FD :", np.abs(F_der - fd(F_h, U, dim)).max())


# ===========================================================================
# HERMITE — base degré ≤ 2 : [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]
# H_0=1, H_1=x, H_2=(x²-1)/√2
# H'_0=0, H'_1=1, H'_2=√2·x
# ===========================================================================
print("=== HERMITE ===")
np.random.seed(0)
N  = 5
UH = np.random.randn(N, 2)
U0, U1 = UH[:, 0], UH[:, 1]

Idx_H = np.array([[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]])
sel   = list(range(6))
pt_H  = ['hermite', 'hermite']

F_h_H = make_trend_handle(sel, Idx_H, pt_H)

# der=0
run_test("der=0", make_trend_handle_deriv(sel, Idx_H, pt_H, 0)(UH), F_h_H, UH, 0,
    np.column_stack([np.zeros(N), np.ones(N), np.zeros(N),
                     np.sqrt(2)*U0, U1, np.zeros(N)]))

# der=1
run_test("der=1", make_trend_handle_deriv(sel, Idx_H, pt_H, 1)(UH), F_h_H, UH, 1,
    np.column_stack([np.zeros(N), np.zeros(N), np.ones(N),
                     np.zeros(N), U0, np.sqrt(2)*U1]))


# ===========================================================================
# LEGENDRE — base degré ≤ 2 : [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]
# L_0=1/√2, L_1=√(3/2)·x, L_2=√(5/2)·(3x²-1)/2
# L'_0=0, L'_1=√(3/2), L'_2=3√(5/2)·x
# ===========================================================================
print("\n=== LEGENDRE ===")
np.random.seed(1)
UL = np.random.uniform(-1, 1, (N, 2))
V0, V1 = UL[:, 0], UL[:, 1]

# Normalisation (vérifiée) : L_n = sqrt(2n+1) * P_n
# L_0=1, L_1=√3·x, L_2=√5·(3x²-1)/2
# L'_0=0, L'_1=√3, L'_2=3√5·x

Idx_L = np.array([[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]])
pt_L  = ['legendre', 'legendre']

F_h_L = make_trend_handle(sel, Idx_L, pt_L)

# der=0 :
# [0,0] : L'_0(V0)·L_0(V1)   = 0·1     = 0
# [1,0] : L'_1(V0)·L_0(V1)   = √3·1    = √3
# [0,1] : L'_0(V0)·L_1(V1)   = 0       = 0
# [2,0] : L'_2(V0)·L_0(V1)   = 3√5·V0  = 3√5·V0
# [1,1] : L'_1(V0)·L_1(V1)   = √3·√3·V1= 3·V1
# [0,2] : L'_0(V0)·L_2(V1)   = 0       = 0
exp_der0_L = np.column_stack([
    np.zeros(N),
    np.full(N, np.sqrt(3)),
    np.zeros(N),
    3 * np.sqrt(5) * V0,
    3 * V1,
    np.zeros(N),
])

run_test("der=0", make_trend_handle_deriv(sel, Idx_L, pt_L, 0)(UL), F_h_L, UL, 0,
         exp_der0_L)

# der=1 : symétrique
# [0,0] : 0
# [1,0] : L_1(V0)·L'_0(V1)   = 0
# [0,1] : L_0(V0)·L'_1(V1)   = 1·√3   = √3
# [2,0] : L_2(V0)·L'_0(V1)   = 0
# [1,1] : L_1(V0)·L'_1(V1)   = √3·V0·√3 = 3·V0
# [0,2] : L_0(V0)·L'_2(V1)   = 3√5·V1
exp_der1_L = np.column_stack([
    np.zeros(N),
    np.zeros(N),
    np.full(N, np.sqrt(3)),
    np.zeros(N),
    3 * V0,
    3 * np.sqrt(5) * V1,
])

run_test("der=1", make_trend_handle_deriv(sel, Idx_L, pt_L, 1)(UL), F_h_L, UL, 1,
         exp_der1_L)


# ---------------------------------------------------------------------------
# make_trend_global_handle  (prototype)
# ---------------------------------------------------------------------------
def make_trend_global_handle(selected_idx, Indices, poly_types):
    Idx_sel = Indices[np.array(selected_idx), :]
    M = Idx_sel.shape[1]
    F0 = make_trend_handle(selected_idx, Indices, poly_types)
    Fk = [make_trend_handle_deriv(selected_idx, Indices, poly_types, k)
          for k in range(M)]
    def F_global(U):
        blocks = [F0(U)] + [fk(U) for fk in Fk]
        return np.vstack(blocks)
    return F_global


def run_global_test(label, F_glob, F_h, U, n, p_sel):
    Fg = F_glob(U)
    M  = (U.shape[1])
    ok_shape = Fg.shape == (n * (M + 1), p_sel)
    ok_bloc0 = np.allclose(Fg[:n, :],     F_h(U),  atol=1e-14)
    ok_blocs = all(
        np.allclose(Fg[(k+1)*n:(k+2)*n, :],
                    fd(F_h, U, k), atol=1e-7)
        for k in range(M)
    )
    ok_deriv = all(
        np.allclose(Fg[(k+1)*n:(k+2)*n, :],
                    fd(F_h, U, k), atol=1e-7)
        for k in range(M)
    )
    status = "PASS" if (ok_shape and ok_bloc0 and ok_blocs) else "FAIL"
    print(f"  {label} — shape:{ok_shape}  bloc0:{ok_bloc0}  FD_blocs:{ok_blocs}  [{status}]")
    if not ok_shape: print("    shape obtenu:", Fg.shape, "attendu:", (n*(M+1), p_sel))
    if not ok_bloc0: print("    err max bloc0:", np.abs(Fg[:n, :] - F_h(U)).max())
    for k in range(M):
        fd_k = fd(F_h, U, k)
        err  = np.abs(Fg[(k+1)*n:(k+2)*n, :] - fd_k).max()
        if err > 1e-7:
            print(f"    err max FD bloc{k+1}:", err)


# ===========================================================================
# GLOBAL — HERMITE
# ===========================================================================
print("\n=== GLOBAL HERMITE ===")
F_glob_H = make_trend_global_handle(sel, Idx_H, pt_H)
run_global_test("Hermite global", F_glob_H, F_h_H, UH, N, len(sel))

# Vérification bloc par bloc contre make_trend_handle_deriv
Fg_H = F_glob_H(UH)
for k in range(2):
    ref = make_trend_handle_deriv(sel, Idx_H, pt_H, k)(UH)
    ok  = np.allclose(Fg_H[(k+1)*N:(k+2)*N, :], ref, atol=1e-14)
    print(f"  bloc{k+1} == make_trend_handle_deriv(der={k}) : {ok}")


# ===========================================================================
# GLOBAL — LEGENDRE
# ===========================================================================
print("\n=== GLOBAL LEGENDRE ===")
F_glob_L = make_trend_global_handle(sel, Idx_L, pt_L)
run_global_test("Legendre global", F_glob_L, F_h_L, UL, N, len(sel))

Fg_L = F_glob_L(UL)
for k in range(2):
    ref = make_trend_handle_deriv(sel, Idx_L, pt_L, k)(UL)
    ok  = np.allclose(Fg_L[(k+1)*N:(k+2)*N, :], ref, atol=1e-14)
    print(f"  bloc{k+1} == make_trend_handle_deriv(der={k}) : {ok}")
