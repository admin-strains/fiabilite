"""
test_eval_global_kernel.py
Valide uq_eval_global_Kernel dual-use :
  Gram     -> R_tilde  (N*(M+1), N*(M+1))
  non-Gram -> r0_tilde (N_test,  N_train*(M+1))

Convention r0_tilde :
  r0_tilde[i, cb*N:(cb+1)*N] = Cov(y(x_i), y_aug_bloc_cb(X_train))
                              = dk(X_test, X_train) / dX_train_{cb-1}
                              = kernel_deriv_factory(None, cb-1)(X_test, X_train)
  = premier bloc-ligne de uq_assemble_global_Kernel(X_test, X_train).

Tests :
  T1 — Formes de sortie (Gram et non-Gram)
  T2 — r0_tilde == premier bloc-ligne de uq_assemble_global_Kernel
  T3 — Valeurs : bloc cb == kernel_deriv_factory(None, dp)(X_test, X_train)
  T4 — Symetrie et def. semi-positive de R_tilde
  T5 — Interpolation GEK : r0_tilde(X_train) == R_tilde[:N, :]
       => r0 @ Rinv @ y_aug == y_train (propriete fondamentale du BLUP)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from branche5 import (
    uq_eval_global_Kernel,
    uq_assemble_global_Kernel,
    kernel_deriv_factory,
)


def run(label, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    msg = f'  [{status}] {label}'
    if detail:
        msg += f'\n         {detail}'
    print(msg)
    return ok


# ---------------------------------------------------------------------------
# Parametres communs
# ---------------------------------------------------------------------------
np.random.seed(42)
M       = 2
N_train = 6
N_test  = 4

X_train = np.random.randn(N_train, M)
X_test  = np.random.randn(N_test,  M)
theta   = np.array([0.8, 1.2])

opts_gauss  = {'Family': 'gaussian',   'Nugget': 0.0}
opts_matern = {'Family': 'matern-5_2', 'Nugget': 0.0}
opts_nugget = {'Family': 'gaussian',   'Nugget': 1e-6}

all_pass = True

# ===========================================================================
print('=== T1 : Formes de sortie ===')
# ===========================================================================
for label, opts in [('gaussian', opts_gauss), ('matern', opts_matern)]:
    R  = uq_eval_global_Kernel(X_train, X_train, theta, opts)
    r0 = uq_eval_global_Kernel(X_test,  X_train, theta, opts)

    ok1 = R.shape  == (N_train*(M+1), N_train*(M+1))
    ok2 = r0.shape == (N_test,        N_train*(M+1))
    all_pass &= ok1 and ok2
    run(f'{label} Gram     shape {R.shape}',  ok1,
        f'attendu ({N_train*(M+1)}, {N_train*(M+1)})')
    run(f'{label} non-Gram shape {r0.shape}', ok2,
        f'attendu ({N_test}, {N_train*(M+1)})')


# ===========================================================================
print('\n=== T2 : r0_tilde == premier bloc-ligne de uq_assemble_global_Kernel ===')
# ===========================================================================
for label, family in [('gaussian', 'gaussian'), ('matern', 'matern-5_2')]:
    opts   = {'Family': family, 'Nugget': 0.0}
    r0     = uq_eval_global_Kernel(X_test, X_train, theta, opts)
    R_full = uq_assemble_global_Kernel(X_test, X_train, theta, family)
    ref    = R_full[:N_test, :]    # premier bloc-ligne (rb=0)
    ok = np.allclose(r0, ref, atol=1e-14)
    all_pass &= ok
    run(f'{label} r0_tilde == R_assemble[:N_test, :]', ok,
        f'err max = {np.abs(r0 - ref).max():.2e}')


# ===========================================================================
print('\n=== T3 : Valeurs — kernel_deriv_factory(None, dp) pour chaque bloc ===')
# ===========================================================================
for label, family in [('gaussian', 'gaussian'), ('matern', 'matern-5_2')]:
    opts   = {'Family': family, 'Nugget': 0.0}
    r0     = uq_eval_global_Kernel(X_test, X_train, theta, opts)
    ok_all = True
    for cb in range(M + 1):
        der  = cb - 1 if cb > 0 else None
        ref  = kernel_deriv_factory(family, der, None)(X_test, X_train, theta)
        bloc = r0[:, cb*N_train : (cb+1)*N_train]
        ok_cb = np.allclose(bloc, ref, atol=1e-14)
        ok_all &= ok_cb
        if not ok_cb:
            print(f'    cb={cb} dp={dp} err={np.abs(bloc-ref).max():.2e}')
    all_pass &= ok_all
    run(f'{label} blocs cb=0..{M} corrects', ok_all)


# ===========================================================================
print('\n=== T4 : Symetrie et def. semi-positive de R_tilde ===')
# ===========================================================================
for label, opts in [('gaussian', opts_gauss), ('matern', opts_matern),
                    ('gaussian+nugget', opts_nugget)]:
    R    = uq_eval_global_Kernel(X_train, X_train, theta, opts)
    sym  = np.allclose(R, R.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(R)
    psd  = bool(eigs.min() >= -1e-10)
    ok   = sym and psd
    all_pass &= ok
    run(f'{label} symetrie:{sym} PSD:{psd} (eig_min={eigs.min():.2e})', ok)


# ===========================================================================
print('\n=== T5 : Interpolation GEK ===')
# Propriete cle du BLUP :
#   r0_tilde(X_train) = R_tilde[:N_train, :]  (T2 + T3 etabli ci-dessus)
#   => r0_tilde @ R_tilde^{-1} @ y_aug = y_train
#
# Note : X_train.copy() declenche toujours le Gram (np.array_equal sur valeurs).
# On construit r0_train directement via kernel_deriv_factory pour contourner.
# ===========================================================================
def f_true(X):
    return np.sin(X[:, 0]) + np.cos(X[:, 1])

def grad_true(X):
    G = np.zeros_like(X)
    G[:, 0] =  np.cos(X[:, 0])
    G[:, 1] = -np.sin(X[:, 1])
    return G

for label, family in [('gaussian', 'gaussian'), ('matern', 'matern-5_2')]:
    opts = {'Family': family, 'Nugget': 1e-10}

    y     = f_true(X_train)
    G     = grad_true(X_train)
    y_aug = np.concatenate([y, G[:, 0], G[:, 1]])   # (N*(M+1),)

    R_tilde = uq_eval_global_Kernel(X_train, X_train, theta, opts)
    c       = np.linalg.solve(R_tilde, y_aug)

    # r0_train = premier bloc-ligne de R_tilde (= R_tilde[:N_train, :])
    # construit manuellement pour eviter la detection Gram
    r0_train = np.empty((N_train, N_train * (M + 1)))
    for cb in range(M + 1):
        der = cb - 1 if cb > 0 else None
        f_  = kernel_deriv_factory(family, der, None)
        r0_train[:, cb*N_train:(cb+1)*N_train] = f_(X_train, X_train, theta)

    # Verification 1 : r0_train == R_tilde[:N_train, :]
    ok_eq = np.allclose(r0_train, R_tilde[:N_train, :], atol=1e-12)

    # Verification 2 : r0_train @ c == y_train (interpolation)
    y_pred = r0_train @ c
    ok_interp = np.allclose(y_pred, y, atol=1e-6)

    ok = ok_eq and ok_interp
    all_pass &= ok
    run(f'{label} r0==R[:N,:]: {ok_eq}  interp err={np.abs(y_pred-y).max():.2e}', ok)

# ---------------------------------------------------------------------------
print()
print('=== BILAN :', 'PASS' if all_pass else 'FAIL', '===')
