"""
Regeneration des fichiers de reference figes (tests/golden/*.json).

A LANCER SCIEMMENT, jamais automatiquement : ecraser un golden efface la
memoire de ce que le code faisait avant. La procedure est :

  1. le test golden tombe apres une modification ;
  2. on DEMONTRE que le nouveau comportement est meilleur (les tests oracle
     de test_20 / test_40 doivent rester verts, et de preference s'ameliorer) ;
  3. on relance ce script ;
  4. on commite le golden AVEC la modification, en expliquant l'ecart dans le
     message de commit.

Usage :  python tests/make_golden.py
"""

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
from reference.limit_states import FlexionLS, LinearLS  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')

CONFIGS = [
    ('flexion', FlexionLS, 24, 3),
    ('linear', LinearLS, 24, 3),
]


def build(case, cls, n_doe, max_degree):
    ls = cls()
    X = harness.make_doe(n_doe, 2)
    out = {
        'case': case,
        'n_doe': n_doe,
        'max_degree': max_degree,
        'doe': X.tolist(),
        'probe': harness.PROBE.tolist(),
        'g_doe': np.asarray(ls.g(X)).ravel().tolist(),
        'grad_doe': np.asarray(ls.grad(X)).tolist(),
        'beta_exact': float(ls.beta_exact()),
        'u_star_exact': np.asarray(ls.u_star_exact()).ravel().tolist(),
        'models': {},
    }
    for kind in ('PCK', 'GEPCK'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fm = harness.fit(kind, X, ls, max_degree=max_degree)
        out['models'][kind] = harness.signature(fm, harness.PROBE, kind)
    return out


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for case, cls, n_doe, max_degree in CONFIGS:
        data = build(case, cls, n_doe, max_degree)
        path = os.path.join(GOLDEN_DIR, f'{case}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, sort_keys=True)
        print(f'ecrit {path}')
        for kind, sig in data['models'].items():
            print(f"   {kind:6s} LOO={sig['LOO']:.6e}  npoly={sig['NumberOfPoly']}"
                  f"  theta={np.round(sig['theta'], 6).tolist()}")


if __name__ == '__main__':
    main()
