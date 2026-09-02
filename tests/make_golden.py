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

Usage :  python tests/make_golden.py            (tous les cas)
         python tests/make_golden.py console    (celui-la seulement)


LE MEME ENVIRONNEMENT NUMERIQUE QUE LA SUITE, ET POURQUOI
----------------------------------------------------------
`tests/conftest.py` bride le BLAS a UN thread avant d'importer numpy : sans
cela le nombre de coeurs change l'ordre des reductions, donc les derniers
chiffres. Ce script n'est pas un test -- `conftest` ne s'y applique pas --,
et il produisait donc ses goldens a sept threads pour une suite qui les
compare a un.

Mesure du 02/09/2026 sur le cas `console`, meme machine, meme plan :

    grandeur    7 threads          1 thread           ecart relatif
    LOO GEPCK   4,403651e-06       4,403640e-06       2,5e-06
    beta_pce    0,35408452         0,35408472         5,6e-07

Au-dessus de la tolerance de 1e-06 de `test_30`, qui refusait donc un golden
tout juste ecrit.

Les deux premiers cas, eux, sont INSENSIBLES au nombre de threads : mesure le
02/09/2026, leurs goldens sont BIT-A-BIT identiques a un et a sept threads,
sur `theta`, `LOO`, `sigma^2` et `beta_pce`. C'est le troisieme, a trois
variables et dans un regime ou la tendance PCE ne suffit pas, qui a rendu
l'ecart visible -- un seul cas peut cacher un defaut.

Un generateur qui ne peut pas produire un golden acceptable est un piege : le
bridage est donc ici AUSSI, en tete, avant numpy.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json                                                  # noqa: E402
import sys                                                   # noqa: E402
import warnings                                              # noqa: E402

import numpy as np                                           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
from reference.limit_states import ConsoleLS, FlexionLS, LinearLS  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')

CONFIGS = [
    ('flexion', FlexionLS, 24, 3),
    ('linear', LinearLS, 24, 3),
    ('console', ConsoleLS, 24, 3),
]


def build(case, cls, n_doe, max_degree):
    ls = cls()
    # La dimension vient du CAS, jamais d'une constante : le troisieme cas de
    # reference en a trois. Ce `2` en dur aurait produit un plan a deux
    # colonnes pour un etat limite qui en attend trois.
    X = harness.make_doe(n_doe, ls.n_var)
    out = {
        'case': case,
        'n_doe': n_doe,
        'max_degree': max_degree,
        'doe': X.tolist(),
        'probe': harness.probe(ls.n_var).tolist(),
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
        out['models'][kind] = harness.signature(fm, harness.probe(ls.n_var), kind)
    return out


def main(demandes=()):
    """Regenere les goldens. Sans argument : tous.

    UN FILTRE, ET POURQUOI IL COMPTE. Ajouter un cas de reference ne doit pas
    obliger a reecrire les goldens des autres : ecraser un golden efface la
    memoire de ce que le code faisait avant, et le faire par effet de bord
    d'un ajout serait exactement la faute que l'en-tete de ce fichier
    interdit. `python tests/make_golden.py console` n'ecrit que celui-la.
    """
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    inconnus = [d for d in demandes if d not in {c[0] for c in CONFIGS}]
    if inconnus:
        raise SystemExit("cas inconnu(s) : %s -- connus : %s"
                         % (", ".join(inconnus),
                            ", ".join(c[0] for c in CONFIGS)))
    for case, cls, n_doe, max_degree in CONFIGS:
        if demandes and case not in demandes:
            continue
        data = build(case, cls, n_doe, max_degree)
        path = os.path.join(GOLDEN_DIR, f'{case}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, sort_keys=True)
        print(f'ecrit {path}')
        for kind, sig in data['models'].items():
            print(f"   {kind:6s} LOO={sig['LOO']:.6e}  npoly={sig['NumberOfPoly']}"
                  f"  theta={np.round(sig['theta'], 6).tolist()}")


if __name__ == '__main__':
    main(tuple(sys.argv[1:]))
