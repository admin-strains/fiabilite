"""
Contrat d'API de _lib/ : ce que les scripts AC importent doit exister et
garder sa signature. Premiere ligne de defense d'un refactoring : si un
symbole disparait ou change d'arite, ce test tombe avant tout le reste.
"""

import inspect

import pytest

# (module, symbole, parametres positionnels attendus)
API = [
    ('branche1', 'fit_pck', ['X', 'Y', 'options', 'marginals', 'copula']),
    ('branche1', 'fit_gepck', ['X', 'Y_aug', 'options', 'marginals', 'copula']),
    ('branche1', 'predict_pck', ['fitted_model', 'X_test']),
    ('branche1', 'predict_gepck', ['fitted_model', 'X_test']),
    ('branche1', 'predict_deriv_gepck', ['fitted_model', 'X_test', 'der_var']),
    ('branche1', 'predict_gradient_gepck', ['fitted_model', 'X_test']),
    ('branche1', 'generate_doe', ['N', 'marginals']),
    ('branche2', 'uq_PCK_initialize', ['current_model']),
    ('branche3', 'uq_PCK_calculate_coefficients', ['X', 'Y', 'pck_config']),
    ('branche3', 'uq_GEPCK_calculate_coefficients', ['X', 'Y_aug', 'pck_config']),
    ('branche4', 'uq_PCK_eval', ['fitted_model', 'X_test']),
    ('branche4', 'uq_GEPCK_eval', ['fitted_model', 'X_test']),
    ('branche4', 'uq_GEPCK_eval_deriv', ['fitted_model', 'X_test', 'der_var']),
    ('branche5', 'uq_eval_Kernel', ['X1', 'X2', 'theta', 'options']),
    ('branche5', 'uq_eval_global_Kernel', ['X1', 'X2', 'theta', 'options']),
    ('branche_lars', 'uq_lar', None),
]


@pytest.mark.parametrize('mod,sym,params', API, ids=[f'{m}.{s}' for m, s, _ in API])
def test_api_surface(mod, sym, params):
    module = __import__(mod)
    assert hasattr(module, sym), f'{mod}.{sym} a disparu'
    fn = getattr(module, sym)
    assert callable(fn)
    if params is None:
        return
    got = list(inspect.signature(fn).parameters)
    assert got[:len(params)] == params, \
        f'{mod}.{sym} : parametres {got[:len(params)]} au lieu de {params}'


def test_lib_nimporte_pas_strains(tmp_path):
    """_lib doit rester utilisable sans STRAINS, OpenTURNS ni sklearn.

    Le controle tourne en SOUS-PROCESSUS : dans le processus pytest, d'autres
    fichiers de tests (test_60_environnement) importent openturns des la
    collecte, ce qui rendrait l'assertion vide de sens. Version precedente :
    elle passait pour cette raison, pas parce que la propriete etait vraie.
    """
    import os
    import subprocess
    import sys

    lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_lib')
    script = tmp_path / 'sonde_imports.py'
    script.write_text(
        'import sys\n'
        'sys.path.insert(0, %r)\n'
        'for m in ("branche1","branche2","branche3","branche4","branche5","branche_lars"):\n'
        '    __import__(m)\n'
        'interdits = {"openturns","STRAINS","smt","sklearn","autograd","nlopt","matplotlib"}\n'
        'print(",".join(sorted(interdits & set(sys.modules))))\n' % lib,
        encoding='utf-8')

    p = subprocess.run([sys.executable, str(script)], capture_output=True,
                       text=True, errors='replace', timeout=300)
    assert p.returncode == 0, f'_lib ne s importe pas seul :\n{p.stderr[-2000:]}'
    charges = [m for m in p.stdout.strip().split(',') if m]
    assert not charges, f'_lib a tire des dependances lourdes : {charges}'
