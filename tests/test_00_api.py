"""
Contrat d'API de _lib/ : ce que les scripts AC importent doit exister et
garder sa signature. Premiere ligne de defense d'un refactoring : si un
symbole disparait ou change d'arite, ce test tombe avant tout le reste.
"""

import inspect

import pytest

# (module, symbole, parametres positionnels attendus)
API = [
    ('api', 'fit_pck', ['X', 'Y', 'options', 'marginals', 'copula']),
    ('api', 'fit_gepck', ['X', 'Y_aug', 'options', 'marginals', 'copula']),
    ('api', 'predict_pck', ['fitted_model', 'X_test']),
    ('api', 'predict_gepck', ['fitted_model', 'X_test']),
    ('api', 'predict_deriv_gepck', ['fitted_model', 'X_test', 'der_var']),
    ('api', 'predict_gradient_gepck', ['fitted_model', 'X_test']),
    ('api', 'generate_doe', ['N', 'marginals']),
    ('options', 'uq_PCK_initialize', ['current_model']),
    ('branche3', 'uq_PCK_calculate_coefficients', ['X', 'Y', 'pck_config']),
    ('branche3', 'uq_GEPCK_calculate_coefficients', ['X', 'Y_aug', 'pck_config']),
    ('predict', 'uq_PCK_eval', ['fitted_model', 'X_test']),
    ('predict', 'uq_GEPCK_eval', ['fitted_model', 'X_test']),
    ('predict', 'uq_GEPCK_eval_deriv', ['fitted_model', 'X_test', 'der_var']),
    ('branche5', 'uq_eval_Kernel', ['X1', 'X2', 'theta', 'options']),
    ('branche5', 'uq_eval_global_Kernel', ['X1', 'X2', 'theta', 'options']),
    ('lars', 'uq_lar', None),
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


# --------------------------------------------------------------------------- #
# Coquilles de compatibilite (phase 2)                                        #
# --------------------------------------------------------------------------- #
COQUILLES = [('branche1', 'api'), ('branche2', 'options'),
             ('branche4', 'predict'), ('branche_lars', 'lars')]


@pytest.mark.parametrize('vieux,neuf', COQUILLES, ids=[v for v, _ in COQUILLES])
def test_ancien_nom_delegue_au_nouveau(vieux, neuf):
    """Les suites de tests/unit/ sont conservees telles quelles : elles
    importent les anciens noms, qui doivent continuer a fonctionner."""
    a, b = __import__(vieux), __import__(neuf)
    exportes = [n for n in dir(b) if not n.startswith('_')]
    assert exportes, f'{neuf} n expose rien'
    for nom in exportes:
        assert getattr(a, nom) is getattr(b, nom), \
            f'{vieux}.{nom} ne pointe pas sur {neuf}.{nom}'


@pytest.mark.parametrize('vieux,neuf', COQUILLES, ids=[v for v, _ in COQUILLES])
def test_la_coquille_suit_l_instrumentation(vieux, neuf):
    """
    Liaison TARDIVE, et non `from X import *`.

    tools/telemetry.py instrumente en remplacant les fonctions dans le module.
    Avec un `import *`, la coquille aurait fige les references au chargement et
    l'ancien nom aurait continue de rendre la fonction NON instrumentee -- deux
    chemins d'import donnant deux objets differents, silencieusement.
    """
    a, b = __import__(vieux), __import__(neuf)
    cible = [n for n in dir(b) if not n.startswith('_')][0]
    original = getattr(b, cible)
    sentinelle = object()
    try:
        setattr(b, cible, sentinelle)
        assert getattr(a, cible) is sentinelle, \
            f'{vieux} a fige {cible} au chargement : la deleguation n est pas tardive'
    finally:
        setattr(b, cible, original)


@pytest.mark.parametrize('vieux,neuf', COQUILLES, ids=[v for v, _ in COQUILLES])
def test_la_coquille_annonce_son_retrait(vieux, neuf):
    """Une coquille sans date de retrait devient permanente. Celle-ci doit
    prevenir a l'import et dire quand elle disparait."""
    import importlib
    import warnings
    mod = importlib.import_module(vieux)
    with warnings.catch_warnings(record=True) as captures:
        warnings.simplefilter('always')
        importlib.reload(mod)
    messages = [str(w.message) for w in captures
                if issubclass(w.category, DeprecationWarning)]
    assert messages, f'{vieux} ne previent pas qu il est obsolete'
    assert neuf in messages[0] and 'phase 3' in messages[0], \
        f'le message ne dit pas le nouveau nom et la date de retrait : {messages[0]}'


# --------------------------------------------------------------------------- #
# Coquilles federatrices (phase 3 : scission de branche3 et branche5)         #
# --------------------------------------------------------------------------- #
SCISSIONS = [('branche3', ['kriging', 'pce_basis', 'fit']),
             ('branche5', ['polynomials', 'transform', 'kernels'])]


@pytest.mark.parametrize('vieux,cibles', SCISSIONS, ids=[v for v, _ in SCISSIONS])
def test_la_coquille_federe_les_trois_modules(vieux, cibles):
    coquille = __import__(vieux)
    vus = set()
    for nom_cible in cibles:
        cible = __import__(nom_cible)
        exportes = [n for n in dir(cible) if not n.startswith('_')]
        assert exportes, f'{nom_cible} n expose rien'
        for nom in exportes:
            assert getattr(coquille, nom) is getattr(cible, nom), \
                f'{vieux}.{nom} ne pointe pas sur {nom_cible}.{nom}'
        vus.update(exportes)
    assert vus, f'{vieux} ne federe rien'


@pytest.mark.parametrize('vieux,cibles', SCISSIONS, ids=[v for v, _ in SCISSIONS])
def test_les_modules_scindes_sont_disjoints(vieux, cibles):
    """Une fonction ne doit exister que dans UN des modules issus de la
    scission : un doublon signifierait deux copies divergentes a terme."""
    vu = {}
    for nom_cible in cibles:
        cible = __import__(nom_cible)
        for nom in dir(cible):
            if nom.startswith('_') or not callable(getattr(cible, nom, None)):
                continue
            fn = getattr(cible, nom)
            # ignorer ce qui est simplement importe d'un module voisin
            if getattr(fn, '__module__', None) != nom_cible:
                continue
            assert nom not in vu, \
                f'{nom} est defini dans {vu[nom]} ET dans {nom_cible}'
            vu[nom] = nom_cible
    assert vu
