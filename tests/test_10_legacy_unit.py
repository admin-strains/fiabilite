"""
Reactivation des suites unitaires historiques.

Ces 10 fichiers (tests/unit/) existaient sur la branche `fiabilite` puis ont
ete SUPPRIMES lors des reorganisations vers `flexion` / `moulin_blanc` /
`dir-fiabilite`. Ils sont restaures tels quels : aucune ligne n'est modifiee,
pour qu'ils restent un temoin fidele du contrat d'origine de la librairie.

Ils sont ecrits en style script (compteur PASS/FAIL + print, pas d'assert et
pas de code retour), donc pytest ne peut pas les evaluer directement : ce
wrapper les execute en sous-processus et lit leur bilan.

STATUT_CONNU consigne l'etat mesure sur la branche `cleaning` au moment de la
restauration. Un test qui passe de 'ok' a 'ko' est une regression ; un test
'ko' qui passe a 'ok' doit faire mettre a jour ce tableau.
"""

import os
import re
import subprocess
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
UNIT_DIR = os.path.join(TESTS_DIR, 'unit')

# suite -> (statut attendu, raison si ko)
STATUT_CONNU = {
    'test_branche1': ('ok', None),
    'test_branche2': ('ok', None),
    'test_branche3': ('ok', None),
    'test_branche4': ('ok', None),
    'test_branche_lars': ('ok', None),
    'test_f_deriv_handles': ('ok', None),
    'test_gek_kernel': ('ok', None),
    'test_make_trend_deriv': ('ok', None),
    'test_eval_global_kernel': (
        'ko',
        "TRANCHE le 26/08/2026 (defaut 4, phase 6) : c'est LE TEST qui a tort. "
        "T3 attend r0_tilde[:, cb] == kernel_deriv_factory(der=cb-1, dp=None), "
        "or la l-ieme observation augmentee est dy/dx_l AU POINT D'APPRENTISSAGE, "
        "donc Cov(y(x*), dy/dx_l(x^j)) = dk(x*,x^j)/dx^j_l -- une derivee du "
        "SECOND argument, soit (der=None, dp=l). Verifie par differences finies : "
        "voir tests/test_51_convention_derivees.py, 20 tests. Les deux docstrings "
        "de kernels.py, elles, etaient fausses et ont ete corrigees. "
        "Ce test reste INCHANGE : les suites d'origine servent de temoin, on ne "
        "reecrit pas un temoin. Le NameError 'dp' l.103 est un typo latent."),
    # CORRIGE le 26/08/2026 (defauts 2 et 3, phase 6). La piste enregistree
    # etait la bonne : c'etait bien le conditionnement de R_tilde. Avec la
    # pepite par defaut a 1e-8, |grad_analytique - FD| passe de 3,7e-3 a sous
    # 1e-6 et les 9 sous-tests passent. Rien n'a ete touche dans ce test.
    'test_predict_deriv_gepck': ('ok', None),
}

SUITES = sorted(f[:-3] for f in os.listdir(UNIT_DIR)
                if f.startswith('test_') and f.endswith('.py'))

_BILAN = re.compile(r'(?:Results|BILAN)\s*[:=]?\s*(\d+)\s*PASS\s*\|?\s*(\d+)\s*FAIL')


def _run(name):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.path.join(ROOT, '_lib')
    env['PYTHONIOENCODING'] = 'utf-8'
    env['MPLBACKEND'] = 'Agg'
    p = subprocess.run([sys.executable, os.path.join(UNIT_DIR, name + '.py')],
                       capture_output=True, text=True, errors='replace',
                       env=env, cwd=UNIT_DIR, timeout=900)
    n_pass = n_fail = 0
    for m in _BILAN.finditer(p.stdout):
        n_pass, n_fail = int(m.group(1)), int(m.group(2))
    if n_pass == 0:
        # test_gek_kernel.py est deja au format pytest
        m = re.search(r'(\d+) passed', p.stdout)
        if m:
            mf = re.search(r'(\d+) failed', p.stdout)
            n_pass = int(m.group(1))
            n_fail = int(mf.group(1)) if mf else 0
    if n_pass == 0:
        # test_make_trend_deriv.py : marqueurs [PASS]/[FAIL], pas de bilan final
        n_pass = len(re.findall(r'\[PASS\]', p.stdout))
        n_fail = len(re.findall(r'\[FAIL\]', p.stdout))
        # ce fichier imprime aussi des verdicts bruts 'xxx : True/False'
        n_fail += len(re.findall(r':\s*False', p.stdout))
    return p, n_pass, n_fail


@pytest.mark.slow
@pytest.mark.parametrize('name', SUITES)
def test_suite_legacy(name):
    attendu, raison = STATUT_CONNU.get(name, ('ok', None))
    p, n_pass, n_fail = _run(name)
    crash = p.returncode != 0
    ko = crash or n_fail > 0

    if attendu == 'ok':
        assert not crash, (
            f'{name} sort en erreur (rc={p.returncode}) alors qu il passait :\n'
            + p.stdout[-3000:] + '\n' + p.stderr[-3000:])
        assert n_pass > 0, f'{name} : aucune assertion executee (bilan illisible)'
        assert n_fail == 0, f'{name} : {n_fail} FAIL / {n_pass + n_fail}\n' + p.stdout[-3000:]
    else:
        if not ko:
            pytest.fail(
                f'{name} est marque "ko" mais passe maintenant ({n_pass} PASS). '
                f'Corriger STATUT_CONNU dans ce fichier.\nRaison enregistree : {raison}')
        pytest.xfail(f'{name} : defaut connu -- {raison}')
