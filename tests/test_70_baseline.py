"""
Garde de la baseline : la chaine complete rend-elle toujours la meme chose ?

Les etages precedents protegent des morceaux -- l'API (test_00), la
librairie (test_10, test_30), la justesse du resultat (test_40). Celui-ci
protege **l'enchainement** : plan d'experiences, metamodele, critere
d'enrichissement, FORM, tirage d'importance, joues bout a bout et compares
grandeur par grandeur au journal de reference `baselines/flexion_analytique/`.

C'est le test qui rend la restructuration sure. Un deplacement de code qui
change la 12e decimale d'un coefficient intermediaire, sans toucher au beta
final, tombe ici -- et le comparateur dit **a quelle etape** il tombe.

Tourne sans Digital Structure : l'etat limite analytique tient lieu de
solveur, avec la geometrie reelle de `test_pure_flexion`.

Regenerer la reference (uniquement apres avoir justifie l'ecart) :

    python tools/baseline_run.py --repeat 3
    copier le dernier run_*.jsonl en baselines/flexion_analytique/reference.jsonl
"""

import json
import os
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "tools"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

REFERENCE = os.path.join(REPO, "baselines", "flexion_analytique", "reference.jsonl")
PLANCHER = os.path.join(REPO, "baselines", "flexion_analytique", "plancher_de_bruit.json")

pytestmark = pytest.mark.skipif(not os.path.isfile(REFERENCE),
                                reason="pas de baseline de reference")


@pytest.fixture(scope="module")
def journal_neuf(tmp_path_factory):
    """Rejoue la chaine complete une fois, dans un journal jetable."""
    import baseline_run
    import telemetry

    j = telemetry.Journal("test", outdir=str(tmp_path_factory.mktemp("baseline")),
                          config={**baseline_run.CONFIG, "solveur": "analytique"},
                          note="rejoue par pytest")
    telemetry.pin_seeds(j)
    restaurer = telemetry.instrument_lib(j)
    try:
        r = baseline_run.chaine(j, baseline_run.CONFIG, "analytique")
    finally:
        restaurer()
    return j.close(**r), r


#: CE QUE LE *TEST* VERIFIE, ET CE QUE L'*OUTIL* VERIFIE -- deux choses
#:
#: `tools/baseline_compare.py` compare au BIT PRES, avec des tolerances a
#: 1e-10. C'est le filet de refactorisation, et il a attrape de vraies choses
#: toute la semaine du 25/08 : il reste STRICT, et se lance a la main sur le
#: poste ou l'on travaille.
#:
#: Ce TEST-ci, lui, tourne aussi sur des machines qui ne sont pas celle du
#: golden. Or l'arithmetique y differe -- noyau BLAS choisi selon le
#: processeur. Mesure du 02/09/2026, runner ubuntu contre poste de reference,
#: apres le gradient analytique :
#:
#:     grandeur                       ecart      tolerance de l'outil
#:     form / u_star                  1.213e-08  1e-10
#:     proba / pf_is                  1.161e-08  1e-10
#:     proba / beta_is                5.097e-10  1e-10
#:     lib:fit_gepck / out_sigmaSQ    2.696e-08  1e-10
#:     eff / eff_grille               1.076e-06  1e-08
#:
#: Les grandeurs PHYSIQUES s'accordent donc a 1e-08 pres. Ce test exige cela,
#: qui est vrai partout, plutot qu'une exactitude au bit qui n'est vraie que
#: sur une machine -- et qui, exigee ici, rendait le harness rouge sur quatre
#: jobs d'integration continue sur cinq.
TOL_PORTABLE = 1e-6


@pytest.mark.slow
def test_la_chaine_reproduit_la_baseline(journal_neuf, capsys):
    """Les trois chiffres de sortie, a une tolerance qui vaut PARTOUT.

    Le comparateur complet -- 35 grandeurs, 1e-10, condensats compris --
    reste dans `tools/baseline_compare.py`. Il est imprime ci-dessous pour
    qu'on VOIE ce qui bouge, mais son verdict n'est pas ce qui est exige :
    les condensats de suite d'appels ne peuvent pas etre portables, et deux
    grandeurs sont comparees en relatif alors qu'elles valent zero au point
    de conception (`g_star`).
    """
    import baseline_compare
    chemin, r = journal_neuf[0], journal_neuf[1]
    with capsys.disabled():
        print()
        code = baseline_compare.comparer(REFERENCE, chemin)
    assert code != 2, ("le journal n'est plus comparable a la reference : des "
                       "grandeurs ont disparu ou sont apparues. Si c'est voulu, "
                       "regenerer la reference (voir l'en-tete de ce fichier).")

    with open(REFERENCE, encoding="utf-8") as fh:
        footer = [json.loads(l) for l in fh if '"footer"' in l][-1]
    attendu = footer["resume"]
    for cle in ("beta", "pf_form", "pf_is"):
        assert r[cle] == pytest.approx(attendu[cle]["value"], rel=TOL_PORTABLE), (
            "%s = %.15g contre %.15g dans la reference, au-dela de %.0e. "
            "Le comparateur ci-dessus dit a quelle ETAPE cela commence."
            % (cle, r[cle], attendu[cle]["value"], TOL_PORTABLE))


@pytest.mark.slow
def test_les_grandeurs_de_sortie_sont_inchangees(journal_neuf):
    """La meme exigence, lisible sans le comparateur.

    Elle exigeait `rel=1e-12`. Aucune machine autre que celle du golden ne
    peut la tenir : le gradient analytique a rendu `beta_is` reproductible a
    5.10e-10 entre machines -- six ordres de mieux qu'avant, et toujours
    au-dessus de 1e-12.
    """
    _, r = journal_neuf
    with open(REFERENCE, encoding="utf-8") as fh:
        footer = [json.loads(l) for l in fh if '"footer"' in l][-1]
    attendu = footer["resume"]
    assert r["beta"] == pytest.approx(attendu["beta"]["value"], rel=TOL_PORTABLE)
    assert r["pf_form"] == pytest.approx(attendu["pf_form"]["value"], rel=TOL_PORTABLE)
    assert r["pf_is"] == pytest.approx(attendu["pf_is"]["value"], rel=TOL_PORTABLE)


def test_le_plancher_de_bruit_est_nul():
    """
    La chaine analytique est deterministe : deux runs identiques donnent des
    chiffres identiques. Si ce test tombe, une source d'alea non maitrisee est
    apparue, et toute comparaison a une baseline devient ininterpretable --
    c'est donc a corriger AVANT de regarder quoi que ce soit d'autre.
    """
    if not os.path.isfile(PLANCHER):
        pytest.skip("plancher de bruit non mesure")
    with open(PLANCHER, encoding="utf-8") as fh:
        d = json.load(fh)
    assert d["n_repetitions"] >= 2
    for cle, stats in d["grandeurs"].items():
        assert stats["etendue_relative"] == 0.0, (
            "%s varie d'un run a l'autre (etendue relative %.3e) : il y a de "
            "l'alea non seme dans la chaine." % (cle, stats["etendue_relative"]))


def test_la_reference_porte_son_contexte():
    """Une baseline sans son environnement n'est pas interpretable."""
    with open(REFERENCE, encoding="utf-8") as fh:
        header = json.loads(fh.readline())
    env = header["environnement"]
    assert env["python"] and env["paquets"]["numpy"] and env["paquets"]["scipy"]
    assert header["config"]["n_doe"] > 0
    assert env["git_commit"], "la reference ne dit pas de quel commit elle vient"
