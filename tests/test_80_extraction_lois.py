"""
Les lois extraites reproduisent-elles celles des scripts AC ?

Les sept lois de `_model/lois.py` viennent de `if __name__ == '__main__':`
des scripts AC : du code que ni les goldens ni la baseline ne couvraient, et
qui n'etait meme pas importable.

L'oracle est `tests/golden/lois_originales.json`, produit par
`tools/golden_lois.py` a partir des definitions **telles qu'elles etaient
avant l'extraction**, lues a une revision git. C'est ce qui rend ce test
durable : les definitions ne sont plus dans les scripts AC, mais leur
comportement est fige.

Ces tests exigent OpenTURNS (pas Digital Structure) et sautent proprement la
ou il n'est pas installe.

Regenerer l'oracle est un acte grave : il n'existe qu'un seul etat
d'avant-extraction, et l'ecraser efface la seule trace de ce que le code
faisait. Si un ecart apparait, la cause est dans `_model/lois.py`.
"""

import json
import os
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "_model")):
    if p not in sys.path:
        sys.path.insert(0, p)

ot = pytest.importorskip("openturns", reason="couche fiabilite : OpenTURNS requis")

import numpy as np  # noqa: E402

import lois  # noqa: E402
from extraction_temoin import variables_libres  # noqa: E402

GOLDEN = os.path.join(TESTS, "golden", "lois_originales.json")

pytestmark = pytest.mark.skipif(not os.path.isfile(GOLDEN),
                                reason="oracle des lois d'origine absent")


@pytest.fixture(scope="module")
def attendu():
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)["lois"]


def _verifier(nom, attendu, fonction):
    for cas in attendu[nom]:
        obtenu = list(fonction(*cas["args"]).getParameter())
        assert obtenu == pytest.approx(cas["parametres"], rel=1e-15, abs=1e-300), \
            f"{nom}{tuple(cas['args'])} : {obtenu} au lieu de {cas['parametres']}"


def test_loi_fy(attendu):
    _verifier("loi_fy", attendu, lois.loi_fy)


def test_loi_fc(attendu):
    _verifier("loi_fc", attendu, lois.loi_fc)


def test_loi_F_permanente(attendu):
    _verifier("loi_F_permanente", attendu, lois.loi_F_permanente)


def test_loi_F_exploitation(attendu):
    _verifier("loi_F_exploitation", attendu, lois.loi_F_exploitation)


def test_loi_F_intermittente(attendu):
    _verifier("loi_F_intermittente", attendu, lois.loi_F_intermittente)


def test_loi_uni_approx(attendu):
    """Loi de Tukey : distribution Python, sans vecteur de parametres --
    on compare densite, repartition et quantiles."""
    for cas in attendu["loi_uni_approx"]:
        d = lois.loi_uni_approx(*cas["args"])
        for x, pdf, cdf in zip(cas["x"], cas["pdf"], cas["cdf"]):
            assert d.computePDF([x]) == pytest.approx(pdf, rel=1e-14, abs=1e-15)
            assert d.computeCDF([x]) == pytest.approx(cdf, rel=1e-14, abs=1e-15)
        for p, q in cas["quantiles"].items():
            assert d.computeQuantile(float(p))[0] == pytest.approx(q, rel=1e-10)


def test_dist_jointe(attendu):
    """`PARAM_CONFIG` et `params_names` etaient des variables libres de main.
    Les rendre explicites ne doit rien changer -- ni a la loi jointe, ni a la
    transformation isoprobabiliste, qui est ce qui sert en aval."""
    ref = attendu["dist_jointe"]
    config = {"fc": {"loi": lois.loi_fc, "args": (48, 0.12)},
              "fy": {"loi": lois.loi_fy, "args": (550, None)}}
    d = lois.dist_jointe(config, ["fc", "fy"])
    assert list(d.getParameter()) == pytest.approx(ref["parametres"], rel=1e-15)
    t = d.getInverseIsoProbabilisticTransformation()
    for u, x in zip(ref["u"], ref["x"]):
        assert list(t(u)) == pytest.approx(x, rel=1e-14)


# --------------------------------------------------------------------------- #
# Ce que l'extraction devait accomplir                                        #
# --------------------------------------------------------------------------- #
NOMS = ["loi_fy", "loi_fc", "loi_F_permanente", "loi_F_exploitation",
        "loi_F_intermittente", "loi_uni_approx", "dist_jointe"]


def test_plus_aucune_variable_libre():
    """Le but de l'extraction. Une fonction qui se ferme sur l'espace de noms
    de `main` ne peut etre appelee par aucun test ; c'est ce lien qui devait
    disparaitre."""
    chemin = os.path.join(REPO, "_model", "lois.py")
    autorises = {"np", "ot", "math", "SIGMA_ACIER_JCSS", "SIGMA_ACIER_LOT",
                 "SIGMA_ACIER_BARRE", "SIGMA_ACIER_ESSAI"}
    for nom in NOMS:
        restantes = set(variables_libres(chemin, nom)) - autorises
        assert not restantes, f"{nom} depend encore de {sorted(restantes)}"


def test_les_scripts_ac_ne_redefinissent_plus_les_lois():
    """Extraire sans retirer l'original laisse deux copies qui divergeront."""
    import re
    for rel in ("pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py"):
        with open(os.path.join(REPO, rel), encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        for nom in NOMS:
            if nom == "dist_jointe":
                continue          # delegue local assume, retire avec le script
            assert not re.search(r"(?m)^\s+def %s\(" % nom, src), \
                f"{rel} redefinit {nom}"


def test_la_constante_sigma_acier_est_inchangee():
    assert lois.SIGMA_ACIER_JCSS == float(np.sqrt(19.0 ** 2 + 22.0 ** 2 + 8.0 ** 2))


def test_le_module_ne_tire_pas_digital_structure():
    """La couche modele a besoin d'OpenTURNS, pas du solveur."""
    import subprocess
    p = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s'); import lois; "
         "print(','.join(sorted({'STRAINS','smt','sklearn'} & set(sys.modules))))"
         % os.path.join(REPO, "_model")],
        capture_output=True, text=True, errors="replace", timeout=300)
    assert p.returncode == 0, p.stderr[-1500:]
    assert not p.stdout.strip(), f"lois.py a tire : {p.stdout.strip()}"
