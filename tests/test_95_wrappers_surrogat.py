r"""Les six enveloppes OpenTURNS, sorties des scripts d'etude.

CE QU'ELLES ETAIENT -- 27/08/2026
----------------------------------
193 lignes definies dans le bloc `__main__` de CHACUN des deux scripts,
identiques au caractere pres. Une classe definie dans `__main__` n'est ni
importable, ni testable, ni isolable : aucune de ces six n'avait jamais ete
couverte par un test, alors qu'elles portent tout ce qu'OpenTURNS voit du
metamodele -- la valeur, l'echantillon, le gradient et l'ecart-type.

DEUX DEFAUTS QUE L'EXTRACTION A REVELES
----------------------------------------
1. `_exec_sigma` -- 25 lignes de variance posterieure a noyau augmente --
   etait recopiee dans `oldGEPCKFunction` ET dans `GEKPLSFunction`, dans le
   MEME fichier, deux fois, soit quatre exemplaires au total. Une correction
   sur l'une n'aurait pas touche les trois autres.
2. Quatre lignes de trace supposaient `n_var == 2` en dur (`u[0]`, `u[1]`).
   Avec trois variables elles tronquaient ; avec une seule, `IndexError` au
   milieu d'un run.
"""

import ast
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _d in (_REPO, os.path.join(_REPO, "_surrogate")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# La couche etudes n'est pas installee partout (test_05).
ot = pytest.importorskip("openturns")
np = pytest.importorskip("numpy")

import wrappers                                      # noqa: E402

SCRIPTS = ["pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py"]
CLASSES = ("HFFunction", "PCKRGFunction", "oldGEPCKFunction",
           "GEPCKFunction", "PCKFunction", "GEKPLSFunction")


# --------------------------------------------------------------------- #
# elles ont bien quitte les scripts, et les deux etudes partagent le meme
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("script", SCRIPTS)
def test_les_enveloppes_ne_sont_plus_definies_dans_l_etude(script):
    chemin = os.path.join(_REPO, script)
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        arbre = ast.parse(fh.read(), filename=chemin)
    main = [n for n in arbre.body if isinstance(n, ast.If)][0]
    definies = {n.name for n in main.body if isinstance(n, ast.ClassDef)}
    restantes = sorted(definies & set(CLASSES))
    assert not restantes, (
        "%s redefinit %s. Ces classes ne dependent pas de l'etude : leurs "
        "seules variables libres etaient n_var, l'evaluateur haute fidelite "
        "et un interrupteur de trace." % (script, restantes))


@pytest.mark.parametrize("script", SCRIPTS)
def test_l_etude_n_a_plus_a_connaitre_les_enveloppes(script):
    """Elles ont d'abord ete importees par l'etude, le temps que le dispatch
    y reste. Depuis que `construire_surrogate` choisit la famille, l'etude
    n'a plus a savoir qu'elles existent : elle demande un modele par son nom
    et recoit une `ot.Function`."""
    with open(os.path.join(_REPO, script), encoding="utf-8",
              errors="replace") as fh:
        src = fh.read()
    assert "from wrappers import" not in src, (
        "%s connait encore les enveloppes OpenTURNS : le choix de la famille "
        "appartient a `_surrogate.construire_surrogate`." % script)
    assert "construire_surrogate(" in src, script


def test_les_six_enveloppes_existent():
    for nom in CLASSES:
        assert hasattr(wrappers, nom), nom


# --------------------------------------------------------------------- #
# la variance GEK est ecrite UNE fois
# --------------------------------------------------------------------- #
def test_la_variance_GEK_n_est_ecrite_qu_une_fois():
    """Elle existait en quatre exemplaires identiques. Le test compte les
    occurrences du calcul, reperees par un terme qui n'apparait nulle part
    ailleurs."""
    src = open(os.path.join(_REPO, "_surrogate", "wrappers.py"),
               encoding="utf-8").read()
    assert src.count("np.einsum('ija,ijb->ijab'") == 1, (
        "la variance posterieure a noyau augmente est recopiee : elle doit "
        "rester dans `sigma_gek` seule.")


def test_les_deux_familles_GEK_passent_par_sigma_gek():
    src = open(os.path.join(_REPO, "_surrogate", "wrappers.py"),
               encoding="utf-8").read()
    arbre = ast.parse(src)
    vues = {}
    for n in arbre.body:
        if isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and m.name == "_exec_sigma":
                    vues[n.name] = ast.dump(m)
    for nom in ("oldGEPCKFunction", "GEKPLSFunction"):
        assert nom in vues, nom
        assert "sigma_gek" in vues[nom], (
            "%s recalcule la variance au lieu d'appeler `sigma_gek`" % nom)


# --------------------------------------------------------------------- #
# les traces ne supposent plus deux variables
# --------------------------------------------------------------------- #
def test_le_format_de_trace_est_inchange_a_deux_variables():
    """Le journal d'un run doit rester comparable a ceux d'avant l'extraction."""
    assert wrappers._liste([-2.797, 3.841], "+.4f") == "[-2.7970, +3.8410]"
    assert wrappers._liste([1.5], "+.6f") == "[+1.500000]"
    assert wrappers._liste([1.0, 2.0, 3.0], "+.4f") == \
        "[+1.0000, +2.0000, +3.0000]"


def test_le_format_de_trace_accepte_les_scalaires_numpy():
    assert wrappers._liste(np.array([-0.5, 0.25]), "+.4f") == "[-0.5000, +0.2500]"


def test_aucune_trace_ne_code_en_dur_deux_variables():
    """`u[0]` et `u[1]` dans une chaine de format : c'est la forme exacte du
    defaut, et elle ne doit pas revenir."""
    src = open(os.path.join(_REPO, "_surrogate", "wrappers.py"),
               encoding="utf-8").read()
    for motif in ("u[0]:", "u[1]:", "u_arr[0]:", "grad[0]:", "grad[1]:"):
        assert motif not in src, (
            "trace codee pour n_var == 2 : %r" % motif)


# --------------------------------------------------------------------- #
# HFFunction : le cache n'est pas un detail
# --------------------------------------------------------------------- #
class _Compteur:
    """Faux evaluateur d'etat limite : g = somme(u), gradient constant."""

    def __init__(self, n_var):
        self.n_var = n_var
        self.appels = []

    def __call__(self, u):
        self.appels.append(list(np.array(u, float)))
        g = float(np.sum(np.array(u, float)))
        return g, [1.0] * self.n_var, None


def test_HFFunction_n_evalue_qu_une_fois_par_point():
    """OpenTURNS demande la valeur PUIS le gradient en deux appels separes.
    Sans cache, chaque point du plan serait resolu deux fois -- 466 s l'un sur
    le Moulin Blanc."""
    ev = _Compteur(2)
    f = wrappers.HFFunction(2, ev)
    assert f._exec([1.0, 2.0]) == [3.0]
    f._gradient([1.0, 2.0])
    assert len(ev.appels) == 1, "le gradient a relance une evaluation"
    f._exec([1.0, 2.0000000000001])   # sous la tolerance atol=1e-12
    assert len(ev.appels) == 1
    f._exec([1.0, 2.5])
    assert len(ev.appels) == 2, "un point vraiment different doit etre evalue"
    assert f.n_hf_calls == 2


def test_HFFunction_rend_le_gradient_au_format_OpenTURNS():
    f = wrappers.HFFunction(3, _Compteur(3))
    assert f._gradient([0.0, 0.0, 0.0]) == [[1.0], [1.0], [1.0]]


def test_HFFunction_fonctionne_a_une_seule_variable():
    """L'ancienne trace lisait `u_arr[1]` : avec une variable, IndexError au
    premier appel -- au milieu d'un run, pas a la compilation."""
    f = wrappers.HFFunction(1, _Compteur(1))
    assert f._exec([2.0]) == [2.0]


def test_HFFunction_fonctionne_a_trois_variables():
    """L'ancienne trace n'en montrait que deux : le journal mentait sur le
    point evalue."""
    ev = _Compteur(3)
    f = wrappers.HFFunction(3, ev)
    assert f._exec([1.0, 2.0, 4.0]) == [7.0]
    assert ev.appels == [[1.0, 2.0, 4.0]]


# --------------------------------------------------------------------- #
# PCKRG : la somme des deux composantes, valeur ET gradient
# --------------------------------------------------------------------- #
def test_PCKRGFunction_somme_les_deux_composantes():
    pce = ot.SymbolicFunction(["u1", "u2"], ["2*u1 + 3*u2"])
    krg = ot.SymbolicFunction(["u1", "u2"], ["u1 - u2"])
    f = wrappers.PCKRGFunction(2, pce, krg)
    assert f._exec([1.0, 1.0]) == pytest.approx([5.0 + 0.0])
    grad = f._gradient([1.0, 1.0])
    # d/du1 = 2 + 1 = 3 ; d/du2 = 3 - 1 = 2
    assert grad[0][0] == pytest.approx(3.0)
    assert grad[1][0] == pytest.approx(2.0)


def test_PCKRGFunction_traite_un_echantillon_comme_les_points_un_a_un():
    pce = ot.SymbolicFunction(["u1", "u2"], ["2*u1 + 3*u2"])
    krg = ot.SymbolicFunction(["u1", "u2"], ["u1 - u2"])
    f = wrappers.PCKRGFunction(2, pce, krg)
    points = [[1.0, 1.0], [-2.0, 0.5], [0.0, 0.0]]
    par_echantillon = [v for (v,) in f._exec_sample(points)]
    un_a_un = [f._exec(p)[0] for p in points]
    assert par_echantillon == pytest.approx(un_a_un)


# --------------------------------------------------------------------- #
# le module reste chargeable sans etude ni licence
# --------------------------------------------------------------------- #
def test_le_module_ne_depend_ni_de_l_etude_ni_du_solveur():
    src = open(os.path.join(_REPO, "_surrogate", "wrappers.py"),
               encoding="utf-8").read()
    for interdit in ("import schema", "import fabrique", "digital_structure",
                     "run_HF", "CFG"):
        assert interdit not in src, (
            "wrappers.py mentionne %r : l'evaluateur haute fidelite lui est "
            "PASSE, il ne va pas le chercher." % interdit)
