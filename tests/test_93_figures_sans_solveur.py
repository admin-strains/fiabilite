r"""Une figure ne doit JAMAIS pouvoir appeler le solveur.

LE DEFAUT QUE CE FICHIER REND IMPOSSIBLE -- 26/08/2026
-------------------------------------------------------
Sept des onze fonctions `print_*` des scripts AC peuvent declencher un appel
au solveur. Le chemin est indirect, donc invisible a la lecture :

    print_planche_EFF -> _hf_from_custom_points -> run_HF
    print_visu        -> _hf_from_custom_points -> run_HF

Une fonction dont le nom dit « imprime » lance ainsi jusqu'a 225 appels, soit
29 heures sur le Moulin Blanc.

Ce n'est pas une hypothese : c'est ce qui m'a fait annoncer le matin meme, a
tort, que la grille arrivait EN DERNIER et qu'on pouvait l'interrompre sans
rien perdre. Elle arrive AVANT l'enrichissement -- `print_planche_EFF` est
appelee ligne 2758, `run_EFF` ligne 2759 -- et je ne l'ai vu qu'en lisant le
journal d'execution, pas le code. Une fonction `print_*` n'eveille aucun
soupcon.

CE QUE LE TEST FAIT
--------------------
Il construit le graphe d'appel des fonctions d'un module (ou du bloc `__main__`
d'un script) et calcule la FERMETURE TRANSITIVE : une fonction de trace
peut-elle, par n'importe quel chemin, atteindre une fonction qui evalue l'etat
limite ?

Sur `_etapes/figurer.py`, la reponse devra etre non -- et ce sera une propriete
verifiee, pas une intention.

Sur les scripts AC, le test MESURE l'avancement de la phase 0 : il est
`xfail` aujourd'hui, et deviendra vert quand la separation sera faite.
"""

import ast
import os

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

#: Les fonctions qui evaluent l'etat limite, c'est-a-dire qui coutent un SOCP.
EVALUENT = {"run_HF", "run_one_SOL", "run_DOE_parallel", "run_HF_grid_parallel",
            "evaluer", "SOLV", "ANISO_MESH"}


def _graphe(chemin, dans_main=True):
    """(fonctions, appels) d'un fichier. `dans_main` cible le bloc __main__."""
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        arbre = ast.parse(fh.read(), filename=chemin)
    corps = arbre.body
    if dans_main:
        mains = [n for n in corps if isinstance(n, ast.If)]
        if not mains:
            return {}, {}
        corps = mains[0].body
    fns = {n.name: n for n in corps if isinstance(n, ast.FunctionDef)}
    appels = {}
    for nom, noeud in fns.items():
        cibles = set()
        for x in ast.walk(noeud):
            if isinstance(x, ast.Call):
                f = x.func
                if isinstance(f, ast.Name):
                    cibles.add(f.id)
                elif isinstance(f, ast.Attribute):
                    cibles.add(f.attr)
        appels[nom] = cibles
    return fns, appels


def _atteint_le_solveur(nom, appels, pile=()):
    """Fermeture transitive : `nom` peut-il aboutir a une evaluation ?"""
    if nom in EVALUENT:
        return True
    if nom in pile or nom not in appels:
        return False
    return any(_atteint_le_solveur(c, appels, pile + (nom,)) for c in appels[nom])


def _chemin_vers_le_solveur(nom, appels, pile=()):
    if nom in EVALUENT:
        return [nom]
    if nom in pile or nom not in appels:
        return None
    for c in sorted(appels[nom]):
        suite = _chemin_vers_le_solveur(c, appels, pile + (nom,))
        if suite:
            return [nom] + suite
    return None


SCRIPTS = ["pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py"]


# --------------------------------------------------------------------- #
# l'outil de detection lui-meme doit etre juste
# --------------------------------------------------------------------- #
def test_la_detection_suit_bien_un_chemin_indirect(tmp_path):
    """Garde-fou du garde-fou : si la fermeture transitive ne suivait que les
    appels DIRECTS, elle ne verrait rien -- le defaut du 26/08 passe par un
    intermediaire."""
    f = tmp_path / "faux.py"
    f.write_text(
        "if __name__ == '__main__':\n"
        "    def run_HF(u):\n        return 0\n"
        "    def intermediaire():\n        return run_HF(1)\n"
        "    def print_joli():\n        return intermediaire()\n"
        "    def print_sage():\n        return 42\n",
        encoding="utf-8")
    _, appels = _graphe(str(f))
    assert _atteint_le_solveur("print_joli", appels), "chemin indirect manque"
    assert not _atteint_le_solveur("print_sage", appels)
    assert _chemin_vers_le_solveur("print_joli", appels) == \
        ["print_joli", "intermediaire", "run_HF"]


def test_la_detection_ne_boucle_pas_sur_une_recursion(tmp_path):
    f = tmp_path / "recursif.py"
    f.write_text(
        "if __name__ == '__main__':\n"
        "    def a():\n        return b()\n"
        "    def b():\n        return a()\n",
        encoding="utf-8")
    _, appels = _graphe(str(f))
    assert not _atteint_le_solveur("a", appels)


# --------------------------------------------------------------------- #
# la mesure sur les scripts d'aujourd'hui
# --------------------------------------------------------------------- #
@pytest.mark.xfail(strict=False, reason="phase 0 en cours : le trace et le "
                                        "calcul de grille sont encore melanges")
@pytest.mark.parametrize("script", SCRIPTS)
def test_aucune_fonction_de_trace_n_appelle_le_solveur(script):
    """LE critere structurel. Tant qu'il est rouge, `n_grid_hf` est un budget
    de calcul deguise en resolution d'image."""
    chemin = os.path.join(_REPO, script)
    fns, appels = _graphe(chemin)
    traces = sorted(n for n in fns if n.startswith("print_"))
    coupables = {n: _chemin_vers_le_solveur(n, appels)
                 for n in traces if _atteint_le_solveur(n, appels)}
    assert not coupables, (
        "%s : %d des %d fonctions de trace peuvent appeler le solveur.\n%s"
        % (script, len(coupables), len(traces),
           "\n".join("  " + " -> ".join(c) for c in coupables.values())))


@pytest.mark.parametrize("script", SCRIPTS)
def test_le_nombre_de_coupables_ne_REMONTE_pas(script):
    """Cliquet : la separation ne peut que progresser. Relever ce plafond pour
    faire passer un test reviendrait a supprimer la mesure."""
    PLAFOND = 3
    fns, appels = _graphe(os.path.join(_REPO, script))
    traces = [n for n in fns if n.startswith("print_")]
    coupables = [n for n in traces if _atteint_le_solveur(n, appels)]
    assert len(coupables) <= PLAFOND, (
        "%s : %d fonctions de trace appellent le solveur, au-dela du plafond "
        "%d. ABAISSER le plafond quand la separation progresse ; jamais le "
        "relever." % (script, len(coupables), PLAFOND))


# --------------------------------------------------------------------- #
# la garantie, des que le module existera
# --------------------------------------------------------------------- #
def test_le_module_de_figures_ne_touche_jamais_le_solveur():
    """Se declenchera des que `_etapes/figurer.py` existera. En attendant, il
    signale que la migration reste a faire -- un test qui passe parce que le
    fichier est absent ne garantit rien."""
    chemin = os.path.join(_REPO, "_etapes", "figurer.py")
    if not os.path.exists(chemin):
        pytest.skip("_etapes/figurer.py pas encore ecrit (phase 0 en cours)")
    fns, appels = _graphe(chemin, dans_main=False)
    coupables = {n: _chemin_vers_le_solveur(n, appels)
                 for n in fns if _atteint_le_solveur(n, appels)}
    assert not coupables, (
        "figurer.py doit couter ZERO appel solveur :\n%s"
        % "\n".join("  " + " -> ".join(c) for c in coupables.values()))


def test_le_contrat_des_actions_est_declare():
    """`_etapes/__init__.py` doit dire ce que chaque action coute -- c'est ce
    qui manquait quand une fonction `print_*` lancait 29 heures de calcul."""
    import sys
    sys.path.insert(0, os.path.join(_REPO, "_etapes"))
    import importlib
    paquet = importlib.import_module("_etapes")
    noms = {a[0] for a in paquet.ACTIONS}
    assert noms == {"plan", "enrichir", "grille", "analyser", "figurer"}
    assert set(paquet.SANS_SOLVEUR) <= noms
    for nom, _desc, cout in paquet.ACTIONS:
        assert cout, "%s ne declare pas son cout" % nom
        if nom in paquet.SANS_SOLVEUR:
            assert cout == "0", "%s est declaree sans solveur mais coute %r" % (nom, cout)
