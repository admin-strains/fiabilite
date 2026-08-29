r"""Un run qui n'entre jamais dans la boucle d'enrichissement.

LE DEFAUT -- D'ORIGINE, CORRIGE LE 28/08/2026
----------------------------------------------
`run_EFF` se termine par une ligne de bilan :

    if print_Pf:
        print(f"  [BB informatif final] ratio = {_ratio_bb} ...")

Or `_ratio_bb` n'etait affecte QUE dans le corps de la boucle. Quand la
boucle ne tourne pas une seule fois, la variable n'existe pas :

    UnboundLocalError: local variable '_ratio_bb' referenced before assignment

Le run meurt APRES tout le travail d'enrichissement, sur une ligne de
journal.

QUAND LA BOUCLE NE TOURNE PAS
------------------------------
`ArretEFF.continuer` rend False des l'entree dans deux cas parfaitement
legitimes :

  1. le budget d'enrichissement est deja epuise -- `len(xt_eff) >=
     n_max_EFF_points`. C'est exactement ce qui arrive quand on REPREND une
     etude qui a consomme ses points ;
  2. le critere EFF est deja satisfait -- le metamodele est assez bon.

Les deux ont ete reproduits en situation, sur l'etude analytique.

POURQUOI PERSONNE NE L'AVAIT VU
--------------------------------
Il faut `print_Pf = true`, et les trois etudes de reference le mettent a
`false`. C'est le quatrieme defaut de la journee qu'une configuration non
exercee par la chaine de controle tenait cache -- apres la branche
parallele du batch EFF, la greffe d'un plan interrompu, et le plan
recopiable.

LE CORRECTIF
-------------
`_ratio_bb = None` avant la boucle. Le bilan imprime alors `ratio = None`,
ce qui est la verite : aucun ratio BB n'a ete mesure pendant ce run.
"""

import ast
import io
import os

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

ETUDES = ("pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py")

#: La boucle vit ici depuis le 28/08/2026. Elle etait recopiee dans les deux
#: etudes, et ce fichier devait donc verifier deux fois le meme invariant sur
#: deux textes qu'il fallait esperer identiques.
BOUCLE = "_reliability/enrichissement.py"

#: le nom que porte le ratio du bilan dans la boucle
RATIO = "ratio_bb"


def _run_EFF():
    """Le corps de la boucle d'enrichissement, et son arbre."""
    s = io.open(os.path.join(_REPO, BOUCLE), encoding="utf-8",
                errors="replace").read()
    classe = {n.name: n for n in ast.parse(s).body
              if isinstance(n, ast.ClassDef)}["BoucleEFF"]
    return {f.name: f for f in classe.body
            if isinstance(f, ast.FunctionDef)}["enrichir"]


def _noms_affectes(noeud, avant_ligne=None):
    """Les noms affectes dans `noeud`, hors corps de boucle si demande."""
    noms = set()
    for n in ast.walk(noeud):
        if isinstance(n, (ast.While, ast.For)) and avant_ligne == "hors_boucle":
            continue
        if isinstance(n, ast.Assign):
            for cible in n.targets:
                for x in ast.walk(cible):
                    if isinstance(x, ast.Name):
                        noms.add(x.id)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            for x in ast.walk(n.target):
                if isinstance(x, ast.Name):
                    noms.add(x.id)
        elif isinstance(n, ast.Tuple):
            pass
    return noms


# --------------------------------------------------------------------------- #
# L'INVARIANT : CE QUE LE BILAN LIT EXISTE MEME SI LA BOUCLE N'A PAS TOURNE     #
# --------------------------------------------------------------------------- #
def test_le_ratio_du_bilan_est_defini_avant_la_boucle():
    """L'invariant, verifie sur l'arbre : le ratio doit etre affecte quelque
    part AVANT le `while`, pas seulement dedans."""
    f = _run_EFF()
    boucles = [n for n in ast.walk(f) if isinstance(n, ast.While)]
    assert len(boucles) == 1, len(boucles)
    debut_boucle = boucles[0].lineno

    affectations = [n.lineno for n in ast.walk(f)
                    if isinstance(n, ast.Assign)
                    for c in n.targets
                    for x in ast.walk(c)
                    if isinstance(x, ast.Name) and x.id == RATIO]
    assert affectations, "`%s` n'est plus affecte du tout" % RATIO
    assert min(affectations) < debut_boucle, (
        "`%s` n'est affecte qu'a partir de la ligne %d, alors que la boucle "
        "commence l. %d et que le bilan le lit APRES. Un run qui n'entre "
        "jamais dans la boucle -- budget epuise a la reprise, ou critere EFF "
        "deja satisfait -- leve UnboundLocalError."
        % (RATIO, min(affectations), debut_boucle))


def test_toute_variable_lue_apres_la_boucle_est_definie_avant():
    """Le meme controle, generalise : aucune variable lue APRES la boucle ne
    doit dependre du fait qu'elle ait tourne.

    C'est ce controle-la qui aurait trouve le defaut sans qu'on le cherche.
    """
    f = _run_EFF()
    boucle = [n for n in ast.walk(f) if isinstance(n, ast.While)][0]

    def _affectes(noeuds):
        noms = set()
        for racine in noeuds:
            for n in ast.walk(racine):
                if isinstance(n, ast.Assign):
                    cibles = n.targets
                elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                    cibles = [n.target]
                elif isinstance(n, ast.arguments):
                    noms.update(a.arg for a in n.args)
                    continue
                else:
                    continue
                for c in cibles:
                    for x in ast.walk(c):
                        if isinstance(x, ast.Name):
                            noms.add(x.id)
        return noms

    corps = list(f.body)
    i = corps.index(boucle)
    avant = _affectes(corps[:i]) | {a.arg for a in f.args.args}
    dans = _affectes([boucle])
    apres = corps[i + 1:]

    lus_apres = set()
    for n in apres:
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                lus_apres.add(x.id)

    # ce qui n'est defini QUE dans la boucle et lu apres
    fragiles = sorted((lus_apres & dans) - avant)
    assert not fragiles, (
        "%s n'est affecte que dans la boucle d'enrichissement et lu apres "
        "elle. Si la boucle ne tourne pas -- budget deja epuise a la reprise, "
        "critere EFF deja satisfait -- c'est un UnboundLocalError au moment "
        "du bilan." % ", ".join("`%s`" % n for n in fragiles))


# --------------------------------------------------------------------------- #
# LE JOURNAL RESTE HONNETE                                                     #
# --------------------------------------------------------------------------- #
def test_le_bilan_BB_n_est_ecrit_qu_une_fois():
    """La ligne etait recopiee dans les deux branches d'un `if/else`. Une
    seule ecriture : deux copies d'un format, c'est deux occasions de le
    faire diverger."""
    s = io.open(os.path.join(_REPO, BOUCLE), encoding="utf-8",
                errors="replace").read()
    assert s.count("[BB informatif final] ratio =") == 1, (
        "%s : la ligne de bilan BB est ecrite plusieurs fois." % BOUCLE)


@pytest.mark.parametrize("script", ETUDES)
def test_le_bilan_n_est_pas_revenu_dans_une_etude(script):
    """Le pendant negatif : une etude qui reecrirait cette ligne aurait
    reintroduit une copie de la boucle."""
    s = io.open(os.path.join(_REPO, script), encoding="utf-8",
                errors="replace").read()
    assert "[BB informatif final]" not in s, (
        "%s : le bilan de la boucle est revenu dans l'etude." % script)
