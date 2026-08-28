r"""Toute difference entre les deux etudes doit etre DECLAREE.

LE PROBLEME QUE CE FICHIER RESOUT
----------------------------------
Les deux scripts d'etude sont la meme implementation, copiee. Sur les lignes
de fonctions communes, 94,2 % etaient identiques au caractere pres le
26/08/2026. Les 5,8 % restants ne sont pas des differences d'etude : ce sont
des DERIVES, et elles se sont accumulees sans que rien ne les signale.

Six ont ete trouvees a la main, une par une, en lisant :

1. `run_HF` et `run_one_SOL` ne maillaient pas pareil -- deux tailles de
   maille differentes alimentaient le MEME metamodele ;
2. les deux `InitSolver.py` ne demandaient pas le meme solveur lineaire
   (MUMPS contre CuDss) ;
3. `print_visu_EFF` et `print_visu_sigma` n'existaient que d'un cote (et
   n'etaient appelees nulle part) ;
4. `build_metamodel_KRG` bornait theta a [1, 100] d'un cote, [0, 100] de
   l'autre -- a 0, le krigeage peut degenerer et interpoler le bruit ;
5. `build_DOE` tirait le plan initial sur un domaine CODE EN DUR (+/- 7,5)
   dans la flexion pure, et sur `eff_bounds` dans le Moulin Blanc : borner le
   domaine n'avait donc d'effet que d'un cote ;
6. `fond_hf_pour_figures` ne cadrait pas le fond haute fidelite sur les
   memes bornes.

Les deux derives sur le domaine de tirage (5) ont ete corrigees le
27/08/2026, gratuitement : les deux bornes coincidaient encore. C'est
exactement le genre de correction qui devient couteuse des qu'on attend.

Chacune a coute une lecture attentive. Aucune n'aurait ete trouvee par un
test -- il n'y en avait pas.

CE QUE FAIT CE TEST
--------------------
Il compare la STRUCTURE des fonctions communes -- l'arbre syntaxique, donc
sans les commentaires, sans les docstrings, sans la mise en page -- et exige
que toute difference figure dans un inventaire ecrit, avec sa raison.

Deux gardes, pas un :
* une divergence NON DECLAREE fait echouer la suite ;
* une divergence DECLAREE QUI A DISPARU la fait echouer aussi, pour que
  l'inventaire ne pourrisse pas en liste de vieilles excuses.

C'est la seule facon de vivre avec la duplication tant que la phase 0 n'est
pas finie : on ne l'interdit pas, on la rend visible.
"""

import ast
import os
import textwrap

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)

FLEXION = "pure_flexion/AC3_pure_flexion.py"
MOULIN = "Moulinblanc/AC3_moulinblanc.py"

#: Les differences LEGITIMES entre les deux etudes, avec leur raison.
#:
#: Une entree ici est un engagement : quelqu'un a lu les deux versions et a
#: conclu que l'ecart est voulu. Tout ce qui n'y figure pas est une derive.
DIVERGENCES_DECLAREES = {
    "print_visu":
        "DIFFERENCE D'ETUDE, reduite a sa plus simple expression : la flexion "
        "pure passe une SURCOUCHE (`_surcouche_analytique`) et une entree de "
        "legende, le Moulin Blanc passe None. Cinq lignes d'ecart sur 70, la "
        "ou il y en avait 31 sur 158. La surcouche est un ARGUMENT : le "
        "dessin, lui, est le meme des deux cotes, dans `_etapes/figurer.py`.",
    "print_3D_HF":
        "Meme raison que `print_visu` : la surface `flexion_claude` tracee en "
        "relief en regard du calcul haute fidelite n'existe que pour la "
        "flexion pure. Reste a traiter comme `print_visu` -- par une "
        "surcouche -- quand cette figure rejoindra le module.",
}


class _SansDocstring(ast.NodeTransformer):
    """Retire les docstrings : elles racontent, elles n'executent pas."""

    def _nettoyer(self, noeud):
        self.generic_visit(noeud)
        corps = noeud.body
        if (corps and isinstance(corps[0], ast.Expr)
                and isinstance(corps[0].value, ast.Constant)
                and isinstance(corps[0].value.value, str)):
            noeud.body = corps[1:] or [ast.Pass()]
        return noeud

    visit_FunctionDef = _nettoyer
    visit_AsyncFunctionDef = _nettoyer
    visit_ClassDef = _nettoyer
    visit_Module = _nettoyer


def _structure(source_lignes):
    """Empreinte structurelle d'une fonction : l'arbre, sans les docstrings.

    Les commentaires et la mise en page disparaissent d'eux-memes -- ils
    n'entrent pas dans l'arbre syntaxique. Ce qui reste est ce qui s'execute.
    """
    src = textwrap.dedent("".join(source_lignes))
    arbre = _SansDocstring().visit(ast.parse(src))
    return ast.dump(arbre, annotate_fields=True, include_attributes=False)


def _fonctions(script):
    chemin = os.path.join(_REPO, script)
    with open(chemin, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    lignes = src.splitlines(True)
    main = [n for n in ast.parse(src, filename=chemin).body
            if isinstance(n, ast.If)][0]
    return {n.name: lignes[n.lineno - 1:n.end_lineno]
            for n in main.body if isinstance(n, ast.FunctionDef)}


@pytest.fixture(scope="module")
def comparaison():
    A, B = _fonctions(FLEXION), _fonctions(MOULIN)
    communes = sorted(set(A) & set(B))
    differentes = {n for n in communes
                   if _structure(A[n]) != _structure(B[n])}
    return communes, differentes, A, B


# --------------------------------------------------------------------- #
# le garde principal
# --------------------------------------------------------------------- #
def test_aucune_divergence_non_declaree(comparaison):
    """Toute difference de STRUCTURE entre les deux etudes doit etre inscrite
    dans `DIVERGENCES_DECLAREES` avec sa raison."""
    communes, differentes, A, _ = comparaison
    non_declarees = sorted(differentes - set(DIVERGENCES_DECLAREES))
    assert not non_declarees, (
        "Divergence(s) non declaree(s) entre les deux etudes : %s\n"
        "Ces deux fichiers sont la MEME implementation copiee : un ecart de "
        "structure est une derive jusqu'a preuve du contraire. Six ont deja "
        "ete trouvees a la main, dont un solveur lineaire different et une "
        "taille de maille differente alimentant le meme metamodele.\n"
        "Soit l'ecart est voulu -- l'inscrire dans DIVERGENCES_DECLAREES avec "
        "sa raison -- soit il ne l'est pas, et il faut le corriger."
        % ", ".join("`%s` (%d l.)" % (n, len(A[n])) for n in non_declarees))


def test_aucune_declaration_perimee(comparaison):
    """L'inventaire ne doit pas devenir une liste de vieilles excuses : une
    divergence declaree qui a disparu doit sortir de la liste."""
    communes, differentes, _, _ = comparaison
    perimees = sorted(n for n in DIVERGENCES_DECLAREES
                      if n in communes and n not in differentes)
    assert not perimees, (
        "Ces divergences sont declarees mais n'existent plus : %s.\n"
        "Les retirer de DIVERGENCES_DECLAREES -- une exception qui ne "
        "s'applique plus masque la prochaine." % ", ".join(perimees))


def test_chaque_divergence_porte_une_raison():
    for nom, raison in DIVERGENCES_DECLAREES.items():
        assert raison and len(raison) > 40, (
            "%s : une divergence declaree sans raison lisible n'est pas une "
            "declaration, c'est un contournement." % nom)


# --------------------------------------------------------------------- #
# le cliquet : la duplication ne doit que decroitre
# --------------------------------------------------------------------- #
def test_la_duplication_ne_REGROSSIT_pas(comparaison):
    """La charge de copie : TOUTES les lignes de fonctions presentes des deux
    cotes, identiques ou non.

    CE CLIQUET A ETE CORRIGE LE 27/08/2026, ET C'EST INSTRUCTIF
    ------------------------------------------------------------
    Il comptait d'abord les lignes IDENTIQUES. Unifier deux copies qui avaient
    DIVERGE -- ce qui est un progres -- faisait donc REMONTER le chiffre, et
    le cliquet refusait le commit. Autrement dit : la mesure recompensait la
    divergence.

    Elle compte desormais la charge de copie totale. Une fonction presente
    dans les deux scripts coute deux corrections a chaque defaut et deux tests
    a chaque garantie, qu'elle soit identique ou non -- et si elle differe,
    elle coute EN PLUS une divergence a trouver. Seule l'extraction fait
    baisser ce nombre.

    Changer la definition d'un cliquet n'est legitime que dans ce sens-la :
    quand l'ancienne mesure recompensait le mauvais comportement. Jamais pour
    faire passer un commit.
    """
    #: 26/08 : 1 975 lignes de fonctions communes.
    #: 27/08, apres figurer + wrappers + ajuster + projection + grille +
    #: evaluation + plan : 1 451.
    #: 27/08 (suite) : l'etat de reprise part dans `_cache/reprise.py`.
    #: `_save_restart_state` passe de 54 a 21 lignes. 803 -> 767.
    #: 27/08 (suite) : l'evaluation du batch EFF part dans
    #: `_doe/evaluation.py`. 767 -> 738.
    #: 27/08 (suite) : le bilan de fin d'enrichissement rejoint les notions
    #: qu'il rapporte -- la decomposition dans `_reliability/eff.py`, les
    #: compteurs dans `ArretEFF.bilan`. 738 -> 709.
    #: 27/08 (suite) : le choix des points d'enrichissement (maximisation
    #: globale + Kriging Believer) part dans `_reliability/eff_ot.py`.
    #: 709 -> 664.
    #: 27/08 (suite) : le montage du plan -- greffe d'une reprise, assemblage
    #: des trois tableaux, plan recopiable -- part dans `_cache/doe.py` et
    #: `_doe/plan.py`. 664 -> 643.
    #: 27/08 (suite) : l'evenement `g < 0`, les points de depart, le warm
    #: start et le choix de coupe partent dans `_reliability/form.py`.
    #: 643 -> 631.
    #: 27/08 (suite) : les historiques sont vides en place, plus rebindes.
    #: 631 -> 630.
    #: 27/08 (suite) : les trois champs d'une coupe -- ecrits DEUX FOIS par
    #: etude -- rejoignent `_reliability/eff_ot`. 630 -> 621.
    #: 27/08 (suite) : la coupe finale se decide dans le flux, et l'enveloppe
    #: locale `_coupe_la_plus_parlante` disparait. 621 -> 615.
    #: 28/08 : « juger une iteration » -- et la decision de payer ou non
    #: l'encadrement -- rejoint `ControleurFORM`. 615 -> 598.
    #: 28/08 (suite) : preparation des historiques et amorce du plan
    #: initial. 598 -> 583.
    #: 28/08 (suite) : le journal des points devient un objet. 583 -> 569.
    #: 28/08 (suite) : rejouer l'enrichissement, et les diagnostics du
    #: metamodele. 569 -> 555.
    #: 28/08 (suite) : le fond de contour haute fidelite rejoint la
    #: grille qui le calcule. 555 -> 520.
    PLAFOND = 520
    communes, differentes, A, B = comparaison
    total = sum(len(A[n]) for n in communes)
    identiques = sum(len(A[n]) for n in communes
                     if "".join(A[n]) == "".join(B[n]))
    assert total <= PLAFOND, (
        "%d lignes de fonctions presentes dans les DEUX etudes (%d identiques, "
        "%d divergentes), au-dela du plafond %d.\n"
        "Chacune coute deux corrections par defaut et deux tests par garantie. "
        "ABAISSER le plafond quand une extraction en retire ; jamais le "
        "relever." % (total, identiques, total - identiques, PLAFOND))


# --------------------------------------------------------------------- #
# l'outil lui-meme doit etre juste
# --------------------------------------------------------------------- #
def test_la_comparaison_ignore_commentaires_et_docstrings(tmp_path):
    """Sans cela, `_hf_from_custom_points` -- 102 lignes qui ne different que
    par un commentaire -- serait signalee comme une divergence, et le bruit
    ferait ignorer les vraies."""
    a = ["def f(x):\n", '    """Un texte."""\n', "    # un commentaire\n",
         "    return x + 1\n"]
    b = ["def f(x):\n", '    """Un AUTRE texte."""\n',
         "    # un commentaire different\n", "    return x + 1\n"]
    assert _structure(a) == _structure(b)


def test_la_comparaison_voit_un_ecart_de_valeur(tmp_path):
    """C'est la forme exacte de la divergence `theta_min` : un litteral, au
    milieu de deux fonctions par ailleurs identiques."""
    a = ["def f(n):\n", "    return borne([1.0] * n, [100.0] * n)\n"]
    b = ["def f(n):\n", "    return borne([0.0] * n, [100.0] * n)\n"]
    assert _structure(a) != _structure(b)


def test_la_comparaison_voit_un_ecart_d_appel():
    """Forme de la divergence de maillage : le meme code appelle une taille
    differente."""
    a = ["def f():\n", "    return mailler(global_size=0.05)\n"]
    b = ["def f():\n", "    return mailler(global_size=0.15)\n"]
    assert _structure(a) != _structure(b)
