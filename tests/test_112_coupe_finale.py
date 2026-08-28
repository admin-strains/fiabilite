r"""La coupe finale se decide AVANT les figures, pas pendant.

LE PIEGE D'ORDRE -- 27/08/2026
-------------------------------
`print_visu` choisissait la coupe finale dans son corps, et la publiait par
effet de bord sur une globale :

    global slice_def_final
    if slice_def_final is None:
        slice_def_final = _coupe_la_plus_parlante(best_result)

Or ses fonds de contour lui sont passes EN ARGUMENT, donc evalues AVANT que
la fonction ne s'execute :

    print_visu(..., fond_hf_final=fond_hf_pour_figures(slice_def_final, ...))

Sur le Moulin Blanc, `slice_def_final` part a `None`. Au moment ou l'argument
est evalue, `fond_hf_pour_figures` retombe donc sur `slice_def`. La figure de
synthese recevait un fond calcule pour UNE coupe et etait tracee sur une
AUTRE. La planche globale, appelee juste apres, recevait elle le fond de la
vraie coupe finale -- deux figures du meme run, deux coupes.

POURQUOI PERSONNE NE L'AVAIT VU
--------------------------------
A DEUX variables, `coupe_la_plus_parlante` rend `(0, 1, {})`, c'est-a-dire
exactement `slice_def`. Les deux coupes coincident, et le defaut est
strictement invisible. Il n'apparait qu'a partir de trois variables -- ou la
coupe finale fige les variables secondaires a `u*`, et ne ressemble plus du
tout au plan par defaut.

C'est le meme genre de defaut que le fond haute fidelite trace sur le cadre
des figures : des valeurs affichees ailleurs que la ou elles ont ete
calculees. Et masque par la meme cause -- l'etude de reference a deux
variables.

CE QUI A CHANGE
----------------
La coupe est decidee une fois, dans le flux principal, avant tout appel de
figure. `print_visu` la RECOIT et ne mute plus rien. L'enveloppe locale
`_coupe_la_plus_parlante` n'avait plus d'appelant : elle disparait, le flux
appelle `_reliability/form.py` directement.
"""

import ast
import io
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_reliability"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ot = pytest.importorskip("openturns")
_form = pytest.importorskip("form", reason="FORM multimodal")

ETUDES = ("pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py")


def _source(script):
    return io.open(os.path.join(_REPO, script), encoding="utf-8",
                   errors="replace").read()


# --------------------------------------------------------------------------- #
# LE DEFAUT, EN CHIFFRES                                                       #
# --------------------------------------------------------------------------- #
class _Resultat:
    def __init__(self, importance, u_star):
        self._i, self._u = importance, u_star

    def getImportanceFactors(self):
        return ot.Point(self._i)

    def getStandardSpaceDesignPoint(self):
        return ot.Point(self._u)


def test_a_deux_variables_les_deux_coupes_coincident():
    """C'est ce qui rendait le defaut invisible : l'etude de reference et le
    Moulin Blanc ont deux variables, et la coupe finale y vaut exactement la
    coupe par defaut."""
    r = _Resultat([0.4, 0.6], [-2.8, -3.8])
    assert _form.coupe_la_plus_parlante(r, 2, (0, 1, {})) == (0, 1, {})


def test_a_trois_variables_elles_different_vraiment():
    """La coupe finale fige la variable secondaire a `u*`. Un fond calcule
    pour `(0, 1, {})` n'a alors plus rien a voir avec le plan trace."""
    r = _Resultat([0.5, 0.05, 0.45], [-2.8, 7.5, -3.8])
    coupe = _form.coupe_la_plus_parlante(r, 3, (0, 1, {}))
    assert coupe != (0, 1, {})
    assert coupe == (0, 2, {1: 7.5})


def test_la_valeur_figee_vient_du_point_de_conception():
    """Ce n'est pas une valeur neutre : c'est la ou FORM dit que la
    defaillance se joue. Figer ailleurs montrerait une coupe ou il ne se
    passe rien."""
    r = _Resultat([0.5, 0.05, 0.45], [-2.8, 7.5, -3.8])
    _, _, figees = _form.coupe_la_plus_parlante(r, 3, (0, 1, {}))
    assert figees == {1: 7.5}


# --------------------------------------------------------------------------- #
# LA DECISION EST PRISE AVANT, ET UNE SEULE FOIS                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ETUDES)
def test_la_coupe_est_decidee_dans_le_flux_pas_dans_une_figure(script):
    s = _source(script)
    assert "slice_def_final = _form.coupe_la_plus_parlante(" in s, (
        "%s : la coupe finale n'est plus decidee dans le flux principal."
        % script)


@pytest.mark.parametrize("script", ETUDES)
def test_aucune_figure_ne_publie_la_coupe_par_effet_de_bord(script):
    """`global slice_def_final` dans une fonction de trace, c'est la cause
    exacte du piege : la valeur d'un argument dependait d'un effet de bord de
    la fonction a qui on le passait."""
    s = _source(script)
    assert "global slice_def_final" not in s, (
        "%s : une fonction publie de nouveau la coupe finale par effet de "
        "bord." % script)


@pytest.mark.parametrize("script", ETUDES)
def test_la_coupe_est_decidee_AVANT_le_premier_fond_de_contour(script):
    """L'invariant qui compte, verifie sur l'ordre reel des lignes du flux
    principal -- pas sur une intention ecrite en commentaire."""
    s = _source(script)
    lignes = s.splitlines()
    decision = [i for i, l in enumerate(lignes, 1)
                if "slice_def_final = _form.coupe_la_plus_parlante(" in l]
    assert len(decision) == 1, decision

    # Seuls comptent les fonds demandes POUR LA COUPE FINALE. Les autres
    # appels (`fond_hf_pour_figures()` sans argument) visent la coupe
    # courante et ne dependent pas de cette decision.
    main = [n for n in ast.parse(s).body if isinstance(n, ast.If)][0]
    dans_une_fonction = [(f.lineno, f.end_lineno) for f in main.body
                         if isinstance(f, ast.FunctionDef)]
    fonds = []
    for n in ast.walk(main):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "fond_hf_pour_figures"):
            continue
        if any(a <= n.lineno <= b for a, b in dans_une_fonction):
            continue
        cite = any(isinstance(x, ast.Name) and x.id == "slice_def_final"
                   for arg in n.args for x in ast.walk(arg))
        if cite:
            fonds.append(n.lineno)
    assert fonds, "aucun fond de contour demande pour la coupe finale"
    assert decision[0] < min(fonds), (
        "%s : la coupe finale (l. %d) est decidee APRES le premier fond de "
        "contour (l. %d) -- le fond sera calcule pour la mauvaise coupe."
        % (script, decision[0], min(fonds)))


@pytest.mark.parametrize("script", ETUDES)
def test_la_figure_de_synthese_recoit_sa_coupe(script):
    """Elle ne la lit plus dans une globale : elle la prend en argument,
    comme tout le reste de ce qu'elle dessine."""
    s = _source(script)
    main = [n for n in ast.parse(s).body if isinstance(n, ast.If)][0]
    fns = {f.name: f for f in main.body if isinstance(f, ast.FunctionDef)}
    assert "coupe" in [a.arg for a in fns["print_visu"].args.args], (
        "%s : `print_visu` ne recoit pas la coupe." % script)


@pytest.mark.parametrize("script", ETUDES)
def test_l_enveloppe_locale_a_disparu(script):
    """Elle ne servait qu'a lier trois variables de `main` ; son seul
    appelant restant est le flux, qui peut appeler le module directement."""
    s = _source(script)
    assert "_coupe_la_plus_parlante" not in s, (
        "%s : l'enveloppe locale est revenue." % script)
