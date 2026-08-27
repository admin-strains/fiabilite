r"""Le FLUX des deux etudes, et non plus seulement leurs fonctions.

LE TROU DANS MON PROPRE OUTILLAGE -- 27/08/2026
------------------------------------------------
`test_97_divergences_declarees` compare les FONCTIONS des deux etudes, une a
une, et refuse toute divergence non declaree. Il a rendu six derives
visibles. Mais il ne compare que des fonctions -- or le corps de
`if __name__ == '__main__':` porte AUSSI 150 instructions au niveau du flux,
soit environ 440 lignes, presque 40 % de chaque etude. Rien ne les comparait.

C'est la qu'etait la derive suivante :

    Moulin Blanc : xt_eff = list(_restart_xt_eff)
    flexion pure : xt_eff = None

Le Moulin Blanc portait deja le correctif ; la flexion pure gardait le
defaut. C'est le TROISIEME cas de la journee ou un correctif n'existait que
d'un cote (le lanceur parallele, la reprise des compteurs EFF, celui-ci).

CE QUE CE DEFAUT COUTAIT
-------------------------
Reprendre une etude pour REFAIRE LES FIGURES -- `restart_enrich_only = true`
avec `do_EFF = false` -- rechargeait les points d'enrichissement puis
reecrivait le dump sans eux. Mesure, dump de 13 points dont 8 enrichis :

    avant le run figures : xt_eff = 8   round_sizes = [13]
    apres                : xt_eff = 0   round_sizes = [13, 0]

Les 13 points restent dans `xt` : rien n'est perdu du CALCUL. Ce qui
disparait, c'est de savoir lesquels venaient de l'enrichissement -- donc le
budget EFF tenu (une reprise suivante repartirait de zero et paierait
jusqu'a `n_max_EFF_points` appels solveur de plus), le decoupage en rounds,
et les points marques sur les figures.

CE QUE CE FICHIER AJOUTE
-------------------------
La comparaison structurelle du flux, avec le meme principe que
`test_97` : les ecarts restants sont ENUMERES par le message d'echec, et un
cliquet interdit qu'il en apparaisse de nouveaux.
"""

import ast
import difflib
import io
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
if _ICI not in sys.path:
    sys.path.insert(0, _ICI)

from test_97_divergences_declarees import _SansDocstring   # noqa: E402

ETUDES = ("pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py")


def _flux(script):
    """Les instructions du flux principal : tout `main` SAUF les `def`.

    Docstrings et commentaires retires -- ils racontent, ils n'executent pas.
    Retourne `(cles structurelles, premieres lignes lisibles)`.
    """
    s = io.open(os.path.join(_REPO, script), encoding="utf-8",
                errors="replace").read()
    lignes = s.splitlines()
    main = [n for n in ast.parse(s).body if isinstance(n, ast.If)][0]
    cles, textes = [], []
    for n in main.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        cles.append(ast.dump(ast.fix_missing_locations(_SansDocstring().visit(n))))
        textes.append(lignes[n.lineno - 1].strip()[:78])
    return cles, textes


@pytest.fixture(scope="module")
def comparaison_du_flux():
    ka, ta = _flux(ETUDES[0])
    kb, tb = _flux(ETUDES[1])
    ecarts = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, ka, kb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        ecarts += [("PF", t) for t in ta[i1:i2]]
        ecarts += [("MB", t) for t in tb[j1:j2]]
    return len(ka), len(kb), ecarts


def test_les_deux_etudes_ont_un_flux_de_meme_longueur(comparaison_du_flux):
    """Elles font la meme chose, dans le meme ordre. Une difference de
    nombre d'etapes serait un ecart de DEROULE, pas de reglage."""
    n_pf, n_mb, _ = comparaison_du_flux
    assert abs(n_pf - n_mb) <= 3, (
        "le flux principal ne compte pas le meme nombre d'etapes : "
        "%d (flexion pure) contre %d (Moulin Blanc)" % (n_pf, n_mb))


def test_le_flux_principal_ne_DIVERGE_pas_davantage(comparaison_du_flux):
    """Le cliquet, sur la partie que `test_97` ne regardait pas.

    Les ecarts restants sont de vraies differences d'etude : deux modeles
    differents, deux jeux de constantes, deux facons de nommer les aciers.
    Ils sont enumeres par le message d'echec -- si le nombre monte, c'est
    qu'un ecart NOUVEAU est apparu, et il faut le lire.
    """
    #: 27/08/2026, apres alignement de `xt_eff` sur le Moulin Blanc.
    PLAFOND = 26
    n_pf, n_mb, ecarts = comparaison_du_flux
    listing = "\n  ".join("%s  %s" % (cote, txt) for cote, txt in ecarts)
    assert len(ecarts) <= PLAFOND, (
        "%d instructions divergentes dans le flux principal (plafond %d) :\n"
        "  %s\n"
        "Ces deux fichiers sont la MEME implementation copiee. Un ecart de "
        "flux est une derive jusqu'a preuve du contraire -- trois correctifs "
        "n'existaient que d'un seul cote. ABAISSER le plafond quand on en "
        "retire un ; jamais le relever." % (len(ecarts), PLAFOND, listing))


# --------------------------------------------------------------------------- #
# LA DERIVE QUE CE DETECTEUR A TROUVEE                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ETUDES)
def test_une_reprise_garde_ses_points_d_enrichissement(script):
    """Sans cela, un run fait pour REFAIRE LES FIGURES effacait du dump la
    liste des points enrichis -- 8 sur 13 dans la mesure ci-dessus."""
    s = io.open(os.path.join(_REPO, script), encoding="utf-8",
                errors="replace").read()
    assert "xt_eff = list(_restart_xt_eff)   # survit sans enrichissement" in s, (
        "%s : la branche de reprise ne reprend plus ses points "
        "d'enrichissement." % script)


@pytest.mark.parametrize("script", ETUDES)
def test_un_run_neuf_part_bien_de_zero(script):
    """L'autre branche, elle, DOIT partir vide : un run neuf n'a rien
    charge, et seeder `xt_eff` y serait faux."""
    s = io.open(os.path.join(_REPO, script), encoding="utf-8",
                errors="replace").read()
    assert "        xt_eff = None\n" in s, (
        "%s : la branche d'un run neuf ne part plus d'un `xt_eff` vide."
        % script)


# --------------------------------------------------------------------------- #
# CE QUE LE DETECTEUR SIGNALE ET QUI RESTE A ARBITRER                          #
# --------------------------------------------------------------------------- #
def test_seule_la_flexion_pure_verifie_ses_parametres_CAD():
    """CONSTATE, NON CORRIGE -- a arbitrer avec Agnes.

    La flexion pure refuse de demarrer si un `params_names` n'est pas connu
    de `PARAM_CONFIG_CAD` ; le Moulin Blanc ne verifie rien. Un parametre mal
    orthographie y serait donc simplement ignore : la variable aleatoire
    n'atteindrait jamais le modele, et le calcul tournerait sur un ouvrage
    ou elle est figee -- sans une ligne de journal.

    Ce test FIGE l'asymetrie plutot que de la corriger : ajouter le controle
    au Moulin Blanc peut faire echouer une etude qui tourne aujourd'hui, et
    c'est une decision d'etude.
    """
    controle = "if not set(params_names) <= set(PARAM_CONFIG_CAD.keys()):"
    presents = [s for s in ETUDES
                if controle in io.open(os.path.join(_REPO, s),
                                       encoding="utf-8", errors="replace").read()]
    assert presents == ["pure_flexion/AC3_pure_flexion.py"], (
        "l'asymetrie a bouge : %s. Si le controle a ete ajoute au Moulin "
        "Blanc, supprimer ce test." % presents)


def test_un_compteur_temporaire_traine_dans_le_moulin_blanc():
    """CONSTATE, NON CORRIGE. `_run_HF_count = [0]  # temporaire` date d'un
    diagnostic memoire. Il n'existe pas en flexion pure. Le retirer est sans
    risque, mais c'est du code de l'auteur : on le signale."""
    s = io.open(os.path.join(_REPO, ETUDES[1]), encoding="utf-8",
                errors="replace").read()
    assert "_run_HF_count = [0]" in s, (
        "le compteur temporaire a disparu du Moulin Blanc -- supprimer ce "
        "test, il a fait son office.")
