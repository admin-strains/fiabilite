"""
Que fait la chaine d'un point que le solveur declare non converge ?

DEFAUT 6 du plan de nettoyage. Les scripts AC lisaient `Primal_bound` sans
jamais regarder `converged` ni `solver_status` -- zero occurrence dans les
deux fichiers. Mesure du 25/08/2026 : a `global_physical_size = 0.018`,
Digital Structure sort NUMERICAL_ERROR avec un ecart primal-dual de 16 % et
alpha = 1,5197 au lieu de ~1,3188. Un tel point entrait au plan d'experiences
comme une evaluation valide, et contaminait le metamodele.

LA DECISION, ET POURQUOI ELLE N'EST PAS TECHNIQUE
--------------------------------------------------
Agnes, 26/08/2026 : les criteres de convergence rendus par Digital Structure
NE SONT PAS ENCORE FIABLES. Exclure aujourd'hui reviendrait a jeter des
evaluations correctes -- chacune coutant un SOCP, soit des secondes en
flexion pure et des minutes sur le Moulin Blanc. Le jour ou ces criteres
seront fiables, ces points seront exclus de l'enrichissement.

D'ou le comportement actuel, qui n'est ni l'ancien ni le futur :

    on SIGNALE chaque point suspect, avec son statut et son alpha
    on le GARDE
    et la bascule tient dans un booleen du fichier d'etude

Ces tests verifient les trois. Ils ne demandent ni Digital Structure ni
OpenTURNS : le contrat `solver/interface.py` suffit.
"""

import ast
import io
import os
import re
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "solver"), os.path.join(REPO, "_config")):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("tomli", reason="lecture TOML") if sys.version_info < (3, 11) else None

from interface import Evaluation, SolveurNonConverge          # noqa: E402
from schema import Configuration, charger                     # noqa: E402

ETUDES = {
    "pure_flexion": os.path.join(REPO, "pure_flexion", "AC3_pure_flexion.py"),
    "moulin_blanc": os.path.join(REPO, "Moulinblanc", "AC3_moulinblanc.py"),
}


# --------------------------------------------------------------------------- #
# 1. Le defaut est-il seulement DETECTABLE ?                                  #
# --------------------------------------------------------------------------- #
def test_un_point_non_converge_est_reconnaissable():
    """Le point mesure le 25/08 : NUMERICAL_ERROR, alpha=1,5197 au lieu de
    ~1,3188. Avant la phase 5, rien dans la chaine ne pouvait le distinguer
    d'une evaluation valide."""
    mauvais = Evaluation(g=0.5197, alpha=1.5197, sain=False,
                         diagnostic={"solver_status": "NUMERICAL_ERROR",
                                     "converged": False, "gap_relatif": 0.16})
    bon = Evaluation(g=0.3188, alpha=1.3188, sain=True,
                     diagnostic={"solver_status": "OPTIMAL", "converged": True})
    assert not mauvais.sain and bon.sain
    assert mauvais.diagnostic["gap_relatif"] > 0.10


def test_exige_sain_nomme_le_point_en_cause():
    """Un message qui ne dit pas OU le probleme s'est produit oblige a relancer
    le calcul pour le savoir -- des minutes sur le Moulin Blanc."""
    mauvais = Evaluation(g=0.0, alpha=1.0, sain=False,
                         diagnostic={"solver_status": "NUMERICAL_ERROR"})
    with pytest.raises(SolveurNonConverge, match=r"point \[48\.0, 550\.0\]"):
        mauvais.exige_sain("point [48.0, 550.0] du plan d'experiences")


# --------------------------------------------------------------------------- #
# 2. Le choix par defaut est-il bien celui qu'Agnes a arrete ?                #
# --------------------------------------------------------------------------- #
def test_par_defaut_les_points_suspects_sont_conserves():
    """FAUX par defaut, et c'est delibere : les criteres rendus par Digital
    Structure ne sont pas encore fiables. Si ce test tombe, c'est que quelqu'un
    a change une decision d'exploitation en croyant corriger un defaut."""
    assert Configuration(modelname="x").exclure_points_non_converges is False


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_aucune_etude_n_active_l_exclusion_pour_l_instant(nom):
    cfg = charger(os.path.join(REPO, "studies", nom + ".toml"))
    assert cfg.exclure_points_non_converges is False


def test_la_bascule_tient_dans_le_fichier_d_etude(tmp_path):
    """Le jour ou les criteres seront fiables, il ne doit y avoir aucune ligne
    de code a modifier."""
    f = tmp_path / "x.toml"
    f.write_text('modelname = "x"\nexclure_points_non_converges = true\n',
                 encoding="utf-8")
    assert charger(str(f)).exclure_points_non_converges is True


# --------------------------------------------------------------------------- #
# 3. Les scripts font-ils vraiment ce qui est annonce ?                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_les_deux_points_d_appel_consultent_l_etat_de_sante(nom):
    """Le plan d'experiences ET l'enrichissement. `run_HF` alimente le meme
    metamodele que `run_one_SOL` : les traiter differemment reintroduirait
    exactement le genre de divergence que la phase 5 a supprime."""
    src = io.open(ETUDES[nom], encoding="utf-8", errors="replace").read()
    assert src.count("if not ev.sain:") == 2, \
        "%s : %d point(s) d'appel consultent ev.sain, attendu 2" % (
            nom, src.count("if not ev.sain:"))


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_un_point_suspect_est_signale_avec_son_statut_et_son_alpha(nom):
    """Un journal qui dit « non converge » sans dire lequel ni de combien
    n'aide personne a juger si le point etait recuperable."""
    src = io.open(ETUDES[nom], encoding="utf-8", errors="replace").read()
    assert src.count("NON CONVERGE") == 2
    for attendu in ('ev.diagnostic.get("solver_status")', "ev.alpha"):
        assert src.count(attendu) >= 2, "%s : %s absent d'un des deux messages" % (nom, attendu)


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_l_exclusion_passe_par_le_contrat_et_non_par_un_test_local(nom):
    """`exige_sain()` porte le message et la regle. Les scripts ne doivent pas
    reimplementer leur propre verdict -- c'est ainsi que les quatre copies de
    l'appel solveur avaient diverge."""
    src = io.open(ETUDES[nom], encoding="utf-8", errors="replace").read()
    assert src.count("CFG.exclure_points_non_converges") >= 2
    assert src.count("ev.exige_sain(") == 2


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_le_script_compile_toujours(nom):
    ast.parse(io.open(ETUDES[nom], encoding="utf-8", errors="replace").read())


@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_la_raison_de_l_attente_est_ecrite_dans_le_script(nom):
    """Sans elle, le prochain lecteur corrigera « l'oubli » -- et jettera des
    evaluations valides a chaque run."""
    src = io.open(ETUDES[nom], encoding="utf-8", errors="replace").read()
    assert re.search(r"pas encore fiables", src), \
        "%s : la raison de conserver les points suspects n'est pas ecrite" % nom


# --------------------------------------------------------------------------- #
# Reprise d'un enrichissement : echouer tot, et le dire                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nom", sorted(ETUDES))
def test_une_reprise_sans_etat_a_reprendre_est_annoncee(nom):
    """`restart_enrich_only = true` demande un `restart_state.json` produit par
    un run precedent. Ce fichier vit dans le `.ds` du modele, PAS dans le
    depot : le cas se produit des qu'on reprend une etude sur un autre poste.

    Le code levait alors un `FileNotFoundError` brut -- et APRES plusieurs
    minutes de construction du modele CAD, qui prend 15 346 aciers sur le
    Moulin Blanc. Le controle est desormais fait tot, avec le chemin manquant
    et la ligne exacte a changer.
    """
    src = io.open(ETUDES[nom], encoding="utf-8", errors="replace").read()
    assert "os.path.isfile(_RESTART_STATE_FILE)" in src, \
        "%s : rien ne verifie que l'etat a reprendre existe" % nom
    # le message doit nommer le fichier ET la sortie
    i_garde = src.index("os.path.isfile(_RESTART_STATE_FILE)")
    extrait = src[i_garde:i_garde + 900]
    assert "restart_enrich_only = false" in extrait, \
        "%s : le message ne dit pas comment repartir de zero" % nom
    assert "_RESTART_STATE_FILE" in extrait


def test_aucune_etude_ne_demande_de_reprise():
    """Le script du Moulin Blanc portait `restart_enrich_only = true` : l'etat
    de TRAVAIL de son auteur, qui reprenait un enrichissement depuis un
    `restart_state.json`. Ce dump vit dans le `.ds` du modele et n'est pas
    dans le depot -- l'etude etait donc injouable ailleurs que sur le poste ou
    il existait.

    Decision d'Agnes, 26/08/2026 : on REJOUE LE RUN COMPLET. Les deux etudes
    partent donc du plan d'experiences. Le garde-fou ci-dessus reste utile
    pour qui remettra `true` en connaissance de cause."""
    for nom in sorted(ETUDES):
        cfg = charger(os.path.join(REPO, "studies", nom + ".toml"))
        assert cfg.restart_enrich_only is False, nom
