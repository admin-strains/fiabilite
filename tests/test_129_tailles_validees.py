r"""Les tailles d'une etude : zero et negatif etaient acceptes partout.

CE QUI A ETE MESURE -- 29/08/2026
----------------------------------
Dix-sept valeurs absurdes essayees sur `Configuration.valider` : les dix-sept
passaient. `n_grid_hf = 0`, `n_batch_EFF = -3`, `n_IS = 0`, `global_size = 0`,
`epsilon_factor = -2`...

La consequence n'est pas un plantage. Avec `n_batch_EFF = 0`, la boucle
d'enrichissement TOURNE, ne paie aucun point, et s'arrete en se declarant
convergee -- le critere BS est satisfait puisque beta ne bouge pas, faute de
point nouveau. Mesure, budget de trois points :

    n_batch_EFF = 1   4 tours, 3 points payes
    n_batch_EFF = 0   4 tours, 0 point paye     <- « convergé »
    n_batch_EFF = -3  4 tours, 0 point paye
    n_max_EFF_points = -1   1 tour, 0 point paye

Une etude reglee ainsi rend un beta et une Pf issus du seul plan initial, sans
qu'une ligne ne le signale.

D'OU VIENNENT LES BORNES
-------------------------
De ce que les CINQ etudes du depot respectent deja, releve avant d'ecrire la
regle : `n_grid_hf` descend a 2 (run de fumee), `eps_taylor` et
`theta_min_krg` valent 0 au Moulin Blanc, `n_max_EFF_points` vaut au moins 5.
Aucune n'est refusee -- ce fichier le verifie.
"""

import glob
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
if os.path.join(_REPO, "_config") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "_config"))

import schema   # noqa: E402


def _valide(**kw):
    schema.Configuration(modelname="x", **kw).valider()


# --------------------------------------------------------------------------- #
# 1. CE QUI DOIT ETRE REFUSE                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nom,valeur", [
    ("n_batch_EFF", 0), ("n_batch_EFF", -3),
    ("n_max_EFF_points", -1),
    ("n_IS", 0), ("n_max_FORM", 0), ("n_sp", 0), ("n_NLopt_EFF", 0),
    ("n_grid", 0), ("n_grid", 1), ("n_grid_hf", 0), ("n_grid_hf", 1),
    ("geo_min_approx", -5),
    ("eps_taylor", -0.1), ("theta_min_krg", -1.0),
    ("epsilon_factor", 0.0), ("epsilon_factor", -2.0),
    ("global_size", 0.0), ("global_size", -1.0),
])
def test_une_taille_absurde_est_refusee(nom, valeur):
    with pytest.raises(ValueError, match=nom):
        _valide(**{nom: valeur})


def test_le_message_dit_la_CONSEQUENCE_pas_la_regle():
    """« doit etre positif » n'apprend rien. Ce qu'il faut savoir, c'est ce
    que la valeur produirait."""
    with pytest.raises(ValueError) as exc:
        _valide(n_batch_EFF=0)
    assert "convergee" in str(exc.value), str(exc.value)


# --------------------------------------------------------------------------- #
# 2. CE QUI DOIT RESTER ACCEPTE                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nom,valeur", [
    ("n_grid_hf", 2),          # le run de fumee du Moulin Blanc
    ("eps_taylor", 0.0),       # les cinq etudes
    ("theta_min_krg", 0.0),    # le Moulin Blanc
    ("n_max_EFF_points", 0),   # « aucun enrichissement » est un choix
    ("geo_min_approx", 0),
    ("n_batch_EFF", 1),
])
def test_une_valeur_legitime_reste_acceptee(nom, valeur):
    _valide(**{nom: valeur})


@pytest.mark.parametrize("chemin", sorted(glob.glob(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "studies", "*.toml"))))
def test_aucune_etude_du_depot_n_est_refusee(chemin):
    """La regle a ete tiree de ces fichiers : elle ne doit pas les mordre."""
    schema.charger(chemin)


def test_les_cinq_etudes_sont_bien_toutes_la():
    """Un balayage qui ne trouve rien approuve tout."""
    fichiers = glob.glob(os.path.join(_REPO, "studies", "*.toml"))
    assert len(fichiers) >= 5, fichiers


# --------------------------------------------------------------------------- #
# 3. LA REGLE : aucune taille sans borne                                       #
# --------------------------------------------------------------------------- #
#: Champs numeriques dont l'absence de borne est assumee, avec la raison.
SANS_BORNE = {
    "u1_min": "borne de trace, signee ; l'ordre est verifie ailleurs",
    "u1_max": "idem", "u2_min": "idem", "u2_max": "idem",
    "eff_bound_min": "borne de domaine, signee ; l'ordre est verifie ailleurs",
    "eff_bound_max": "idem",
    "max_degree": "compare a max_of_maxdegree",
    "max_of_maxdegree": "compare a max_degree",
    "n0": "verifie separement (>= 1)",
    "n_workers_DOE": "verifie separement (>= 1)",
    "q": "verifie separement (dans ]0, 1])",
    "tol_EFF": "verifie separement (> 0)", "tol_BB": "idem",
    "tol_BS": "idem", "tol_FORM": "idem", "cov_IS": "idem",
    "tol_all_modes": "idem",
    "tol_warmstart": "tolerance de relance ; zero relance toujours, ce qui est "
                     "un choix legitime",
    "seuil_pce": "sans effet (cf. SANS_EFFET)",
    "reduc_PLS": "sans effet (cf. SANS_EFFET)",
    "max_size": "None = suit global_size ; borne avec lui",
}


def test_aucun_reglage_numerique_n_echappe_a_une_borne():
    """La regle qui a trouve les onze tailles.

    Tout champ numerique doit etre borne quelque part, ou figurer dans
    `SANS_BORNE` avec sa raison.
    """
    import io
    src = io.open(os.path.join(_REPO, "_config", "schema.py"),
                  encoding="utf-8").read()
    cfg = schema.Configuration(modelname="x")
    numeriques = {nom for nom, v in cfg.en_dict().items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
    oublies = sorted(n for n in numeriques
                     if n not in SANS_BORNE and '"%s"' % n not in src)
    assert not oublies, (
        "reglage(s) numerique(s) qu'aucune borne ne garde : %s.\n"
        "Zero et negatif y sont acceptes -- et la consequence n'est pas un "
        "plantage mais un run qui ne fait rien en se declarant convergé."
        % ", ".join(oublies))
