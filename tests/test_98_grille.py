r"""La grille haute fidelite : reprise apres interruption, et cache qui ne ment pas.

POURQUOI CES DEUX PROPRIETES-LA
--------------------------------
Une grille 15x15 coute 225 appels solveur, soit 29 heures sur le Moulin Blanc.
Deux choses seulement rendent ce cout supportable, et ni l'une ni l'autre
n'etait couverte par un test avant le 27/08/2026 :

1. **La reprise.** Chaque point est ecrit dans un `.partial` des qu'il est
   calcule. Un run tue a la 200e evaluation doit reprendre a la 201e. Si cette
   propriete casse, personne ne s'en apercoit avant la prochaine interruption
   -- c'est-a-dire au pire moment.

2. **Le cache qui ne ment pas.** Le 26/08/2026, apres avoir borne le domaine a
   +/- 6 a cause d'un crash du solveur, il restait sur le disque un
   `hf_grid_cache.json.partial` contenant `g = -0,8556` calcule a
   `u = [-7,5, -7,5]` sous cuDSS. Sans verification de signature, cette valeur
   aurait ete relue comme etant celle du nouveau domaine, sous MUMPS. Le
   fichier a ete mis de cote a la main ; le garde-fou, lui, doit etre teste.

Tout se teste ici sans licence : le seul lien avec le solveur est un
`evaluer(u)` que ces tests remplacent par un compteur.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _d in (os.path.join(_REPO, "_cache"), os.path.join(_REPO, "_etapes")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

np = pytest.importorskip("numpy")

import grille as _grille                              # noqa: E402

COUPE = (0, 1, {})


class _Compteur:
    """Faux etat limite : `g = u1 - u2`, et il compte ce qu'on lui demande.

    `mourir_apres` simule une interruption -- une machine eteinte, un solveur
    qui termine le processus, un Ctrl-C.
    """

    def __init__(self, mourir_apres=None):
        self.points = []
        self.mourir_apres = mourir_apres

    def __call__(self, u):
        if self.mourir_apres is not None and len(self.points) >= self.mourir_apres:
            raise KeyboardInterrupt("interruption simulee")
        self.points.append(list(np.array(u, float)))
        return (float(u[0]) - float(u[1]), None, None)


def _grille_de_test(tmp_path, evaluer, cote=4, signature="sig-A", bornes=None):
    return _grille.Grille(
        evaluer=evaluer, n_var=2, cote=cote,
        bornes=bornes or (-3.0, 3.0, -3.0, 3.0),
        fichier_cache=str(tmp_path / "hf_grid_cache.json"),
        fichier_cache_complet=str(tmp_path / "hf_grid_full_cache.json"),
        signature=signature, config_identique=True,
        tracer=lambda _m: None)


def _points(g):
    UX, UY = g.maillage_2d()
    return np.column_stack([UX.ravel(), UY.ravel()])


# --------------------------------------------------------------------- #
# le cout annonce est le cout paye
# --------------------------------------------------------------------- #
def test_une_grille_froide_coute_exactement_son_carre(tmp_path):
    ev = _Compteur()
    g = _grille_de_test(tmp_path, ev, cote=4)
    Z, _ = g.calculer_2d(_points(g), coupe=COUPE)
    assert len(ev.points) == 16, "4x4 doit couter 16 appels, pas %d" % len(ev.points)
    assert Z.shape == (4, 4)
    # g = u1 - u2, et le maillage varie u1 le long des colonnes
    UX, UY = g.maillage_2d()
    assert Z == pytest.approx(UX - UY)


def test_une_grille_deja_calculee_ne_coute_rien(tmp_path):
    ev = _Compteur()
    g = _grille_de_test(tmp_path, ev, cote=4)
    g.calculer_2d(_points(g), coupe=COUPE)
    n_premier = len(ev.points)

    ev2 = _Compteur()
    g2 = _grille_de_test(tmp_path, ev2, cote=4)
    Z2, _ = g2.calculer_2d(_points(g2), coupe=COUPE)
    assert ev2.points == [], "le cache complet doit couter ZERO appel"
    assert n_premier == 16
    assert Z2.shape == (4, 4)


# --------------------------------------------------------------------- #
# LA REPRISE -- la propriete qui vaut des heures
# --------------------------------------------------------------------- #
def test_une_interruption_ne_perd_pas_les_points_deja_payes(tmp_path):
    ev = _Compteur(mourir_apres=7)
    g = _grille_de_test(tmp_path, ev, cote=4)
    with pytest.raises(KeyboardInterrupt):
        g.calculer_2d(_points(g), coupe=COUPE)
    assert len(ev.points) == 7

    ev2 = _Compteur()
    g2 = _grille_de_test(tmp_path, ev2, cote=4)
    Z, _ = g2.calculer_2d(_points(g2), coupe=COUPE)
    assert len(ev2.points) == 9, (
        "la reprise doit payer les 9 points manquants, pas %d. Sur le Moulin "
        "Blanc, tout recalculer coute 29 h." % len(ev2.points))
    UX, UY = g2.maillage_2d()
    assert Z == pytest.approx(UX - UY), (
        "les points repris et les points recalcules doivent se recoller dans "
        "le BON ORDRE")


def test_la_reprise_efface_le_partiel_une_fois_la_grille_finie(tmp_path):
    ev = _Compteur()
    g = _grille_de_test(tmp_path, ev, cote=3)
    g.calculer_2d(_points(g), coupe=COUPE)
    assert not os.path.exists(str(tmp_path / "hf_grid_cache.json.partial")), (
        "un `.partial` survivant serait relu au prochain run comme une "
        "reprise, alors que la grille est complete")


# --------------------------------------------------------------------- #
# LE CACHE NE MENT PAS -- le defaut du 26/08/2026
# --------------------------------------------------------------------- #
def test_un_cache_d_une_autre_configuration_est_refuse(tmp_path):
    """C'est exactement le fichier qui trainait le 26/08 : des valeurs
    calculees sur un autre domaine, sous un autre solveur lineaire."""
    ev = _Compteur()
    g = _grille_de_test(tmp_path, ev, cote=4, signature="cuDSS|+-7.5")
    g.calculer_2d(_points(g), coupe=COUPE)

    ev2 = _Compteur()
    g2 = _grille_de_test(tmp_path, ev2, cote=4, signature="MUMPS|+-6.0")
    g2.calculer_2d(_points(g2), coupe=COUPE)
    assert len(ev2.points) == 16, (
        "un cache d'une AUTRE configuration a ete servi tel quel : c'est le "
        "defaut du 26/08/2026, avec des valeurs a +/- 7,5 relues comme "
        "etant celles de +/- 6")


def test_un_partiel_d_une_autre_configuration_est_refuse(tmp_path):
    ev = _Compteur(mourir_apres=5)
    g = _grille_de_test(tmp_path, ev, cote=4, signature="cuDSS|+-7.5")
    with pytest.raises(KeyboardInterrupt):
        g.calculer_2d(_points(g), coupe=COUPE)

    ev2 = _Compteur()
    g2 = _grille_de_test(tmp_path, ev2, cote=4, signature="MUMPS|+-6.0")
    g2.calculer_2d(_points(g2), coupe=COUPE)
    assert len(ev2.points) == 16, (
        "le cache PARTIEL est le plus dangereux : c'est lui qui trainait le "
        "26/08, et il ne portait aucune signature")


def test_un_cote_different_ne_reutilise_pas_la_grille(tmp_path):
    ev = _Compteur()
    g = _grille_de_test(tmp_path, ev, cote=3)
    g.calculer_2d(_points(g), coupe=COUPE)

    ev2 = _Compteur()
    g2 = _grille_de_test(tmp_path, ev2, cote=5)
    g2.calculer_2d(_points(g2), coupe=COUPE)
    assert len(ev2.points) == 25, (
        "une grille 3x3 a ete relue pour une demande 5x5")


# --------------------------------------------------------------------- #
# la grille complete, et les coupes gratuites qu'elle paie
# --------------------------------------------------------------------- #
def test_une_coupe_dans_la_grille_complete_ne_coute_rien(tmp_path):
    """C'est tout l'interet de payer `cote^n_var` une fois."""
    ev = _Compteur()
    g = _grille_de_test(tmp_path, ev, cote=4)
    g.calculer_complete()
    assert len(ev.points) == 16          # 4^2
    n_apres_grille = len(ev.points)
    Z = g.coupe_depuis_complete(COUPE)
    assert len(ev.points) == n_apres_grille, "une coupe interpolee est gratuite"
    UX, UY = g.maillage_2d()
    assert Z == pytest.approx(UX - UY, abs=1e-9)


def test_une_coupe_sans_grille_complete_refuse_au_lieu_de_calculer(tmp_path):
    """Elle pourrait « rendre service » en lancant la grille complete. Ce
    serait `cote^n_var` appels declenches par une fonction qui a l'air d'un
    accesseur -- exactement le defaut qu'on vient de corriger ailleurs."""
    g = _grille_de_test(tmp_path, _Compteur(), cote=4)
    with pytest.raises(ValueError) as err:
        g.coupe_depuis_complete(COUPE)
    assert "calculer_complete" in str(err.value)


def test_la_grille_complete_relue_ne_coute_rien(tmp_path):
    ev = _Compteur()
    _grille_de_test(tmp_path, ev, cote=4).calculer_complete()
    ev2 = _Compteur()
    g2 = _grille_de_test(tmp_path, ev2, cote=4)
    g2.calculer_complete()
    assert ev2.points == []
    assert g2.complete is not None and g2.axes_complets is not None, (
        "la grille relue doit etre utilisable pour des coupes, sinon le cache "
        "ne sert a rien")


# --------------------------------------------------------------------- #
# l'hypothese sur les bornes, dite tout haut
# --------------------------------------------------------------------- #
def test_des_bornes_differentes_entre_u1_et_u2_sont_signalees(tmp_path):
    """La grille complete construit TOUS ses axes sur les bornes de u1. Les
    deux coincident dans toutes les etudes actuelles ; le jour ou elles
    differeront, une coupe interpolee sortira du domaine."""
    g = _grille_de_test(tmp_path, _Compteur(), cote=3,
                        bornes=(-3.0, 3.0, -6.0, 6.0))
    message = g.verifier_bornes()
    assert message and "u2" in message and "domaine" in message


def test_des_bornes_identiques_ne_declenchent_aucune_alerte(tmp_path):
    g = _grille_de_test(tmp_path, _Compteur(), cote=3,
                        bornes=(-3.0, 3.0, -3.0, 3.0))
    assert g.verifier_bornes() is None


# --------------------------------------------------------------------- #
# le module reste sans licence
# --------------------------------------------------------------------- #
def test_le_module_ne_connait_ni_le_modele_ni_la_licence():
    src = open(os.path.join(_REPO, "_etapes", "grille.py"),
               encoding="utf-8").read()
    for interdit in ("import schema", "import fabrique", "digital_structure",
                     "STRAINS", "CetSOLV"):
        assert interdit not in src, (
            "grille.py mentionne %r : l'evaluateur lui est PASSE." % interdit)


# --------------------------------------------------------------------- #
# une grille de points CHOISIS plutot qu'un quadrillage
# --------------------------------------------------------------------- #
# Un quadrillage regulier depense la moitie de son budget loin de l'etat
# limite. Quand on sait deja ou il passe, on place les points a la main et on
# interpole entre eux -- c'est `hf_custom_points`.
#
# Ces 102 lignes etaient recopiees a l'identique dans les deux etudes et
# n'avaient jamais ete couvertes.
def _grille_points(tmp_path, evaluer, signature="sig-A", evaluer_lot=None):
    return _grille.Grille(
        evaluer=evaluer, n_var=2, cote=4, bornes=(-3.0, 3.0, -3.0, 3.0),
        fichier_cache=str(tmp_path / "hf_grid_cache.json"),
        fichier_cache_complet=str(tmp_path / "hf_grid_full_cache.json"),
        fichier_cache_points=str(tmp_path / "hf_custom_cache.json"),
        evaluer_lot=evaluer_lot,
        signature=signature, config_identique=True,
        tracer=lambda _m: None)


CHOISIS = [[-1.0, -1.0], [0.0, 0.5], [1.0, -0.5], [2.0, 1.0], [-2.0, 2.0]]


def test_sans_points_choisis_rien_n_est_calcule(tmp_path):
    ev = _Compteur()
    g = _grille_points(tmp_path, ev)
    assert g.depuis_points_libres(None) == (None, None, None)
    assert ev.points == []


def test_chaque_point_choisi_coute_un_appel(tmp_path):
    ev = _Compteur()
    g = _grille_points(tmp_path, ev)
    Z, UX, UY = g.depuis_points_libres(CHOISIS)
    assert len(ev.points) == len(CHOISIS)
    assert Z.shape == UX.shape == UY.shape


def test_le_resultat_est_garde_en_memoire(tmp_path):
    """Plusieurs figures redemandent le meme fond ; relire le JSON a chaque
    fois etait le motif du memo d'origine."""
    ev = _Compteur()
    g = _grille_points(tmp_path, ev)
    a = g.depuis_points_libres(CHOISIS)
    b = g.depuis_points_libres(CHOISIS)
    assert a is b
    assert len(ev.points) == len(CHOISIS)


def test_un_second_objet_relit_le_cache_sans_rien_payer(tmp_path):
    _grille_points(tmp_path, _Compteur()).depuis_points_libres(CHOISIS)
    ev2 = _Compteur()
    Z, _UX, _UY = _grille_points(tmp_path, ev2).depuis_points_libres(CHOISIS)
    assert ev2.points == [], "le cache complet doit couter ZERO appel"
    assert Z is not None


def test_le_cache_des_points_choisis_porte_sa_signature(tmp_path):
    """Il ne la portait pas : une grille calculee sous un autre solveur, un
    autre maillage ou d'autres bornes etait relue telle quelle."""
    _grille_points(tmp_path, _Compteur(), signature="cuDSS").depuis_points_libres(CHOISIS)
    ev2 = _Compteur()
    _grille_points(tmp_path, ev2, signature="MUMPS").depuis_points_libres(CHOISIS)
    assert len(ev2.points) == len(CHOISIS), (
        "un cache d'une AUTRE configuration a ete servi tel quel")


def test_une_interruption_ne_perd_pas_les_points_choisis_deja_payes(tmp_path):
    ev = _Compteur(mourir_apres=3)
    g = _grille_points(tmp_path, ev)
    with pytest.raises(KeyboardInterrupt):
        g.depuis_points_libres(CHOISIS)
    assert len(ev.points) == 3

    ev2 = _Compteur()
    g2 = _grille_points(tmp_path, ev2)
    Z, _UX, _UY = g2.depuis_points_libres(CHOISIS)
    assert len(ev2.points) == len(CHOISIS) - 3, (
        "la reprise doit payer les %d points manquants, pas %d"
        % (len(CHOISIS) - 3, len(ev2.points)))
    assert Z is not None


def test_un_partiel_d_une_autre_configuration_ne_sert_pas(tmp_path):
    ev = _Compteur(mourir_apres=2)
    with pytest.raises(KeyboardInterrupt):
        _grille_points(tmp_path, ev, signature="cuDSS").depuis_points_libres(CHOISIS)
    ev2 = _Compteur()
    _grille_points(tmp_path, ev2, signature="MUMPS").depuis_points_libres(CHOISIS)
    assert len(ev2.points) == len(CHOISIS)


def test_l_evaluation_en_lot_est_utilisee_quand_elle_existe(tmp_path):
    """Le chemin parallele : un seul appel groupe au lieu de N appels."""
    lots = []

    def en_lot(points):
        lots.append(list(points))
        return [float(len(p)) for p in points]

    ev = _Compteur()
    g = _grille_points(tmp_path, ev, evaluer_lot=en_lot)
    g.depuis_points_libres(CHOISIS)
    assert ev.points == [], "l'evaluation point par point a ete utilisee quand meme"
    assert len(lots) == 1 and len(lots[0]) == len(CHOISIS)


def test_la_surface_interpolee_encadre_les_points(tmp_path):
    """La grille d'interpolation est cadree sur l'enveloppe des points,
    elargie d'une marge -- sinon les points du bord tombent hors domaine et
    `griddata` rend des NaN."""
    g = _grille_points(tmp_path, _Compteur())
    _Z, UX, UY = g.depuis_points_libres(CHOISIS, marge=0.1, n_interp=20)
    pts = np.array(CHOISIS)
    assert UX.min() == pytest.approx(pts[:, 0].min() - 0.1)
    assert UX.max() == pytest.approx(pts[:, 0].max() + 0.1)
    assert UY.min() == pytest.approx(pts[:, 1].min() - 0.1)
    assert UY.max() == pytest.approx(pts[:, 1].max() + 0.1)
    assert UX.shape == (20, 20)


def test_la_grille_d_interpolation_n_est_construite_qu_une_fois():
    """L'original la construisait DEUX FOIS -- une fois dans la branche
    « cache complet », une fois a la fin -- avec les memes constantes
    recopiees."""
    src = open(os.path.join(_REPO, "_etapes", "grille.py"),
               encoding="utf-8").read()
    import ast as _ast
    assert src.count("griddata(") == 1, "l'interpolation est ecrite deux fois"
    appels = [n for n in _ast.walk(_ast.parse(src))
              if isinstance(n, _ast.Call)
              and isinstance(n.func, _ast.Attribute)
              and n.func.attr == "_interpoler"]
    assert len(appels) == 1, (
        "la grille d'interpolation doit etre construite en UN endroit "
        "(%d appels a `_interpoler`)" % len(appels))


# --------------------------------------------------------------------- #
# LE CACHE DES POINTS CHOISIS VERIFIE SES POINTS (28/08/2026)
#
# Meme defaut que celui ferme le 26/08 sur le cache du plan d'experiences,
# et meme remede -- il n'avait pas ete porte ici. `hf_custom_points` existe
# pour placer les points a la main la ou l'on sait que l'etat limite passe :
# les DEPLACER est l'usage normal. Le cache validait sa signature et le
# NOMBRE de points, jamais les points eux-memes, alors qu'il les stockait.
#
# Mesure faite le 28/08 sur l'etude analytique, avant correctif : six points
# remplaces par six autres, entierement differents. Journal :
#     [HF CUSTOM] cache complet charge (6 pts) -> 0 SOCP
# Les anciennes valeurs etaient alors appariees aux NOUVELLES coordonnees
# par `_interpoler` -- la surface de reference « g = 0 HF » construite de
# couples faux, en silence, sous un message qui annonce une economie.
# --------------------------------------------------------------------- #
def _grille_pts_libres(tmp_path, evaluer=None, tracer=None):
    appels = []

    def _defaut(u):
        appels.append(tuple(u))
        return float(u[0] + u[1]), [0.0, 0.0], [0.0, 0.0]

    g = _grille.Grille(
        evaluer=evaluer or _defaut, n_var=2, cote=4,
        bornes=(-1.0, 1.0, -1.0, 1.0),
        fichier_cache=str(tmp_path / "c.json"),
        fichier_cache_complet=str(tmp_path / "f.json"),
        fichier_cache_points=str(tmp_path / "pts.json"),
        config_identique=True, signature={"solveur": "analytique"},
        tracer=tracer or (lambda _m: None))
    return g, appels


#: QUATRE points NON ALIGNES : trois points colineaires ne se
#: triangulent pas, et `_interpoler` s'appuie sur Qhull.
PTS_A = [[-3.0, -3.0], [-2.0, -3.5], [-1.0, -4.0], [-2.5, -1.0]]
PTS_B = [[3.0, 3.0], [2.0, 3.5], [1.0, 4.0], [2.5, 1.0]]


def test_des_points_INCHANGES_reutilisent_le_cache(tmp_path):
    """Une garde qui jette un cache legitime ne vaut rien : trois points, ce
    sont trois appels solveur -- 23 minutes sur le Moulin Blanc."""
    g, appels = _grille_pts_libres(tmp_path)
    g.depuis_points_libres(PTS_A)
    n = len(appels)
    g2, appels2 = _grille_pts_libres(tmp_path)
    g2.depuis_points_libres(PTS_A)
    assert n == 4
    assert appels2 == [], "le cache legitime n'a pas ete reutilise"


def test_des_points_DEPLACES_sont_recalcules(tmp_path):
    """LE defaut. Sans ce controle, les valeurs de PTS_A etaient rendues pour
    PTS_B -- meme nombre, meme signature, coordonnees sans rapport."""
    g, _ = _grille_pts_libres(tmp_path)
    g.depuis_points_libres(PTS_A)
    g2, appels2 = _grille_pts_libres(tmp_path)
    g2.depuis_points_libres(PTS_B)
    assert [list(u) for u in appels2] == PTS_B, (
        "les points deplaces n'ont pas ete recalcules : la surface serait "
        "faite des anciennes valeurs aux nouvelles coordonnees")


def test_le_refus_est_ANNONCE_avec_l_ecart(tmp_path):
    """Un cache refuse en silence se lit comme un cache absent. Le journal
    doit dire ce qui a change."""
    g, _ = _grille_pts_libres(tmp_path)
    g.depuis_points_libres(PTS_A)
    j = []
    g2, _ = _grille_pts_libres(tmp_path, tracer=j.append)
    g2.depuis_points_libres(PTS_B)
    ligne = [m for m in j if "les points ont CHANGE" in m]
    assert ligne, j
    assert "ecart max" in ligne[0] and "8.000e+00" in ligne[0], (
        "le journal doit porter l ecart REEL : max sur les quatre paires, "
        "soit |-1 - 1| et |-4 - 4| -> 8.0")


def test_un_nombre_de_points_different_est_deja_refuse(tmp_path):
    """Ce controle-la existait ; il ne doit pas disparaitre."""
    g, _ = _grille_pts_libres(tmp_path)
    g.depuis_points_libres(PTS_A)
    g2, appels2 = _grille_pts_libres(tmp_path)
    g2.depuis_points_libres(PTS_A + [[0.0, 0.0]])
    assert len(appels2) == 5


def test_un_cache_sans_coordonnees_est_refuse(tmp_path):
    """Les caches ecrits AVANT ce controle n'ont pas de `pts`. On ne peut pas
    les valider, donc on ne les reprend pas -- en le disant."""
    import json as _json
    fichier = str(tmp_path / "pts.json")
    _json.dump({"n_total": 4, "complet": True, "g_vals": [1.0, 2.0, 3.0, 4.0],
                "signature": {"solveur": "analytique"}},
               open(fichier, "w"))
    j = []
    g, appels = _grille_pts_libres(tmp_path, tracer=j.append)
    g.depuis_points_libres(PTS_A)
    assert len(appels) == 4
    assert any("sans coordonnees" in m for m in j), j


def test_le_cache_PARTIEL_verifie_aussi_ses_points(tmp_path):
    """Il etait plus expose encore que le complet : il reprenait des valeurs
    PAR INDICE, et n'ecrivait meme pas ses coordonnees."""
    import json as _json
    partiel = str(tmp_path / "pts.json.partial")
    _json.dump({"n_total": 4, "g_vals": [1.0, None, None, None],
                "pts": PTS_A, "signature": {"solveur": "analytique"}},
               open(partiel, "w"))
    g, appels = _grille_pts_libres(tmp_path)
    g.depuis_points_libres(PTS_A)
    assert len(appels) == 3, "le point deja calcule devait etre repris"

    _json.dump({"n_total": 4, "g_vals": [1.0, None, None, None],
                "pts": PTS_A, "signature": {"solveur": "analytique"}},
               open(partiel, "w"))
    g2, appels2 = _grille_pts_libres(tmp_path)
    g2.depuis_points_libres(PTS_B)
    assert len(appels2) == 4, (
        "le cache partiel a rendu une valeur de PTS_A pour un point de PTS_B")


def test_le_cache_partiel_ecrit_ses_coordonnees(tmp_path):
    """Sans elles, la verification precedente n'aurait rien a comparer."""
    import json as _json

    def evaluer_qui_meurt(u):
        if tuple(u) == tuple(PTS_A[2]):
            raise RuntimeError("interruption simulee")
        return float(u[0]), [0.0, 0.0], [0.0, 0.0]

    g, _ = _grille_pts_libres(tmp_path, evaluer=evaluer_qui_meurt)
    try:
        g.depuis_points_libres(PTS_A)
    except RuntimeError:
        pass
    d = _json.load(open(str(tmp_path / "pts.json.partial")))
    assert d["pts"] == PTS_A
    assert d["g_vals"][0] is not None and d["g_vals"][2] is None


# --------------------------------------------------------------------- #
# UNE COUPE, C'EST TROIS CHOSES -- pas deux (28/08/2026)
#
# Le cache ne comparait que les deux AXES. Les valeurs auxquelles les autres
# variables sont figees etaient ECRITES dans le fichier et jamais relues :
# `(0, 2, {1: 7.5})` et `(0, 2, {1: -3.0})` sont deux surfaces sans rapport,
# et le cache servait l'une pour l'autre en annoncant « coupe OK ».
#
# Sans effet a deux variables, ou le dictionnaire est toujours vide. Faux
# au-dela -- et c'est justement au-dela que la coupe finale fige les
# variables secondaires a `u*`. Meme cause masquante que le fond trace sur
# le cadre des figures et que la coupe finale decidee trop tard : l'etude de
# reference a deux variables.
# --------------------------------------------------------------------- #
def _cache_hf():
    import sys, os as _os
    chemin = _os.path.join(_REPO, "_cache")
    if chemin not in sys.path:
        sys.path.insert(0, chemin)
    import hf
    return hf


SIG = {"solveur": "analytique"}


def test_une_autre_valeur_figee_n_est_PAS_la_meme_coupe(tmp_path):
    hf = _cache_hf()
    f = str(tmp_path / "c.json")
    hf.save_hf_cache(np.arange(9.0).reshape(3, 3), 3, f, (0, 2, {1: 7.5}),
                     signature=SIG)
    assert hf.load_hf_cache(3, f, (0, 2, {1: -3.0}), signature=SIG) is None, (
        "une surface calculee a u1 = 7,5 a ete servie pour u1 = -3,0")


def test_la_MEME_coupe_est_toujours_servie(tmp_path):
    """Une garde qui refuse un cache legitime coute `cote^2` appels
    solveur -- 29 heures sur le Moulin Blanc a 15."""
    hf = _cache_hf()
    f = str(tmp_path / "c.json")
    hf.save_hf_cache(np.arange(9.0).reshape(3, 3), 3, f, (0, 2, {1: 7.5}),
                     signature=SIG)
    Z = hf.load_hf_cache(3, f, (0, 2, {1: 7.5}), signature=SIG)
    assert Z is not None and Z.shape == (3, 3)


def test_a_deux_variables_rien_ne_change(tmp_path):
    """Le dictionnaire est vide des deux cotes : c'est pourquoi le defaut
    etait invisible dans les deux etudes du depot."""
    hf = _cache_hf()
    f = str(tmp_path / "c.json")
    hf.save_hf_cache(np.arange(16.0).reshape(4, 4), 4, f, (0, 1, {}),
                     signature=SIG)
    assert hf.load_hf_cache(4, f, (0, 1, {}), signature=SIG) is not None


def test_un_ecart_infime_sur_une_valeur_figee_ne_refuse_pas(tmp_path):
    """Les valeurs figees viennent de `u*`, donc de flottants. Une egalite
    stricte refuserait des caches legitimes pour un dernier bit."""
    hf = _cache_hf()
    f = str(tmp_path / "c.json")
    hf.save_hf_cache(np.zeros((3, 3)), 3, f, (0, 2, {1: 7.5}), signature=SIG)
    assert hf.load_hf_cache(3, f, (0, 2, {1: 7.5 + 1e-12}),
                            signature=SIG) is not None


def test_des_variables_figees_DIFFERENTES_sont_refusees(tmp_path):
    """Meme axes, mais ce n'est pas la meme variable qu'on immobilise."""
    hf = _cache_hf()
    f = str(tmp_path / "c.json")
    hf.save_hf_cache(np.zeros((3, 3)), 3, f, (0, 3, {1: 1.0}), signature=SIG)
    assert hf.load_hf_cache(3, f, (0, 3, {2: 1.0}), signature=SIG) is None


def test_un_cache_anterieur_au_controle_est_refuse_SI_ON_FIGE(tmp_path):
    """Un fichier sans valeurs figees ne peut pas etre valide quand la coupe
    demandee en a. Mais a deux variables il reste utilisable."""
    import json as _json
    hf = _cache_hf()
    f = str(tmp_path / "c.json")
    _json.dump({"Z": [[0.0]], "n_grid_hf": 1, "signature": SIG,
                "slice_def": [0, 2]}, open(f, "w"))
    assert hf.load_hf_cache(1, f, (0, 2, {1: 7.5}), signature=SIG) is None
    assert hf.load_hf_cache(1, f, (0, 2, {}), signature=SIG) is not None


def test_le_cache_PARTIEL_compare_lui_aussi_la_coupe_entiere(tmp_path):
    hf = _cache_hf()
    f = str(tmp_path / "c.json")
    hf.save_hf_cache_partial([1.0, None, None], 3, f, (0, 2, {1: 7.5}),
                             signature=SIG)
    assert hf.load_hf_cache_partial(f, (0, 2, {1: -3.0}), 3,
                                    signature=SIG) is None
    assert hf.load_hf_cache_partial(f, (0, 2, {1: 7.5}), 3,
                                    signature=SIG) is not None


# --------------------------------------------------------------------- #
# UN ECHEC DE CACHE NE DOIT PAS ETRE MUET (28/08/2026)
#
# Quatre `except Exception: pass` avalaient les echecs du cache de points.
# Les deux pires portaient l'ECRITURE : sans le fichier, la grille entiere
# est repayee a chaque run, indefiniment, sans qu'une ligne ne dise
# pourquoi. Sur le Moulin Blanc, vingt points choisis valent deux heures et
# demie. Le module voisin (`_cache/hf.py`) annonce ses echecs depuis
# toujours ; celui-ci ne le faisait pas.
# --------------------------------------------------------------------- #
def test_un_cache_de_points_illisible_est_annonce(tmp_path):
    """Un cache corrompu se lit comme un cache absent : le run recalcule.
    C'est le bon comportement, mais il doit se voir."""
    fichier = tmp_path / "pts.json"
    fichier.write_text("{ceci n'est pas du JSON", encoding="utf-8")
    j = []
    g, appels = _grille_pts_libres(tmp_path, tracer=j.append)
    g.depuis_points_libres(PTS_A)
    assert len(appels) == 4, "le run doit recalculer"
    assert any("cache complet illisible" in m for m in j), j


def test_un_partiel_illisible_est_annonce(tmp_path):
    fichier = tmp_path / "pts.json.partial"
    fichier.write_text("tronque", encoding="utf-8")
    j = []
    g, _ = _grille_pts_libres(tmp_path, tracer=j.append)
    g.depuis_points_libres(PTS_A)
    assert any("cache partiel illisible" in m for m in j), j


def test_un_echec_d_ECRITURE_du_cache_est_annonce(tmp_path):
    """Le plus couteux des quatre : sans ce message, la grille est repayee a
    chaque run et rien ne dit pourquoi."""
    j = []
    g, _ = _grille_pts_libres(tmp_path, tracer=j.append)
    # un dossier la ou le fichier devrait aller : l'ecriture echouera
    (tmp_path / "pts.json").mkdir()
    g.depuis_points_libres(PTS_A)
    assert any("sauvegarde du cache echouee" in m for m in j), j
    assert any("ne seront pas reutilises" in m for m in j), j


def test_aucun_gestionnaire_muet_dans_la_grille():
    """Le garde : un `except` sans le moindre appel avale l'information.

    Ce module tient le calcul le plus cher du programme -- 225 appels
    solveur pour une grille 15x15. Un echec qui n'ecrit rien s'y paie en
    heures.
    """
    import ast as _ast
    src = open(os.path.join(_REPO, "_etapes", "grille.py"),
               encoding="utf-8").read()
    muets = [n.lineno for n in _ast.walk(_ast.parse(src))
             if isinstance(n, _ast.ExceptHandler)
             and not any(isinstance(x, (_ast.Call, _ast.Raise))
                         for b in n.body for x in _ast.walk(b))]
    assert not muets, (
        "gestionnaire(s) d'exception muet(s) aux lignes %s : ils avalent un "
        "echec sans laisser de trace." % muets)
