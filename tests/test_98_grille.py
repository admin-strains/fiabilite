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
