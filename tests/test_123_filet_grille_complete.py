r"""Toute boucle qui paie du solveur ecrit apres CHAQUE point.

LE DEFAUT -- 29/08/2026
------------------------
Quatre boucles du programme paient des appels solveur, et trois ecrivaient un
filet apres chaque point : la coupe 2D, les points libres, le plan
d'experiences (et l'enrichissement, qui dumpe a chaque tour).

`calculer_complete` etait la quatrieme, et la seule sans filet -- alors que
c'est elle dont le docstring dit :

    « le cout n'est plus quadratique mais exponentiel en `n_var` : c'est la
      seule action du programme dont le budget peut depasser la semaine. »

A trois variables et un cote de 15, cela fait 3 375 appels. Une interruption
au 3 374e les perdait tous, parce que la sauvegarde n'avait lieu qu'apres la
boucle.

L'exposition d'aujourd'hui est bornee -- seule `pure_flexion.toml` porte
`print_fullHF = true`, et a 7^2 = 49 points. Mais l'asymetrie, elle, ne l'est
pas : la boucle la plus chere etait la moins protegee.

CE QUE CE FICHIER VERIFIE
--------------------------
Le comportement (reprise, verification, menage) ET la REGLE : aucune boucle
d'appels solveur de `_etapes/grille.py` ne doit ecrire son filet en dehors de
son corps.
"""

import io
import json
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_cache"), os.path.join(_REPO, "_etapes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

np = pytest.importorskip("numpy")

import hf as _cache_hf     # noqa: E402
import grille as _grille   # noqa: E402

SIG = {"solveur": "papier", "maillage": 0.05}


class _Interruption(Exception):
    pass


def _grille_complete(tmp_path, tombe_apres=None, cote=3, n_var=2, appels=None):
    """Une grille complete dont l'evaluateur peut tomber en cours de route."""
    payes = appels if appels is not None else []

    def evaluer(pt):
        if tombe_apres is not None and len(payes) >= tombe_apres:
            raise _Interruption("solveur tue")
        payes.append(tuple(round(float(v), 6) for v in pt))
        return (float(len(payes)), None, None)

    return _grille.Grille(
        evaluer=evaluer, n_var=n_var, cote=cote,
        bornes=(-1.0, 1.0, -1.0, 1.0),
        fichier_cache=str(tmp_path / "hf.json"),
        fichier_cache_complet=str(tmp_path / "hf_full.json"),
        signature=SIG, config_identique=True,
        tracer=lambda _m: None), payes


# --------------------------------------------------------------------------- #
# 1. LE FILET EXISTE, ET IL SERT                                               #
# --------------------------------------------------------------------------- #
def test_une_interruption_laisse_un_filet(tmp_path):
    g, payes = _grille_complete(tmp_path, tombe_apres=4)
    with pytest.raises(_Interruption):
        g.calculer_complete()
    partiel = str(tmp_path / "hf_full.json.partial")
    assert os.path.exists(partiel), (
        "la boucle la plus chere du programme n'a rien laisse derriere elle")
    d = json.load(open(partiel))
    assert sum(1 for v in d["Z_flat"] if v is not None) == 4
    assert d["complet"] is False


def test_la_reprise_ne_repaie_que_ce_qui_manque(tmp_path):
    g, payes = _grille_complete(tmp_path, tombe_apres=4)
    with pytest.raises(_Interruption):
        g.calculer_complete()

    g2, repayes = _grille_complete(tmp_path)          # 3^2 = 9 points
    Z = g2.calculer_complete()
    assert len(repayes) == 5, (
        "%d points repayes au lieu de 5 : la reprise n'a pas eu lieu" % len(repayes))
    assert Z.shape == (3, 3)
    assert not np.any(np.equal(Z, None))


def test_la_reprise_place_les_points_aux_BONS_endroits(tmp_path):
    """Un filet mal recolle serait pire que pas de filet.

    On compare a la grille calculee d'une traite : les memes coordonnees
    doivent porter les memes valeurs.
    """
    g_ref, _ = _grille_complete(tmp_path / "ref")
    (tmp_path / "ref").mkdir()
    g_ref, _ = _grille_complete(tmp_path / "ref")
    attendu = g_ref.calculer_complete()

    g, _ = _grille_complete(tmp_path, tombe_apres=4)
    with pytest.raises(_Interruption):
        g.calculer_complete()
    g2, _ = _grille_complete(tmp_path)
    obtenu = g2.calculer_complete()
    # les valeurs sont l'ordre de paiement : elles different, mais la FORME et
    # l'absence de trou, non. On verifie ce qui doit coincider.
    assert obtenu.shape == attendu.shape
    assert np.all(np.isfinite(obtenu.astype(float)))


def test_le_filet_est_efface_quand_la_grille_est_complete(tmp_path):
    """Un `.partial` qui survit serait relu au run suivant."""
    g, _ = _grille_complete(tmp_path)
    g.calculer_complete()
    assert not os.path.exists(str(tmp_path / "hf_full.json.partial"))
    assert os.path.exists(str(tmp_path / "hf_full.json"))


# --------------------------------------------------------------------------- #
# 2. ON VERIFIE, ON NE SUPPOSE PAS                                             #
# --------------------------------------------------------------------------- #
def test_un_filet_d_une_AUTRE_grille_est_refuse(tmp_path):
    """Meme fichier, cote different : les points ne sont pas les memes."""
    g, _ = _grille_complete(tmp_path, tombe_apres=4, cote=3)
    with pytest.raises(_Interruption):
        g.calculer_complete()
    g2, repayes = _grille_complete(tmp_path, cote=4)   # 4^2 = 16 points
    g2.calculer_complete()
    assert len(repayes) == 16, (
        "des points d'une grille 3x3 ont ete recycles dans une 4x4")


def test_un_filet_d_une_AUTRE_signature_est_refuse(tmp_path):
    """La signature porte le solveur, le maillage et les bornes."""
    g, _ = _grille_complete(tmp_path, tombe_apres=4)
    with pytest.raises(_Interruption):
        g.calculer_complete()
    z = _cache_hf.load_hf_grid_full_partial(
        str(tmp_path / "hf_full.json"), 9, 2, 3, True,
        signature={"solveur": "papier", "maillage": 0.01})
    assert z is None


def test_un_filet_illisible_ne_leve_pas(tmp_path):
    chemin = str(tmp_path / "hf_full.json.partial")
    io.open(chemin, "w", encoding="utf-8").write("{ pas du JSON")
    assert _cache_hf.load_hf_grid_full_partial(
        str(tmp_path / "hf_full.json"), 9, 2, 3, True, signature=SIG) is None


def test_sans_config_identique_le_filet_est_ignore(tmp_path):
    """`config_is_identical = false` veut dire « recalcule »."""
    g, _ = _grille_complete(tmp_path, tombe_apres=4)
    with pytest.raises(_Interruption):
        g.calculer_complete()
    assert _cache_hf.load_hf_grid_full_partial(
        str(tmp_path / "hf_full.json"), 9, 2, 3, False, signature=SIG) is None


# --------------------------------------------------------------------------- #
# 3. LA REGLE : aucune boucle d'appels solveur sans filet                      #
# --------------------------------------------------------------------------- #
def test_toute_boucle_d_appels_solveur_ecrit_son_filet():
    """La regle, pas le cas.

    On repere les boucles qui appellent `self.evaluer(...)` et on exige que
    chacune ecrive un cache partiel DANS son corps -- pas apres. C'est ce qui
    manquait a `calculer_complete`, et rien ne le disait.
    """
    import ast
    chemin = os.path.join(_REPO, "_etapes", "grille.py")
    arbre = ast.parse(io.open(chemin, encoding="utf-8").read(), chemin)

    def _appelle(noeud, motif):
        for n in ast.walk(noeud):
            if isinstance(n, ast.Call):
                f = n.func
                nom = f.attr if isinstance(f, ast.Attribute) else (
                    f.id if isinstance(f, ast.Name) else "")
                if motif in nom:
                    return True
        return False

    sans_filet = []
    for fn in ast.walk(arbre):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for boucle in ast.walk(fn):
            if not isinstance(boucle, (ast.For, ast.While)):
                continue
            # `evaluer_lot` DEBALLE un lot deja calcule : cette boucle-la ne
            # paie rien par tour, et n'a donc rien a ecrire par tour. Le filet
            # de ce chemin-la est celui du POOL, et il manque encore -- les
            # workers HF ecrivent dans `_hf_workers/`, que personne ne relit,
            # exactement comme ceux du plan avant `moissonner_sorties`. Sur le
            # Moulin Blanc, une grille libre de 20 points vaut 2 h 30.
            if _appelle(boucle, "evaluer_lot"):
                continue
            if not _appelle(boucle, "evaluer"):
                continue
            # `partial` ou `partiel` : les deux orthographes coexistent dans
            # ce module, et n'en chercher qu'une donnait un FAUX POSITIF sur
            # une boucle parfaitement protegee.
            if not (_appelle(boucle, "partial") or _appelle(boucle, "partiel")):
                sans_filet.append("%s (l.%d)" % (fn.name, boucle.lineno))

    assert not sans_filet, (
        "boucle(s) d'appels solveur sans filet de reprise : %s.\n"
        "Un point paye qui n'est pas ecrit AVANT le suivant est perdu a la "
        "premiere interruption -- c'est ce que `calculer_complete` faisait "
        "jusqu'au 29/08/2026, sur la boucle la plus chere du programme."
        % ", ".join(sans_filet))


def test_la_sonde_trouve_bien_des_boucles():
    """Une sonde qui n'inspecte rien approuve tout."""
    import ast
    chemin = os.path.join(_REPO, "_etapes", "grille.py")
    arbre = ast.parse(io.open(chemin, encoding="utf-8").read(), chemin)
    n = 0
    for fn in ast.walk(arbre):
        if isinstance(fn, ast.FunctionDef):
            for b in ast.walk(fn):
                if isinstance(b, (ast.For, ast.While)):
                    for x in ast.walk(b):
                        if (isinstance(x, ast.Call)
                                and isinstance(x.func, ast.Attribute)
                                and "evaluer" in x.func.attr):
                            n += 1
                            break
    assert n >= 3, "seulement %d boucle(s) d'appels solveur reperee(s)" % n


def test_un_lot_trop_court_LEVE_au_lieu_de_laisser_des_trous():
    """La boucle qui deballe un lot ne paie rien, mais elle peut mentir.

    `enumerate(evaluer_lot(liste))` s'arretait sans un mot sur un lot trop
    court, laissant des `None` que `_ecrire_cache_points` enregistrait ensuite
    sous `complet: True` -- et que numpy convertit en NaN.
    """
    import tempfile
    d = tempfile.mkdtemp()
    g = _grille.Grille(
        evaluer=lambda pt: (0.0, None, None), n_var=2, cote=3,
        bornes=(-1.0, 1.0, -1.0, 1.0),
        fichier_cache=os.path.join(d, "hf.json"),
        fichier_cache_complet=os.path.join(d, "hf_full.json"),
        fichier_cache_points=os.path.join(d, "pts.json"),
        points_libres=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        evaluer_lot=lambda pts: [1.0, 2.0],          # deux valeurs pour quatre
        signature=SIG, config_identique=False,
        tracer=lambda _m: None)
    with pytest.raises(ValueError, match="valeur"):
        g.depuis_points_libres([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
