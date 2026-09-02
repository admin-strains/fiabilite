r"""Ce que la boucle d'enrichissement se donne elle-meme, et ce qu'elle exige.

POURQUOI CE FICHIER
--------------------
Le 02/09/2026, cinq collaborateurs sont sortis des etudes pour etre
construits par `BoucleEFF` : le critere EFF, l'encadrement `g +/- 2 sigma`,
le choix du prochain batch, le tirage d'importance de repli, et la loi
jointe. C'etaient des DELEGUES d'une ligne, recopies dans les deux etudes,
dont la seule fonction etait de lier `n_var` et `cfg.*` a un appel de module.

Deplacer du code sans filet, c'est deplacer un defaut. `test_119` exerce
l'ORDRE DES GESTES avec des collaborateurs de papier -- il stube donc
justement ceux-la, et ne les voit pas. Ce fichier les regarde.

CE QU'IL VERIFIE
-----------------
1. chaque collaborateur derive est bien construit, et depuis les bons
   reglages -- verifie en CHANGEANT le reglage et en observant l'effet ;
2. une injection explicite l'emporte toujours (c'est ce qui garde `test_119`
   possible) ;
3. l'encadrement REFUSE d'etre fabrique sans predicteur, plutot que de
   fabriquer quelque chose de faux ;
4. les etudes ne les passent plus.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in (os.path.join(_REPO, "_reliability"), os.path.join(_REPO, "_doe"),
           os.path.join(_REPO, "_model"), os.path.join(_REPO, "_lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ot = pytest.importorskip("openturns", reason="la couche etudes n'est pas installee")

import enrichissement as _enr        # noqa: E402

ETUDES = ("pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py")


class _Cfg:
    """Les reglages que les collaborateurs derives consomment, et eux seuls."""

    def __init__(self, **kw):
        self.do_PCK = True
        self.do_GEPCK = False
        self.epsilon_factor = 2.0
        self.eff_bound_min = -6.0
        self.eff_bound_max = 6.0
        self.n_IS = 1234
        self.cov_IS = 0.05
        self.n_batch_EFF = 1
        self.n_NLopt_EFF = 30
        self.__dict__.update(kw)


def _boucle(**kw):
    """Une boucle nue : rien que ce que la construction exige."""
    reglages = dict(journal=None, historiques={}, ajuster=lambda *a, **k: None,
                    evaluer_un_point=lambda u: (0.0, None))
    cfg = kw.pop("cfg", _Cfg())
    n_var = kw.pop("n_var", 2)
    reglages.update(kw)
    return _enr.BoucleEFF(cfg, n_var, **reglages)


# --------------------------------------------------------------------------- #
# 1. LE DOMAINE, DERIVE DE `eff_bound_*` ET DE `n_var`                         #
# --------------------------------------------------------------------------- #
def test_le_domaine_repete_les_bornes_sur_chaque_variable():
    """Les etudes l'ecrivaient en deux lignes chacune."""
    b = _boucle(cfg=_Cfg(eff_bound_min=-7.5, eff_bound_max=7.5), n_var=3)
    assert b._bornes == ([-7.5, -7.5, -7.5], [7.5, 7.5, 7.5])


def test_le_domaine_suit_le_fichier_d_etude():
    """La preuve que le reglage AGIT : on le change, le domaine change."""
    serre = _boucle(cfg=_Cfg(eff_bound_min=-6.0, eff_bound_max=6.0))._bornes
    large = _boucle(cfg=_Cfg(eff_bound_min=-7.5, eff_bound_max=7.5))._bornes
    assert serre != large
    assert serre[0][0] == -6.0 and large[0][0] == -7.5


# --------------------------------------------------------------------------- #
# 2. LE CRITERE EFF                                                            #
# --------------------------------------------------------------------------- #
class _Surrogate:
    """Un metamodele de papier : `g` affine, `sigma` constant."""

    def __init__(self, sigma=0.5):
        self._sigma = sigma

    def __call__(self, u):
        return [u[0] + u[1]]

    def sigma(self, u):
        import numpy as np
        u = np.atleast_2d(u)
        return np.full((len(u), 1), self._sigma)


def test_le_critere_EFF_est_construit_et_evaluable():
    b = _boucle()
    s = _Surrogate()
    f = b.fonction_EFF(s, s.sigma)
    valeur = ot.Function(f)([0.5, -0.5])[0]
    assert valeur == pytest.approx(valeur)      # evaluable, fini
    assert valeur >= 0.0


def test_le_critere_EFF_lit_epsilon_factor():
    """`epsilon_factor` elargit la bande autour de `g = 0` : a metamodele
    identique, le critere doit changer avec lui. Sans quoi le reglage ne
    serait pas cable."""
    s = _Surrogate()
    petit = ot.Function(_boucle(cfg=_Cfg(epsilon_factor=0.5))
                        .fonction_EFF(s, s.sigma))([0.5, -0.5])[0]
    grand = ot.Function(_boucle(cfg=_Cfg(epsilon_factor=4.0))
                        .fonction_EFF(s, s.sigma))([0.5, -0.5])[0]
    assert petit != pytest.approx(grand), (
        "le critere ne depend pas de `epsilon_factor` : le reglage n'est pas "
        "transmis (petit=%r, grand=%r)" % (petit, grand))


# --------------------------------------------------------------------------- #
# 3. L'ENCADREMENT -- ET SON REFUS                                             #
# --------------------------------------------------------------------------- #
def test_l_encadrement_exige_le_predicteur_et_le_DIT():
    """Le seul element que la boucle ne peut pas deduire de `cfg` : les
    fonctions de prediction vivent dans `_lib`, et `_reliability` n'a pas a en
    dependre. Mieux vaut un refus lisible qu'un encadrement faux."""
    b = _boucle()                     # sans `predire`
    with pytest.raises(ValueError) as capture:
        b.bornes_surrogate(_Surrogate(), _Surrogate().sigma, +1)
    texte = str(capture.value)
    assert "predire" in texte and "predicteur" in texte, texte


def test_l_encadrement_utilise_le_predicteur_donne():
    """Et quand il est fourni, c'est bien LUI qui est appele.

    LE CHEMIN EXACT COMPTE, et ma premiere version de ce temoin l'avait
    manque : `bound_surrogate_function` n'appelle le predicteur que dans
    `_exec_sample`, et seulement si `sigma_func.__self__.fm` existe -- c'est
    le chemin en LOT, celui qui rend l'encadrement abordable pour FORM.
    Evalue point par point, il passe par `_exec` et le predicteur ne sert
    pas. Il faut donc un metamodele qui porte un `fm`, et une evaluation sur
    un echantillon.
    """
    import numpy as np
    appels = []

    class _AvecFm(_Surrogate):
        fm = "le metamodele"           # ce que `_exec_sample` va chercher

    def _predire(fm, U, return_var=False):
        appels.append((fm, np.asarray(U).shape))
        mu = np.zeros((len(np.atleast_2d(U)), 1))
        return (mu, np.full_like(mu, 0.25)) if return_var else mu

    s = _AvecFm()
    b = _boucle(predire=_predire)
    haut = ot.Function(b.bornes_surrogate(s, s.sigma, +1))
    valeurs = haut(ot.Sample([[0.0, 0.0], [1.0, 1.0]]))
    assert appels, (
        "le predicteur fourni n'a pas ete appele : `bornes_surrogate` ne "
        "transmet pas `predire` a `form.bound_surrogate_function`.")
    assert appels[0][0] == "le metamodele"
    # mu = 0, sigma = sqrt(0.25) = 0.5, signe +1 -> 0 + 2*0.5 = 1.0
    assert [v[0] for v in valeurs] == pytest.approx([1.0, 1.0]), (
        "l'encadrement ne vaut pas `mu + 2 sigma` avec ce que le predicteur "
        "a rendu : %s" % [v[0] for v in valeurs])


# --------------------------------------------------------------------------- #
# 4. LE TIRAGE D'IMPORTANCE DE REPLI                                           #
# --------------------------------------------------------------------------- #
def test_le_tirage_d_importance_lit_n_IS_et_cov_IS(monkeypatch):
    """Les deux reglages qui coutent : le nombre de tirages et la cible de
    coefficient de variation. Ils etaient lies par un delegue de l'etude."""
    import form as _form
    vus = {}

    def _faux(modes, evenement, n_var, n_IS, cov_IS):
        vus.update(n_var=n_var, n_IS=n_IS, cov_IS=cov_IS)
        return None

    monkeypatch.setattr(_form, "run_IS", _faux)
    _boucle(cfg=_Cfg(n_IS=4321, cov_IS=0.02), n_var=3).executer_is([], None)
    assert vus == {"n_var": 3, "n_IS": 4321, "cov_IS": 0.02}


# --------------------------------------------------------------------------- #
# 5. LE PROCHAIN BATCH                                                         #
# --------------------------------------------------------------------------- #
def test_le_batch_recoit_les_reglages_le_domaine_et_le_reajustement(monkeypatch):
    """`_find_batch_EFF_points` liait sept choses. Elles doivent toutes
    arriver -- surtout `reajuster`, qui est le metamodele de l'etude, et
    `gradient_du_surrogate`, qui depend du modele choisi."""
    import eff_ot as _eff_ot
    vus = {}

    def _faux(g_ot, sigma_func, xt, yt, all_grad, **kw):
        vus.update(kw)
        return ([], 0.0)

    monkeypatch.setattr(_eff_ot, "batch_kriging_believer", _faux)
    ajuster = lambda *a, **k: None                       # noqa: E731
    b = _boucle(cfg=_Cfg(n_batch_EFF=2, n_NLopt_EFF=99, epsilon_factor=3.0,
                         do_GEPCK=True, do_PCK=False),
                n_var=2, ajuster=ajuster)
    b.points_EFF(None, None, None, None, None)
    assert vus["n_batch"] == 2
    assert vus["n_appels"] == 99
    assert vus["epsilon_factor"] == 3.0
    assert vus["n_var"] == 2
    assert vus["bornes_min"] == [-6.0, -6.0]
    assert vus["bornes_max"] == [6.0, 6.0]
    assert vus["gradient_du_surrogate"] is True
    assert vus["reajuster"] is ajuster, (
        "le reajustement passe au batch n'est pas celui de l'etude : le "
        "Kriging Believer imputerait ses points sur un autre metamodele.")


# --------------------------------------------------------------------------- #
# 6. LA LOI JOINTE -- ET SON ABSENCE ASSUMEE                                   #
# --------------------------------------------------------------------------- #
def test_la_loi_jointe_reste_None_sans_catalogue():
    """Le chemin parallele teste `dist_jointe is not None`. Une loi fabriquee
    a vide serait pire que son absence : elle ferait repasser les points dans
    l'espace physique avec les mauvaises lois."""
    assert _boucle().dist_jointe is None


def test_la_loi_jointe_est_construite_avec_le_catalogue():
    import lois as _lois
    pc = {"fc": {"loi": _lois.loi_fc, "args": (48, 0.12)},
          "fy": {"loi": _lois.loi_fy, "args": (550, None)}}
    b = _boucle(param_config=pc, params_names=["fc", "fy"])
    assert b.dist_jointe is not None
    loi = b.dist_jointe()
    assert loi.getDimension() == 2


# --------------------------------------------------------------------------- #
# 7. UNE INJECTION EXPLICITE L'EMPORTE TOUJOURS                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nom", ["fonction_EFF", "bornes_surrogate",
                                 "executer_is", "points_EFF"])
def test_ce_qui_est_injecte_gagne(nom):
    """C'est ce qui garde `test_119_boucle_eff.py` possible : il stube ces
    quatre-la pour eprouver l'ordre des gestes sans metamodele."""
    temoin = lambda *a, **k: "le mien"                   # noqa: E731
    b = _boucle(**{nom: temoin})
    assert getattr(b, nom) is temoin


# --------------------------------------------------------------------------- #
# 8. LES ETUDES NE LES PASSENT PLUS                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ETUDES)
def test_l_etude_ne_fabrique_plus_ces_collaborateurs(script):
    """Le pendant cote etude : si l'un revenait, il y aurait de nouveau DEUX
    endroits qui lient les memes reglages -- et ils divergeraient."""
    import io
    src = io.open(os.path.join(_REPO, script), encoding="utf-8",
                  errors="replace").read()
    for interdit in ("def EFFFunction", "def BoundSurrogateFunction",
                     "def _find_batch_EFF_points",
                     "fonction_EFF=", "bornes_surrogate=", "points_EFF="):
        assert interdit not in src, (
            "%s : %r est revenu. `BoucleEFF` construit ce collaborateur a "
            "partir de `cfg`, `n_var` et `predire`." % (script, interdit))
    assert "predire=_PREDICT" in src, (
        "%s ne passe plus le predicteur : l'encadrement `g +/- 2 sigma` "
        "refuserait d'etre construit." % script)
