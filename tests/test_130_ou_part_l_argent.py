r"""Ou l'enrichissement depense-t-il ses appels solveur ?

CONSTATE, PAS APPROUVE -- 29/08/2026
-------------------------------------
Chaque point d'enrichissement coute 466 s sur le Moulin Blanc, et l'etude
reelle en prevoit 360 : quarante-six heures. La question « ou tombent-ils ? »
n'avait jamais ete posee.

Mesure sur la flexion pure analytique, huit iterations depuis le meme plan
initial de cinq points :

    beta VRAI (FORM sur l'etat limite EXACT)   4,7728
    u* vrai                                    [-2,573 ; -4,020]   ||u*|| = 4,77

    budget   beta      ecart     ||u|| moyen   distance moyenne a u*
    30       4,8030    +0,63 %      8,19             7,24
    100      4,8075    +0,73 %      8,60             7,73
    300      4,8074    +0,73 %      8,62             7,74
    1000     4,8074    +0,73 %      8,62             7,74

Les points payes tombent AUX COINS du domaine, a une distance de u* superieure
a la norme de u* lui-meme. Ce n'est pas un defaut de l'optimiseur : le critere
EFF est grand la ou le metamodele est incertain, et il l'est enormement loin
de tout point d'apprentissage. C'est le DOMAINE de recherche -- `eff_bound_min`
/ `eff_bound_max`, +/- 7,5 ici -- qui gouverne le phenomene. A +/- 7,5, la
densite vaut p ~ 3e-14 : on paie 466 s pour explorer une region qui ne pese
rien dans la probabilite de defaillance.

CE QUE CE FICHIER NE FAIT PAS
------------------------------
Il ne corrige rien. Le domaine a deja ete arbitre une fois -- Agnes,
26/08/2026, `_RAISON_DOMAINE` dans `test_85` : ramener le Moulin Blanc de
+/- 7,5 a +/- 6,0, « un compromis assume » entre contenir le point de
conception et garder les cones SOCP bien conditionnes. La presente mesure
ajoute un CHIFFRE a cette question ; elle ne la retranche pas.

UNE ERREUR DE MA PART, CONSIGNEE
---------------------------------
J'ai d'abord conclu que `n_NLopt_EFF = 30` etait trop petit -- l'optimiseur
trouvait un point valant parfois 6,8 %% du critere atteignable. C'est vrai, et
sans consequence : le tableau ci-dessus montre que le budget deplace beta de
0,1 %%. J'avais mesure la valeur du CRITERE et parle comme si j'avais mesure la
JUSTESSE du resultat. Le reglage est reste a 30.
"""

import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
for _p in ("_reliability", "_doe", "_surrogate", "_lib", "_model", "_cache",
           "_config", "_etapes", "solver"):
    _c = os.path.join(_REPO, _p)
    if _c not in sys.path:
        sys.path.insert(0, _c)

np = pytest.importorskip("numpy")
ot = pytest.importorskip("openturns")
pytest.importorskip("smt")

import schema      # noqa: E402
import lois        # noqa: E402

#: Releve du 29/08/2026, a reproduire ici. `beta` vient d'un FORM sur l'etat
#: limite EXACT -- aucun metamodele, aucune approximation.
BETA_VRAI = 4.7728
U_ETOILE = np.array([-2.573, -4.020])
PARAMS = ["fc", "fy"]
CONFIG = {"fc": {"loi": lois.loi_fc, "args": (48, 0.12)},
          "fy": {"loi": lois.loi_fy, "args": (550, None)}}


@pytest.fixture(scope="module")
def etude():
    cfg = schema.charger(os.path.join(_REPO, "studies",
                                      "pure_flexion_analytique.toml"))
    chemin = cfg.chemin_ds        # jamais recalcule, cf. test_128
    if not os.path.isdir(chemin):
        pytest.skip("le modele de la flexion pure n'est ni dans le storage "
                    "ni dans `modeles/` du depot")
    from fabrique import solveur as fabriquer
    sol = fabriquer(cfg.solveur, chemin_ds=chemin,
                    dossier_etude=os.path.join(_REPO, "pure_flexion"),
                    params_names=PARAMS,
                    regions=[{"param": "COMPRESSIVE_STRENGTH", "region_key": "fc"},
                             {"param": "YIELD_STRENGTH", "region_key": "fy"}])
    return cfg, sol


def _dist():
    return lois.dist_jointe(CONFIG, PARAMS)


def _enrichir(cfg, sol, n_tours=5):
    """Enrichit reellement, et rend les points payes."""
    import ajuster as _fit
    import eff_ot as _eff_ot
    import plan as _plan

    bmin, bmax = [cfg.eff_bound_min] * 2, [cfg.eff_bound_max] * 2
    T = _dist().getIsoProbabilisticTransformation()
    T_inv = _dist().getInverseIsoProbabilisticTransformation()

    def evaluer_plan(SOL):
        for s in SOL:
            ev = sol.evaluer({p: s[p] for p in PARAMS}, sensibilite=True)
            s["g"] = ev.g
            s["_u"] = [float(v) for v in
                       T(ot.Point([float(s[p]) for p in PARAMS]))]
            for j, p in enumerate(PARAMS):
                s["dg_%s" % p] = float(ev.grad_x[j])
        return SOL

    class _Diag:
        def enregistrer(self, d):
            pass

    ot.RandomGenerator.SetSeed(77)
    xt, yt, ag = _plan.construire_plan_initial(
        cfg, cfg.n0, dist_jointe=_dist, params_names=PARAMS,
        bornes_min=bmin, bornes_max=bmax,
        fichier_cache=os.path.join(cfg.chemin_ds, "_test130.json"),
        signature={"test": 130}, executer_plan=evaluer_plan,
        moissonner=lambda SOL, noms: {}, tracer=lambda m: None)

    payes = []
    for _ in range(n_tours):
        g_ot, sigma, _, _, _ = _fit.ajuster_sur_le_plan(
            cfg, xt, yt, ag, max_degree=cfg.max_degree, dist_X=_dist(),
            diagnostics=_Diag(), tracer=lambda m: None)
        f_eff = ot.Function(_eff_ot.eff_function(g_ot, sigma, 2,
                                                 cfg.epsilon_factor))
        u, _v = _eff_ot.maximiser_EFF(f_eff, bmin, bmax, 2, cfg.n_NLopt_EFF)
        payes.append(np.array(u))

        x = T_inv(ot.Point(list(u)))
        ev = sol.evaluer({p: float(x[j]) for j, p in enumerate(PARAMS)},
                         sensibilite=True)
        gx = [float(v) for v in ev.grad_x]
        gu = []
        for j in range(2):
            h, up = 1e-6, list(u)
            up[j] += h
            xp = T_inv(ot.Point(up))
            gu.append(sum(gx[k] * (float(xp[k]) - float(x[k])) / h
                          for k in range(2)))
        xt = np.vstack([xt, [u]])
        yt = np.vstack([yt, [[ev.g]]])
        ag = np.vstack([ag, [gu]])
    return np.array(payes)


# --------------------------------------------------------------------------- #
# LA MESURE, FIGEE                                                             #
# --------------------------------------------------------------------------- #
def test_l_enrichissement_paie_aux_COINS_du_domaine(etude):
    """Le constat. Les seuils sont larges : c'est un ORDRE DE GRANDEUR qu'on
    fige, pas une valeur -- un ecart de quelques pour cent ne veut rien dire,
    un point paye pres de u* voudrait dire beaucoup."""
    cfg, sol = etude
    payes = _enrichir(cfg, sol)
    norme = float(np.mean(np.linalg.norm(payes, axis=1)))
    distance = float(np.mean(np.linalg.norm(payes - U_ETOILE, axis=1)))

    assert norme > 6.0, (
        "les points payes tombent maintenant a ||u|| = %.2f, contre 8,2 le "
        "29/08/2026. Si le domaine ou le critere ont change, c'est une bonne "
        "nouvelle -- mettre a jour ce temoin." % norme)
    assert distance > 4.0, (
        "les points payes tombent maintenant a %.2f de u*, contre 7,2 le "
        "29/08/2026." % distance)
    assert norme > float(np.linalg.norm(U_ETOILE)), (
        "les points payes sont desormais plus pres de l'origine que u* : "
        "l'enrichissement a cesse d'explorer les coins.")


def test_le_domaine_de_recherche_est_bien_celui_qu_on_croit(etude):
    """Le phenomene tient au DOMAINE, pas a l'optimiseur : si les bornes
    changent, ce temoin doit etre relu."""
    cfg, _ = etude
    assert (cfg.eff_bound_min, cfg.eff_bound_max) == (-7.5, 7.5)


def test_le_budget_de_recherche_est_reste_a_30():
    """Consigne : j'avais propose de le porter a 300 sur une mesure qui ne
    prouvait pas ce que j'en disais. Le tableau du docstring montre que le
    budget deplace beta de 0,1 %."""
    cfg = schema.Configuration(modelname="x")
    assert cfg.n_NLopt_EFF == 30
