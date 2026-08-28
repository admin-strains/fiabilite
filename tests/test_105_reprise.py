r"""L'etat de reprise : le fichier le plus cher du depot, enfin testable.

CE QU'IL ETAIT
---------------
115 lignes -- ecriture et relecture -- recopiees a l'identique dans les deux
etudes, a l'interieur de `if __name__ == '__main__':`. Aucun test ne pouvait
les atteindre. Sur le Moulin Blanc un point coute 466 s : ce fichier porte
jusqu'a quatre-vingt-dix heures de calcul.

CE QUE CES TESTS METTENT PAR ECRIT
-----------------------------------
1. Les trous des historiques restent des trous. En mode `BB`, `hist_BS` ne
   recoit rien et un round peut ne produire aucun ratio : `float(None)`
   leverait, ecraser en 0.0 mentirait sur la courbe de convergence.
2. Ecrire un dump n'emporte JAMAIS le run. Un echec de serialisation
   survient a la fin d'un round d'enrichissement -- c'est-a-dire apres le
   calcul, pas avant.
3. Relire un dump refuse une configuration differente, et le dit champ par
   champ. C'est le controle ajoute le 26/08/2026 : la signature etait ecrite
   depuis toujours, personne ne la lisait.
4. C'est la signature SOLVEUR qui protege, pas la signature faible. Un
   `modelname` different se relit sans broncher -- deliberement.
5. L'ordre qui rend inutile le `try/except` retire de `_save_restart_state` :
   `_GRILLE` est construite avant les deux appels.
"""

import io
import json
import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
if os.path.join(_REPO, "_cache") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "_cache"))

import reprise as _reprise                            # noqa: E402


SIG_FAIBLE = {"n0": 24, "params": ["fc", "fy"], "n_var": 2, "modelname": "flexion"}
SIG_SOLVEUR = {"solveur": "DS", "solveur_lineaire": "MUMPS", "taille_maille": 0.05}

HIST = {"EFF": [1.0, 0.5], "BB": [None, 0.02], "BS": [0.3, None],
        "theta": [[1.0, 2.0], [1.1, 2.1]], "beta_IS": [None, 4.7],
        # Le sixieme depuis le 28/08/2026 : il etait tenu pendant le run et
        # jamais dumpe. Voir `tests/test_118_historiques_reprise.py`, qui
        # ferme la REGLE dont il n'etait qu'un cas.
        "Pf": [None, {"mid": 1e-6, "sup": 2e-7, "inf": 4e-6}]}


def _champs(**surcharge):
    """Un jeu d'arguments complet, que chaque test deforme sur un point."""
    base = dict(
        signature=SIG_FAIBLE, signature_solveur=SIG_SOLVEUR,
        modele="flexion", timestamp="20260827_1200", max_degree=3, n0=24,
        xt=np.arange(8, dtype=float).reshape(4, 2),
        yt=np.array([1.0, -1.0, 0.5, -0.5]),
        all_grad=np.zeros((4, 2)), xt_eff=[np.array([0.1, 0.2])],
        enrich_round=0, round_sizes_prev=[], historiques=HIST,
        coupe_hf=None, best_result=None, best_sp=None, modes=None,
        result_IS=None)
    base.update(surcharge)
    return base


# --------------------------------------------------------------------------- #
# CE QUI EST ECRIT                                                             #
# --------------------------------------------------------------------------- #
def test_le_dump_porte_les_points_et_leurs_gradients():
    st = _reprise.construire_etat(**_champs())
    assert st["xt"] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    assert st["yt"] == [1.0, -1.0, 0.5, -0.5]
    assert st["all_grad"] == [[0.0, 0.0]] * 4
    assert st["xt_eff"] == [[0.1, 0.2]]
    assert st["n_total"] == 4 and st["n_doe"] == 24


def test_les_tableaux_absents_restent_absents_sans_lever():
    st = _reprise.construire_etat(**_champs(xt=None, yt=None, all_grad=None,
                                            xt_eff=[]))
    assert st["xt"] is None and st["yt"] is None and st["all_grad"] is None
    assert st["xt_eff"] == [] and st["n_total"] == 0
    assert st["round_sizes"] == []


def test_les_trous_des_historiques_restent_des_trous():
    """`None` n'est ni une erreur ni un zero.

    En mode `BB` l'historique BS reste vide, et un round peut ne produire
    aucun ratio. `float(None)` leverait ; 0.0 dirait « critere satisfait ».
    """
    st = _reprise.construire_etat(**_champs())
    assert st["hist_BB"] == [None, 0.02]
    assert st["hist_BS"] == [0.3, None]
    assert st["hist_beta_IS"] == [None, 4.7]
    assert st["hist_EFF"] == [1.0, 0.5]
    assert st["hist_theta"] == [[1.0, 2.0], [1.1, 2.1]]


def test_le_premier_round_compte_tout_le_plan():
    st = _reprise.construire_etat(**_champs(enrich_round=0))
    assert st["round_sizes"] == [4]
    assert st["round_boundaries"] == [0, 4]


def test_un_round_suivant_ne_compte_que_ce_qu_il_a_ajoute():
    st = _reprise.construire_etat(**_champs(enrich_round=2,
                                            round_sizes_prev=[24, 6]))
    assert st["round_sizes"] == [24, 6, 4 - 30]
    assert st["round_boundaries"] == [0, 24, 30, 4]


def test_un_degre_illisible_ne_bloque_pas_le_dump():
    st = _reprise.construire_etat(**_champs(max_degree=object()))
    assert st["max_degree"] is None


def test_un_resultat_form_qui_n_en_est_pas_un_vaut_None():
    st = _reprise.construire_etat(**_champs(best_result=object(),
                                            modes=[object(), object()]))
    assert st["best_result"] is None
    assert st["modes"] == [None, None]


class _ResultatFORM:
    def getStandardSpaceDesignPoint(self):
        return [-1.5, 2.5]

    def getHasoferReliabilityIndex(self):
        return 4.7516


def test_un_resultat_form_se_reduit_a_u_star_et_beta():
    st = _reprise.construire_etat(**_champs(best_result=_ResultatFORM(),
                                            best_sp=np.array([0.0, 1.0])))
    assert st["best_result"] == {"u_star": [-1.5, 2.5], "beta": 4.7516}
    assert st["best_sp"] == [0.0, 1.0]


# --------------------------------------------------------------------------- #
# ECRIRE N'EMPORTE JAMAIS LE RUN                                               #
# --------------------------------------------------------------------------- #
def test_enregistrer_ecrit_un_json_relisible(tmp_path, capsys):
    f = str(tmp_path / "restart_state.json")
    st = _reprise.enregistrer(f, **_champs())
    assert st is not None
    relu = json.load(io.open(f, encoding="utf-8"))
    assert relu["xt"] == st["xt"]
    assert "[RESTART DUMP] etat sauve" in capsys.readouterr().out


def test_un_echec_de_serialisation_est_signale_jamais_fatal(tmp_path, capsys):
    """Le dump est ecrit APRES le calcul du round. Lever ici jetterait des
    heures de solveur pour une erreur de mise en forme."""
    f = str(tmp_path / "restart_state.json")
    manquant = {k: v for k, v in HIST.items() if k != "BS"}
    assert _reprise.enregistrer(f, **_champs(historiques=manquant)) is None
    sortie = capsys.readouterr().out
    assert "[RESTART DUMP] sauvegarde echouee" in sortie
    assert "KeyError" in sortie and "BS" in sortie
    assert not os.path.isfile(f)


def test_un_chemin_impossible_est_signale_jamais_fatal(tmp_path, capsys):
    f = str(tmp_path / "dossier_absent" / "restart_state.json")
    assert _reprise.enregistrer(f, **_champs()) is None
    assert "sauvegarde echouee" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# RELIRE REFUSE UNE AUTRE CONFIGURATION                                        #
# --------------------------------------------------------------------------- #
def test_un_dump_absent_dit_quoi_faire(tmp_path):
    """Le message compte : le cas se produit APRES plusieurs minutes de
    construction du modele CAD, sur un poste ou le dump n'a jamais existe."""
    f = str(tmp_path / "jamais_ecrit.json")
    with pytest.raises(SystemExit) as e:
        _reprise.charger(f, SIG_SOLVEUR)
    msg = str(e.value)
    assert "aucun etat a reprendre" in msg
    assert "restart_enrich_only = false" in msg
    assert f in msg


def test_une_signature_solveur_identique_laisse_reprendre(tmp_path):
    f = str(tmp_path / "restart_state.json")
    _reprise.enregistrer(f, **_champs())
    rs = _reprise.charger(f, dict(SIG_SOLVEUR))
    assert rs["n_total"] == 4
    assert rs["hist_BB"] == [None, 0.02]


def test_un_solveur_lineaire_different_refuse_la_reprise(tmp_path):
    """90 heures de points calcules sous un backend, poursuivies sous un
    autre : le melange serait indetectable dans le resultat."""
    f = str(tmp_path / "restart_state.json")
    _reprise.enregistrer(f, **_champs())
    autre = dict(SIG_SOLVEUR, solveur_lineaire="CuDss")
    with pytest.raises(SystemExit) as e:
        _reprise.charger(f, autre)
    msg = str(e.value)
    assert "n'a PAS ete produit" in msg
    assert "solveur_lineaire : dump='MUMPS'  courant='CuDss'" in msg
    # seul le champ qui differe est cite
    assert "taille_maille" not in msg


def test_plusieurs_ecarts_sont_tous_cites(tmp_path):
    f = str(tmp_path / "restart_state.json")
    _reprise.enregistrer(f, **_champs())
    autre = dict(SIG_SOLVEUR, solveur_lineaire="CuDss", taille_maille=0.02)
    with pytest.raises(SystemExit) as e:
        _reprise.charger(f, autre)
    msg = str(e.value)
    assert "solveur_lineaire" in msg and "taille_maille" in msg


def test_un_dump_anterieur_au_controle_le_dit(tmp_path):
    """Les dumps ecrits avant le 26/08/2026 n'ont pas de signature. On ne
    peut pas les valider, donc on ne les reprend pas -- en le disant."""
    f = str(tmp_path / "restart_state.json")
    json.dump({"n_total": 4}, io.open(f, "w", encoding="utf-8"))
    with pytest.raises(SystemExit) as e:
        _reprise.charger(f, SIG_SOLVEUR)
    assert "anterieur au controle" in str(e.value)


def test_la_signature_faible_ne_bloque_rien(tmp_path):
    """Deliberement : `doe_cache_sig` ignore le solveur, le maillage et les
    bornes. Elle estampille, elle ne protege pas -- des outils la lisent."""
    f = str(tmp_path / "restart_state.json")
    _reprise.enregistrer(f, **_champs(signature=dict(SIG_FAIBLE,
                                                     modelname="autre")))
    rs = _reprise.charger(f, dict(SIG_SOLVEUR))
    assert rs["signature"]["modelname"] == "autre"


# --------------------------------------------------------------------------- #
# LE TOUR COMPLET                                                              #
# --------------------------------------------------------------------------- #
def test_aller_retour_les_points_reviennent_a_l_identique(tmp_path):
    f = str(tmp_path / "restart_state.json")
    champs = _champs(coupe_hf={"Z": [[0.0, 1.0], [2.0, 3.0]], "cote": 2})
    _reprise.enregistrer(f, **champs)
    rs = _reprise.charger(f, dict(SIG_SOLVEUR))
    assert np.allclose(np.array(rs["xt"], float), champs["xt"])
    assert np.allclose(np.array(rs["all_grad"], float), champs["all_grad"])
    assert rs["hf_2d_grid"]["cote"] == 2


def test_les_six_historiques_reviennent_sous_leurs_clefs(tmp_path):
    f = str(tmp_path / "restart_state.json")
    _reprise.enregistrer(f, **_champs())
    h = _reprise.historiques_de(_reprise.charger(f, dict(SIG_SOLVEUR)))
    assert set(h) == {"EFF", "BB", "BS", "theta", "beta_IS", "Pf"}
    assert h == HIST


def test_un_historique_absent_du_dump_revient_vide():
    """Un dump anterieur a l'ajout d'un historique se relit sans lever."""
    h = _reprise.historiques_de({"hist_EFF": [1.0]})
    assert h["EFF"] == [1.0]
    assert h["BB"] == [] and h["BS"] == [] and h["theta"] == []
    assert h["beta_IS"] == [] and h["Pf"] == []


def test_le_dump_vit_dans_le_ds_du_modele():
    chemin = _reprise.fichier_de(os.path.join("C:", "modeles", "mb.ds"))
    assert chemin.endswith("restart_state.json")
    assert "mb.ds" in chemin


# --------------------------------------------------------------------------- #
# CE QUI NE DOIT PLUS ETRE DANS LES ETUDES                                     #
# --------------------------------------------------------------------------- #
SCRIPTS = ("pure_flexion/AC3_pure_flexion.py", "Moulinblanc/AC3_moulinblanc.py")


@pytest.mark.parametrize("script", SCRIPTS)
def test_les_etudes_ne_serialisent_plus_elles_memes(script):
    s = io.open(os.path.join(_REPO, script), encoding="utf-8").read()
    for interdit in ('st["hist_BB"]', 'st["round_boundaries"]',
                     "json.dump(st,", "_rs.get('hist_BB'",
                     'os.path.join(_path_ds, "restart_state.json")'):
        assert interdit not in s, (
            "%s : la serialisation de l'etat de reprise est revenue dans "
            "l'etude (%r). Elle appartient a `_cache/reprise.py`."
            % (script, interdit))


@pytest.mark.parametrize("script", SCRIPTS)
def test_la_grille_est_construite_avant_les_deux_dumps(script):
    """L'ordre qui rend inutile le `try/except` retire de
    `_save_restart_state` : `coupe_hf=_GRILLE.coupes.get(...)` est evalue au
    site d'appel, donc hors du filet de `enregistrer`. Si un dump remontait
    avant la construction de `_GRILLE`, ce serait un NameError fatal.
    """
    import ast
    s = io.open(os.path.join(_REPO, script), encoding="utf-8").read()
    lignes = s.splitlines()
    grille = [i for i, l in enumerate(lignes, 1) if "_GRILLE = _grille.Grille(" in l]
    assert len(grille) == 1, script

    main = [n for n in ast.parse(s).body if isinstance(n, ast.If)][0]
    fns = {f.name: (f.lineno, f.end_lineno)
           for f in main.body if isinstance(f, ast.FunctionDef)}

    def flux_principal(i):
        """La ligne, ou -- si elle est dans une fonction -- l'appel de
        cette fonction depuis le flux principal."""
        dedans = [n for n, (a, b) in fns.items() if a <= i <= b]
        if not dedans:
            return i
        appels = [j for j, l in enumerate(lignes, 1)
                  if (dedans[0] + "(") in l and "def " not in l
                  and not any(a <= j <= b for a, b in fns.values())]
        assert appels, (script, dedans[0])
        return min(appels)

    dumps = [i for i, l in enumerate(lignes, 1)
             if "_save_restart_state(" in l and "def " not in l]
    assert len(dumps) == 2, (script, dumps)
    for i in dumps:
        assert flux_principal(i) > grille[0], (
            "%s : un dump (l. %d) s'execute avant la construction de "
            "`_GRILLE` (l. %d) -- `coupe_hf=` leverait NameError."
            % (script, i, grille[0]))
