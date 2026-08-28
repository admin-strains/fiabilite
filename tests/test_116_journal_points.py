r"""Le journal des points : une ligne JSON par appel au solveur.

CE QUE C'ETAIT
---------------
Une seule notion tenue en HUIT endroits de chaque etude : une fonction de
quatorze lignes recopiee, deux etats globaux ecrits en listes d'un seul
element (`_point_log_phase = ["?"]`, l'idiome qu'on emploie pour muter depuis
une fermeture faute d'objet), le chemin du fichier, la remise a zero, la
ligne de reprise, et deux sites d'appel.

POURQUOI CE FICHIER COMPTE
---------------------------
C'est la trace de ce qui a REELLEMENT ete calcule. Sur le Moulin Blanc un
point coute 466 s : c'est le seul endroit ou l'on peut, apres coup, repondre
a « combien d'appels, ou, et pour quel resultat ».

CE QUE CES TESTS METTENT PAR ECRIT
-----------------------------------
1. Chaque point porte son resultat sous ses DEUX noms : `g` et
   `lambda = g + 1`. L'etat limite du calcul a la rupture s'ecrit
   `g = alpha - 1` : `lambda` est le facteur de charge lui-meme, celui que
   l'ingenieur lit.
2. Ecrire n'emporte JAMAIS le run -- mise en forme comprise, pas seulement
   l'ecriture disque.
3. Une reprise ne VIDE pas le journal : les points du run precedent ont ete
   payes. Un run neuf, si.
"""

import io
import json
import os
import sys

import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
if os.path.join(_REPO, "_cache") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "_cache"))

import journal_points as _jp                            # noqa: E402


PARAMS = ["fc", "fy"]


class _Trace(list):
    def __call__(self, message):
        self.append(message)


def _journal(tmp_path, tracer=None):
    # `is None`, PAS `tracer or ...` : `_Trace` herite de `list`, donc un
    # traceur vide est FAUX et se ferait remplacer par un autre. C'est le
    # meme piege que celui corrige dans `_reliability/eff.py` le 27/08 --
    # tendu ici par le test lui-meme.
    return _jp.JournalDesPoints(str(tmp_path / "points_log.jsonl"), PARAMS,
                                tracer=_Trace() if tracer is None else tracer)


def _lignes(journal):
    """Les lignes ecrites. Un journal ou rien n'a pu s'ecrire n'existe meme
    pas -- et c'est un etat legitime, pas une erreur de test."""
    if not os.path.exists(journal.fichier):
        return []
    with io.open(journal.fichier, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# --------------------------------------------------------------------------- #
# CE QUE PORTE UNE LIGNE                                                       #
# --------------------------------------------------------------------------- #
def test_un_point_porte_ses_deux_espaces_et_ses_deux_noms(tmp_path):
    j = _journal(tmp_path)
    j.marquer("EFF")
    j.enregistrer([-2.8, -3.8], [38.5, 440.0], 0.0636)
    (ligne,) = _lignes(j)
    assert ligne["phase"] == "EFF" and ligne["round"] == 0
    assert ligne["u_fc"] == -2.8 and ligne["u_fy"] == -3.8
    assert ligne["x_fc"] == 38.5 and ligne["x_fy"] == 440.0
    assert ligne["g"] == pytest.approx(0.0636)
    assert ligne["lambda"] == pytest.approx(1.0636)


def test_lambda_est_le_facteur_de_charge_pas_une_decoration():
    """`g = alpha - 1` : `lambda` est `alpha`. Un `g` nul est un ouvrage
    exactement a la rupture, donc un facteur de charge de 1."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        j = _jp.JournalDesPoints(os.path.join(d, "p.jsonl"), PARAMS,
                                 tracer=_Trace())
        for g in (0.0, -0.25, 0.5):
            j.enregistrer([0.0, 0.0], [0.0, 0.0], g)
        assert [l["lambda"] for l in _lignes(j)] == [1.0, 0.75, 1.5]


def test_un_point_sans_resultat_ne_fabrique_pas_de_facteur(tmp_path):
    """`g = None` -- le solveur n'a rien rendu. Ecrire `lambda = 1.0`
    affirmerait un ouvrage juste a la rupture."""
    j = _journal(tmp_path)
    j.enregistrer([0.0, 0.0], [1.0, 2.0], None)
    (ligne,) = _lignes(j)
    assert ligne["g"] is None and ligne["lambda"] is None


def test_une_coordonnee_absente_vaut_None(tmp_path):
    """L'appelant ne connait pas toujours les deux espaces."""
    j = _journal(tmp_path)
    j.enregistrer([-2.8, -3.8], None, 1.0)
    (ligne,) = _lignes(j)
    assert ligne["x_fc"] is None and ligne["x_fy"] is None
    assert ligne["u_fc"] == -2.8


def test_l_ordre_des_parametres_nomme_les_colonnes(tmp_path):
    j = _jp.JournalDesPoints(str(tmp_path / "p.jsonl"), ["a", "b", "c"],
                             tracer=_Trace())
    j.enregistrer([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 0.0)
    (ligne,) = _lignes(j)
    assert (ligne["u_a"], ligne["u_b"], ligne["u_c"]) == (1.0, 2.0, 3.0)
    assert (ligne["x_a"], ligne["x_b"], ligne["x_c"]) == (4.0, 5.0, 6.0)


# --------------------------------------------------------------------------- #
# LA PHASE ET LE ROUND ESTAMPILLENT CE QUI SUIT                                #
# --------------------------------------------------------------------------- #
def test_la_phase_vaut_jusqu_a_ce_qu_on_la_change(tmp_path):
    j = _journal(tmp_path)
    j.marquer("DOE")
    j.enregistrer([0.0, 0.0], [0.0, 0.0], 1.0)
    j.marquer("EFF")
    j.enregistrer([0.0, 0.0], [0.0, 0.0], 1.0)
    j.enregistrer([0.0, 0.0], [0.0, 0.0], 1.0)
    assert [l["phase"] for l in _lignes(j)] == ["DOE", "EFF", "EFF"]


def test_une_phase_passee_au_point_prime_sans_changer_l_etat(tmp_path):
    """Le plan d'experiences logue « DOE » sans deranger la phase courante."""
    j = _journal(tmp_path)
    j.marquer("EFF")
    j.enregistrer([0.0, 0.0], [0.0, 0.0], 1.0, phase="DOE")
    j.enregistrer([0.0, 0.0], [0.0, 0.0], 1.0)
    assert [l["phase"] for l in _lignes(j)] == ["DOE", "EFF"]


# --------------------------------------------------------------------------- #
# REMISE A ZERO ET REPRISE                                                     #
# --------------------------------------------------------------------------- #
def test_un_run_neuf_vide_le_journal(tmp_path):
    """Sinon les rounds du run precedent se melangent aux nouveaux."""
    t = _Trace()
    j = _journal(tmp_path, tracer=t)
    j.enregistrer([0.0, 0.0], [0.0, 0.0], 1.0)
    j.reinitialiser()
    assert _lignes(j) == []
    assert any("[POINT LOG] reset" in m for m in t)


def test_une_reprise_ne_vide_PAS_le_journal(tmp_path):
    """Les points du run precedent ont ete payes -- 466 s piece sur le
    Moulin Blanc. Ils restent, et une ligne dit ou la reprise commence."""
    j = _journal(tmp_path)
    j.enregistrer([0.0, 0.0], [0.0, 0.0], 1.0)
    j.marquer_reprise(2, n_total=13, n_eff=8)
    lignes = _lignes(j)
    assert len(lignes) == 2
    assert lignes[1] == {"phase": "_RESTART", "round": 2,
                         "n_total": 13, "n_eff": 8}


def test_la_reprise_estampille_les_points_suivants(tmp_path):
    j = _journal(tmp_path)
    j.marquer_reprise(2, n_total=13, n_eff=8)
    j.marquer("EFF")
    j.enregistrer([0.0, 0.0], [0.0, 0.0], 1.0)
    assert _lignes(j)[-1]["round"] == 2


# --------------------------------------------------------------------------- #
# ECRIRE N'EMPORTE JAMAIS LE RUN                                               #
# --------------------------------------------------------------------------- #
def test_un_chemin_impossible_est_signale_jamais_fatal(tmp_path):
    t = _Trace()
    j = _jp.JournalDesPoints(str(tmp_path / "absent" / "p.jsonl"), PARAMS,
                             tracer=t)
    j.enregistrer([0.0, 0.0], [0.0, 0.0], 1.0)
    assert any("[POINT LOG] append echoue" in m for m in t)


def test_une_coordonnee_non_convertible_est_signalee_jamais_fatale(tmp_path):
    """La MISE EN FORME est protegee, pas seulement l'ecriture. Un point qui
    vient de couter un appel solveur ne doit pas etre emporte par un `float`
    qui echoue."""
    t = _Trace()
    j = _journal(tmp_path, tracer=t)
    j.enregistrer(["pas un nombre", 0.0], [0.0, 0.0], 1.0)
    assert any("[POINT LOG] append echoue" in m for m in t)
    assert _lignes(j) == [], "rien n'a ete ecrit, et le run continue"


def test_le_journal_reste_relisible_apres_un_echec(tmp_path):
    """Une ligne ratee ne doit pas laisser un JSON tronque derriere elle."""
    j = _journal(tmp_path)
    j.enregistrer([1.0, 2.0], [3.0, 4.0], 1.0)
    j.enregistrer(["x", 0.0], [0.0, 0.0], 1.0)      # echoue
    j.enregistrer([5.0, 6.0], [7.0, 8.0], 2.0)
    lignes = _lignes(j)
    assert len(lignes) == 2
    assert lignes[0]["u_fc"] == 1.0 and lignes[1]["u_fc"] == 5.0


# --------------------------------------------------------------------------- #
# CE QUI NE DOIT PLUS ETRE DANS LES ETUDES                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", ["pure_flexion/AC3_pure_flexion.py",
                                    "Moulinblanc/AC3_moulinblanc.py"])
def test_les_etudes_ne_tiennent_plus_le_journal_elles_memes(script):
    s = io.open(os.path.join(_REPO, script), encoding="utf-8",
                errors="replace").read()
    for interdit in ("_point_log_phase", "_point_log_round", "_POINT_LOG_FILE",
                     "_append_point_log", '"lambda":'):
        assert interdit not in s, (
            "%s : le journal des points est revenu dans l'etude (%r)."
            % (script, interdit))
    assert "_journal_points.JournalDesPoints(" in s
