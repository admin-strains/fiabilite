r"""Le journal des points : l'artefact sur lequel repose le decoupage.

CE QU'IL DOIT GARANTIR
-----------------------
Le principe est : **ne persister que ce qui coute un appel au solveur, et
recalculer tout le reste**. Le journal des points est donc le seul bien
irremplacable d'une etude -- 592 appels sur le Moulin Blanc, ~76 heures.

Trois proprietes en decoulent, et ce sont elles qu'on teste :

1. **ajout seul** -- un point ecrit est acquis ; une interruption ne peut pas
   abimer ce qui precede. C'est ce qui manquait au plan d'experiences, dont le
   cache incremental etait ecrit puis JETE a la relecture ;
2. **signature** -- un journal produit sous un autre solveur, un autre
   maillage ou un autre domaine est refuse, pas melange. C'etait le defaut
   commun aux huit points de reprise du depot ;
3. **aucun gradient fabrique** -- `np.asarray([[None]], dtype=float)` rend
   `[[nan]]` sans rien dire ; les NaN se propageraient dans l'ajustement.

Aucune dependance lourde : ni OpenTURNS, ni Digital Structure.
"""

import json
import os
import sys

import numpy as np
import pytest

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, os.path.join(_REPO, "_etapes"))

import points as mod                                  # noqa: E402
from points import Point, JournalPoints               # noqa: E402

SIG = {"solveur_lineaire": "mumps", "global_size": 0.05}


def _pt(g=0.5, grad=(0.1, 0.2), origine="plan", sain=True, u=(1.0, 2.0)):
    return Point(u=u, x=(300.0, 400.0), g=g, grad_u=grad, origine=origine, sain=sain)


# --------------------------------------------------------------------- #
# le point lui-meme
# --------------------------------------------------------------------- #
def test_un_gradient_partiel_vaut_pas_de_gradient():
    """Un seul None suffit : un gradient a moitie connu n'est pas utilisable,
    et le completer par zero serait une invention."""
    assert Point(u=(0, 0), x=None, g=1.0, grad_u=(0.1, None)).grad_u is None
    assert not Point(u=(0, 0), x=None, g=1.0, grad_u=(0.1, None)).gradient_complet
    assert Point(u=(0, 0), x=None, g=1.0, grad_u=(0.1, 0.2)).gradient_complet


def test_une_origine_inconnue_est_refusee():
    """L'origine distingue ce qui a nourri le metamodele de ce qui n'a servi
    qu'a une figure -- distinction qu'aucun cache actuel ne fait."""
    with pytest.raises(ValueError, match="origine"):
        Point(u=(0, 0), x=None, g=1.0, origine="nimportequoi")


def test_aller_retour_dict():
    p = _pt(g=-0.25, grad=(1.5, -2.5), origine="enrichissement", sain=False)
    q = Point.depuis_dict(json.loads(json.dumps(p.en_dict())))
    assert (q.u, q.g, q.grad_u, q.origine, q.sain) == (p.u, p.g, p.grad_u, p.origine, p.sain)


# --------------------------------------------------------------------- #
# 1. ajout seul : survivre a une interruption
# --------------------------------------------------------------------- #
def test_chaque_point_est_ecrit_immediatement(tmp_path):
    """Un appel solveur coute de la dizaine de secondes a plusieurs minutes.
    Le perdre parce que le run s'arrete deux points plus loin n'est pas
    acceptable."""
    f = str(tmp_path / "etat" / "points.jsonl")
    j = JournalPoints(f, SIG)
    j.ajouter(_pt(g=1.0))
    assert os.path.exists(f), "le premier point doit etre sur le disque"
    j.ajouter(_pt(g=2.0))
    with open(f, encoding="utf-8") as fh:
        assert len(fh.read().splitlines()) == 3       # en-tete + 2 points


def test_une_ligne_tronquee_ne_perd_que_le_dernier_point(tmp_path):
    """LE scenario d'interruption : coupure pendant une ecriture. Tout ce qui
    precede doit survivre -- c'est l'interet de l'ajout seul face a un
    `json.dump` complet, qui laisserait un fichier illisible."""
    f = str(tmp_path / "points.jsonl")
    j = JournalPoints(f, SIG)
    for k in range(4):
        j.ajouter(_pt(g=float(k)))
    with open(f, "a", encoding="utf-8") as fh:
        fh.write('{"u": [1.0, 2.0], "g": 4.0, "gr')   # coupe net

    relu = JournalPoints.relire(f, SIG, tracer=None)
    assert len(relu) == 4
    assert [p.g for p in relu] == [0.0, 1.0, 2.0, 3.0]


def test_un_journal_absent_donne_un_journal_vide(tmp_path):
    """Ne jamais lever : un cache absent conduit a recalculer, pas a
    interrompre."""
    j = JournalPoints.relire(str(tmp_path / "rien.jsonl"), SIG, tracer=None)
    assert len(j) == 0


def test_un_fichier_illisible_ne_leve_pas(tmp_path):
    f = str(tmp_path / "points.jsonl")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("ceci n'est pas du json\nni ca\n")
    assert len(JournalPoints.relire(f, SIG, tracer=None)) == 0


# --------------------------------------------------------------------- #
# 2. signature : refuser un melange
# --------------------------------------------------------------------- #
def test_un_journal_d_un_autre_backend_est_refuse(tmp_path):
    """Le defaut commun aux huit points de reprise du depot."""
    f = str(tmp_path / "points.jsonl")
    j = JournalPoints(f, {"solveur_lineaire": "cudss"})
    j.ajouter(_pt())
    autre = JournalPoints.relire(f, {"solveur_lineaire": "mumps"}, tracer=None)
    assert len(autre) == 0


def test_un_journal_sans_signature_est_refuse(tmp_path):
    f = str(tmp_path / "points.jsonl")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"pas_de_signature": 1}) + "\n")
        fh.write(json.dumps(_pt().en_dict()) + "\n")
    assert len(JournalPoints.relire(f, SIG, tracer=None)) == 0


def test_le_refus_est_DIT(tmp_path):
    """Un cache ecarte en silence est aussi dangereux qu'un cache accepte a
    tort : dans les deux cas on ne sait pas ce qui a servi."""
    f = str(tmp_path / "points.jsonl")
    JournalPoints(f, {"solveur_lineaire": "cudss"}).ajouter(_pt())
    dits = []
    JournalPoints.relire(f, {"solveur_lineaire": "mumps"}, tracer=dits.append)
    assert dits and "solveur_lineaire" in dits[0]


def test_sans_signature_demandee_tout_est_relu(tmp_path):
    f = str(tmp_path / "points.jsonl")
    JournalPoints(f, SIG).ajouter(_pt())
    assert len(JournalPoints.relire(f, None, tracer=None)) == 1


# --------------------------------------------------------------------- #
# 3. aucun gradient fabrique
# --------------------------------------------------------------------- #
def test_les_tableaux_refusent_un_point_sans_gradient(tmp_path):
    """`np.asarray([[None]], dtype=float)` rend `[[nan]]` SANS RIEN DIRE."""
    j = JournalPoints(str(tmp_path / "p.jsonl"), SIG)
    j.ajouter(_pt(grad=(0.1, 0.2)))
    j.ajouter(_pt(grad=(None, None)))
    with pytest.raises(ValueError, match="sans gradient"):
        j.tableaux()


def test_les_tableaux_marchent_sur_les_points_filtres(tmp_path):
    j = JournalPoints(str(tmp_path / "p.jsonl"), SIG)
    j.ajouter(_pt(g=1.0, grad=(0.1, 0.2)))
    j.ajouter(_pt(g=2.0, grad=(None, None)))
    j.ajouter(_pt(g=3.0, grad=(0.5, 0.6)))
    u, g, grad = j.tableaux(j.selon(avec_gradient=True))
    assert u.shape == (2, 2) and g.shape == (2, 1) and grad.shape == (2, 2)
    assert list(g.ravel()) == [1.0, 3.0]
    assert bool(np.all(np.isfinite(grad)))


def test_le_journal_vide_donne_des_tableaux_vides(tmp_path):
    u, g, grad = JournalPoints(str(tmp_path / "p.jsonl"), SIG).tableaux()
    assert u.size == g.size == grad.size == 0


# --------------------------------------------------------------------- #
# selection et lisibilite
# --------------------------------------------------------------------- #
def test_on_distingue_ce_qui_a_nourri_le_metamodele(tmp_path):
    """Un point de GRILLE sert a une figure ; un point de PLAN nourrit le
    metamodele. Aucun cache actuel ne fait la difference."""
    j = JournalPoints(str(tmp_path / "p.jsonl"), SIG)
    j.ajouter(_pt(origine="plan"))
    j.ajouter(_pt(origine="enrichissement"))
    j.ajouter(_pt(origine="grille"))
    j.ajouter(_pt(origine="grille"))
    assert len(j.selon(origine="grille")) == 2
    assert len(j.selon(origine=("plan", "enrichissement"))) == 2


def test_on_retrouve_les_points_non_converges(tmp_path):
    j = JournalPoints(str(tmp_path / "p.jsonl"), SIG)
    j.ajouter(_pt(sain=True))
    j.ajouter(_pt(sain=False))
    assert len(j.selon(sain=False)) == 1


def test_le_resume_dit_l_essentiel(tmp_path):
    j = JournalPoints(str(tmp_path / "p.jsonl"), SIG)
    j.ajouter(_pt(origine="plan"))
    j.ajouter(_pt(origine="grille", grad=(None, None), sain=False))
    r = j.resume()
    assert "2 point(s)" in r and "grille=1" in r
    assert "1 sans gradient" in r and "1 non converge" in r


def test_le_module_reste_leger():
    """Il doit tourner en integration continue, sans licence."""
    src = open(os.path.join(_REPO, "_etapes", "points.py"), encoding="utf-8").read()
    for interdit in ("import openturns", "from openturns", "STRAINS",
                     "matplotlib", "sklearn", "smt"):
        assert interdit not in src, "_etapes/points.py importe %r" % interdit
