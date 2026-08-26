r"""
Le contrat entre la chaine de fiabilite et ce qui evalue l'etat limite.

PHASE 5 du plan de nettoyage. Jusqu'ici, la chaine appelait Digital Structure
directement, depuis deux fonctions imbriquees dans chaque script AC. Ce
module definit la SEULE frontiere autorisee.

CE QUE LA MESURE A MONTRE
--------------------------
La surface de couplage etait minuscule et pourtant dupliquee quatre fois :

    run_one_SOL   le plan d'experiences        ~90 lignes
    run_HF        l'enrichissement et la HF    ~90 lignes
                  ... les deux, dans les deux scripts AC

Ces quatre exemplaires du meme appel avaient DIVERGE. `run_one_SOL` lisait la
taille de maille dans la configuration ; `run_HF` la codait en dur a 0.05,
avec `geometric_approximation_min` a "4". Or `run_HF` sert aux points
d'enrichissement EFF, qui rejoignent le plan d'experiences : regler
`global_size = 0.007` aurait entraine le metamodele sur des points calcules
avec DEUX MAILLAGES DIFFERENTS, sans un mot. Le defaut est dormant -- les
deux etudes sont a 0.05 -- mais le commentaire d'origine invitait justement a
changer ce reglage (« 0.05 = rapide FORM, 0.007 = tres fin »).

Une seule implementation ne peut plus diverger d'elle-meme.

CE QUI EST DANS LE CONTRAT, ET CE QUI N'Y EST PAS
-------------------------------------------------
Le solveur rend `g` et son gradient DANS L'ESPACE PHYSIQUE X. Le passage en
espace standard U est l'affaire de la fiabilite, pas du solveur : c'est la
transformation isoprobabiliste de la loi jointe, qui ne connait pas le
maillage. La frontiere passe donc la.

Le solveur rend aussi son ETAT DE SANTE. Les scripts AC lisaient
`Primal_bound` sans jamais regarder `converged` ni `solver_status` -- zero
occurrence dans les deux fichiers. Mesure du 25/08/2026 : a
`global_physical_size = 0.018`, le solveur sort NUMERICAL_ERROR avec un ecart
primal-dual de 16 % et alpha = 1.5197 au lieu de ~1.3188. Un tel point
entrait dans le plan d'experiences comme une evaluation valide. Le contrat
oblige desormais a recevoir cette information ; libre a l'appelant d'en faire
ce qu'il veut, mais plus de l'ignorer par omission.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Evaluation:
    """Ce qu'un solveur rend pour UN point de l'espace physique."""

    #: g = alpha - 1. Negatif = defaillance, par convention du calcul a la rupture.
    g: float
    #: le multiplicateur de rupture lui-meme (`Primal_bound`), avant decalage
    alpha: float
    #: dg/dx_i, un par variable, dans l'ordre de `params_names`. None si
    #: la sensibilite n'a pas ete demandee ou n'a pas ete rendue.
    grad_x: Tuple[Optional[float], ...] = ()
    #: False si le solveur n'a pas converge : le resultat n'est pas exploitable
    sain: bool = True
    #: tout ce que le solveur sait dire de lui-meme (statut, iterations,
    #: nombre de tetraedres, ecart primal-dual, durees). Journalise, jamais
    #: interprete par la chaine.
    diagnostic: dict = field(default_factory=dict)

    @property
    def gradient_complet(self) -> bool:
        """Vrai si toutes les composantes du gradient sont disponibles."""
        return bool(self.grad_x) and all(v is not None for v in self.grad_x)

    def exige_sain(self, contexte: str = "") -> "Evaluation":
        """Leve si le solveur n'a pas converge. A utiliser la ou un point faux
        contaminerait le metamodele."""
        if not self.sain:
            raise SolveurNonConverge(
                "%s : le solveur n'a pas converge (%s), alpha=%.6f. "
                "Ce point ne doit pas entrer dans le plan d'experiences."
                % (contexte or "evaluation", self.diagnostic.get("solver_status"),
                   self.alpha))
        return self


class SolveurNonConverge(RuntimeError):
    """Le solveur a rendu un resultat qu'il declare lui-meme non converge."""


class Solveur:
    """Contrat minimal. Deux implementations : `digital_structure`, `analytique`.

    Une implementation n'a pas a heriter de cette classe -- seule la signature
    compte. Elle est ecrite pour etre lue, et pour que `tests/` puisse verifier
    que les deux implementations repondent bien la meme chose.
    """

    #: noms des variables, dans l'ordre ou `grad_x` est rendu
    params_names: Sequence[str] = ()

    def evaluer(self, valeurs, sensibilite=True, etiquette=None) -> Evaluation:
        """Evalue l'etat limite au point `valeurs` (dict {nom: valeur physique}).

        `sensibilite` demande le gradient ; `etiquette` sert aux traces et a
        l'archivage des sorties, jamais au calcul.
        """
        raise NotImplementedError

    @property
    def cout_par_appel(self) -> str:
        """Ordre de grandeur, pour que l'appelant sache ce qu'il declenche."""
        return "inconnu"
