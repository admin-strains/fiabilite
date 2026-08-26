r"""
L'implementation ANALYTIQUE du contrat `solver/interface.py`.

Meme contrat que `digital_structure`, meme geometrie, meme parametrage -- mais
l'etat limite est ferme, donc l'appel coute des microsecondes au lieu de
plusieurs secondes, et ne demande ni licence, ni GPU, ni Python 3.10.

A QUOI CELA SERT
----------------
A tester la CHAINE, pas la physique. Plan d'experiences, metamodele PCK/GEPCK,
enrichissement EFF, FORM multimodal, tirage d'importance : ces 2 700 lignes ne
savent pas d'ou vient `g`. Jusqu'a la phase 5, on ne pouvait pourtant pas les
exercer sans Digital Structure -- et la mesure du 25/08/2026 a montre que la
chaine complete sur DS n'est meme pas reproductible d'un run a l'autre
(12,3 % d'etendue sur `Pf_IS`). Un etat limite ferme, lui, l'est au bit pres :
c'est la seule facon d'attribuer un ecart au code plutot qu'au solveur.

CE QU'ELLE CALCULE
-------------------
La section rectangulaire en flexion simple, branche « aciers plastifies »,
telle que la porte `flexion_claude` dans le script d'etude :

    M_R(fc, fy) = A.fy + B.fy^2 / fc      A = As.d / gamma_s
                                          B = -As^2.gamma_c / (2.b.gamma_s^2)
    g = (M_R - Med) / Med

La geometrie est lue dans les MEMES fichiers texte que le solveur reel --
`dsCad.txt` et `dsLoad.txt` -- qui ne demandent rien d'autre qu'un
`open()`. Les deux solveurs travaillent donc sur la meme section, ce qui rend
la comparaison legitime.

CE QU'ELLE N'EST PAS
---------------------
Ce n'est pas un modele de remplacement. La branche « beton ecrase » n'est pas
implementee, et le calcul a la rupture tridimensionnel ne se resume pas a une
formule de section. Hors du domaine plastifie, `evaluer` le DIT au lieu de
rendre un chiffre faux : c'est precisement le defaut que la phase 5 corrige
cote Digital Structure, il n'y a pas de raison de l'introduire ici.
"""

import math
import os
import re

from interface import Evaluation


def _parse(texte, nom):
    """Valeur d'une affectation `nom = <nombre>` dans un dsCad/dsLoad.

    Recopie de `AC3_pure_flexion._parse`.
    """
    m = re.search(r'(?m)^\s*%s\s*=\s*([\d.]+)' % re.escape(nom), texte)
    if m is None:
        raise ValueError("parametre %r introuvable dans le fichier modele" % nom)
    return float(m.group(1))


def lire_section(chemin_ds):
    """Geometrie et chargement, lus dans les fichiers texte du modele.

    Aucun appel a Digital Structure : `dsCad.txt` et `dsLoad.txt` sont du
    texte. C'est ce qui permet aux deux solveurs de partager la meme section.
    """
    with open(os.path.join(chemin_ds, 'dsCad.txt'), 'r') as fh:
        cad = fh.read()
    with open(os.path.join(chemin_ds, 'dsLoad.txt'), 'r') as fh:
        load = fh.read()

    b = _parse(cad, 'b')
    h = _parse(cad, 'h')
    L = _parse(cad, 'L')
    phi = _parse(cad, 'phi')
    gamma_c = _parse(cad, 'gamma_c')
    gamma_s = _parse(cad, 'gamma_s')

    n_bars = len(re.findall(r'REBAR\(', cad))
    As = n_bars * math.pi * (phi / 2e3) ** 2

    z_rebar = [float(v) for v in re.findall(
        r'pts\d+\.append\(POINT\([^,]+,\s*[^,]+,\s*([\d.]+)\)', cad)]
    if not z_rebar:
        raise ValueError("aucune position d'acier trouvee dans dsCad.txt")
    d = h / 2 + sum(z_rebar) / len(z_rebar)

    F = abs(float(re.search(r"Z='(-?[\d.]+)'", load).group(1)))
    return {"b": b, "h": h, "L": L, "As": As, "d": d, "n_bars": n_bars,
            "gamma_c": gamma_c, "gamma_s": gamma_s, "F": F, "Med": F * L}


class SolveurAnalytique:
    """Evalue g = (M_R - Med) / Med en forme fermee, sans Digital Structure."""

    #: module d'Young de l'acier et raccourcissement ultime du beton, tels que
    #: poses dans les scripts d'etude. Servent a situer la limite de
    #: plastification, pas au calcul de M_R.
    Es = 200000.0
    ecu = 0.0035

    def __init__(self, chemin_ds, params_names, dossier_etude=None, regions=None,
                 global_size=None, geo_min_approx=None, archiver=False,
                 verbeux=False, **ignores):
        """Accepte la meme signature que `SolveurDS` -- c'est le principe d'un
        contrat. `dossier_etude`, `global_size`, `geo_min_approx` et
        `archiver` decrivent un maillage : sans objet ici, ils sont acceptes
        et ignores plutot que de forcer l'appelant a savoir a qui il parle."""
        self.chemin_ds = chemin_ds
        self.params_names = tuple(params_names)
        self.verbeux = verbeux
        self.section = lire_section(chemin_ds)
        s = self.section
        self.A = s["As"] * s["d"] / s["gamma_s"]
        self.B = -s["As"] ** 2 * s["gamma_c"] / (2.0 * s["b"] * s["gamma_s"] ** 2)
        self.Med = s["Med"]
        self._appels = 0

        inconnues = set(self.params_names) - {"fc", "fy"}
        if inconnues:
            raise ValueError(
                "solveur analytique : variables %s non prevues. La forme "
                "fermee ne connait que fc et fy ; pour les autres, il faut "
                "Digital Structure." % sorted(inconnues))

    # ------------------------------------------------------------------ #
    @property
    def cout_par_appel(self) -> str:
        return "une formule fermee : quelques microsecondes"

    @property
    def nb_appels(self) -> int:
        return self._appels

    # ------------------------------------------------------------------ #
    def moment_resistant(self, fc, fy):
        return self.A * fy + self.B * fy ** 2 / fc

    def domaine_plastifie(self, fc, fy):
        """La branche « aciers plastifies » est-elle celle qui gouverne ?

        Condition classique : la hauteur de beton comprime necessaire a
        l'equilibre doit rester sous le pivot. En dessous, c'est le beton qui
        ecrase, et la formule ci-dessus surestime le moment resistant.
        """
        s = self.section
        x = s["As"] * fy / (s["gamma_s"] * 0.8 * s["b"] * fc / s["gamma_c"])
        x_lim = s["d"] * self.ecu / (self.ecu + fy / (s["gamma_s"] * self.Es))
        return x <= x_lim, x, x_lim

    # ------------------------------------------------------------------ #
    def evaluer(self, valeurs, sensibilite=True, etiquette=None) -> Evaluation:
        fc = float(valeurs["fc"])
        fy = float(valeurs["fy"])
        self._appels += 1

        plastifie, x, x_lim = self.domaine_plastifie(fc, fy)
        M_R = self.moment_resistant(fc, fy)
        alpha = M_R / self.Med
        g = alpha - 1.0

        grad_x = tuple([None] * len(self.params_names))
        if sensibilite:
            derivees = {
                "fc": -self.B * fy ** 2 / fc ** 2 / self.Med,
                "fy": (self.A + 2.0 * self.B * fy / fc) / self.Med,
            }
            grad_x = tuple(derivees[p] for p in self.params_names)

        diagnostic = {
            "solver_status": "OPTIMAL" if plastifie else "HORS_DOMAINE",
            "converged": True,
            "solverIterations": 0,
            "x_comprime": x,
            "x_limite": x_lim,
            "etiquette": etiquette,
            "t_mesh": 0.0,
            "t_solv": 0.0,
        }
        if not plastifie and self.verbeux:
            print("  [ANALYTIQUE] fc=%.3f fy=%.3f : hors du domaine plastifie "
                  "(x=%.4f > x_lim=%.4f). La forme fermee surestime M_R."
                  % (fc, fy, x, x_lim), flush=True)

        return Evaluation(g=g, alpha=alpha, grad_x=grad_x,
                          sain=plastifie, diagnostic=diagnostic)
