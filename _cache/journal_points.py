# -*- coding: utf-8 -*-
"""
Le journal des points : une ligne JSON par appel au solveur.

C'est la trace de ce qui a REELLEMENT ete calcule, dans l'ordre. Sur le
Moulin Blanc, un point coute 466 s : ce fichier est le seul endroit ou l'on
peut, apres coup, repondre a « combien d'appels, ou, et pour quel resultat ».

Il etait tenu par une fonction de quatorze lignes recopiee dans les deux
etudes, plus trois variables globales, plus deux ecritures directes ailleurs
dans le fichier -- huit endroits pour une seule notion.

CE QUE PORTE CHAQUE LIGNE
--------------------------
La phase (`DOE`, `EFF`, `HF`, `USTAR`), le round de re-enrichissement, les
coordonnees du point dans les DEUX espaces -- norme `u_*` et physique `x_*`
-- et le resultat sous ses deux noms : `g`, et `lambda = g + 1`.

Les deux noms comptent. L'etat limite du calcul a la rupture s'ecrit
`g = alpha - 1` ou `alpha` est le facteur de charge : `lambda` est donc le
facteur lui-meme, celui que l'ingenieur lit. `g` est ce que la fiabilite
manipule, `lambda` est ce que l'ouvrage subit.

ECRIRE NE DOIT JAMAIS EMPORTER LE RUN
--------------------------------------
Une ligne de journal perdue coute une ligne de journal. Un point perdu coute
466 s. Toute erreur -- mise en forme comprise, pas seulement l'ecriture --
est donc signalee et avalee.
"""

import json
import os


NOM_PAR_DEFAUT = "points_log.jsonl"


def _ecrire(message):
    print(message, flush=True)


def fichier_de(path_ds, nom=NOM_PAR_DEFAUT):
    """Le journal vit dans le `.ds` du modele, a cote de ce qu'il decrit."""
    return os.path.join(path_ds, nom)


class JournalDesPoints:
    """La trace des appels solveur d'un run.

    `phase` et `round` sont de l'ETAT : ils changent au fil du run et
    estampillent les lignes suivantes. C'etaient deux listes d'un seul
    element (`_point_log_phase = ["?"]`) -- l'idiome qu'on emploie pour muter
    une variable depuis une fermeture, faute d'objet.
    """

    def __init__(self, fichier, params_names, tracer=_ecrire):
        self.fichier = fichier
        self.params_names = list(params_names)
        self.phase = "?"
        self.round = 0
        self.tracer = tracer

    # ------------------------------------------------------------------ #
    def marquer(self, phase):
        """Estampille les lignes suivantes.

        Passee en rappel a la grille et au plan d'experiences, qui n'ont pas
        a connaitre le journal.
        """
        self.phase = phase

    def reinitialiser(self):
        """Vide le journal : un run NEUF ne prolonge pas la trace du
        precedent, sinon les rounds se melangent."""
        open(self.fichier, "w").close()
        self.tracer("[POINT LOG] reset -> %s" % self.fichier)

    def marquer_reprise(self, round_, n_total, n_eff):
        """Une ligne de separation, pour qu'on voie ou une reprise commence.

        Le journal n'est PAS vide dans ce cas : les points du run precedent
        ont ete payes, ils restent.
        """
        self.round = round_
        self._ecrire_ligne({"phase": "_RESTART", "round": round_,
                            "n_total": int(n_total), "n_eff": int(n_eff)})

    # ------------------------------------------------------------------ #
    def enregistrer(self, u, x, g, phase=None):
        """Un point calcule : ou il est, dans les deux espaces, et ce qu'il a
        rendu.

        `u` ou `x` peuvent manquer -- l'appelant ne connait pas toujours les
        deux -- et les colonnes absentes valent None plutot que de faire
        echouer l'ecriture.
        """
        try:
            u = list(u) if u is not None else []
            x = list(x) if x is not None else []
            ligne = {"phase": self.phase if phase is None else phase,
                     "round": self.round,
                     "g": None if g is None else float(g),
                     "lambda": None if g is None else float(g) + 1.0}
            for i, p in enumerate(self.params_names):
                ligne["u_%s" % p] = float(u[i]) if i < len(u) else None
                ligne["x_%s" % p] = float(x[i]) if i < len(x) else None
            self._ecrire_ligne(ligne)
        except Exception as e:
            # La MISE EN FORME est protegee, pas seulement l'ecriture : une
            # coordonnee non convertible ne doit pas emporter un point qui
            # vient de couter un appel solveur.
            self.tracer("[POINT LOG] append echoue (%s: %s)"
                        % (type(e).__name__, e))

    def _ecrire_ligne(self, ligne):
        with open(self.fichier, "a") as fh:
            fh.write(json.dumps(ligne) + "\n")
