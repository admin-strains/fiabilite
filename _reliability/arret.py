r"""Quand arreter d'enrichir : les quatre criteres, et ce qu'ils ne disent pas.

LE PROBLEME
------------
Le critere EFF dit ou ajouter un point, jamais quand s'arreter. Un
metamodele peut coller a l'etat limite partout sauf la ou beta se joue ; a
l'inverse, il peut donner un beta stable alors qu'il reste tres incertain
loin du point de conception. Les deux mesures ci-dessous repondent chacune a
une moitie de la question :

* **BB** -- ce que l'incertitude du metamodele coute EN FIABILITE :
  `|beta(g + 2 sigma) - beta(g - 2 sigma)| / |beta|`. Tant qu'il est large,
  deux surfaces egalement plausibles donnent deux reponses.
* **BS** -- la stabilite de beta d'une iteration a l'autre :
  `|beta - beta_precedent| / |beta|`. Il dit que le calcul ne bouge plus ; il
  ne dit pas qu'il est juste.

Un critere doit etre satisfait TROIS fois de suite (deux pour `both`) : une
seule iteration stable ne prouve rien, l'enrichissement peut traverser un
palier.

MESURER N'EST PAS DECIDER -- decision d'Agnes, 28/08/2026
----------------------------------------------------------
Ce module transcrivait quatre branches quasi identiques, une par critere,
chacune ne remplissant que ce dont SON critere se servait. Deux asymetries en
resultaient, portees a l'arbitrage et tranchees :

1. **Tout ce qui est PAYE est ENREGISTRE.** Le ratio BS ne coute rien de plus
   que `beta`, calcule a chaque iteration de toute facon : il est donc
   toujours mesurable, et desormais toujours enregistre. Le ratio BB, lui,
   coute un encadrement `g +/- 2 sigma` -- trois FORM+IS de plus par
   iteration. Quand il n'est pas calcule il vaut None ; mais quand il l'est
   -- parce que le critere en a besoin, ou parce que `print_Pf` l'a demande
   pour les courbes de Pf -- il est range, quel que soit le critere. Avant,
   le mode `BS` avec `print_Pf` payait ce ratio et le jetait.

2. **Les trois compteurs sont tenus dans tous les modes.** En mode `both`,
   `n_BB` et `n_BS` restaient a zero pendant tout un run : le bilan
   affichait `count_valid_BB=0 count_valid_BS=0` alors que les deux criteres
   pouvaient etre satisfaits a chaque iteration. Ils sont desormais tenus
   comme dans `at_least_one`.

Ce que le critere commande, et lui seul : L'ARRET (`continuer`). Aucune
decision d'arret ne change -- chaque mode lit exactement le compteur qu'il
lisait deja. Ce qui change, c'est ce qui est OBSERVE et RAPPORTE.

Consequence : `reprendre_depuis_historique` recompte aussi `n_both`. Sans
cela, une reprise en mode `both` repartait de zero et repayait des appels
solveur pour reprouver ce qui l'etait deja -- exactement le defaut que cette
methode existe pour fermer, mais pour le troisieme compteur.
"""


def _ecrire(message):
    print(message, flush=True)


#: Combien de fois de suite un critere doit etre satisfait pour arreter.
REPETITIONS = {"BB": 3, "BS": 3, "both": 2}

#: Ce que le journal ecrit apres le ratio BS, pour dire quel critere est en
#: vigueur. Les libelles sont ceux d'origine : ils rendent les journaux de
#: deux runs comparables.
SUFFIXE_JOURNAL = {"BB": " bb", "BS": "", "both": " both",
                   "at_least_one": " alo"}

#: Quels criteres ont besoin de l'encadrement `g +/- 2 sigma` -- qui coute
#: DEUX FORM+IS de plus par iteration.
UTILISE_BB = ("BB", "both", "at_least_one")
UTILISE_BS = ("BS", "both", "at_least_one")


class ArretEFF:
    """Les compteurs de convergence d'un enrichissement.

    `hist_BB` et `hist_BS` sont les listes que l'etude trace en fin de run :
    elles sont remplies ici, au moment ou la decision se prend.
    """

    def __init__(self, critere, tol_BB, tol_BS, n_max_points, tol_EFF,
                 hist_BB=None, hist_BS=None, tracer=_ecrire):
        self.critere = critere
        self.tol_BB = tol_BB
        self.tol_BS = tol_BS
        self.n_max_points = n_max_points
        self.tol_EFF = tol_EFF
        # CONTRAT : ces deux listes sont PARTAGEES avec l'appelant, qui les
        # relit pour le dump de reprise et pour la courbe de convergence.
        # L'appelant doit donc les vider EN PLACE (`del l[:]`), jamais les
        # rebinder : `hist = []` creerait une nouvelle liste et laisserait
        # cet objet ecrire dans celle que plus personne ne lit. C'est arrive
        # le 27/08/2026 -- les ratios BB et BS ont disparu du bilan de fin de
        # run, et 623 tests verts ne l'ont pas vu. Seule la chaine analytique,
        # journal compare LIGNE A LIGNE, l'a attrape.
        self.hist_BB = hist_BB if hist_BB is not None else []
        self.hist_BS = hist_BS if hist_BS is not None else []
        self.tracer = tracer
        self.n_BB = 0
        self.n_BS = 0
        self.n_both = 0

    # ------------------------------------------------------------------ #
    @property
    def a_besoin_de_l_encadrement(self):
        """L'encadrement `g +/- 2 sigma` coute deux FORM+IS par iteration."""
        return self.critere in UTILISE_BB

    def amorcer(self, ratio_bb):
        """Le ratio BB mesure sur le plan INITIAL, avant tout enrichissement.

        Il compte deja pour une iteration valide : un plan qui part deja
        convergent ne doit pas payer trois iterations pour le prouver.

        Appelee seulement quand l'encadrement initial a ete calcule -- donc
        quand ce ratio a ete PAYE. Il est alors range quel que soit le
        critere, et il n'a pas de contrepartie BS : a la premiere mesure il
        n'existe pas d'iteration precedente. `hist_BB` porte donc une entree
        de plus que `hist_BS`, celle du plan initial ; les courbes de
        convergence sont alignees a droite pour cette raison.
        """
        self.hist_BB.append(ratio_bb)
        if ratio_bb is not None and ratio_bb < self.tol_BB:
            self.n_BB = 1

    def reprendre_depuis_historique(self):
        """Recompte les iterations valides consecutives depuis les historiques.

        Sans cela, une reprise d'enrichissement repart de zero compteur : un
        run interrompu apres deux iterations valides sur trois en redemande
        trois, soit deux appels solveur pour rien -- une demi-heure sur le
        Moulin Blanc.

        Cette reprise n'existait que d'un cote (`AC3_moulinblanc.py`), et sa
        boucle comparait `_v < tol` SANS ecarter les `None` que les
        historiques contiennent : une reprise apres une iteration ou le FORM
        avait echoue levait `TypeError`.
        """
        self.n_BS = _valides_consecutifs(self.hist_BS, self.tol_BS)
        self.n_BB = _valides_consecutifs(self.hist_BB, self.tol_BB)
        # `n_both` aussi (28/08/2026) : sans lui, une reprise en mode `both`
        # repartait de zero et repayait des appels solveur pour reprouver ce
        # qui l'etait deja -- le defaut meme que cette methode ferme, mais
        # pour le troisieme compteur. Les historiques sont alignes a DROITE :
        # `hist_BB` peut porter une entree de plus, celle du plan initial.
        n = min(len(self.hist_BB), len(self.hist_BS))
        self.n_both = _valides_consecutifs_deux(
            list(self.hist_BB)[len(self.hist_BB) - n:],
            list(self.hist_BS)[len(self.hist_BS) - n:],
            self.tol_BB, self.tol_BS)
        if self.n_BS or self.n_BB or self.n_both:
            self.tracer("  [RESTART] compteurs repris : count_valid_BB=%d  "
                        "count_valid_BS=%d  count_valid_both=%d"
                        % (self.n_BB, self.n_BS, self.n_both))
        return self.n_BB, self.n_BS

    def continuer(self, n_points, valeur_eff):
        """Faut-il une iteration de plus ?

        Trois raisons d'arreter, dans cet ordre : le budget de points, le
        critere EFF lui-meme, et la convergence de beta.
        """
        if n_points >= self.n_max_points:
            return False
        if abs(valeur_eff) <= self.tol_EFF:
            return False
        if self.critere == "BB":
            return self.n_BB < REPETITIONS["BB"]
        if self.critere == "BS":
            return self.n_BS < REPETITIONS["BS"]
        if self.critere == "both":
            return self.n_both < REPETITIONS["both"]
        if self.critere == "at_least_one":
            return not (self.n_BB >= REPETITIONS["BB"]
                        or self.n_BS >= REPETITIONS["BS"]
                        or self.n_both >= REPETITIONS["both"])
        return True

    # ------------------------------------------------------------------ #
    def _ratio_bs(self, beta, beta_precedent, prefixe, suffixe):
        """`|beta - beta_precedent| / |beta|`, ou None si incalculable.

        Incalculable veut dire : pas de beta courant (le FORM a echoue), pas
        d'iteration precedente, ou beta nul.
        """
        if beta is None or beta_precedent is None or beta == 0:
            return None
        r = abs(beta - beta_precedent) / abs(beta)
        self.tracer("  [%s%s] |beta_IS - beta_IS_prec| / beta_IS = %.4f"
                    % (prefixe, suffixe, r))
        return r

    def enregistrer(self, ratio_bb, beta, beta_precedent, prefixe=""):
        """Mesure une iteration : les deux ratios, les trois compteurs, les
        deux historiques.

        UNE SEULE VOIE, quel que soit le critere -- c'etait quatre branches
        quasi identiques, chacune ne remplissant que ce dont son critere se
        servait. Le critere ne commande plus que L'ARRET (`continuer`).

        `ratio_bb` peut valoir None : il demande un encadrement que seuls
        certains modes paient. `ratio_bs` est calcule ici, parce qu'il ne
        coute rien de plus que `beta`.

        Retourne `(ratio_bb, ratio_bs)`.
        """
        ratio_bs = self._ratio_bs(beta, beta_precedent, prefixe,
                                  SUFFIXE_JOURNAL.get(self.critere, ""))
        self._compter_bb(ratio_bb)
        self._compter_bs(ratio_bs)
        self.n_both = (self.n_both + 1) if self._les_deux(ratio_bb, ratio_bs) else 0
        self.hist_BB.append(ratio_bb)
        self.hist_BS.append(ratio_bs)
        return ratio_bb, ratio_bs

    def _compter_bb(self, ratio):
        self.n_BB = (self.n_BB + 1) if (ratio is not None
                                        and ratio < self.tol_BB) else 0

    def _compter_bs(self, ratio):
        self.n_BS = (self.n_BS + 1) if (ratio is not None
                                        and ratio < self.tol_BS) else 0

    def _les_deux(self, ratio_bb, ratio_bs):
        return (ratio_bb is not None and ratio_bb < self.tol_BB
                and ratio_bs is not None and ratio_bs < self.tol_BS)

    # ------------------------------------------------------------------ #
    def bilan(self, eff_final, n_points_ajoutes):
        """Ce qui a arrete l'enrichissement, et sur quel historique.

        LES TROIS COMPTEURS SONT TENUS DANS TOUS LES MODES depuis le
        28/08/2026, et les deux historiques sont remplis des que leur ratio
        est disponible. Le bilan dit donc ce qui s'est passe, et pas
        seulement ce que le critere en vigueur a regarde : un run en mode
        `both` affichait `count_valid_BB=0 count_valid_BS=0` alors que les
        deux criteres pouvaient etre satisfaits a chaque iteration.

        `?` comme raison n'est pas un defaut d'affichage : c'est un run
        arrete par le budget de points, ni par EFF ni par la convergence.
        Le distinguer compte -- un plafond atteint ne prouve rien.
        """
        if abs(eff_final) <= self.tol_EFF:
            raison = "EFF"
        else:
            bb, bs, both = self.raisons()
            raison = ("BB (3 iter valides)" if bb else
                      "BS (3 iter valides)" if bs else
                      "both (2 iter valides)" if both else "?")
        self.tracer("  EFF converge [%s] : EFF(u_opt)=%.4f  count_valid_BB=%d"
                    "  count_valid_BS=%d  count_valid_both=%d  (%d point(s) "
                    "ajoutes)" % (raison, eff_final, self.n_BB, self.n_BS,
                                  self.n_both, n_points_ajoutes))
        arrondi = lambda lst: [round(r, 4) if r is not None else None for r in lst]
        if self.hist_BB:
            self.tracer("  [historique ratio BB] %s  tol=%s"
                        % (arrondi(self.hist_BB), self.tol_BB))
        if self.hist_BS:
            self.tracer("  [historique ratio BS] %s  tol=%s"
                        % (arrondi(self.hist_BS), self.tol_BS))
        return raison

    def raisons(self):
        """`(BB, BS, both)` -- lequel des criteres a effectivement arrete.

        Un run qui s'arrete sur le budget de points ou sur EFF rend trois
        `False` : ce n'est pas la convergence qui a decide, et le bilan doit
        le dire.
        """
        return (self.n_BB >= REPETITIONS["BB"] and self.critere in ("BB", "at_least_one"),
                self.n_BS >= REPETITIONS["BS"] and self.critere in ("BS", "at_least_one"),
                self.n_both >= REPETITIONS["both"] and self.critere in ("both", "at_least_one"))


def _valides_consecutifs(historique, tolerance):
    """Combien d'iterations valides de suite, en partant de la fin.

    Un `None` -- iteration ou le FORM a echoue -- ARRETE le compte : on ne
    sait pas si elle etait valide, et la supposer valide raccourcirait
    l'enrichissement sur une ignorance.
    """
    n = 0
    for valeur in reversed(list(historique or ())):
        if valeur is None or valeur >= tolerance:
            break
        n += 1
    return n


def _valides_consecutifs_deux(hist_a, hist_b, tol_a, tol_b):
    """Combien d'iterations de suite ou LES DEUX ratios sont valides.

    Le compteur du mode `both`, recompute depuis les historiques. Un `None`
    d'un cote ou de l'autre arrete le compte, pour la meme raison que dans
    `_valides_consecutifs` : on ne sait pas, et supposer raccourcirait
    l'enrichissement sur une ignorance.
    """
    n = 0
    for a, b in zip(reversed(list(hist_a or ())), reversed(list(hist_b or ()))):
        if a is None or b is None or a >= tol_a or b >= tol_b:
            break
        n += 1
    return n
