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

CE QUE CE MODULE NE TRANCHE PAS
--------------------------------
Agnes, 27/08/2026 : « on n'est pas au clair sur nos criteres de convergence ».
Ce module ne les change donc PAS -- il les rassemble, les nomme et les rend
testables, pour qu'on puisse ensuite en discuter sur pieces.

Deux asymetries transcrites telles quelles, et qui meritent une decision :

1. **Les historiques ne sont pas remplis de la meme facon selon le critere.**
   En mode `BB`, `hist_BS` ne recoit rien ; en mode `BS`, `hist_BB` ne recoit
   rien. La courbe de convergence tracee en fin de run montre donc des
   choses differentes selon le critere choisi.
2. **En mode `both`, les compteurs BB et BS ne sont jamais mis a jour** --
   seul `both` l'est. En mode `at_least_one`, les trois le sont. Le bilan de
   fin de run affiche pourtant les trois compteurs dans les deux cas.

Ni l'une ni l'autre n'est forcement un defaut ; aucune n'etait ecrite.
"""


def _ecrire(message):
    print(message, flush=True)


#: Combien de fois de suite un critere doit etre satisfait pour arreter.
REPETITIONS = {"BB": 3, "BS": 3, "both": 2}

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
        """
        if self.critere not in UTILISE_BB:
            return
        self.hist_BB.append(ratio_bb)
        if self.critere in ("BB", "at_least_one") \
                and ratio_bb is not None and ratio_bb < self.tol_BB:
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
        if self.n_BS or self.n_BB:
            self.tracer("  [RESTART] compteurs repris : count_valid_BB=%d  "
                        "count_valid_BS=%d" % (self.n_BB, self.n_BS))
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
        """Met a jour les compteurs pour une iteration, et remplit les
        historiques.

        Retourne `(ratio_bb, ratio_bs)` -- les deux valeurs retenues, dont
        l'une peut etre None selon le critere.
        """
        if self.critere == "BB":
            self._compter_bb(ratio_bb)
            self.hist_BB.append(ratio_bb)
            return ratio_bb, None

        if self.critere == "BS":
            ratio_bs = self._ratio_bs(beta, beta_precedent, prefixe, "")
            if ratio_bs is None:
                self.n_BS = 0
            else:
                self._compter_bs(ratio_bs)
            self.hist_BS.append(ratio_bs)
            return None, ratio_bs

        if self.critere == "both":
            ratio_bs = self._ratio_bs(beta, beta_precedent, prefixe, " both")
            self.n_both = (self.n_both + 1) if self._les_deux(ratio_bb, ratio_bs) else 0
            self.hist_BB.append(ratio_bb)
            self.hist_BS.append(ratio_bs)
            return ratio_bb, ratio_bs

        if self.critere == "at_least_one":
            ratio_bs = self._ratio_bs(beta, beta_precedent, prefixe, " alo")
            self._compter_bb(ratio_bb)
            self._compter_bs(ratio_bs)
            self.n_both = (self.n_both + 1) if self._les_deux(ratio_bb, ratio_bs) else 0
            self.hist_BB.append(ratio_bb)
            self.hist_BS.append(ratio_bs)
            return ratio_bb, ratio_bs

        return ratio_bb, None

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

        LE BILAN AFFICHE TROIS COMPTEURS, MAIS EN MODE `both` DEUX D'ENTRE
        EUX NE BOUGENT JAMAIS -- `n_BB` et `n_BS` restent a zero pendant que
        seul `n_both` est tenu. De meme, en mode `BB` l'historique BS reste
        vide, et la ligne correspondante ne s'imprime pas. Ces deux
        asymetries sont DECRITES, pas corrigees : Agnes, 27/08/2026, « on
        n'est pas au clair sur nos criteres de convergence ». Les changer
        avant cet arbitrage reviendrait a trancher a sa place.

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
