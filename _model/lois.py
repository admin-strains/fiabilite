"""
Lois de probabilite des variables de base.

Extrait de `AC3_pure_flexion.py` / `AC3_moulinblanc.py`, ou ces sept
fonctions etaient definies a l'INTERIEUR de `if __name__ == '__main__':`,
donc ni importables ni testables -- et recopiees a l'identique dans les deux
scripts (comparaison caractere par caractere : identiques).

PHASE 3 du plan de nettoyage. Les corps sont repris VERBATIM, a deux
exceptions signalees en commentaire : les variables libres `SIGMA` et
`PARAM_CONFIG` deviennent des parametres explicites. C'etait le but --
une fonction qui se ferme sur l'espace de noms de `main` ne peut pas etre
appelee depuis un test.

L'equivalence avec les originaux est verifiee par
`tests/test_80_extraction_lois.py`, qui recupere les fonctions encore
imbriquees dans les scripts AC et compare les distributions produites.

Reference : JCSS Probabilistic Model Code.
"""

import numpy as np
import openturns as ot

# --------------------------------------------------------------------------- #
# Dispersion de la limite d'elasticite de l'acier (JCSS).
# Etait une variable libre `SIGMA` de main :
#     SIGMA_11, SIGMA_12, SIGMA_13 = 19.0, 22.0, 8.0
#     SIGMA = np.sqrt(SIGMA_11**2 + SIGMA_12**2 + SIGMA_13**2)
# Les trois composantes sont conservees, nommees, et la somme quadratique
# reste identique au chiffre pres.
# --------------------------------------------------------------------------- #
SIGMA_ACIER_LOT = 19.0        #: variabilite entre lots
SIGMA_ACIER_BARRE = 22.0      #: variabilite entre barres d'un meme lot
SIGMA_ACIER_ESSAI = 8.0       #: incertitude de mesure

SIGMA_ACIER_JCSS = float(np.sqrt(SIGMA_ACIER_LOT ** 2
                                 + SIGMA_ACIER_BARRE ** 2
                                 + SIGMA_ACIER_ESSAI ** 2))   # ~30,15 MPa


def loi_fy(fym, cov=None, sigma_defaut=SIGMA_ACIER_JCSS):
    """Limite d'elasticite de l'acier -- Normale (JCSS).

    `sigma_defaut` remplace la variable libre `SIGMA` de main.
    La valeur par defaut reproduit exactement le comportement d'origine.
    """
    if cov is not None:
        sig_ec = cov * fym
    else:
        sig_ec = sigma_defaut

    dist = ot.Normal(fym, sig_ec)
    return dist

def loi_fc(fcm, cov=None):
    COV_TABLE = {"C15": 0.14, "C25": 0.12, "C35": 0.09, "C45": 0.07}
    fck_eq = fcm - 8.0
    classe = min(COV_TABLE, key=lambda c: abs(int(c[1:]) - fck_eq))
    v = cov if cov is not None else COV_TABLE[classe]

    sigma_ln = np.sqrt(np.log(1 + v**2))
    mu_ln    = np.log(fcm) - 0.5 * sigma_ln**2

    dist = ot.LogNormal(mu_ln, sigma_ln, 0.0)
    return dist

def loi_F_permanente(Fm, cov=None):
    """Charge permanente (poids propre). JCSS PMC 2.1.
    Distribution : Normale. Moyenne = nominale. COV par defaut = 0.05 (beton ordinaire).
    COV recommandes : 0.02-0.04 (acier), 0.05 (beton), 0.08-0.10 (avec non-structuraux), 0.10 (bois)."""
    if cov is None:
        cov = 0.05
    return ot.Normal(Fm, cov * Fm)

def loi_F_exploitation(Fm, cov=None, usage='office'):
    """Charge d'exploitation soutenue. JCSS PMC 2.2, Table 2.2.1.
    Distribution : Gamma. Parametres selon la categorie d'usage.
    Si cov est fourni, utilise Fm et cov directement. Sinon, utilise les parametres JCSS.
    Si cov=None et usage=None, utilise 'office' par defaut."""
    PARAMS = {
        'office':     {'mq': 0.5, 'sv': 0.3, 'su': 0.6},
        'residence':  {'mq': 0.3, 'sv': 0.15, 'su': 0.3},
        'hotel':      {'mq': 0.3, 'sv': 0.05, 'su': 0.1},
        'hospital':   {'mq': 0.4, 'sv': 0.3, 'su': 0.6},
        'laboratory': {'mq': 0.7, 'sv': 0.4, 'su': 0.8},
        'library':    {'mq': 1.7, 'sv': 0.5, 'su': 1.0},
        'classroom':  {'mq': 0.6, 'sv': 0.15, 'su': 0.4},
        'retail':     {'mq': 0.9, 'sv': 0.6, 'su': 1.6},
        'storage':    {'mq': 3.5, 'sv': 2.5, 'su': 6.9},
        'industrial_light':  {'mq': 1.0, 'sv': 1.0, 'su': 2.8},
        'industrial_heavy':  {'mq': 3.0, 'sv': 1.5, 'su': 4.1},
    }
    if cov is not None:
        sigma = cov * Fm
    else:
        p = PARAMS.get(usage, PARAMS['office'])
        sigma = np.sqrt(p['sv']**2 + p['su']**2)
        Fm = p['mq']
    k = (Fm / sigma)**2
    lam = Fm / sigma**2
    return ot.Gamma(k, lam, 0.0)

def loi_F_intermittente(usage='office'):
    """Charge d'exploitation intermittente. JCSS PMC 2.2, Table 2.2.1.
    Distribution : Exponentielle (sigma = moyenne).
    Si usage=None, utilise 'office' par defaut."""
    PARAMS = {
        'office':     {'mp': 0.2},
        'residence':  {'mp': 0.3},
        'hotel':      {'mp': 0.2},
        'hospital':   {'mp': 0.2},
        'classroom':  {'mp': 0.5},
        'retail':     {'mp': 0.4},
        'crowd':      {'mp': 1.25},
    }
    p = PARAMS.get(usage, PARAMS['office'])
    return ot.Exponential(1.0 / p['mp'], 0.0)

def loi_uni_approx(a, b, alpha=0.5):
    """Loi uniforme approchee (fenetre de Tukey normalisee).
    Support [a, b]. alpha=0 -> uniforme exacte, alpha=1 -> Hann."""

    class TukeyDistribution(ot.PythonDistribution):
        def __init__(self, a, b, alpha):
            super().__init__(1)
            self.a = float(a)
            self.b = float(b)
            self.alpha = float(alpha)
            self.L = self.b - self.a
            self.C = 1.0 - self.alpha / 2.0  # integrale de w sur [0,1]

        def getRange(self):
            return ot.Interval([self.a], [self.b])

        def computePDF(self, X):
            x = X[0]
            if x < self.a or x > self.b:
                return 0.0
            t = (x - self.a) / self.L          # t in [0, 1]
            al = self.alpha
            if al <= 0.0:
                w = 1.0
            elif t < al / 2.0:
                w = 0.5 * (1.0 + np.cos(2.0 * np.pi / al * (t - al / 2.0)))
            elif t > 1.0 - al / 2.0:
                w = 0.5 * (1.0 + np.cos(2.0 * np.pi / al * (t - 1.0 + al / 2.0)))
            else:
                w = 1.0
            return w / (self.L * self.C)

        def computeCDF(self, X):
            x = X[0]
            if x <= self.a:
                return 0.0
            if x >= self.b:
                return 1.0
            t = (x - self.a) / self.L
            al = self.alpha
            if al <= 0.0:
                return t
            if t <= al / 2.0:
                F = t / 2.0 + al / (4.0 * np.pi) * np.sin(2.0 * np.pi / al * (t - al / 2.0))
            elif t <= 1.0 - al / 2.0:
                F = t - al / 4.0
            else:
                F = (1.0 - 3.0 * al / 4.0
                     + (t - 1.0 + al / 2.0) / 2.0
                     + al / (4.0 * np.pi) * np.sin(2.0 * np.pi / al * (t - 1.0 + al / 2.0)))
            return F / self.C

        def getMean(self):
            return [(self.a + self.b) / 2.0]

        def computeScalarQuantile(self, p, tail=False):
            if tail:
                p = 1.0 - p
            al = self.alpha
            if al <= 0.0:
                return self.a + p * self.L
            F_left  = (al / 4.0) / self.C
            F_right = (1.0 - 3.0 * al / 4.0) / self.C
            if p <= F_left:
                t = al / 4.0
                for _ in range(20):
                    F = t / 2.0 + al / (4.0 * np.pi) * np.sin(2.0 * np.pi / al * (t - al / 2.0))
                    f = 0.5 * (1.0 + np.cos(2.0 * np.pi / al * (t - al / 2.0)))
                    t -= (F / self.C - p) / (f / self.C)
                    t = max(0.0, min(t, al / 2.0))
            elif p <= F_right:
                t = p * self.C + al / 4.0
            else:
                t = 1.0 - al / 4.0
                for _ in range(20):
                    F = (1.0 - 3.0 * al / 4.0
                         + (t - 1.0 + al / 2.0) / 2.0
                         + al / (4.0 * np.pi) * np.sin(2.0 * np.pi / al * (t - 1.0 + al / 2.0)))
                    f = 0.5 * (1.0 + np.cos(2.0 * np.pi / al * (t - 1.0 + al / 2.0)))
                    t -= (F / self.C - p) / (f / self.C)
                    t = max(1.0 - al / 2.0, min(t, 1.0))
            return self.a + t * self.L

        def isContinuous(self):
            return True

    return ot.Distribution(TukeyDistribution(a, b, alpha))

def dist_jointe(param_config, params_names):
    """Loi jointe des variables de base, dans l'ordre de `params_names`.

    `param_config` et `params_names` etaient des variables libres de main.
    """
    return ot.JointDistribution([param_config[p]['loi'](*param_config[p]['args'])
                                 for p in params_names])


# --------------------------------------------------------------------------- #
# LES LOIS, PAR LEUR NOM                                                      #
# --------------------------------------------------------------------------- #
#: Ce que le fichier `.toml` d'une etude peut nommer dans `loi = "..."`.
#:
#: POURQUOI UN REGISTRE, ET PAS UN `getattr`
#: Un `getattr(lois, "loi_" + nom)` accepterait n'importe quel nom de ce
#: module -- `dist_jointe`, `SIGMA_ACIER_LOT` -- et rendrait un objet que le
#: metamodele appellerait sans savoir ce qu'il est. Un registre dit
#: exactement ce qui est declarable, et le refus peut lister les choix.
#:
#: `loi_F_intermittente` n'y figure pas : elle ne prend pas de moyenne mais
#: un `usage`, et aucune etude ne l'emploie. L'ajouter quand une le fera.
LOIS = {
    "fy": loi_fy,
    "fc": loi_fc,
    "F_permanente": loi_F_permanente,
    "F_exploitation": loi_F_exploitation,
    "uni_approx": loi_uni_approx,
}


def loi_nommee(nom):
    """La loi que designe `nom`, ou un refus qui dit les choix possibles.

    C'est le point ou une faute de frappe dans un fichier d'etude doit
    s'arreter. Sans lui, `loi = "fyd"` aurait passe la lecture du TOML et
    plante -- ou pire, silencieusement fait autre chose -- au premier tirage.
    """
    if nom not in LOIS:
        raise ValueError(
            "loi %r inconnue. Declarables : %s.\n"
            "Les lois vivent dans `_model/lois.py` ; en ajouter une consiste "
            "a l'ecrire puis a la nommer dans `LOIS`."
            % (nom, ", ".join(sorted(LOIS))))
    return LOIS[nom]
