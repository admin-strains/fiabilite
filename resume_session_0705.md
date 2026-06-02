# Resume session 07/05 -- refactoring flexion_claude + FORM HF direct

---

## PARAMETRES GLOBAUX (etat actuel)

```python
Es = 200000        # MPa
ecu = 0.0035
eud = 0.045
gamma_c = 1.5
gamma_s = 1.15

params_names = ['fc', 'fy']
fcm, fym = 28, 550        # MPa (caracteristiques EC2)
cov_fc, cov_fy = 0.12, None
fc_otparams = (fcm, cov_fc)
fy_otparams = (fym, cov_fy)

n0 = 10                   # taille DOE LHS
do_KRG = False
do_GEK = False
do_HF  = True
print_ana = True
print_DOE = True

u1_min, u1_max = -3.0, 3.0
u2_min, u2_max = -3.0, 3.0
n_grid = 300
```

Distributions :
- fc : lognormale, fcm = fck+8 = 36 MPa, COV = 0.12 (classe C25, table JCSS)
- fy : normale, mu = fym + 1.645*SIGMA ≈ 599.6 MPa, sigma = SIGMA ≈ 30 MPa

---

## STRUCTURE DU CODE (apres refactoring de la session)

### class flexion_claude (lignes ~450-506)

Classe completement refactorisee par l'utilisatrice. Plus de parametres en argument -- lit directement dsCad.txt et dsLoad.txt dans __init__.

```python
class flexion_claude:
    def __init__(self):
        # Lecture dsCad/dsLoad -> b, h, L, phi, As, d, Med
        # Distributions OT -> T_inv, T
        # Constantes pivot B plastique :
        self.A  = As * d / gamma_s
        self.B  = - As**2 * gamma_c / (2 * b * gamma_s**2)
        self.C  = -Med
        # Limite plastique (calcul analytique) :
        self.Ap = 0.8*d*b / (As*gamma_c*Es*ecu)
        self.Bp = 0.8*b*d**2 / gamma_c
        K  = -2*self.C*self.Ap / self.Bp
        s  = ((0.8 + K) + np.sqrt((K + 2.8)**2 - 6.4)) / 2
        x1_lim_plast = (s**2 - 1) / (4*self.Ap)
        self.u1_lim_plast = self.T(ot.Point([x1_lim_plast, 0.0]))[0]
        # u1_lim_plast : valeur de u1 au seuil aciers non plastifies

    def u2p_LS(self, u1):
        # Remplace gp_pivotB. Retourne u2 sur la surface limite (pivot B plastique).
        x_point = self.T_inv(ot.Point([u1, 0.0]))
        x1 = x_point[0]                         # fc physique [MPa]
        a, b, c = self.B, self.A * x1, self.C * x1
        Delta = b**2 - 4 * a * c
        fy = (-b + Delta**0.5) / (2 * a)        # racine physique (< fy_grande_racine)
        return self.T(ot.Point([0.0, fy]))[1]   # u2 standard
```

**Note** : la formule `(-b + Delta**0.5) / (2*a)` avec a=B<0 donne bien la petite racine physique (aciers a la limite d'elasticite). La grande racine non physique serait `(-b - Delta**0.5) / (2*a)`.

**Delta > 0 toujours** : prouve par le calcul (condition mu < 1 toujours satisfaite en flexion RC standard).

---

### def print_visu_ana() (lignes ~509-539)

Fonction standalone (non integree a print_visu) qui trace la courbe analytique seule :
- Branche plastique : courbe u2p_LS(u1) de u1_lim_plast a u1_max
- Branche non plastique : segment vertical a u1 = u1_lim_plast de u2_lim a u2_max
- Point de raccord (u1_lim, u2_lim) marque en noir

Non appelee dans le main actuel -- a appeler explicitement si besoin.

---

### def build_DOE() (lignes ~394-440)

Modifiee pour fonctionner dans les deux cas :
- `do_HF = False` : appelle STRAINS, retourne (xt, yt, all_grad)
- `do_HF = True`  : genere uniquement le DOE LHS (pas d'appel STRAINS), retourne xt

```python
if not do_HF:
    # appelle STRAINS, calcule yt et all_grad
    return xt, yt, all_grad
return xt   # do_HF : juste le DOE pour multistart
```

---

### Bloc principal do_HF (lignes ~949-952)

```python
elif do_HF:
    xt = build_DOE()                   # DOE LHS n0=10 pts (pas d'appel STRAINS)
    g_ot_HF = ot.Function(HFFunction())
    event = FORM_event(g_ot_HF)

starting_points = np.vstack([xt, [[0.0, 0.0]]])   # 10 pts LHS + origine
best_result, best_sp = FORM_multistart(starting_points)
```

Le DOE sert uniquement de points de depart pour FORM_multistart (pas de surrogate).

---

### def print_visu (lignes ~818-906)

Integre la courbe analytique directement (plus de print_visu_claude) :
```python
if print_ana:
    calc = flexion_claude()
    u1_lim_a = calc.u1_lim_plast
    u2_lim_a = calc.u2p_LS(u1_lim_a)
    u1_g_a   = np.linspace(u1_lim_a, u1_max, n_grid)
    u2_g_a   = np.array([calc.u2p_LS(u) for u in u1_g_a])
    ax.plot(u1_g_a, u2_g_a, color='green', linestyle='-.', linewidth=2)
    ax.plot([u1_lim_a, u1_lim_a], [u2_lim_a, u2_max], color='green', ...)
    ax.plot(u1_lim_a, u2_lim_a, 'ko', ...)
```

---

## RESULTATS FORM (run du 07/05, do_HF=True, n0=10, F=0.08 MN)

- **beta = 1.758**
- **u* = [-1.287, -1.199]**
- 2 points de depart (LHS + [0,0]) -> meme resultat
- Med = F * L = 0.08 * 5 = 0.40 MN.m

---

## INVESTIGATION ot.Scalar (non pertinente selon utilisatrice)

Investigation menee pour expliquer pourquoi gp_pivotB / u2p_LS echouait.
Hypothese (ecartee) : T()[0] retourne un ot.Scalar incompatible avec ot.Point([...]).
**Conclusion utilisatrice : ce n'est pas un probleme.**