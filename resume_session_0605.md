# Resume session 06/05 -- nouvelles fonctions analytiques et visu

## Contexte
La methode `print_ana` de `flexion_simple` produisait une courbe hors de la fenetre standard
(u1 in [18,44], u2 in [0,16000]) a cause de 4 bugs. Une nouvelle classe `flexion_claude` a ete
creee pour test independant. Le run du 06/05 a 15h59 montre la courbe verte dans la bonne fenetre.

Resultats FORM du run 0605_1559 (inchanges vs sessions precedentes) :
- beta = 2.1172, Pf = 1.7121e-02, u* = [-1.3805, -1.6053], Imp = [0.4251, 0.5749]

---

## VERSION UTILISATRICE (originale, dans flexion_simple -- commentee)

### gp_pivotB (version utilisatrice -- BUGGUEE)
```python
def gp_pivotB(self, u1):
    x_point = self.T_inv(ot.Point([u1, 0.0]))
    x1 = x_point[0]
    a = self.B
    b = self.A * x1
    c = self.C * x1
    Delta = b**2 - 4*a*c
    return -(self.A*x1 + Delta**0.5) / (2*self.B)
    # Bug 1 : signe + au lieu de - devant Delta**0.5 -> mauvaise racine (~3983 MPa au lieu de ~578)
    # Bug 2 : retourne fy en MPa (espace physique) au lieu de u2 (espace standard)
```

### gnp_pivotB (version utilisatrice -- BUGGUEE)
```python
def gnp_pivotB(self):
    a = self.Ap * self.Bp
    b = self.Ap * (2*self.Bp + self.C) - 0.4*self.Bp
    c = 2 * self.Ap * self.C
    Delta = b**2 - 4.0 * a * c
    u = (-b + Delta**0.5) / (2.0 * a)
    return u*(u+2) / (4*self.Ap)
    # Bug : incoherence dimensionnelle (additionne 2*Bp [m3] et C [MN.m]) -> fc >> 1e7 MPa
```

### limite_pivotA (version utilisatrice -- BUGGUEE)
```python
def limite_pivotA(self):
    return (4 * self.B * self.C) / (self.A**2 - (self.A - (-2*self.B) * self.A3 * self.A2 / self.A1)**2)
    # Bug : incoherence dimensionnelle -> valeur enorme
```

### print_ana (version utilisatrice -- BUGGUEE)
```python
def print_ana(self, ax):
    origin = self.T_inv(ot.Point([0.0, 0.0]))
    x1_ref, x2_ref = origin[0], origin[1]

    def u1_de_x1(x1):
        return self.T(ot.Point([x1, x2_ref]))[0]

    def u2_de_x2(x2):
        return self.T(ot.Point([x1_ref, x2]))[1]

    x1_npB  = self.gnp_pivotB()
    x1_limA = self.limite_pivotA()
    u1_npB  = u1_de_x1(x1_npB)
    u1_limA = u1_de_x1(x1_limA)

    u1_B = np.linspace(u1_npB, u1_limA, 300)
    u2_B = []
    for u1 in u1_B:
        try:
            u2_B.append(u2_de_x2(self.gp_pivotB(u1)))  # Bug : double transfo (gp_pivotB retourne deja u2 apres fix)
        except Exception:
            u2_B.append(np.nan)
    u2_B = np.array(u2_B)

    u2_cst = u2_de_x2(self.g_pivotA())  # Bug : g_pivotA() n'est pas definie
    xlim   = ax.get_xlim()
    u1_max = xlim[1] if xlim[1] > u1_limA + 0.5 else u1_limA + 4.0
    u1_A   = np.linspace(u1_limA, u1_max, 100)
    u2_A   = np.full_like(u1_A, u2_cst)

    ax.axvline(u1_npB, color='green', linestyle='-.', linewidth=1.5)
    ax.plot(u1_B, u2_B, color='green', linestyle='-.', linewidth=2, label='g=0 ana')
    ax.plot(u1_A, u2_A, color='green', linestyle='-.', linewidth=2)
```

---

## VERSION CLAUDE (class flexion_claude -- active, lignes ~595-679)

Toutes les fonctions ci-dessous sont des methodes de `class flexion_claude`.
Pour restaurer : coller le bloc complet apres `class flexion_simple`, avant `def _parse`.

### __init__
```python
def __init__(self, Med, As, b, h, d, fc_otparams, fy_otparams,
             Es=200000, ecu=0.0035, eud=0.045, gamma_c=1.5, gamma_s=1.15):
    self.Med = Med
    self.As  = As
    self.b   = b
    self.h   = h
    self.d   = d
    self.Es  = Es
    self.ecu = ecu
    self.fyk = fy_otparams[0]   # nouveau vs flexion_simple

    dist = []
    if 'fc' in params_names:
        dist.append(loi_fc(*fc_otparams))
    if 'fy' in params_names:
        dist.append(loi_fy(*fy_otparams))
    dist_X     = ot.JointDistribution(dist)
    self.T_inv = dist_X.getInverseIsoProbabilisticTransformation()
    self.T     = dist_X.getIsoProbabilisticTransformation()

    self.A  = As * d / gamma_s
    self.B  = -0.5 * As**2 / b * gamma_c / gamma_s**2
    self.C  = -Med
    self.A1 = As * Es * ecu / (0.8 * b * d) * gamma_c
    self.A2 = Es * ecu * gamma_s
    self.A3 = ecu / (ecu + eud)
```

### gnp_pivotB -- CORRIGEE
Formule issue de l'equilibre des forces au seuil eps_s = eps_yd, eps_c = eps_cu.
Retourne fc [MPa]. Valeur attendue : ~31 MPa -> u1_npB ~ -1.3.
```python
def gnp_pivotB(self):
    gamma_s = self.A2 / (self.Es * self.ecu)
    gamma_c = self.A1 * 0.8 * self.b * self.d / (self.As * self.Es * self.ecu)
    fyd = self.fyk / gamma_s
    eyd = fyd / self.Es
    return self.As * fyd * gamma_c * (self.ecu + eyd) / (0.8 * self.b * self.d * self.ecu)
```

### limite_pivotA -- SUPPRIMEE (session 0605)
Fonction supprimee de flexion_claude -- inutile depuis que print_ana ne trace plus de zone pivot A.
La formule corrigee est conservee ici pour reference :
```python
# def limite_pivotA(self):
#     gamma_c = self.A1 * 0.8 * self.b * self.d / (self.As * self.Es * self.ecu)
#     x_limA  = self.d * self.A3
#     return self.Med * gamma_c / (0.8 * self.b * x_limA * (self.d - 0.4 * x_limA))
# Valeur attendue : ~242 MPa -> u1_limA ~ +13 (hors fenetre [-4,7])
```

### gp_pivotB -- CORRIGEE
Retourne u2 (espace standard). Fix signe + transformation T finale.
```python
def gp_pivotB(self, u1):
    x_point = self.T_inv(ot.Point([u1, 0.0]))
    x1 = x_point[0]
    a  = self.B
    b  = self.A * x1
    c  = self.C * x1
    Delta = b**2 - 4 * a * c
    fy = -(self.A * x1 - Delta**0.5) / (2 * self.B)   # - au lieu de +
    return self.T(ot.Point([x1, fy]))[1]               # retourne u2
```

### print_ana -- VERSION ACTUELLE (apres modifs session 0605)
Trace la courbe g=0 analytique sur ax (espace standard u1/u2).
Deux elements :
- Segment vertical de gp_pivotB(u1_npB) vers u2_max (bord haut du plot) -- acier non plastifie
- Courbe pivot B plastique de u1_npB jusqu'au bord droit du plot (pas de zone pivot A)

Modifications vs version initiale :
- limite_pivotA supprimee (inutile)
- plus de zone pivot A horizontale
- axvline remplace par ax.plot segment [u2_npB, u2_max] (ne descend pas sous la courbe)
- variables renommees u1_pB / u2_pB
- [A VALIDER] fix type ot.Scalar : u1_npB issu de T()[0] est un ot.Scalar, pas un float Python.
  ot.Point([u1_npB, 0.0]) echouait avec InvalidArgumentException.
  Correction : float(u1_npB) avant l'appel a gp_pivotB pour le segment vertical.

```python
def print_ana(self, ax):
    origin = self.T_inv(ot.Point([0.0, 0.0]))
    x1_ref, x2_ref = origin[0], origin[1]

    def u1_de_x1(x1):
        return self.T(ot.Point([x1, x2_ref]))[0]

    x1_npB = self.gnp_pivotB()
    u1_npB = u1_de_x1(x1_npB)

    # Courbe pivot B plastique jusqu'au bord droit
    xlim   = ax.get_xlim()
    u1_max = xlim[1]
    u1_pB = np.linspace(u1_npB, u1_max, 300)
    u2_pB = []
    for u1 in u1_pB:
        try:
            u2_pB.append(self.gp_pivotB(u1))
        except Exception:
            u2_pB.append(np.nan)
    u2_pB = np.array(u2_pB)

    # Segment vertical de gp_pivotB(u1_npB) vers le haut
    # float() : u1_npB est un ot.Scalar (depuis T()[0]), pas accepte directement par ot.Point([...])
    u2_npB = self.gp_pivotB(float(u1_npB))
    u2_max = ax.get_ylim()[1]
    ax.plot([u1_npB, u1_npB], [u2_npB, u2_max], color='green', linestyle='-.', linewidth=1.5)
    ax.plot(u1_pB, u2_pB, color='green', linestyle='-.', linewidth=2, label='g=0 ana')
```

---

## FONCTIONS DE VISU (bloc complet pour do_visu_claude)

### calc_ana_claude() -- lignes ~714-737
Factory qui lit dsCad.txt / dsLoad.txt et retourne une instance de flexion_claude.
Identique a calc_ana() mais avec flexion_claude au lieu de flexion_simple.
```python
def calc_ana_claude():
    path = os.path.join(r'C:\workspace\storage\admin\SF', modelname + '.ds')
    with open(os.path.join(path, 'dsCad.txt'), 'r') as f:
        _cad = f.read()
    with open(os.path.join(path, 'dsLoad.txt'), 'r') as f:
        _load = f.read()

    b   = _parse(_cad, 'b')
    h   = _parse(_cad, 'h')
    L   = _parse(_cad, 'L')
    phi = _parse(_cad, 'phi')

    n_bars = len(re.findall(r'REBAR\(', _cad))
    As = n_bars * math.pi * (phi / 2e3) ** 2

    z_rebar = [float(v) for v in re.findall(
        r'pts\d+\.append\(POINT\([^,]+,\s*[^,]+,\s*([\d.]+)\)', _cad)]
    d = h/2 + sum(z_rebar) / len(z_rebar)

    F = abs(float(re.search(r"Z='(-?[\d.]+)'", _load).group(1)))
    Med = F * L

    return flexion_claude(Med=Med, As=As, b=b, h=h, d=d,
                          fc_otparams=(fck, cov_fck), fy_otparams=(fyk, cov_fyk))
```

### print_visu_claude(best_result, best_sp, xt, sm_GEK, g_ot_KRG, g_hf, modes, calc)
-- lignes ~1151-1221
Copie de print_visu avec parametre calc explicite et appel calc.print_ana(ax).
Legende : 'g=0 ana (claude)'. Titre : 'FORM sur GEKPLS -- courbe analytique corrigee'.
La seule difference fonctionnelle vs print_visu est :
```python
# apres le bloc contour HF, avant les scatter points :
if calc is not None:
    calc.print_ana(ax)
# et dans la legende :
if calc is not None:
    legend_lines.append(Line2D([0], [0], color='green', linestyle='-.', linewidth=2, label='g=0 ana (claude)'))
```

### Flag et bloc main
```python
# OPTIONS TEST (ligne ~101)
do_visu_claude = True   # True -> print_visu_claude | False -> print_visu classique

# Bloc main (lignes ~1286-1299)
if do_visu_claude:
    calc_claude = calc_ana_claude()
    if not try_pce:
        print_visu_claude(best_result, best_sp, xt, sm_GEK, g_ot_KRG, run_HF, modes, calc_claude)
    else:
        print_visu_claude(best_result, best_sp, xt, sm_GEPCK, g_ot_PCKRG, run_HF, modes, calc_claude)
else:
    if print_ana:
        calc = calc_ana()
        calc.print_f()
    if not try_pce:
        print_visu(best_result, best_sp, xt, sm_GEK, g_ot_KRG, run_HF, modes)
    else:
        print_visu(best_result, best_sp, xt, sm_GEPCK, g_ot_PCKRG, run_HF, modes)
```
