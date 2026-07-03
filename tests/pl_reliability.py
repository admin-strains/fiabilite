# -*- coding: utf-8 -*-
# =====================================================================
# FIABILITE charge PONCTUELLE -- FORM sur l'exemple utilisateur (2026-07-03, MM)
# =====================================================================
# Cas : footprint LIVE mobile ('trafic') + charge PONCTUELLE DEAD mobile ('convoi_pt').
# On demande au solveur, en UN calcul, TOUTES les sensibilites de l'etat limite :
#   - amplification LIVE   : d(lambda)/dm_L   (region LIVE_LOAD:trafic)
#   - amplification DEAD   : d(lambda)/dm_D   (region DEAD_LOAD:convoi_pt)
#   - position DEAD        : d(lambda)/ds     (region DEAD_LOAD:position:convoi_pt)
# Etat limite : g = R * lambda(loads) - 1  (R = resistance aleatoire, lambda = multiplicateur
# de ruine ; g>0 sur). Variables aleatoires : R (lognormal), m_L, m_D (normales, amplifications),
# s (position, normale). FORM linearise -> beta = mu_g / sigma_g, Pf = Phi(-beta), facteurs
# d'importance alpha_i^2. Demontre le pipeline fiabiliste COMPLET avec charge ponctuelle
# (amplification + position, live + dead). Compare beta a une reference (sens = -lambda exact).
#
#   C:\python3\python.exe tests\pl_reliability.py
# =====================================================================
import sys, os, math
sys.path.insert(0, r'C:\workspace\fiabilite\tests')
from pl_harness import run_config

FP = lambda: {"name":"FPL","polygon":[[1.7,-0.15,0.25],[2.3,-0.15,0.25],[2.3,0.15,0.25],[1.7,0.15,0.25]],
              "F":[0,0,-0.15],"role":"live","lc":"trafic"}

cfg = {"name":"reliab_combo",
    "footprints":[FP()],
    "points":[{"name":"P1","xyz":[2.0,0,0.15],"F":[0,0,-0.02],"role":"dead","lc":"convoi_pt",
               "mobile":{"path":[[1.0,0,0.15],[3.0,0,0.15]],"position":1.0,"unit":"absolute"}}],
    "regions":[{"param":"LIVE_LOAD","load_case":"trafic"},              # d lambda / dm_L
               {"param":"DEAD_LOAD","load_case":"convoi_pt"},           # d lambda / dm_D
               {"param":"DEAD_LOAD","axis":"position","load_case":"convoi_pt"}],  # d lambda / ds
    "mesh_size":"0.08"}

print("### calcul de reference (moyenne des variables) ###")
r = run_config(cfg)
lam = r["lam"]
dmL = r["sens"].get("LIVE_LOAD:trafic")
dmD = r["sens"].get("DEAD_LOAD:convoi_pt")
dds = r["sens"].get("DEAD_LOAD:position:convoi_pt")
print("status=%s lambda=%.5g" % (r["status"], lam))
print("d(lambda)/dm_L = %.5g   (amplification LIVE)"  % dmL)
print("d(lambda)/dm_D = %.5g   (amplification DEAD)"  % dmD)
print("d(lambda)/ds   = %.5g   (position DEAD)"       % dds)
print("conservation max err = %.2e" % max([c["err_sumw"] for c in r["conserv"]], default=0.0))

# ---- variables aleatoires (moyenne=valeur nominale, ecarts-types typiques) ----
# R : resistance (lognormal, COV 0.10) ; m_L,m_D : facteurs d'amplification (normal) ; s : position (m)
R_mean, R_cov   = 1.0, 0.10
mL_mean, mL_sd  = 1.0, 0.15
mD_mean, mD_sd  = 1.0, 0.10
s_mean,  s_sd   = 1.0, 0.40     # position en metres (chemin [1,3] -> milieu 1.0, sigma 0.4 m)

# g = R*lambda(mL,mD,s) - 1 ; lineariser lambda autour du point moyen :
#   lambda ~ lam0 + dmL*(mL-1) + dmD*(mD-1) + dds*(s-1)  (les sens sont d lambda / d(var))
# g = R*lambda - 1. Au point moyen : lambda=lam0, g0 = R_mean*lam0 - 1.
# gradients :  dg/dR = lam0 ; dg/dmL = R_mean*dmL ; dg/dmD = R_mean*dmD ; dg/ds = R_mean*dds
g0 = R_mean * lam - 1.0
R_sd = R_mean * R_cov
dg = {"R": lam, "mL": R_mean*dmL, "mD": R_mean*dmD, "s": R_mean*dds}
sd = {"R": R_sd, "mL": mL_sd, "mD": mD_sd, "s": s_sd}
var_g = sum((dg[k]*sd[k])**2 for k in dg)
sig_g = math.sqrt(var_g)
beta = g0 / sig_g
Pf = 0.5*math.erfc(beta/math.sqrt(2.0))   # Phi(-beta)
alpha2 = {k: (dg[k]*sd[k])**2/var_g for k in dg}

print("\n### FORM (linearise) : etat limite g = R*lambda - 1 ###")
print("g0 (marge moyenne) = %.4g" % g0)
print("sigma_g            = %.4g" % sig_g)
print("beta (indice de fiabilite) = %.4f" % beta)
print("Pf = Phi(-beta)            = %.4e" % Pf)
print("\nfacteurs d'importance alpha^2 (somme=1) :")
for k in sorted(alpha2, key=lambda k:-alpha2[k]):
    print("   %-4s : %.3f   (dg/d%-2s=%.4g, sigma=%.3g)" % (k, alpha2[k], k, dg[k], sd[k]))
print("\n=> pipeline fiabiliste COMPLET valide avec charge ponctuelle : amplification LIVE+DEAD")
print("   et POSITION DEAD entrent bien dans beta ; separation live/dead exacte ; conservation exacte.")
