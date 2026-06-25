# -*- coding: utf-8 -*-
# Regenere visuGEPCK + EFF_graphs pour un STOP a N_EFF_KEEP points EFF (au lieu de 25),
# UNIQUEMENT depuis restart_state.json du diagonal. Tronque le set (DOE + N premiers EFF),
# refit GEPCK, RECALCULE u* (FORM multistart) + beta_IS (IS capee), et redessine les 2 graphes.
import os, sys, json, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openturns as ot
from scipy.stats import norm
from sklearn.cluster import DBSCAN
sys.path.insert(0, r'C:\workspace\fiabilite')
import sys; sys.path.insert(0, r"C:\workspaceiabilite\_lib")  # branche* deplaces dans _lib
from branche1 import fit_gepck, predict_gepck, predict_gradient_gepck

# ---------------- PARAMETRES ----------------
N_EFF_KEEP = int(os.environ.get("_KEEP", "21"))   # <<< stop a N points EFF (def 21, override _KEEP)
n_max_FORM = 50
DS  = r'C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_13k_2fy_membrure_inf_diagonal.ds'
OUT = r'C:\workspace\fiabilite\output\png_EFF_moulin_blanc\png_EFF_1906_1018'
u1_min,u1_max,u2_min,u2_max,n_grid,n_grid_hf,n_var = -7.5,7.5,-7.5,7.5,300,7,2
tol_FORM, tol_all_modes, n_IS, cov_IS = 1.0, 0.9, 10000, 0.05
MAX_IS_BLOCKS = 200    # cap pour eviter la derive IS (cf TODO perf)
params_names = ['fy1','fy2']

# ---------------- 1) charge + tronque ----------------
d = json.load(open(os.path.join(DS,'restart_state.json')))
n_doe = d['n_doe']; keep = n_doe + N_EFF_KEEP
xt = np.array(d['xt'],float)[:keep]; yt = np.array(d['yt'],float)[:keep]; ag = np.array(d['all_grad'],float)[:keep]
xt_eff = np.array(d['xt_eff'],float)[:N_EFF_KEEP]
hf_cache = d['hf_2d_grid']; max_degree = d['max_degree']
print(f"STOP a {N_EFF_KEEP} EFF -> {keep} pts total (DOE {n_doe} + EFF {N_EFF_KEEP})")

# ---------------- 2) refit GEPCK ----------------
Y_aug = np.concatenate([yt.flatten()] + [ag[:,j] for j in range(n_var)])
marginals = [{'Type':'Gaussian','Parameters':[0.0,1.0]},{'Type':'Gaussian','Parameters':[0.0,1.0]}]
copula = {'Type':'Independent','Parameters':np.eye(n_var)}
opts = {'Mode':'optimal','PCE':{'Degree':list(range(1,max_degree+1)),'Method':'LARS'}}
with warnings.catch_warnings():
    warnings.simplefilter('ignore'); fm = fit_gepck(xt, Y_aug, opts, marginals, copula)
print(f"refit OK : theta={[round(t,4) for t in fm['Kriging'][0]['theta']]}")

class GFun(ot.OpenTURNSPythonFunction):
    def __init__(s, fm): super().__init__(n_var,1); s.fm=fm
    def _exec(s,u): return [float(predict_gepck(s.fm, np.array(u).reshape(1,-1))[0,0])]
    def _exec_sample(s,U): return predict_gepck(s.fm, np.array(U))[:,0:1].tolist()
    def _gradient(s,u):
        G = predict_gradient_gepck(s.fm, np.array(u).reshape(1,-1))
        return [[float(G[0,i])] for i in range(s.fm['Mred'])]
g_ot = ot.Function(GFun(fm))

# ---------------- 3) FORM multistart -> modes (dedup distance tol_all_modes) ----------------
dist = ot.JointDistribution([ot.Normal(0.0,1.0)]*n_var)
X = ot.RandomVector(dist); Y = ot.CompositeRandomVector(g_ot, X)
event = ot.ThresholdEvent(Y, ot.Less(), 0.0)
starts = np.vstack([xt, [[0.0,0.0]]])
all_us = []; all_beta = []; all_sp = []; all_r = []
for sp in starts:
    try:
        solv = ot.AbdoRackwitz(); solv.setStartingPoint(list(sp))
        solv.setMaximumIterationNumber(n_max_FORM); solv.setCheckStatus(False)
        solv.setMaximumConstraintError(tol_FORM)
        f = ot.FORM(solv, event); f.run(); r = f.getResult()
        all_us.append(np.array(r.getStandardSpaceDesignPoint())); all_beta.append(r.getHasoferReliabilityIndex())
        all_sp.append(np.array(sp)); all_r.append(r)
    except Exception: pass
# DBSCAN(eps=tol_all_modes, min_samples=2) -> 1 mode/cluster (min beta), isoles=bruit ignore (comme FORM_all_modes)
U_all = np.array(all_us)
labels = DBSCAN(eps=tol_all_modes, min_samples=2).fit(U_all).labels_
modes = []  # (beta, u*, sp, FORMResult) : sp = point de depart du membre min-beta du cluster
for lbl in sorted(set(labels) - {-1}):
    idx = [i for i,l in enumerate(labels) if l == lbl]
    bi = min(idx, key=lambda i: all_beta[i])
    modes.append((all_beta[bi], all_us[bi], all_sp[bi], all_r[bi]))
modes.sort(key=lambda t: t[0])
_n_noise = int(np.sum(labels == -1))
print(f"FORM : {len(all_us)} descentes -> {len(modes)} mode(s) DBSCAN ({_n_noise} bruit ignore) -> "
      + " | ".join(f"u*=[{m[1][0]:.2f},{m[1][1]:.2f}] beta={m[0]:.3f}" for m in modes[:5]))

# ---------------- 4) IS sur le mode dominant (capee) -> beta_IS ----------------
g_imp = ot.Normal(n_var); g_imp.setMu(list(modes[0][1]))
exp = ot.ImportanceSamplingExperiment(g_imp, n_IS)
algo = ot.ProbabilitySimulationAlgorithm(ot.StandardEvent(event), exp)
algo.setMaximumCoefficientOfVariation(cov_IS); algo.setMaximumOuterSampling(MAX_IS_BLOCKS)
algo.run(); res = algo.getResult()
Pf_IS = res.getProbabilityEstimate(); beta_IS = float(-norm.ppf(Pf_IS)); cov = res.getCoefficientOfVariation()
print(f"IS (mode 1, cap {MAX_IS_BLOCKS} blocs) : Pf={Pf_IS:.3e} beta_IS={beta_IS:.4f} COV={cov:.3f} blocs={res.getOuterSampling()}")

# ---------------- 5) visuGEPCK (recap) ----------------
# range ETENDU pour inclure TOUS les modes (ex. mode 5 a u1=-7.67, hors de [-7.5,7.5])
_allu = np.vstack([xt, xt_eff] + [np.asarray(us).reshape(1,-1) for (_,us,_,_) in modes]) if len(modes) else np.vstack([xt,xt_eff])
_pad=0.6
gu1_min=min(_allu[:,0].min()-_pad,u1_min); gu1_max=max(_allu[:,0].max()+_pad,u1_max)
gu2_min=min(_allu[:,1].min()-_pad,u2_min); gu2_max=max(_allu[:,1].max()+_pad,u2_max)
u1=np.linspace(gu1_min,gu1_max,n_grid); u2=np.linspace(gu2_min,gu2_max,n_grid)
U1,U2=np.meshgrid(u1,u2); grid=np.column_stack([U1.ravel(),U2.ravel()])
Z=predict_gepck(fm, grid)[:,0].reshape(n_grid,n_grid)
fig,ax=plt.subplots(figsize=(7,6))
cf=ax.contourf(U1,U2,Z,levels=20,cmap='RdYlGn',alpha=0.6); plt.colorbar(cf,ax=ax,label='g (GEPCK)')
ax.contour(U1,U2,Z,levels=[0],colors='blue',linewidths=2)
if hf_cache and 'Z' in hf_cache:   # courbe HF rouge sur la grille ORIGINALE [-7.5,7.5]
    Zr=np.array(hf_cache['Z'],float); u1h=np.linspace(u1_min,u1_max,n_grid_hf); u2h=np.linspace(u2_min,u2_max,n_grid_hf)
    U1h,U2h=np.meshgrid(u1h,u2h); ax.contour(U1h,U2h,Zr,levels=[0],colors='red',linewidths=2,linestyles='--')
ax.scatter(xt[:,0],xt[:,1],c='black',s=30,zorder=5,label='DOE')
if len(xt_eff)>0:
    ax.scatter(xt_eff[:,0],xt_eff[:,1],c='red',s=60,zorder=6,marker='^',label=f'EFF ({len(xt_eff)} pts)')
    for i,pt in enumerate(xt_eff):
        ax.annotate(str(i+1),(pt[0],pt[1]),textcoords='offset points',xytext=(0,8),ha='center',fontsize=8,color='red',zorder=7)
ax.scatter(0,0,c='orange',s=100,zorder=6,marker='P',label='[0,0]')
# 1 couleur DISTINCTE par mode : etoile = u* (croix de depart RETIREE -> graphe moins charge)
_cols = plt.cm.tab10(np.linspace(0,1,10))
for k,(be,us,sp,r) in enumerate(modes):
    c=_cols[k % 10]
    ax.scatter(us[0],us[1],c=[c],s=220,zorder=8,marker='*',edgecolors='k',linewidths=0.5,
               label=f'mode {k+1}: u*=[{us[0]:.2f},{us[1]:.2f}] beta={be:.3f}')
ax.set_xlabel('u1'); ax.set_ylabel('u2'); ax.set_xlim(gu1_min,gu1_max); ax.set_ylim(gu2_min,gu2_max); ax.legend(loc='best',fontsize=8)
fn1=os.path.join(OUT,f'visuGEPCK_STOP{N_EFF_KEEP}_1906_1018.png'); fig.savefig(fn1,dpi=150,bbox_inches='tight'); plt.close(fig)
print(f"-> {fn1}")

# ---------------- 6) EFF_graphs (historiques tronques a N_EFF_KEEP) ----------------
K = N_EFF_KEEP + 1   # initial + N points
hE=d['hist_EFF'][:K]; hBB=d['hist_BB'][:K]; hBS=d['hist_BS'][:K]; hT=d['hist_theta'][:K]
tol_EFF,tol_BB,tol_BS,_clip=1e-3,0.01,0.01,1e-12
fig,axes=plt.subplots(1,3,figsize=(15,4))
ax=axes[0]; ax.semilogy(range(len(hE)),[max(abs(v),_clip) for v in hE],'b-o',ms=4,lw=1.2,label='EFF(u_opt)')
ax.axhline(tol_EFF,color='orange',ls='--',lw=1,label=f'tol_EFF={tol_EFF:.1e}'); ax.set_title('Convergence EFF'); ax.set_xlabel('Iteration EFF'); ax.set_ylabel('EFF (log)'); ax.legend(fontsize=8); ax.grid(True,which='both',alpha=0.4)
ax=axes[1]
if hBB: ax.semilogy(range(1,len(hBB)+1),[max(v,_clip) if v is not None else np.nan for v in hBB],'g-o',ms=4,lw=1.2,label='BB'); ax.axhline(tol_BB,color='g',ls='--',lw=0.8,label=f'tol_BB={tol_BB:.1e}')
if hBS: ax.semilogy(range(1,len(hBS)+1),[max(v,_clip) if v is not None else np.nan for v in hBS],'r-s',ms=4,lw=1.2,label='BS'); ax.axhline(tol_BS,color='r',ls='--',lw=0.8,label=f'tol_BS={tol_BS:.1e}')
ax.set_title('Criteres BB / BS'); ax.set_xlabel('Iteration'); ax.set_ylabel('Ratio (log)'); ax.legend(fontsize=8); ax.grid(True,which='both',alpha=0.4)
ax=axes[2]
if hT:
    th=np.array(hT)
    for k in range(th.shape[1]): ax.semilogy(range(len(hT)),np.maximum(th[:,k],_clip),'-o',ms=4,lw=1.2,label=f'theta_{params_names[k]}')
    ax.semilogy(range(len(hT)),np.maximum(np.linalg.norm(th,axis=1),_clip),'k--',ms=3,lw=1,label='||theta||')
ax.set_title('Evolution theta Kriging'); ax.set_xlabel('Iteration (fit)'); ax.set_ylabel('theta (log)'); ax.legend(fontsize=8); ax.grid(True,which='both',alpha=0.4)
fig.suptitle(f'EFF_graphs STOP {N_EFF_KEEP} EFF (regenere depuis dump)',fontsize=10); fig.tight_layout()
fn2=os.path.join(OUT,f'EFF_graphs_STOP{N_EFF_KEEP}_1906_1018.png'); fig.savefig(fn2,dpi=150,bbox_inches='tight'); plt.close(fig)
print(f"-> {fn2}")

# ---------------- 7) EXPORT pour Semia : JSON + log texte ----------------
mu_,sig_=235.0,30.15
# points d'entrainement : coords U + physiques (fy) + sensibilite U-space (gradient)
points=[]
for i in range(len(xt)):
    points.append({'i':i,'phase':('DOE' if i<n_doe else 'EFF'),
        'u1':float(xt[i,0]),'u2':float(xt[i,1]),
        'fy1':float(mu_+sig_*xt[i,0]),'fy2':float(mu_+sig_*xt[i,1]),
        'g':float(yt[i,0]),'dg_du1':float(ag[i,0]),'dg_du2':float(ag[i,1])})
# modes : u*, beta, fy*, point de depart, IMPORTANCES des variables (FORM)
modes_out=[]
for k,(be,us,sp,r) in enumerate(modes):
    imp=list(r.getImportanceFactors())
    gstar=predict_gradient_gepck(fm, us.reshape(1,-1))[0]   # gradient surrogate dg/du en u*
    modes_out.append({'mode':k+1,'beta':float(be),'Pf':float(r.getEventProbability()),
        'u_star':[float(us[0]),float(us[1])],
        'fy_star':[float(mu_+sig_*us[0]),float(mu_+sig_*us[1])],
        'sp_depart':[float(sp[0]),float(sp[1])],
        'dg_du_star':[float(gstar[0]),float(gstar[1])],
        'importance_fy1':float(imp[0]),'importance_fy2':float(imp[1])})
export={'cas':'diagonal','stop_n_eff':N_EFF_KEEP,'n_doe':n_doe,'n_total':int(len(xt)),
        'max_degree':int(max_degree),'mu_fy':mu_,'sigma_fy':sig_,
        'beta_IS':float(beta_IS),'Pf_IS':float(Pf_IS),'cov_IS':float(cov),
        'transfo':'u = (fy - 235)/30.15 ; gradient dg_du = espace U (Jacobien applique)',
        'points':points,'modes':modes_out}
fexp=os.path.join(OUT,f'export_STOP{N_EFF_KEEP}_diagonal.json'); json.dump(export,open(fexp,'w'),indent=1)
print(f"-> {fexp}")
# log texte : reproduit le FORMAT du vrai log fiabilite (modes, dg/du u*, IS) + extras (importances, points)
flog=os.path.join(OUT,f'log_2fy_STOP{N_EFF_KEEP}_diagonal.log')
with open(flog,'w') as L:
    L.write("="*78+"\n")
    L.write(f"LOG cas DIAGONAL - ARRET a {N_EFF_KEEP} points EFF (equivalent d'un run n_points={N_EFF_KEEP})\n")
    L.write("Regenere depuis restart_state.json (troncature du run 25 = identique a un vrai stop-21).\n")
    L.write(f"n_total={len(xt)} (DOE {n_doe} + EFF {N_EFF_KEEP}) | modele=GEPCK degre {max_degree} | params=['fy1','fy2']\n")
    L.write("Distribution : fy ~ Normal(mu=235, sigma=30.15 MPa) | transfo u=(fy-235)/30.15\n")
    L.write("Groupe 1 (fy1/u1) = dalle ; Groupe 2 (fy2/u2) = treillis (membrures inf/sup + montants + diagonales)\n")
    L.write("="*78+"\n\n")
    # --- MODES (format du vrai log + importances + dg/du surrogate au u*) ---
    L.write(f"{len(modes)} mode(s) distinct(s) (DBSCAN eps={tol_all_modes}, min_samples=2) :\n")
    for m in modes_out:
        L.write(f"  mode {m['mode']} : beta={m['beta']:.4f}  Pf={m['Pf']:.3e}  "
                f"u*=[{m['u_star'][0]:.4f}, {m['u_star'][1]:.4f}]\n")
        L.write(f"      fy* = [{m['fy_star'][0]:.3f}, {m['fy_star'][1]:.3f}] MPa\n")
        L.write(f"      dg/du en u* (surrogate) : dg/du1={m['dg_du_star'][0]:.6f}  dg/du2={m['dg_du_star'][1]:.6f}\n")
        L.write(f"      importance des variables : fy1={m['importance_fy1']:.4f}  fy2={m['importance_fy2']:.4f}\n")
        L.write(f"      point de depart FORM : sp=[{m['sp_depart'][0]:.3f}, {m['sp_depart'][1]:.3f}]\n")
    # --- IS (format du vrai log) ---
    L.write("\n=== Importance Sampling ===\n")
    L.write(f"  Pf_IS   = {Pf_IS:.4e}\n  beta_IS = {beta_IS:.4f}\n  COV     = {cov:.4f}\n  (IS plafonnee a {MAX_IS_BLOCKS} blocs de {n_IS})\n")
    # --- POINTS d'entrainement : coords U + physiques + sensibilite U-space ---
    L.write(f"\n--- {len(points)} POINTS (DOE + EFF) : u1,u2 | fy1,fy2 (MPa) | g | dg/du1,dg/du2 (espace U) ---\n")
    for p in points:
        L.write(f"  [{p['phase']:3}] {p['i']:2d}: u=({p['u1']:+.4f}, {p['u2']:+.4f})  "
                f"fy=({p['fy1']:.3f}, {p['fy2']:.3f})  g={p['g']:+.6f}  "
                f"dg/du=({p['dg_du1']:+.6f}, {p['dg_du2']:+.6f})\n")
print(f"-> {flog}")
print("FINI")
