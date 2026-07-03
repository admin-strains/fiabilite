# -*- coding: utf-8 -*-
# =====================================================================
# VALIDATION ANALYTIQUE charge PONCTUELLE (2026-07-03, MM)
# =====================================================================
# Une charge ponctuelle sur un continuum est SINGULIERE (poinconnement local) : le lambda
# absolu depend du maillage (non-determinisme meshgems + tetra de localisation). La FD
# croisee-maillage est donc du bruit. Ce qui EST valide analytiquement, par solve :
#   (1) CONSERVATION exacte : SUM(w)=1 -> SUM(w*F)=F, aucune perte (mesh-independant).
#   (2) MAGNITUDE : d(alpha)/dm = -lambda EXACT (scaling). On trace sens vs -lambda sur un
#       balayage de positions -> doit suivre la droite y=x (bande de tolerance poinconnement).
#   (3) POSITION : signe coherent (charge vers l'about -> lambda baisse -> d(alpha)/ds<0),
#       et monotonie.
# Compare aussi au cas FOOTPRINT (charge distribuee, bien posee) qui suit la loi 1D
# alpha = C/x_c proprement -> reference "courbe analytique complete".
#
#   C:\python3\python.exe tests\pl_analytical.py
# =====================================================================
import sys, os, json
sys.path.insert(0, r'C:\workspace\fiabilite\tests')
from pl_harness import run_config

def sweep_point_magnitude():
    """point LIVE a plusieurs positions -> (lambda, sens_mag, sens_pos, conserv)."""
    rows = []
    for x in [0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6]:
        cfg = {"name":"anaP_x%.1f"%x,
               "points":[{"name":"P1","xyz":[x,0,0.1],"F":[0,0,-1.0],"role":"live","lc":"trafic",
                          "mobile":{"path":[[0.5,0,0.1],[3.8,0,0.1]],"position":x-0.5,"unit":"absolute"}}],
               "regions":[{"param":"LIVE_LOAD","load_case":"trafic"},
                          {"param":"LIVE_LOAD","axis":"position","load_case":"trafic"}],
               "mesh_size":"0.10"}
        r = run_config(cfg)
        if r["status"] == "OPTIMAL":
            rows.append({"x":x,"lam":r["lam"],"mag":r["sens"].get("LIVE_LOAD:trafic"),
                         "pos":r["sens"].get("LIVE_LOAD:position:trafic"),
                         "cerr":max([c["err_sumw"] for c in r["conserv"]], default=0.0)})
        print("  point x=%.1f status=%s lam=%.4g mag=%s pos=%s cerr=%.1e"
              % (x, r["status"], (r["lam"] or 0), r["sens"].get("LIVE_LOAD:trafic"),
                 r["sens"].get("LIVE_LOAD:position:trafic"),
                 max([c["err_sumw"] for c in r["conserv"]], default=0.0)))
    return rows

def sweep_footprint_magnitude():
    """footprint LIVE a plusieurs positions -> loi 1D propre (reference)."""
    rows = []
    for xc in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        cfg = {"name":"anaFP_x%.1f"%xc,
               "footprints":[{"name":"FPL","polygon":[[xc-0.25,-0.15,0.25],[xc+0.25,-0.15,0.25],
                              [xc+0.25,0.15,0.25],[xc-0.25,0.15,0.25]],"F":[0,0,-0.15],"role":"live","lc":"trafic"}],
               "regions":[{"param":"LIVE_LOAD","load_case":"trafic"}],"mesh_size":"0.08"}
        r = run_config(cfg)
        if r["status"] == "OPTIMAL":
            rows.append({"xc":xc,"lam":r["lam"],"mag":r["sens"].get("LIVE_LOAD:trafic")})
        print("  fp xc=%.1f status=%s lam=%.4g mag=%s"
              % (xc, r["status"], (r["lam"] or 0), r["sens"].get("LIVE_LOAD:trafic")))
    return rows

if __name__ == "__main__":
    print("\n#### balayage POINT (magnitude+position) ####")
    P = sweep_point_magnitude()
    print("\n#### balayage FOOTPRINT (loi 1D propre) ####")
    FP = sweep_footprint_magnitude()

    def pct(a,b): return abs(a-b)/max(abs(b),1e-30)*100
    print("\n#### (1) CONSERVATION (max err SUM(w)-1) ####")
    cmax = max([r["cerr"] for r in P], default=0.0)
    print("   max conservation error sur %d points = %.2e  (attendu ~0)" % (len(P), cmax))
    print("\n#### (2) MAGNITUDE : sens vs -lambda (poinconnement -> bande ~10%%) ####")
    for r in P:
        print("   x=%.1f  sens=%.4e  -lam=%.4e  ecart=%.1f%%" % (r["x"], r["mag"], -r["lam"], pct(r["mag"], -r["lam"])))
    avg = sum(pct(r["mag"], -r["lam"]) for r in P)/max(len(P),1)
    print("   ecart moyen sens vs -lambda = %.1f%%" % avg)
    print("\n#### (3) POSITION : signe (attendu <0 : charge vers about -> lambda baisse) ####")
    nneg = sum(1 for r in P if r["pos"] is not None and r["pos"] < 0)
    print("   %d/%d positions ont d(alpha)/ds < 0 (coherent)" % (nneg, len(P)))
    print("\n#### FOOTPRINT (reference propre) : sens vs -lambda ####")
    favg = sum(pct(r["mag"], -r["lam"]) for r in FP)/max(len(FP),1)
    for r in FP:
        print("   xc=%.1f sens=%.4e -lam=%.4e ecart=%.2f%%" % (r["xc"], r["mag"], -r["lam"], pct(r["mag"], -r["lam"])))
    print("   footprint ecart moyen = %.2f%% (charge distribuee = bien posee)" % favg)

    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        # (1) conservation
        ax[0].semilogy([r["x"] for r in P], [max(r["cerr"],1e-18) for r in P], 'go-')
        ax[0].axhline(1e-9, color='r', ls='--', label='seuil 1e-9')
        ax[0].set_title("(1) CONSERVATION  |SUM(w)-1|"); ax[0].set_xlabel("position x"); ax[0].set_ylabel("erreur"); ax[0].legend()
        # (2) magnitude sens vs -lam
        xs = [-r["lam"] for r in P]; ys = [r["mag"] for r in P]
        lo = min(xs+ys); hi = max(xs+ys)
        ax[1].plot([lo,hi],[lo,hi],'b--',label='y=x (analytique)')
        ax[1].plot(xs, ys, 'ro', label='point (poinconnement)')
        ax[1].plot([-r["lam"] for r in FP], [r["mag"] for r in FP], 'g^', label='footprint (bien pose)')
        ax[1].set_title("(2) MAGNITUDE : sens vs -lambda"); ax[1].set_xlabel("-lambda"); ax[1].set_ylabel("d(alpha)/dm"); ax[1].legend()
        # (3) position sens
        ax[2].plot([r["x"] for r in P], [r["pos"] for r in P], 'mo-')
        ax[2].axhline(0, color='k', lw=0.5)
        ax[2].set_title("(3) POSITION : d(alpha)/ds (signe<0 coherent)"); ax[2].set_xlabel("position x"); ax[2].set_ylabel("d(alpha)/ds")
        fig.suptitle("VALIDATION charge PONCTUELLE : conservation exacte + magnitude=-lambda + position coherente\n"
                     "(point = poinconnement local, lambda mesh-dependant ; footprint = reference bien posee)")
        fig.tight_layout()
        out = os.path.join(r"C:\workspace\fiabilite\tests", "pl_analytical.png")
        fig.savefig(out, dpi=110); print("\nFIGURE -> %s" % out)
    except Exception as ex:
        print("(matplotlib indispo: %r)" % ex)
