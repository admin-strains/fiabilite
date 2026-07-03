# -*- coding: utf-8 -*-
# =====================================================================
# MATRICE de validation CHARGE PONCTUELLE -- combinaisons massives (2026-07-03, MM)
# =====================================================================
# Genere des centaines de configs combinant : point interieur / sur-noeud / au-dessus
# (projete) / dehors (ignore) ; live / dead ; magnitude & position ; seul ou combine a un
# footprint ; meme ou differentes variables. Chaque calcul verifie :
#   (1) pas d'erreur (exit propre)          (4) sensibilites bien calculees + routees
#   (2) charge CONSERVEE (SUM(w)=1, exact)  (5) separation live/dead exacte
#   (3) point interieur CONVERGE (OPTIMAL)  (6) mode de ruine coherent (lambda>0 fini)
#
# Sliceable pour agents paralleles :  python pl_matrix.py --slice i/N --out res_i.json
# =====================================================================
import sys, os, json, argparse, math
sys.path.insert(0, r'C:\workspace\fiabilite\tests')
from pl_harness import run_config

# ---------- geometrie de reference (cantilever L=4, b=0.4, h=0.5) ----------
# interieur : x in (0,4), |y|<0.2, |z|<0.25 ; surface haute z=0.25 ; encastrement x=0.
FP_LIVE = lambda xc=2.0: {"name":"FPL","polygon":[[xc-0.3,-0.15,0.25],[xc+0.3,-0.15,0.25],
                          [xc+0.3,0.15,0.25],[xc-0.3,0.15,0.25]],"F":[0,0,-0.15],"role":"live","lc":"trafic"}

def R(param, lc, axis=None):
    r = {"param":param, "load_case":lc}
    if axis: r["axis"] = axis
    return r

def build_matrix():
    cfgs = []

    # ---- FAMILLE 1 : LIVE point interieur, grille de positions (conservation + convergence) ----
    xs = [0.8, 1.4, 2.0, 2.6, 3.2, 3.6]
    ys = [-0.12, 0.0, 0.12]
    zs = [-0.18, 0.0, 0.18]
    fam1 = 0
    for x in xs:
        for y in ys:
            for z in zs:
                cfgs.append({"name":"F1_live_int_x%.1f_y%+.2f_z%+.2f"%(x,y,z),
                    "points":[{"name":"P1","xyz":[x,y,z],"F":[0,0,-1.0],"role":"live","lc":"trafic"}],
                    "regions":[R("LIVE_LOAD","trafic")],"mesh_size":"0.10",
                    "expect":{"optimal":True,"locate":["located"],"sens_key":"LIVE_LOAD:trafic",
                              "sens_approx_neglam":0.15}})
                fam1 += 1

    # ---- FAMILLE 2 : LIVE point, magnitudes & directions variees (conservation invariante) ----
    for mag in [0.2, 1.0, 5.0, 50.0]:
        for F,tag in [([0,0,-mag],"Z"),([mag*0.5,0,-mag],"XZ"),([mag*0.4,mag*0.3,-mag],"XYZ")]:
            cfgs.append({"name":"F2_live_mag%.1f_%s"%(mag,tag),
                "points":[{"name":"P1","xyz":[2.0,0.0,0.1],"F":F,"role":"live","lc":"trafic"}],
                "regions":[R("LIVE_LOAD","trafic")],"mesh_size":"0.10",
                "expect":{"optimal":True,"locate":["located"],"sens_key":"LIVE_LOAD:trafic",
                          "sens_approx_neglam":0.15}})

    # ---- FAMILLE 3 : PROJECTION (point au-dessus de la surface -> projete le long de -Z) ----
    for x in [1.0, 2.0, 3.0]:
        for zabove in [0.30, 0.5, 1.0]:
            cfgs.append({"name":"F3_proj_x%.1f_z%.2f"%(x,zabove),
                "points":[{"name":"P1","xyz":[x,0.0,zabove],"F":[0,0,-1.0],"role":"live","lc":"trafic"}],
                "regions":[R("LIVE_LOAD","trafic")],"mesh_size":"0.10",
                "expect":{"optimal":True,"locate":["projected"],"sens_key":"LIVE_LOAD:trafic",
                          "sens_approx_neglam":0.15}})

    # ---- FAMILLE 4 : DEHORS (projection ratee -> ignore) ; un point interieur porte la charge ----
    for y in [1.0, -1.0]:
        cfgs.append({"name":"F4_out_y%+.1f"%y,
            "points":[{"name":"P1","xyz":[2.0,y,0.2],"F":[0,0,-1.0],"role":"live","lc":"trafic"},
                      {"name":"P2","xyz":[2.0,0.0,0.1],"F":[0,0,-1.0],"role":"live","lc":"trafic"}],
            "regions":[R("LIVE_LOAD","trafic")],"mesh_size":"0.10",
            "expect":{"optimal":True,"locate_multiset":{"ignored":1,"located":1}}})
    # point sous la structure, force vers le HAUT (projette vers le haut sur surface basse)
    cfgs.append({"name":"F4_below_upforce",
        "points":[{"name":"P1","xyz":[2.0,0.0,-0.5],"F":[0,0,1.0],"role":"live","lc":"trafic"}],
        "regions":[R("LIVE_LOAD","trafic")],"mesh_size":"0.10",
        "expect":{"optimal":True,"locate":["projected"],"sens_key":"LIVE_LOAD:trafic"}})

    # ---- FAMILLE 5 : LIVE point MOBILE (position sensibilite, signe) ----
    for pos in [0.4, 0.8, 1.2, 1.6]:
        cfgs.append({"name":"F5_live_mobile_pos%.1f"%pos,
            "points":[{"name":"P1","xyz":[2.0,0,0.1],"F":[0,0,-1.0],"role":"live","lc":"trafic",
                       "mobile":{"path":[[1.0,0,0.1],[3.0,0,0.1]],"position":pos,"unit":"absolute"}}],
            "regions":[R("LIVE_LOAD","trafic"),R("LIVE_LOAD","trafic","position")],"mesh_size":"0.10",
            "expect":{"optimal":True,"locate":["located"],"sens_key":"LIVE_LOAD:trafic",
                      "pos_key":"LIVE_LOAD:position:trafic","pos_sign":-1}})

    # ---- FAMILLE 6 : COMBINE footprint LIVE + point DEAD (magnitude) -- l'exemple utilisateur ----
    for xd in [0.8, 1.5, 2.5, 3.2]:
        for md in [0.01, 0.02, 0.04]:
            cfgs.append({"name":"F6_fpLive_ptDead_x%.1f_m%.2f"%(xd,md),
                "footprints":[FP_LIVE(2.0)],
                "points":[{"name":"P1","xyz":[xd,0,0.15],"F":[0,0,-md],"role":"dead","lc":"poids"}],
                "regions":[R("LIVE_LOAD","trafic"),R("DEAD_LOAD","poids"),
                           R("LIVE_LOAD","poids")],  # separation : live de 'poids' -> ~0
                "mesh_size":"0.10",
                "expect":{"optimal":True,"locate":["located"],"sens_key":"LIVE_LOAD:trafic",
                          "sens_approx_neglam":0.05,"dead_key":"DEAD_LOAD:poids",
                          "sep_zero_key":"LIVE_LOAD:poids"}})

    # ---- FAMILLE 7 : COMBINE footprint LIVE + point DEAD MOBILE (position) -- exemple complet ----
    for pos in [0.5, 1.0, 1.5]:
        for md in [0.01, 0.03]:
            cfgs.append({"name":"F7_fpLive_ptDeadMob_p%.1f_m%.2f"%(pos,md),
                "footprints":[FP_LIVE(2.0)],
                "points":[{"name":"P1","xyz":[2.0,0,0.15],"F":[0,0,-md],"role":"dead","lc":"convoi_pt",
                           "mobile":{"path":[[1.0,0,0.15],[3.0,0,0.15]],"position":pos,"unit":"absolute"}}],
                "regions":[R("LIVE_LOAD","trafic"),R("DEAD_LOAD","convoi_pt"),
                           R("DEAD_LOAD","convoi_pt","position"),
                           R("LIVE_LOAD","convoi_pt","position")],  # separation : live pos de dead LC -> ~0
                "mesh_size":"0.10",
                "expect":{"optimal":True,"locate":["located"],"sens_key":"LIVE_LOAD:trafic",
                          "dead_key":"DEAD_LOAD:convoi_pt","pos_key":"DEAD_LOAD:position:convoi_pt",
                          "sep_zero_key":"LIVE_LOAD:position:convoi_pt"}})

    # ---- FAMILLE 8 : DEUX points live mobiles, MEME variable (groupe) vs DIFFERENTES ----
    # meme variable -> le solveur SOMME les dFEXT/ds des deux points sous une cle unique.
    cfgs.append({"name":"F8_two_pts_same_var",
        "points":[{"name":"P1","xyz":[1.5,0,0.1],"F":[0,0,-1.0],"role":"live","lc":"convoi",
                   "mobile":{"path":[[0.8,0,0.1],[2.2,0,0.1]],"position":0.7,"unit":"absolute"}},
                  {"name":"P2","xyz":[2.5,0,0.1],"F":[0,0,-1.0],"role":"live","lc":"convoi",
                   "mobile":{"path":[[1.8,0,0.1],[3.2,0,0.1]],"position":0.7,"unit":"absolute"}}],
        "regions":[R("LIVE_LOAD","convoi"),R("LIVE_LOAD","convoi","position")],"mesh_size":"0.10",
        "expect":{"optimal":True,"locate":["located","located"],"pos_key":"LIVE_LOAD:position:convoi"}})
    cfgs.append({"name":"F8_two_pts_diff_var",
        "points":[{"name":"P1","xyz":[1.5,0,0.1],"F":[0,0,-1.0],"role":"live","lc":"convoiA",
                   "mobile":{"path":[[0.8,0,0.1],[2.2,0,0.1]],"position":0.7,"unit":"absolute"}},
                  {"name":"P2","xyz":[2.5,0,0.1],"F":[0,0,-1.0],"role":"live","lc":"convoiB",
                   "mobile":{"path":[[1.8,0,0.1],[3.2,0,0.1]],"position":0.7,"unit":"absolute"}}],
        "regions":[R("LIVE_LOAD","convoiA","position"),R("LIVE_LOAD","convoiB","position")],"mesh_size":"0.10",
        "expect":{"optimal":True,"locate":["located","located"],
                  "pos_key":"LIVE_LOAD:position:convoiA","pos_key2":"LIVE_LOAD:position:convoiB"}})

    return cfgs

# ---------------- verification par config ----------------
def check(cfg, r):
    e = cfg.get("expect", {})
    fails = []
    if r["err"]: fails.append("ERR:%s" % r["err"])
    # (1)/(3) convergence attendue
    if e.get("optimal") and r["status"] != "OPTIMAL":
        fails.append("status=%s (attendu OPTIMAL)" % r["status"])
    # (6) ruine coherente
    if e.get("optimal") and (r["lam"] is None or not (0 < r["lam"] < 1e6)):
        fails.append("lambda incoherent=%s" % r["lam"])
    # (2) CONSERVATION : tout point localise/projete a SUM(w)=1
    for c in r["conserv"]:
        if c["err_sumw"] > 1e-9:
            fails.append("conserv err=%.2e (ipl %d)" % (c["err_sumw"], c["ipl"]))
    # localisation attendue
    if "locate" in e:
        got = [x for x in r["located"]]
        # on compare l'ensemble des tags attendus present (ordre peut varier)
        for tag in e["locate"]:
            if tag not in got: fails.append("locate manque '%s' (got %s)" % (tag, got))
    if "locate_multiset" in e:
        from collections import Counter
        cnt = Counter(r["located"])
        for tag, n in e["locate_multiset"].items():
            if cnt.get(tag, 0) < n: fails.append("locate '%s' x%d attendu (got %s)" % (tag, n, dict(cnt)))
    # (4) sensibilite magnitude ~ -lambda (tolerance punching)
    if "sens_key" in e and e.get("sens_approx_neglam") is not None and r["status"] == "OPTIMAL":
        s = r["sens"].get(e["sens_key"]); lam = r["lam"]
        if s is None: fails.append("sens '%s' absente" % e["sens_key"])
        elif lam and abs(s - (-lam))/abs(lam) > e["sens_approx_neglam"]:
            fails.append("sens %.4g != -lam %.4g (%.1f%%>%.0f%%)" %
                         (s, -lam, abs(s+lam)/abs(lam)*100, e["sens_approx_neglam"]*100))
    # sensibilite dead presente + non nulle
    if "dead_key" in e and r["status"] == "OPTIMAL":
        s = r["sens"].get(e["dead_key"])
        if s is None or abs(s) < 1e-9: fails.append("dead sens '%s'=%s (attendu non nul)" % (e["dead_key"], s))
    # position : presente, non nulle, signe attendu
    if "pos_key" in e and r["status"] == "OPTIMAL":
        s = r["sens"].get(e["pos_key"])
        if s is None or abs(s) < 1e-9: fails.append("pos sens '%s'=%s (attendu non nul)" % (e["pos_key"], s))
        elif "pos_sign" in e and (s * e["pos_sign"] < 0):
            fails.append("pos sens '%s'=%.4g signe attendu %d" % (e["pos_key"], s, e["pos_sign"]))
    if "pos_key2" in e and r["status"] == "OPTIMAL":
        s = r["sens"].get(e["pos_key2"])
        if s is None or abs(s) < 1e-9: fails.append("pos sens2 '%s'=%s (attendu non nul)" % (e["pos_key2"], s))
    # (5) separation live/dead : cle attendue ~ 0
    if "sep_zero_key" in e and r["status"] == "OPTIMAL":
        s = r["sens"].get(e["sep_zero_key"])
        if s is not None and abs(s) > 1e-6:
            fails.append("separation '%s'=%.4g (attendu ~0)" % (e["sep_zero_key"], s))
    return fails

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="0/1")   # i/N
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    all_cfgs = build_matrix()
    if args.list:
        print("TOTAL configs = %d" % len(all_cfgs))
        for c in all_cfgs: print("  ", c["name"])
        sys.exit(0)
    i, N = map(int, args.slice.split("/"))
    mine = [c for j, c in enumerate(all_cfgs) if j % N == i]
    results = []
    npass = nfail = 0
    for c in mine:
        try:
            r = run_config(c)
        except Exception as ex:
            r = {"name":c["name"],"status":"CRASH","lam":None,"sens":{},"conserv":[],"located":[],"err":repr(ex)[:200]}
        fails = check(c, r)
        ok = (len(fails) == 0)
        npass += ok; nfail += (not ok)
        rec = {"name":c["name"],"status":r["status"],"lam":r["lam"],"ok":ok,"fails":fails,
               "conserv_max_err":max([cc["err_sumw"] for cc in r["conserv"]], default=0.0),
               "located":r["located"],"sens":r["sens"]}
        results.append(rec)
        print("%s %s | %s | lam=%s%s" % ("PASS" if ok else "FAIL", c["name"], r["status"],
              r["lam"], "" if ok else " <<< " + "; ".join(fails)))
    summary = {"slice":args.slice,"npass":npass,"nfail":nfail,"ntotal":len(mine),"results":results}
    if args.out:
        json.dump(summary, open(args.out,"w"), indent=1)
    print("\n==== SLICE %s : %d PASS / %d FAIL / %d total ====" % (args.slice, npass, nfail, len(mine)))
