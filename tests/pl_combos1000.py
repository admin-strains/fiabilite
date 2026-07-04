# -*- coding: utf-8 -*-
# =====================================================================
# VALIDATION EXHAUSTIVE ~1000 COMBOS -- charges ponctuelles + footprints (2026-07-04, MM)
# =====================================================================
# Matrice combinatoire de COUPLES/TRIPLES/QUADS de charges (ponctuelle/footprint x live/dead
# x mobile/statique x meme variable/variables differentes), chaque tuile BALAYEE en position
# et comparee a sa COURBE ANALYTIQUE poutre (cantilever encastre en x=0), SANS calibration :
#   (1) dLambda/dm_L  = -lambda                      (identite exacte, toute config)
#   (2) dLambda/dm_D  = -M_D / M_L                   (M = somme F_i * x_i du role)
#   (3) dLambda/ds_L  = -lambda * tau_x / x_L        (membre live mobile)
#   (4) dLambda/ds_D  = -F_D * tau_x / M_L           (membre dead mobile)
#   meme variable     : cle partagee = somme des contributions (live + dead)
#   groupement multi-LC magnitude : cle groupee == somme des cles individuelles
# + conservation exacte SUM(w)=1 (points), separation live/dead ~0, convergence, signes.
#
# Familles : A couples (pt/fp dead x pt/fp live) | B mixte meme-LC (pt+fp ensemble)
#            C triples + groupement multi-LC     | D robustesse (mesh/oblique/projection/
#            ignore/noeud)                       | E singles regression | R random seedes
#
#   C:\python3\python.exe tests\pl_combos1000.py --list
#   C:\python3\python.exe tests\pl_combos1000.py --slice 0/6 --out res.json
# =====================================================================
import sys, os, json, argparse, random, math
sys.path.insert(0, r'C:\workspace\fiabilite\tests')
from pl_harness import run_config

Z_TOP = 0.25          # surface haute de la poutre (h/2)
X_FIX = 0.0           # encastrement

# ------------------------- membres parametriques -------------------------
def mk_pt(role, lc, x, F=None, z=0.14, y=0.0, mobile=None, name="P"):
    F = F if F is not None else ([0,0,-1.0] if role=="live" else [0,0,-0.02])
    m = {"name": name, "xyz":[x, y, z], "F": F, "role": role, "lc": lc}
    if mobile: m["mobile"] = mobile
    return m

def mk_fp(role, lc, xc, p=None, w=0.6, d=0.3, mobile=None, name="FP"):
    p = p if p is not None else (-0.15 if role=="live" else -0.05)
    m = {"name": name, "polygon":[[xc-w/2,-d/2,Z_TOP],[xc+w/2,-d/2,Z_TOP],
                                  [xc+w/2,d/2,Z_TOP],[xc-w/2,d/2,Z_TOP]],
         "F":[0,0,p], "role": role, "lc": lc}
    if mobile: m["mobile"] = mobile
    return m

def mob(x, z=None, var=None):
    """chemin absolu le long de X centre sur x (tangente x=1). z=None -> Z_TOP (footprint)."""
    zz = Z_TOP if z is None else z
    m = {"path":[[x-1.0,0,zz],[x+1.0,0,zz]], "position":1.0, "unit":"absolute"}
    if var: m["variable"] = var
    return m

def Ftot_x(m):
    """(force totale verticale |Fz_tot|, bras de levier EFFECTIF x) d'un membre.
    2026-07-04 fix : pour une force OBLIQUE, le moment depend de la LIGNE D'ACTION ->
    x_eff = x + (Fx/|Fz|)*z (invariant par projection le long de la force : le travail
    de Fx sur u_x = theta*z ajoute Fx*z au moment, quel que soit le point d'application
    sur la ligne). Sans ce terme, les combos XZ+above etaient sous-estimes de ~20%."""
    if "polygon" in m:
        xs=[q[0] for q in m["polygon"]]; ys=[q[1] for q in m["polygon"]]
        A=(max(xs)-min(xs))*(max(ys)-min(ys))
        return abs(m["F"][2])*A, sum(xs)/len(xs)
    Fx, Fz = m["F"][0], m["F"][2]
    x_eff = m["xyz"][0] + (Fx/abs(Fz))*m["xyz"][2] if abs(Fz) > 1e-12 else m["xyz"][0]
    return abs(Fz), x_eff

# ------------------------- generation d'une config -------------------------
def build_cfg(name, members, var_mode, mesh="0.10", extra_regions=None, tags=None):
    """var_mode: 'diff' (groupes = noms LC) | 'same' (variable partagee 's_sh' pour tous
    les membres mobiles). Regions = mag+pos par LC (+ cles partagees en mode same)."""
    lcs_live = sorted({m["lc"] for m in members if m["role"]=="live"})
    lcs_dead = sorted({m["lc"] for m in members if m["role"]=="dead"})
    regions = []
    for lc in lcs_live: regions.append({"param":"LIVE_LOAD","load_case":lc})
    for lc in lcs_dead: regions.append({"param":"DEAD_LOAD","load_case":lc})
    mobile_live = [m for m in members if m["role"]=="live" and m.get("mobile")]
    mobile_dead = [m for m in members if m["role"]=="dead" and m.get("mobile")]
    if var_mode == "same":
        for m in members:
            if m.get("mobile"): m["mobile"]["variable"] = "s_sh"
        if mobile_live: regions.append({"param":"LIVE_LOAD","axis":"position","load_case":"s_sh"})
        if mobile_dead: regions.append({"param":"DEAD_LOAD","axis":"position","load_case":"s_sh"})
    else:
        for lc in sorted({m["lc"] for m in mobile_live}):
            regions.append({"param":"LIVE_LOAD","axis":"position","load_case":lc})
        for lc in sorted({m["lc"] for m in mobile_dead}):
            regions.append({"param":"DEAD_LOAD","axis":"position","load_case":lc})
        # separation : cle position LIVE d'un LC dead mobile -> ~0
        for lc in sorted({m["lc"] for m in mobile_dead}):
            regions.append({"param":"LIVE_LOAD","axis":"position","load_case":lc})
    if extra_regions: regions += extra_regions
    pts = [m for m in members if "xyz" in m]
    fps = [m for m in members if "polygon" in m]
    return {"name": name, "points": pts, "footprints": fps, "regions": regions,
            "mesh_size": mesh, "_members": members, "_var_mode": var_mode,
            "_tags": tags or {}}

# ------------------------- matrice -------------------------
def build_matrix():
    cfgs = []
    rng = random.Random(42)
    SL = [1.2, 1.8, 2.4, 3.0]          # positions balayage LIVE
    SD = [0.9, 1.7, 2.5]               # positions balayage DEAD

    def add_couple(fam, dt, lt, var_mode, xl, xd, mL=1.0, mD=1.0, mesh="0.10",
                   live_dir="Z", live_place="int", live_mobile=True, dead_mobile=True):
        nm = f"{fam}_{dt}D_{lt}L_{var_mode}_xl{xl:.1f}_xd{xd:.1f}_mL{mL}_mD{mD}_{live_dir}_{live_place}_M{mesh}"
        mem = []
        # LIVE
        if lt == "pt":
            F = [0,0,-1.0*mL] if live_dir=="Z" else ([0.4*mL,0,-1.0*mL] if live_dir=="XZ" else [0.3*mL,0.2*mL,-1.0*mL])
            z = 0.14 if live_place=="int" else 0.6         # 'above' -> projection
            mem.append(mk_pt("live","LCL", xl, F=F, z=z, name="PL",
                             mobile=(mob(xl, z=z) if live_mobile else None)))
        else:
            mem.append(mk_fp("live","LCL", xl, p=-0.15*mL, name="FPL",
                             mobile=(mob(xl) if live_mobile else None)))
        # DEAD
        if dt == "pt":
            mem.append(mk_pt("dead","LCD", xd, F=[0,0,-0.02*mD], z=0.12, name="PD",
                             mobile=(mob(xd, z=0.12) if dead_mobile else None)))
        else:
            mem.append(mk_fp("dead","LCD", xd, p=-0.05*mD, w=0.5, d=0.3, name="FPD",
                             mobile=(mob(xd) if dead_mobile else None)))
        cfgs.append(build_cfg(nm, mem, var_mode, mesh=mesh,
                              tags={"fam":fam,"dt":dt,"lt":lt,"vm":var_mode,"xl":xl,"xd":xd}))

    # ---- FAMILLE A : 4 couples x {diff,same} x SL x SD x 2 magnitudes = 4*2*4*3*2 = 192
    for dt in ("pt","fp"):
        for lt in ("pt","fp"):
            for vm in ("diff","same"):
                for xl in SL:
                    for xd in SD:
                        for (mL,mD) in ((1.0,1.0),(0.6,0.5)):
                            add_couple("A", dt, lt, vm, xl, xd, mL, mD)

    # ---- FAMILLE A2 : variantes live pt (direction oblique / projection) sur sous-grille
    #      2 couples(pt live) x {diff,same} x {XZ,XYZ}x{int} + {Z}x{above} x 2 SL x 2 SD = 96
    for dt in ("pt","fp"):
        for vm in ("diff","same"):
            for (ldir, lpl) in (("XZ","int"), ("XYZ","int"), ("Z","above")):
                for xl in SL[::2]:
                    for xd in SD[::2]:
                        add_couple("A2", dt, "pt", vm, xl, xd, live_dir=ldir, live_place=lpl)

    # ---- FAMILLE B : MEME LC mixte (pt + fp balayes ENSEMBLE dans un LC, philosophie
    #      "par load case") x role du LC mixte {live,dead} x {diff,same} x SL x SD = 96
    for mixed_role in ("live","dead"):
        for vm in ("diff","same"):
            for xl in SL:
                for xd in SD:
                    if mixed_role == "live":     # LC live = pt+fp ensemble ; dead = fp seul
                        mem = [mk_pt("live","LCL", xl, F=[0,0,-0.5], z=0.14, name="PL", mobile=mob(xl, z=0.14)),
                               mk_fp("live","LCL", xl+0.3, p=-0.12, name="FPL", mobile=mob(xl+0.3)),
                               mk_fp("dead","LCD", xd, p=-0.05, w=0.5, name="FPD", mobile=mob(xd))]
                    else:                        # LC dead = pt+fp ensemble ; live = fp seul
                        mem = [mk_fp("live","LCL", xl, p=-0.15, name="FPL", mobile=mob(xl)),
                               mk_pt("dead","LCD", xd, F=[0,0,-0.015], z=0.12, name="PD", mobile=mob(xd, z=0.12)),
                               mk_fp("dead","LCD", xd+0.3, p=-0.04, w=0.4, name="FPD", mobile=mob(xd+0.3))]
                    cfgs.append(build_cfg(f"B_{mixed_role}mix_{vm}_xl{xl:.1f}_xd{xd:.1f}", mem, vm,
                                          tags={"fam":"B","mix":mixed_role,"vm":vm,"xl":xl,"xd":xd}))

    # ---- FAMILLE C : triples + GROUPEMENT multi-LC magnitude (groupe == somme) ----
    # C1 : 2 LC dead (pt + fp) + 1 live {pt|fp} ; region DEAD groupee [LCD1,LCD2] vs individuelles
    for lt in ("pt","fp"):
        for vm in ("diff","same"):
            for xl in SL:
                for xd in SD[::2]:
                    mem = [ (mk_pt("live","LCL", xl, F=[0,0,-1.0], z=0.14, name="PL", mobile=mob(xl,z=0.14))
                             if lt=="pt" else mk_fp("live","LCL", xl, name="FPL", mobile=mob(xl))),
                            mk_pt("dead","LCD1", xd, F=[0,0,-0.015], z=0.12, name="PD", mobile=mob(xd,z=0.12)),
                            mk_fp("dead","LCD2", xd+0.5, p=-0.04, w=0.4, name="FPD", mobile=mob(xd+0.5)) ]
                    extra = [{"param":"DEAD_LOAD","load_case":["LCD1","LCD2"],"region_key":"grpD"}]
                    cfgs.append(build_cfg(f"C1_{lt}L_{vm}_xl{xl:.1f}_xd{xd:.1f}", mem, vm,
                                          extra_regions=extra,
                                          tags={"fam":"C1","lt":lt,"vm":vm,"xl":xl,"xd":xd,"grpD":True}))
    # C2 : 2 LC live (pt + fp) + 1 dead fp ; region LIVE groupee [LCL1,LCL2] vs individuelles
    for vm in ("diff","same"):
        for xl in SL:
            for xd in SD:
                mem = [ mk_pt("live","LCL1", xl, F=[0,0,-0.6], z=0.14, name="PL", mobile=mob(xl,z=0.14)),
                        mk_fp("live","LCL2", xl+0.6, p=-0.12, name="FPL", mobile=mob(xl+0.6)),
                        mk_fp("dead","LCD", xd, p=-0.05, w=0.5, name="FPD", mobile=mob(xd)) ]
                extra = [{"param":"LIVE_LOAD","load_case":["LCL1","LCL2"],"region_key":"grpL"}]
                cfgs.append(build_cfg(f"C2_{vm}_xl{xl:.1f}_xd{xd:.1f}", mem, vm,
                                      extra_regions=extra,
                                      tags={"fam":"C2","vm":vm,"xl":xl,"xd":xd,"grpL":True}))

    # ---- FAMILLE D : robustesse ----
    # D1 mesh x couple pt-dead + fp-live
    for mesh in ("0.08","0.12"):
        for vm in ("diff","same"):
            for xl in SL[::2]:
                for xd in SD[::2]:
                    add_couple("D1","pt","fp",vm,xl,xd,mesh=mesh)
    # D2 : point live IGNORE (hors structure) + compagnon valide -> le reste intact.
    # 2026-07-04 fix : le point dehors dans son PROPRE LC STATIQUE (pas dans le LC mobile !
    # MOVING_LOAD re-ancre le BARYCENTRE du convoi -> un point a y=1.5 dans le convoi
    # decale tout le monde hors poutre -> LC live vide -> pivot singulier).
    for xl in SL[::2]:
        for xd in SD[::2]:
            mem = [ mk_pt("live","LCL", xl, F=[0,0,-1.0], z=0.14, name="PL", mobile=mob(xl,z=0.14)),
                    mk_pt("live","LCX", 2.0, F=[0,0,-0.3], z=0.2, y=1.5, name="PX"),   # dehors -> ignore
                    mk_fp("dead","LCD", xd, p=-0.05, w=0.5, name="FPD", mobile=mob(xd)) ]
            cfgs.append(build_cfg(f"D2_ign_xl{xl:.1f}_xd{xd:.1f}", mem, "diff",
                                  tags={"fam":"D2","xl":xl,"xd":xd,"ignored":1, "lcx_zero":True}))
    # D3 : live statique + dead mobile / live mobile + dead statique
    for (lmob, dmob) in ((False,True),(True,False)):
        for dt in ("pt","fp"):
            for lt in ("pt","fp"):
                for xl in SL[::2]:
                    for xd in SD[::2]:
                        add_couple("D3",dt,lt,"diff",xl,xd,live_mobile=lmob,dead_mobile=dmob)

    # ---- FAMILLE E : quads (2 dead + 2 live, tous types) sous-echantillonnes ----
    for vm in ("diff","same"):
        for xl in SL:
            for xd in SD[::2]:
                mem = [ mk_pt("live","LCL1", xl, F=[0,0,-0.6], z=0.14, name="PL", mobile=mob(xl,z=0.14)),
                        mk_fp("live","LCL2", xl+0.5, p=-0.10, name="FPL", mobile=mob(xl+0.5)),
                        mk_pt("dead","LCD1", xd, F=[0,0,-0.012], z=0.12, name="PD", mobile=mob(xd,z=0.12)),
                        mk_fp("dead","LCD2", xd+0.6, p=-0.035, w=0.4, name="FPD", mobile=mob(xd+0.6)) ]
                cfgs.append(build_cfg(f"E_quad_{vm}_xl{xl:.1f}_xd{xd:.1f}", mem, vm,
                                      tags={"fam":"E","vm":vm,"xl":xl,"xd":xd}))

    # ---- FAMILLE R : tirages aleatoires seedes pour completer ~1000 ----
    while len(cfgs) < 1000:
        dt, lt = rng.choice(("pt","fp")), rng.choice(("pt","fp"))
        vm = rng.choice(("diff","same"))
        xl = round(rng.uniform(1.0, 3.2), 2); xd = round(rng.uniform(0.7, 2.9), 2)
        mL = round(rng.uniform(0.4, 1.6), 2); mD = round(rng.uniform(0.4, 1.4), 2)
        ldir = rng.choice(("Z","Z","XZ")); lpl = rng.choice(("int","int","above")) if lt=="pt" else "int"
        add_couple("R", dt, lt, vm, xl, xd, mL, mD, live_dir=ldir if lt=="pt" else "Z",
                   live_place=lpl)
    return cfgs

# ------------------------- verification analytique -------------------------
def check(cfg, r):
    fails, info = [], {}
    mem = cfg["_members"]; vm = cfg["_var_mode"]; tags = cfg["_tags"]
    if r["err"]: return ["ERR:"+r["err"]], info
    if r["status"] != "OPTIMAL": return ["status=%s" % r["status"]], info
    lam = r["lam"]
    if not (lam and 0 < lam < 1e6): return ["lambda incoherent=%s" % lam], info
    # conservation exacte (chaque point localise/projete)
    for c in r["conserv"]:
        if c["err_sumw"] > 1e-9: fails.append("conserv %.1e" % c["err_sumw"])
    npts_expected = sum(1 for m in mem if "xyz" in m) - int(tags.get("ignored", 0))
    # moments analytiques
    M_L = sum(Ftot_x(m)[0]*Ftot_x(m)[1] for m in mem if m["role"]=="live" and m.get("name")!="PX")
    M_D = sum(Ftot_x(m)[0]*Ftot_x(m)[1] for m in mem if m["role"]=="dead")
    s = r["sens"]
    def rel(a, b): return abs(a-b)/max(abs(b), 1e-12)
    # (1) magnitude LIVE totale == -lambda (somme des cles live individuelles)
    lcs_live = sorted({m["lc"] for m in mem if m["role"]=="live"})
    magL = [s.get("LIVE_LOAD:"+lc) for lc in lcs_live]
    if any(v is None for v in magL): fails.append("cle magL absente")
    else:
        tot = sum(magL); info["magL_err"] = rel(tot, -lam)
        if info["magL_err"] > 0.05: fails.append("magL %.4g != -lam %.4g (%.1f%%)" % (tot, -lam, info["magL_err"]*100))
    # (2) magnitude DEAD totale == -M_D/M_L
    lcs_dead = sorted({m["lc"] for m in mem if m["role"]=="dead"})
    if lcs_dead:
        magD = [s.get("DEAD_LOAD:"+lc) for lc in lcs_dead]
        if any(v is None for v in magD): fails.append("cle magD absente")
        else:
            tot = sum(magD); ana = -M_D/M_L; info["magD_err"] = rel(tot, ana)
            if info["magD_err"] > 0.20: fails.append("magD %.4g != ana %.4g (%.1f%%)" % (tot, ana, info["magD_err"]*100))
    # groupement multi-LC == somme des individuelles (exact)
    if tags.get("grpD"):
        g = s.get("DEAD_LOAD:grpD")
        if g is None: fails.append("cle grpD absente")
        elif rel(g, sum(s.get("DEAD_LOAD:"+lc, 0.0) for lc in lcs_dead)) > 0.01:
            fails.append("grpD != somme")
    if tags.get("grpL"):
        g = s.get("LIVE_LOAD:grpL")
        if g is None: fails.append("cle grpL absente")
        elif rel(g, sum(s.get("LIVE_LOAD:"+lc, 0.0) for lc in lcs_live)) > 0.01:
            fails.append("grpL != somme")
    # (3)/(4) positions : tau_x = D_x du chemin (=2.0 m ici, unit absolute -> tangente unitaire x=1)
    TAUX = 1.0
    def ana_pos(role_members):
        tot = 0.0
        for m in role_members:
            F, x = Ftot_x(m)
            if m["role"] == "live": tot += -lam * TAUX / max(x, 1e-9) * (F*x/M_L)   # ponderation multi-live
            else:                    tot += -F * TAUX / M_L
        return tot
    if vm == "same":
        mobL = [m for m in mem if m["role"]=="live" and m.get("mobile")]
        mobD = [m for m in mem if m["role"]=="dead" and m.get("mobile")]
        if mobL:
            v = s.get("LIVE_LOAD:position:s_sh")
            if v is None: fails.append("cle posL(s_sh) absente")
            else:
                ana = ana_pos(mobL); info["posL_err"] = rel(v, ana)
                if info["posL_err"] > 0.25: fails.append("posL %.4g != ana %.4g (%.1f%%)" % (v, ana, info["posL_err"]*100))
        if mobD:
            v = s.get("DEAD_LOAD:position:s_sh")
            if v is None: fails.append("cle posD(s_sh) absente")
            else:
                ana = ana_pos(mobD); info["posD_err"] = rel(v, ana)
                if info["posD_err"] > 0.25: fails.append("posD %.4g != ana %.4g (%.1f%%)" % (v, ana, info["posD_err"]*100))
    else:
        for lc in sorted({m["lc"] for m in mem if m["role"]=="live" and m.get("mobile")}):
            v = s.get("LIVE_LOAD:position:"+lc)
            ana = ana_pos([m for m in mem if m["lc"]==lc and m.get("mobile")])
            if v is None: fails.append("cle posL(%s) absente" % lc)
            else:
                info.setdefault("posL_err", rel(v, ana))
                if rel(v, ana) > 0.25: fails.append("posL(%s) %.4g != ana %.4g (%.1f%%)" % (lc, v, ana, rel(v, ana)*100))
        for lc in sorted({m["lc"] for m in mem if m["role"]=="dead" and m.get("mobile")}):
            v = s.get("DEAD_LOAD:position:"+lc)
            ana = ana_pos([m for m in mem if m["lc"]==lc and m.get("mobile")])
            if v is None: fails.append("cle posD(%s) absente" % lc)
            else:
                info.setdefault("posD_err", rel(v, ana))
                if rel(v, ana) > 0.25: fails.append("posD(%s) %.4g != ana %.4g (%.1f%%)" % (lc, v, ana, rel(v, ana)*100))
            # SEPARATION : la cle LIVE position d'un LC dead mobile doit etre ~0
            vz = s.get("LIVE_LOAD:position:"+lc)
            if vz is not None and abs(vz) > 1e-6: fails.append("separation posL(%s)=%.3g" % (lc, vz))
    # point ignore attendu
    if tags.get("ignored") and "ignored" not in r["located"]:
        fails.append("point dehors non ignore")
    return fails, info

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="0/1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--names", default=None, help="liste de noms exacts separes par des virgules (rejouer des combos precis)")
    args = ap.parse_args()
    cfgs = build_matrix()
    if args.names:
        wanted = set(args.names.split(","))
        cfgs = [c for c in cfgs if c["name"] in wanted]
        print("REJOUE %d/%d configs nommees" % (len(cfgs), len(wanted)))
    if args.list:
        from collections import Counter
        print("TOTAL =", len(cfgs))
        print(Counter(c["_tags"].get("fam","?") for c in cfgs))
        sys.exit(0)
    i, N = map(int, args.slice.split("/"))
    mine = [c for j, c in enumerate(cfgs) if j % N == i]
    if args.limit: mine = mine[:args.limit]
    results = []; npass = nfail = 0
    for k, c in enumerate(mine):
        try:
            r = run_config(c)
        except Exception as ex:
            r = {"err": repr(ex)[:200], "status":"CRASH", "lam":None, "sens":{}, "conserv":[], "located":[]}
        fails, info = check(c, r)
        ok = not fails; npass += ok; nfail += (not ok)
        results.append({"name": c["name"], "tags": c["_tags"], "ok": ok, "fails": fails,
                        "lam": r.get("lam"), "sens": r.get("sens"), "errs": info})
        print("%s [%d/%d] %s%s" % ("PASS" if ok else "FAIL", k+1, len(mine), c["name"],
              "" if ok else " <<< " + "; ".join(fails)), flush=True)
    if args.out:
        json.dump({"slice": args.slice, "npass": npass, "nfail": nfail, "results": results},
                  open(args.out, "w"), indent=1)
    print("\n==== SLICE %s : %d PASS / %d FAIL / %d ====" % (args.slice, npass, nfail, len(mine)))
