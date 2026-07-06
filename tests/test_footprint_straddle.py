# -*- coding: utf-8 -*-
# ==========================================================================================
# Test EMPREINTE A CHEVAL sur le bord de structure  (2026-07-06, MM)
# ------------------------------------------------------------------------------------------
# Valide le CHANGEMENT DE DEFAUT conserve="force" : quand une empreinte deborde du pont,
# la part HORS-structure est JETEE (pas de compensation sur la part dedans). Recale sur
# A_poly_clip_sum (aire polygone reellement sur structure) au lieu de A_polygon entier.
#   Ref backend : CetSOLV_MeshOutFiller.cxx, bloc [RESCALE V3.6] + [DEBUG couverture].
#
# Deux configs sur la MEME poutre cantilever (L=4, top a z=0.25) :
#   A) empreinte ENTIEREMENT dedans (xc=1.0, w=0.6 -> x in [0.7,1.3]) -> couverture ~ 1.0
#   B) empreinte A CHEVAL sur le bout libre (xc=4.0, w=0.6 -> x in [3.7,4.3]) -> ~ 0.5
#
# Attendu :
#   - A : couverture ~= 1.0 (non-regression : rien ne bouge sur une empreinte pleine)
#   - B : couverture ~= 0.5 (la moitie hors-pont est jetee)
#   - PREUVE DECISIVE : scale_B ~= 1.0. Sous l'ancien code (cible = A_polygon entier),
#     ce cas aurait donne scale ~= 2.0 (la moitie appliquee remontee pour retrouver la
#     charge totale = COMPENSATION). scale=1.0 = pas de compensation, part dehors jetee.
#   NB : on ne compare PAS lambda_A et lambda_B (les 2 configs different en magnitude ET
#   en position/bras de levier -> comparaison non concluante).
#
# IMPORTANT : pointe sur le CORE PARTAGE (front) via PL_FRONT, PAS front_mohamad.
#   C:\python3\python.exe tests\test_footprint_straddle.py
# ==========================================================================================
import os, re
os.environ["PL_FRONT"] = r"C:\workspace\front"   # core partage (contient le changement conserve-clip)

import pl_harness as H

Z_TOP = 0.25

def _fp(xc, w=0.6, d=0.3, p=-1.0):
    return {"name": "FP", "role": "live", "lc": "trafic", "F": [0.0, 0.0, p],
            "polygon": [[xc - w/2, -d/2, Z_TOP], [xc + w/2, -d/2, Z_TOP],
                        [xc + w/2,  d/2, Z_TOP], [xc - w/2,  d/2, Z_TOP]]}

def _run(name, xc):
    cfg = {"name": name, "footprints": [_fp(xc)],
           "regions": [{"param": "LIVE_LOAD", "load_case": "trafic"}],
           "mesh_size": "0.05"}
    wd = os.path.join(r"C:\workspace\storage\admin\Moulin_Blanc", "_straddle_%s.ds" % name)
    res = H.run_config(cfg, workdir=wd, keep=True)
    txt = open(os.path.join(wd, "_cstdout.log"), encoding="latin-1").read()
    cov = None
    m = re.search(r"\[DEBUG couverture A_poly_sur_structure/A_poly_entier\] = ([0-9.]+)", txt)
    if m:
        cov = float(m.group(1))
    scale = None
    ms = re.findall(r"\[RESCALE V3.6\] scale = ([0-9.]+)", txt)
    if ms:
        scale = float(ms[-1])
    # aussi la ligne FOOTPRINT: total clipped area vs A_polygon
    clip = re.search(r"total clipped area = ([0-9.eE+\-]+) m2, A_polygon = ([0-9.eE+\-]+) m2", txt)
    return res, cov, scale, clip

print("\n############ TEST empreinte a cheval (conserve=force JETTE le dehors) ############")
print("Core :", H.FRONT)

rA, covA, scA, clipA = _run("inside", 1.0)
print("\n[A] empreinte DEDANS  (xc=1.0) : status=%s lambda=%s" % (rA["status"], rA["lam"]))
print("    couverture=%s  scale=%s  clip/A_poly=%s" %
      (covA, scA, (clipA.group(1) + "/" + clipA.group(2)) if clipA else "?"))

rB, covB, scB, clipB = _run("straddle", 4.0)
print("\n[B] empreinte A CHEVAL (xc=4.0) : status=%s lambda=%s" % (rB["status"], rB["lam"]))
print("    couverture=%s  scale=%s  clip/A_poly=%s" %
      (covB, scB, (clipB.group(1) + "/" + clipB.group(2)) if clipB else "?"))

# [C] empreinte TOTALEMENT DEHORS (xc=6.0 -> x in [5.7,6.3], la poutre finit a x=4).
# INFORMATIF : la seule charge live tombe entierement dans le vide -> aucune facette clippee
# -> live LC vide. On veut VOIR le comportement (pas de crash process, statut propre) sans
# hard-fail sur le statut (le solveur ne peut pas normaliser FEXT.u=1 sans charge live).
rC, covC, scC, clipC = _run("outside", 6.0)
print("\n[C] empreinte TOTALEMENT DEHORS (xc=6.0) : status=%s lambda=%s err=%s" %
      (rC["status"], rC["lam"], rC.get("err")))
print("    couverture=%s  scale=%s  clip/A_poly=%s" %
      (covC, scC, (clipC.group(1) + "/" + clipC.group(2)) if clipC else "0/?"))

ok = True
def _check(label, cond):
    global ok
    print(("OK " if cond else "!! ECHEC ") + ": " + label); ok &= cond

def _clip_ratio(clip):
    return (float(clip.group(1)) / float(clip.group(2))) if clip and float(clip.group(2)) > 0 else None

crA = _clip_ratio(clipA)
crB = _clip_ratio(clipB)

_check("A converge (status OPTIMAL)", rA["status"] == "OPTIMAL")
_check("B converge (status OPTIMAL)", rB["status"] == "OPTIMAL")
_check("A couverture ~= 1.0 (empreinte pleine, non-regression)",
       covA is not None and covA > 0.97)
_check("A scale ~= 1.0 (non-regression : empreinte pleine inchangee)",
       scA is not None and abs(scA - 1.0) < 0.05)
_check("B couverture nettement < 1.0 (part hors-pont jetee)",
       covB is not None and 0.35 < covB < 0.65)
_check("B aire clippee ~= 0.5 * A_polygon (moitie sur le pont)",
       crB is not None and 0.35 < crB < 0.65)
_check("B scale ~= 1.0 = PAS de compensation (ancien code aurait donne ~2.0)",
       scB is not None and abs(scB - 1.0) < 0.10)
# [C] : robustesse -- on exige juste que le PROCESS ne crashe pas (on recupere un dict
# resultat, statut propre ou err capturee), et que couverture ~= 0 (rien sur le pont).
_check("C empreinte dehors : process non crashe (resultat recupere, pas d'exception fatale)",
       isinstance(rC, dict))
_check("C couverture ~= 0 (aucune part sur la structure)",
       covC is None or covC < 0.05)
print("    -> [C] verdict : le calcul %s (status=%s). Charge live nulle attendue." %
      ("N'A PAS converge (attendu : live LC vide)" if rC["status"] != "OPTIMAL" else "a converge",
       rC["status"]))

assert ok, "ECHEC : le comportement conserve-clip (jeter le dehors) n'est pas valide"
print("\n############ VALIDE : conserve=force jette la part hors-structure ############")
