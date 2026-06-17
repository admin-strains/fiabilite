"""Cas 2 : membrure inferieure basculee dans le TREILLIS (groupe 2).

Base = decoupage du cas 1 (tablier). On bascule G1 -> G2 les barres dont le
barycentre est dans l'un des 2 solides de boxes_membrures_inf.stp ET qui sont
LONGITUDINALES (polyligne ouverte). Les CADRES (boucles fermees = etriers) sont
IGNORES (ils restent en groupe 1 / tablier).

Sorties : rebar_groups_cas2.json + groupes_membrure_inf_dans_treillis.json
+ noms_groupe1_*.txt / noms_groupe2_*.txt + visu.
"""
import os, sys, re, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_solid import load_triangles, points_inside, find_box_step

OUT = os.path.dirname(os.path.abspath(__file__))
DSCAD = r"C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_LM1_PRESSURE.ds\dsCad.txt"
CAS1_JSON = os.path.join(OUT, "..", "cas1_membrure_inf_dans_tablier", "rebar_groups.json")
BOXES = find_box_step(OUT)   # boxes_membrures_inf.stp dans ce dossier
CLOSED_TOL = 1e-3            # boucle fermee si |p0 - p_dernier| < tol -> cadre


def parse_rebars(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    plists = {}
    for m in re.finditer(r"(pts_\w+)\.append\(POINT\(([-\d.eE]+),\s*([-\d.eE]+),\s*([-\d.eE]+)\)\)", txt):
        plists.setdefault(m.group(1), []).append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
    names, polys = [], []
    for m in re.finditer(r"REBAR\('([^']+)',\s*E=EDGE\(PS=(\w+)\)", txt):
        if m.group(2) in plists:
            names.append(m.group(1)); polys.append(np.array(plists[m.group(2)]))
    return names, polys


def main():
    print("=" * 70)
    print("CAS 2 : membrure inferieure -> treillis (groupe 2)")
    print("=" * 70)

    names, polys = parse_rebars(DSCAD)
    C = np.array([p.mean(0) for p in polys])
    is_cadre = np.array([np.linalg.norm(p[0] - p[-1]) < CLOSED_TOL for p in polys])
    print(f"[REBARS] {len(names)} aciers  ({int(is_cadre.sum())} cadres fermes, "
          f"{int((~is_cadre).sum())} longitudinaux)")

    # base cas 1
    cas1 = json.load(open(CAS1_JSON))
    g1_base = set(cas1['group1_in_box'])
    in_g1 = np.array([n in g1_base for n in names])
    print(f"[BASE cas1] G1={int(in_g1.sum())}  G2={len(names)-int(in_g1.sum())}")

    # boites membrure inf
    tris, info = load_triangles(BOXES, scale=0.001)
    print(f"[BOXES] {os.path.basename(BOXES)} : {info['n_advanced_face']} faces, "
          f"{info['n_triangles']} tri, etanche={info['watertight']}, "
          f"bbox Y=[{info['bbox_min'][1]:.2f},{info['bbox_max'][1]:.2f}]")
    in_boxes = points_inside(C, tris)

    # bascule : longitudinaux dans boites, actuellement en G1
    move = in_boxes & (~is_cadre) & in_g1
    print(f"\n[BASCULE G1->G2]")
    print(f"  dans boites           : {int(in_boxes.sum())}")
    print(f"  dont cadres (IGNORES) : {int((in_boxes & is_cadre).sum())}")
    print(f"  dont longitudinaux    : {int((in_boxes & ~is_cadre).sum())}")
    print(f"  bascules (long.+G1)   : {int(move.sum())}")

    inside = in_g1 & (~move)   # groupe 1 final = base G1 moins les bascules
    n1 = int(inside.sum())
    print(f"\n[RESULTAT cas2] G1 tablier (sans membrure inf long.) = {n1}  |  "
          f"G2 structure+membrure inf = {len(names)-n1}")

    g1 = [names[i] for i in range(len(names)) if inside[i]]
    g2 = [names[i] for i in range(len(names)) if not inside[i]]
    moved = [names[i] for i in range(len(names)) if move[i]]

    from collections import Counter
    def pref(n):
        m = re.match(r"(HA_\d+)_", n); return m.group(1) if m else n
    print(f"  diametres bascules : {dict(sorted(Counter(pref(n) for n in moved).items()))}")

    data = {
        'scenario': 'membrure_inf_dans_treillis',
        'description': ('Base cas1 (tablier). Barres LONGITUDINALES de la membrure '
                        'inferieure (barycentre dans boxes_membrures_inf.stp) basculees '
                        'en groupe 2. Cadres (etriers fermes) NON basculees.'),
        'base': 'cas1_membrure_inf_dans_tablier/rebar_groups.json',
        'criterion': 'cas1 - (longitudinaux dans boxes_membrures_inf.stp)',
        'n_groupe1': len(g1), 'n_groupe2': len(g2), 'n_bascules': len(moved),
        'groupe1_tablier': g1, 'groupe2_structure_avec_membrure_inf': g2,
        'bascules_membrure_inf_longitudinale': moved,
    }
    json.dump(data, open(os.path.join(OUT, "groupes_membrure_inf_dans_treillis.json"), 'w'), indent=1)
    json.dump({'group1_in_box': g1, 'group2_rest': g2}, open(os.path.join(OUT, "rebar_groups_cas2.json"), 'w'), indent=1)
    open(os.path.join(OUT, "noms_groupe1_tablier.txt"), 'w').write('\n'.join(g1) + '\n')
    open(os.path.join(OUT, "noms_groupe2_structure_membrureinf.txt"), 'w').write('\n'.join(g2) + '\n')
    open(os.path.join(OUT, "noms_bascules_membrure_inf.txt"), 'w').write('\n'.join(moved) + '\n')
    print("\n-> groupes_membrure_inf_dans_treillis.json, rebar_groups_cas2.json, noms_*.txt")

    # visu : G1 gris, G2 rouge, bascules en bleu vif
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        mv = move
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (i, j, lx, ly) in [(a1, (0, 1, 'X', 'Y')), (a2, (0, 2, 'X', 'Z'))]:
            ax.scatter(C[inside, i], C[inside, j], s=3, c='lightgray', label=f'G1 tablier ({n1})')
            ax.scatter(C[(~inside) & ~mv, i], C[(~inside) & ~mv, j], s=5, c='red', label=f'G2 structure')
            ax.scatter(C[mv, i], C[mv, j], s=18, c='blue', marker='^', label=f'membrure inf basculee ({int(mv.sum())})')
            ax.set_xlabel(lx); ax.set_ylabel(ly); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle('Cas 2 : membrure inf longitudinale basculee en G2 (cadres ignores)')
        fig.savefig(os.path.join(OUT, "cas2_groups.png"), dpi=130, bbox_inches='tight')
        print("-> cas2_groups.png")
    except Exception as e:
        print(f"  [visu skip] {e}")


if __name__ == '__main__':
    main()
