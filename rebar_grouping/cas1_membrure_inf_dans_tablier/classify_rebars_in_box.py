"""Classe les aciers (REBAR) en 2 groupes selon leur appartenance a une box STEP.

Groupe 1 = aciers dont le CENTRE DE GRAVITE est DANS le solide STEP.
Groupe 2 = tout le reste.

- Aciers + coords : parses depuis le dsCad (pts_<name>.append(POINT(x,y,z)) + REBAR(...)).
- Box : solide polyedrique du fichier STEP (faces planes), test point-dans-solide
  par ray-casting vectorise (gere le non-convexe), via step_solid.py.

Usage : python classify_rebars_in_box.py
Sorties : rebar_groups.json + figure rebar_groups.png (centroides colories).
"""
import os
import re
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_solid import load_triangles, points_inside, find_box_step

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DSCAD = r"C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_LM1_PRESSURE.ds\dsCad.txt"
STEP = find_box_step(OUT_DIR)   # box .stp du dossier courant (cas1/cas2/...)


def parse_rebars(path):
    """Retourne (names list, centroids (N,3))."""
    txt = open(path, encoding='utf-8', errors='replace').read()
    point_lists = {}
    for m in re.finditer(
            r"(pts_\w+)\.append\(POINT\(([-\d.eE]+),\s*([-\d.eE]+),\s*([-\d.eE]+)\)\)", txt):
        var = m.group(1)
        point_lists.setdefault(var, []).append(
            (float(m.group(2)), float(m.group(3)), float(m.group(4))))
    names, centroids = [], []
    for m in re.finditer(r"REBAR\('([^']+)',\s*E=EDGE\(PS=(\w+)\)", txt):
        name, var = m.group(1), m.group(2)
        if var in point_lists:
            pts = np.array(point_lists[var])
            names.append(name)
            centroids.append(pts.mean(axis=0))
    return names, np.array(centroids)


def main():
    print("=" * 70)
    print("CLASSIFICATION DES ACIERS PAR APPARTENANCE A LA BOX STEP")
    print("=" * 70)

    # 1) Box STEP -> triangles
    tris, info = load_triangles(STEP, scale=0.001)  # mm -> m
    print(f"\n[STEP] {os.path.basename(STEP)}")
    print(f"  faces ADVANCED_FACE : {info['n_advanced_face']}")
    print(f"  triangles generes   : {info['n_triangles']}")
    print(f"  bbox solide (m)     : min={np.round(info['bbox_min'],3)} "
          f"max={np.round(info['bbox_max'],3)}")
    if info['n_triangles'] == 0:
        print("  [ERREUR] aucun triangle -> parsing STEP echoue"); return

    # 2) Aciers -> centroides
    names, C = parse_rebars(DSCAD)
    print(f"\n[REBARS] {len(names)} aciers parses depuis le dsCad")
    print(f"  bbox centroides (m) : min={np.round(C.min(0),3)} max={np.round(C.max(0),3)}")

    # 3) Validation du test point-dans-solide sur 2 cas connus
    print(f"\n[VALIDATION] test point-dans-solide sur cas connus :")
    bmin, bmax = np.array(info['bbox_min']), np.array(info['bbox_max'])
    center = (bmin + bmax) / 2
    far = bmax + np.array([10.0, 10.0, 10.0])
    chk = points_inside(np.array([center, far]), tris)
    print(f"  centre bbox {np.round(center,2)} -> inside={chk[0]} (attendu True si solide ~ box pleine)")
    print(f"  point loin  {np.round(far,2)} -> inside={chk[1]} (attendu False)")

    # 4) Classification vectorisee de tous les aciers
    inside = points_inside(C, tris)
    # Override : les HA_8 (Ø8mm) sont des aciers de tablier. ~67 tombent hors box
    # aux coins/extremites du tablier (Y niveau tablier mais bord non couvert) ->
    # forces en groupe 1 (tablier). Cf remarque utilisateur 2026-06-17.
    FORCE_GROUP1_PREFIX = ('HA_8_',)
    n_forced = 0
    for i, nm in enumerate(names):
        if nm.startswith(FORCE_GROUP1_PREFIX) and not inside[i]:
            inside[i] = True
            n_forced += 1
    if n_forced:
        print(f"\n[OVERRIDE] {n_forced} aciers {FORCE_GROUP1_PREFIX} hors box -> forces groupe 1 (tablier)")
    n_in = int(inside.sum())
    print(f"\n[RESULTAT]")
    print(f"  Groupe 1 (DANS la box)  : {n_in:5d} aciers")
    print(f"  Groupe 2 (hors box)     : {len(names)-n_in:5d} aciers")

    # apercu par prefixe de nom (HA_5 vs HA_50 etc.)
    from collections import Counter
    def prefix(n):
        m = re.match(r"(HA_\d+)_", n)
        return m.group(1) if m else n.split('_')[0]
    pin = Counter(prefix(names[i]) for i in range(len(names)) if inside[i])
    pout = Counter(prefix(names[i]) for i in range(len(names)) if not inside[i])
    print(f"  Prefixes groupe 1 : {dict(pin)}")
    print(f"  Prefixes groupe 2 : {dict(pout)}")

    # 5) Sauvegarde JSON
    groups = {
        'group1_in_box': [names[i] for i in range(len(names)) if inside[i]],
        'group2_rest':   [names[i] for i in range(len(names)) if not inside[i]],
        'meta': {'n_total': len(names), 'n_group1': n_in, 'n_group2': len(names) - n_in,
                 'step': os.path.basename(STEP), 'criterion': 'centroid_in_solid'},
    }
    out_json = os.path.join(OUT_DIR, 'rebar_groups.json')
    with open(out_json, 'w') as f:
        json.dump(groups, f, indent=1)
    print(f"\n  -> {out_json}")

    # 6) Visualisation (centroides colories par groupe, vues XY et XZ)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (a, b, la, lb) in [(ax1, (0, 1, 'X (m)', 'Y (m)')),
                                    (ax2, (0, 2, 'X (m)', 'Z (m)'))]:
            ax.scatter(C[~inside, a], C[~inside, b], s=6, c='lightgray', label=f'Groupe 2 hors box ({len(names)-n_in})')
            ax.scatter(C[inside, a], C[inside, b], s=8, c='red', label=f'Groupe 1 dans box ({n_in})')
            ax.set_xlabel(la); ax.set_ylabel(lb); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle('Classification aciers / box STEP (centre de gravite)')
        out_png = os.path.join(OUT_DIR, 'rebar_groups.png')
        fig.savefig(out_png, dpi=130, bbox_inches='tight')
        print(f"  -> {out_png}")
    except Exception as e:
        print(f"  [visu skip] {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
