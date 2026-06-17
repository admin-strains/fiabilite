"""Vues 3D de la classification des aciers (groupe 1 dans box = rouge, groupe 2 = gris).
Genere une planche multi-angles + le contour du solide box.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_solid import load_triangles, points_inside
from classify_rebars_in_box import parse_rebars, DSCAD, STEP, OUT_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

names, C = parse_rebars(DSCAD)
tris, info = load_triangles(STEP, scale=0.001)
inside = points_inside(C, tris)
n_in = int(inside.sum())
print(f"Groupe 1 (box): {n_in}  Groupe 2: {len(names)-n_in}")

# remap (X, Z, Y) -> axe vertical matplotlib = notre Y
def xzy(P):
    return P[:, 0], P[:, 2], P[:, 1]

# aretes uniques du solide box (pour wireframe), remappees en (X,Z,Y)
edges = set()
for t in tris:
    pid = [tuple(np.round(p, 3)) for p in t]
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        edges.add(tuple(sorted((pid[a], pid[b]))))
box_segments = [np.array([[a[0], a[2], a[1]], [b[0], b[2], b[1]]]) for a, b in edges]

Cin = C[inside]
Cout = C[~inside]

# sous-echantillonnage leger du gris pour lisibilite (garde tout le rouge)
step_out = max(1, len(Cout) // 4000)
Coutv = Cout[::step_out]

def draw(ax, Cin_v, Cout_v, elev, azim, title, marker_in=5, marker_out=3):
    lc = Line3DCollection(box_segments, colors='steelblue', linewidths=0.6, alpha=0.5)
    ax.add_collection3d(lc)
    xo, zo, yo = xzy(Cout_v)
    xi, zi, yi = xzy(Cin_v)
    ax.scatter(xo, zo, yo, s=marker_out, c='lightgray', alpha=0.5)
    ax.scatter(xi, zi, yi, s=marker_in, c='red', alpha=0.75)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X (long.)'); ax.set_ylabel('Z (larg.)'); ax.set_zlabel('Y (vert.)')
    ax.set_title(title, fontsize=9)

# --- Planche 1 : vue d'ensemble, 6 angles, aspect AUTO (cube rempli) ---
ANGLES = [(20, -60), (15, 35), (30, 110), (8, 88), (88, -90), (3, 0)]
TITLES = ['3/4 avant', '3/4 arriere', 'laterale', 'de cote (axe Z)',
          'de dessus (XZ)', 'de face (XY)']
fig = plt.figure(figsize=(20, 11))
for k, ((elev, azim), title) in enumerate(zip(ANGLES, TITLES)):
    ax = fig.add_subplot(2, 3, k + 1, projection='3d')
    draw(ax, Cin, Coutv, elev, azim, f'{title}  (elev={elev}, azim={azim})')
    if k == 0:
        ax.scatter([], [], [], s=20, c='red', label=f'Groupe 1 box ({n_in})')
        ax.scatter([], [], [], s=20, c='lightgray', label=f'Groupe 2 hors box ({len(names)-n_in})')
        ax.legend(fontsize=8, loc='upper left')
fig.suptitle('Aciers Moulin Blanc — vues 3D ensemble (aspect auto, X long / Y vert / Z larg)', fontsize=13)
out1 = os.path.join(OUT_DIR, 'rebar_groups_3d.png')
fig.savefig(out1, dpi=120, bbox_inches='tight'); print(f"-> {out1}")

# --- Planche 2 : coupe centrale (tranche |X|<8 m) pour voir tablier vs structure ---
slab = np.abs(C[:, 0]) < 8.0
Cin2 = C[inside & slab]; Cout2 = C[(~inside) & slab]
fig2 = plt.figure(figsize=(20, 11))
ANGLES2 = [(18, -70), (12, 20), (5, 90), (3, 0), (88, -90), (40, 135)]
TITLES2 = ['3/4', '3/4 bis', 'de cote (Z)', 'de face (XY)', 'dessus', 'plongee']
for k, ((elev, azim), title) in enumerate(zip(ANGLES2, TITLES2)):
    ax = fig2.add_subplot(2, 3, k + 1, projection='3d')
    draw(ax, Cin2, Cout2, elev, azim, f'tranche |X|<8m — {title}', marker_in=10, marker_out=8)
fig2.suptitle('Aciers Moulin Blanc — COUPE CENTRALE |X|<8m (tablier rouge vs structure montante grise)', fontsize=13)
out2 = os.path.join(OUT_DIR, 'rebar_groups_3d_coupe.png')
fig2.savefig(out2, dpi=120, bbox_inches='tight'); print(f"-> {out2}")
