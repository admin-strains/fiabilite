"""Vues 3D individuelles (1 par image) des aciers classes en 2 groupes.

Genere deux series d'images :
  - vrais_*  : la VRAIE geometrie des aciers (polylignes des points du dsCad)
  - bary_*   : seulement les BARYCENTRES (centres de gravite)
Groupe 1 (dans box) = rouge, Groupe 2 (hors box) = gris. Y vertical (remap X,Z,Y).
"""
import os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_solid import load_triangles, points_inside

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

DSCAD = r"C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_LM1_PRESSURE.ds\dsCad.txt"
STEP = r"C:\workspace\fiabilite\bounding_box_acier_tablier.stp"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_rebars_full(path):
    """Retourne names, polylines (liste d'arrays (k,3)), centroids (N,3)."""
    txt = open(path, encoding='utf-8', errors='replace').read()
    plists = {}
    for m in re.finditer(
            r"(pts_\w+)\.append\(POINT\(([-\d.eE]+),\s*([-\d.eE]+),\s*([-\d.eE]+)\)\)", txt):
        plists.setdefault(m.group(1), []).append(
            (float(m.group(2)), float(m.group(3)), float(m.group(4))))
    names, polylines, centroids = [], [], []
    for m in re.finditer(r"REBAR\('([^']+)',\s*E=EDGE\(PS=(\w+)\)", txt):
        name, var = m.group(1), m.group(2)
        if var in plists:
            P = np.array(plists[var])
            names.append(name); polylines.append(P); centroids.append(P.mean(axis=0))
    return names, polylines, np.array(centroids)


def remap(P):
    """(X,Y,Z) -> (X,Z,Y) pour que Y soit vertical en 3D matplotlib."""
    return P[:, [0, 2, 1]]


# ----- donnees -----
names, polylines, C = parse_rebars_full(DSCAD)
tris, info = load_triangles(STEP, scale=0.001)
inside = points_inside(C, tris)
n_in = int(inside.sum())
print(f"Groupe 1 (box) {n_in} / Groupe 2 {len(names)-n_in}")

# segments polylignes par groupe (remappes)
seg_in, seg_out = [], []
for i, P in enumerate(polylines):
    Pr = remap(P)
    segs = [Pr[j:j+2] for j in range(len(Pr)-1)]
    (seg_in if inside[i] else seg_out).extend(segs)
print(f"segments rouges {len(seg_in)} / gris {len(seg_out)}")

Cin_r = remap(C[inside]); Cout_r = remap(C[~inside])

# contour box (remappe)
edges = set()
for t in tris:
    pid = [tuple(np.round(p, 3)) for p in t]
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        edges.add(tuple(sorted((pid[a], pid[b]))))
box_segments = [np.array([[a[0], a[2], a[1]], [b[0], b[2], b[1]]]) for a, b in edges]

ASPECT = (np.ptp(C[:, 0]), np.ptp(C[:, 2]), np.ptp(C[:, 1]))

# angles : (nom_fichier, elev, azim, use_true_aspect)
# face_XY regarde le long de X -> aspect auto (sinon ecrase par X=96m)
VIEWS = [
    ('3-4',      18, -60, True),
    ('face_XY',   2,   0, False),
    ('cote_Z',    6,  90, True),
    ('dessus',   88, -90, True),
]


def _setup(ax, elev, azim, title, use_aspect=True):
    lc = Line3DCollection(box_segments, colors='steelblue', linewidths=0.5, alpha=0.4)
    ax.add_collection3d(lc)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X (long.)'); ax.set_ylabel('Z (larg.)'); ax.set_zlabel('Y (vert.)')
    ax.set_title(title, fontsize=12)
    if use_aspect:
        try:
            ax.set_box_aspect(ASPECT)
        except Exception:
            pass
    else:
        # coupe transversale : aspect equilibre Z,Y, X compresse
        try:
            ax.set_box_aspect((np.ptp(C[:, 0]) * 0.15, np.ptp(C[:, 2]), np.ptp(C[:, 1])))
        except Exception:
            pass


# ----- serie VRAIS ACIERS (polylignes) -----
for tag, elev, azim, use_asp in VIEWS:
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Line3DCollection(seg_out, colors='lightgray', linewidths=0.4, alpha=0.5))
    ax.add_collection3d(Line3DCollection(seg_in, colors='red', linewidths=0.6, alpha=0.8))
    _setup(ax, elev, azim, f'VRAIS ACIERS (geometrie reelle) — vue {tag}\n'
                           f'rouge=tablier ({n_in})  gris=structure ({len(names)-n_in})', use_asp)
    out = os.path.join(OUT_DIR, f'vrais_aciers_{tag}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f"-> {out}")

# ----- serie BARYCENTRES -----
for tag, elev, azim, use_asp in VIEWS:
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Cout_r[:, 0], Cout_r[:, 1], Cout_r[:, 2], s=4, c='lightgray', alpha=0.5)
    ax.scatter(Cin_r[:, 0], Cin_r[:, 1], Cin_r[:, 2], s=6, c='red', alpha=0.75)
    _setup(ax, elev, azim, f'BARYCENTRES (centres de gravite) — vue {tag}\n'
                           f'rouge=tablier ({n_in})  gris=structure ({len(names)-n_in})', use_asp)
    out = os.path.join(OUT_DIR, f'barycentres_{tag}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f"-> {out}")
