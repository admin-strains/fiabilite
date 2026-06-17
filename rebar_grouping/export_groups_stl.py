"""Exporte les 2 groupes d'aciers en STL (tubes solides) -> visible dans TOUT viewer 3D.

Chaque acier (polyligne) devient un tube a section carree (rayon gonfle pour la
visibilite a l'echelle pont). 1 fichier STL par groupe -> colorier a l'import.
Coords mm (x1000) pour se superposer a la box. Binary STL (compact).
"""
import os, sys, re, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_solid import load_triangles, points_inside

DSCAD = r"C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_LM1_PRESSURE.ds\dsCad.txt"
STEP = r"C:\workspace\fiabilite\bounding_box_acier_tablier.stp"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALE = 1000.0       # m -> mm
RADIUS_MM = 60.0     # rayon tube gonfle pour visibilite (les vrais sont 2.5-25mm)
NSIDE = 4            # section carree (compromis taille/lisibilite)


def parse_rebars_full(path):
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
            P = np.array(plists[var]) * SCALE
            names.append(name); polylines.append(P); centroids.append(P.mean(axis=0))
    return names, polylines, np.array(centroids)


def tube_triangles(P, r, nside):
    """Triangles (liste de (3,3)) d'un tube a section nside autour de la polyligne P (mm)."""
    tris = []
    if len(P) < 2:
        return tris
    angles = np.linspace(0, 2*np.pi, nside, endpoint=False)
    rings = []
    for i in range(len(P)):
        # direction locale
        if i == 0:
            d = P[1] - P[0]
        elif i == len(P)-1:
            d = P[-1] - P[-2]
        else:
            d = P[i+1] - P[i-1]
        n = np.linalg.norm(d)
        if n < 1e-9:
            d = np.array([1.0, 0, 0]); n = 1.0
        d = d / n
        # base perpendiculaire
        ref = np.array([0, 1.0, 0]) if abs(d[1]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(d, ref); u /= np.linalg.norm(u)
        w = np.cross(d, u)
        ring = [P[i] + r*(np.cos(a)*u + np.sin(a)*w) for a in angles]
        rings.append(ring)
    # quads entre anneaux consecutifs -> 2 triangles chacun
    for i in range(len(rings)-1):
        for j in range(nside):
            a = rings[i][j]; b = rings[i][(j+1) % nside]
            c = rings[i+1][j]; e = rings[i+1][(j+1) % nside]
            tris.append((a, c, b)); tris.append((b, c, e))
    return tris


def write_binary_stl(path, all_tris):
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(all_tris)))
        for (a, b, c) in all_tris:
            nrm = np.cross(b - a, c - a)
            ln = np.linalg.norm(nrm)
            nrm = nrm/ln if ln > 1e-12 else np.zeros(3)
            f.write(struct.pack('<3f', *nrm))
            for v in (a, b, c):
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))


def main():
    names, polylines, C = parse_rebars_full(DSCAD)
    tris_box, info = load_triangles(STEP, scale=0.001)
    inside = points_inside(C / SCALE, tris_box)  # classification en metres
    print(f"Groupe 1 (tablier) {int(inside.sum())} / Groupe 2 (structure) {int((~inside).sum())}")

    for grp_mask, fname, lbl in [(inside, 'GROUPE1_tablier.stl', 'G1 tablier'),
                                 (~inside, 'GROUPE2_structure.stl', 'G2 structure')]:
        all_tris = []
        for i in range(len(names)):
            if grp_mask[i]:
                all_tris.extend(tube_triangles(np.asarray(polylines[i], float), RADIUS_MM, NSIDE))
        # convertir en arrays
        all_tris = [tuple(np.asarray(t, float)) for t in all_tris]
        out = os.path.join(OUT_DIR, fname)
        write_binary_stl(out, all_tris)
        print(f"-> {out}  ({os.path.getsize(out)//1024} Ko, {len(all_tris)} triangles)  [{lbl}]")


if __name__ == '__main__':
    main()
