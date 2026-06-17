"""STL tubes des 2 groupes du CAS 2 (membrure inf dans le treillis).

Lit le groupement depuis rebar_groups_cas2.json (G1 tablier / G2 structure+membrure
inf longitudinale). 1 STL binaire par groupe -> visible dans tout viewer 3D.
Coords mm (x1000), rayon gonfle pour visibilite.
"""
import os, sys, re, struct, json
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
DSCAD = r"C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_LM1_PRESSURE.ds\dsCad.txt"
GROUPS_JSON = os.path.join(OUT, "rebar_groups_cas2.json")
SCALE = 1000.0
RADIUS_MM = 60.0
NSIDE = 4


def parse_rebars_full(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    plists = {}
    for m in re.finditer(r"(pts_\w+)\.append\(POINT\(([-\d.eE]+),\s*([-\d.eE]+),\s*([-\d.eE]+)\)\)", txt):
        plists.setdefault(m.group(1), []).append((float(m.group(2)), float(m.group(3)), float(m.group(4))))
    names, polylines = [], []
    for m in re.finditer(r"REBAR\('([^']+)',\s*E=EDGE\(PS=(\w+)\)", txt):
        if m.group(2) in plists:
            names.append(m.group(1)); polylines.append(np.array(plists[m.group(2)]) * SCALE)
    return names, polylines


def tube_triangles(P, r, nside):
    tris = []
    if len(P) < 2:
        return tris
    angles = np.linspace(0, 2*np.pi, nside, endpoint=False)
    rings = []
    for i in range(len(P)):
        if i == 0:
            d = P[1] - P[0]
        elif i == len(P)-1:
            d = P[-1] - P[-2]
        else:
            d = P[i+1] - P[i-1]
        n = np.linalg.norm(d)
        d = d/n if n > 1e-9 else np.array([1.0, 0, 0])
        ref = np.array([0, 1.0, 0]) if abs(d[1]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(d, ref); u /= np.linalg.norm(u)
        w = np.cross(d, u)
        rings.append([P[i] + r*(np.cos(a)*u + np.sin(a)*w) for a in angles])
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
            nrm = np.cross(b - a, c - a); ln = np.linalg.norm(nrm)
            nrm = nrm/ln if ln > 1e-12 else np.zeros(3)
            f.write(struct.pack('<3f', *nrm))
            for v in (a, b, c):
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))


def main():
    names, polys = parse_rebars_full(DSCAD)
    grp = json.load(open(GROUPS_JSON))
    g1 = set(grp['group1_in_box']); g2 = set(grp['group2_rest'])
    idx_g1 = [i for i, n in enumerate(names) if n in g1]
    idx_g2 = [i for i, n in enumerate(names) if n in g2]
    print(f"G1 tablier {len(idx_g1)} / G2 structure+membrure inf {len(idx_g2)}")

    for idxs, fname, lbl in [(idx_g1, 'GROUPE1_tablier.stl', 'G1 tablier'),
                             (idx_g2, 'GROUPE2_structure_membrureinf.stl', 'G2 structure+membrure inf')]:
        all_tris = []
        for i in idxs:
            all_tris.extend(tube_triangles(np.asarray(polys[i], float), RADIUS_MM, NSIDE))
        all_tris = [tuple(np.asarray(t, float)) for t in all_tris]
        out = os.path.join(OUT, fname)
        write_binary_stl(out, all_tris)
        print(f"-> {out}  ({os.path.getsize(out)//1024} Ko, {len(all_tris)} triangles)  [{lbl}]")


if __name__ == '__main__':
    main()
