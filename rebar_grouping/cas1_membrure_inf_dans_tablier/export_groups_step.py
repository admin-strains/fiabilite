"""Exporte les 2 groupes d'aciers en fichiers STEP (wireframe polylignes).

1 fichier par groupe -> chaque groupe = un bloc importable separement dans un
viewer CAD (Rhino/FreeCAD), a colorier pour validation visuelle.
Coords en mm (x1000) pour se superposer a bounding_box_acier_tablier.stp.
"""
import os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_solid import load_triangles, points_inside, find_box_step


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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DSCAD = r"C:\workspace\storage\admin\Moulin_Blanc\Calcul_fiabilite_LM1_PRESSURE.ds\dsCad.txt"
STEP = find_box_step(OUT_DIR)
SCALE = 1000.0  # m -> mm


def write_step_wireframe(path, polylines, product_name):
    """Ecrit un STEP AP214 wireframe : GEOMETRIC_CURVE_SET de POLYLINE."""
    lines = []
    nid = [0]
    def nxt():
        nid[0] += 1; return nid[0]

    # --- boilerplate produit + contexte ---
    app_ctx = nxt()
    proto = nxt()
    prod_ctx = nxt()
    prod = nxt()
    pdf = nxt()
    pd_ctx = nxt()
    pd = nxt()
    pds = nxt()
    # contexte geometrique (unite mm)
    unit_len = nxt(); unit_ang = nxt(); unit_sol = nxt(); uncert = nxt(); geo_ctx = nxt()

    body = []
    body.append(f"#{app_ctx}=APPLICATION_CONTEXT('automotive design');")
    body.append(f"#{proto}=APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app_ctx});")
    body.append(f"#{prod_ctx}=PRODUCT_CONTEXT('',#{app_ctx},'mechanical');")
    body.append(f"#{prod}=PRODUCT('{product_name}','{product_name}','',(#{prod_ctx}));")
    body.append(f"#{pdf}=PRODUCT_DEFINITION_FORMATION('','',#{prod});")
    body.append(f"#{pd_ctx}=PRODUCT_DEFINITION_CONTEXT('',#{app_ctx},'design');")
    body.append(f"#{pd}=PRODUCT_DEFINITION('design','',#{pdf},#{pd_ctx});")
    body.append(f"#{pds}=PRODUCT_DEFINITION_SHAPE('','',#{pd});")
    body.append(f"#{unit_len}=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));")
    body.append(f"#{unit_ang}=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));")
    body.append(f"#{unit_sol}=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());")
    body.append(f"#{uncert}=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.01),#{unit_len},'distance_accuracy_value','');")
    body.append(f"#{geo_ctx}=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncert}))GLOBAL_UNIT_ASSIGNED_CONTEXT((#{unit_len},#{unit_ang},#{unit_sol}))REPRESENTATION_CONTEXT('',''));")

    # --- polylignes ---
    poly_ids = []
    for P in polylines:
        pt_ids = []
        for (x, y, z) in P:
            pid = nxt()
            body.append(f"#{pid}=CARTESIAN_POINT('',({x*SCALE:.4f},{y*SCALE:.4f},{z*SCALE:.4f}));")
            pt_ids.append(pid)
        poly = nxt()
        refs = ','.join(f"#{i}" for i in pt_ids)
        body.append(f"#{poly}=POLYLINE('',({refs}));")
        poly_ids.append(poly)

    curve_set = nxt()
    body.append(f"#{curve_set}=GEOMETRIC_CURVE_SET('',({','.join(f'#{i}' for i in poly_ids)}));")
    wf = nxt()
    body.append(f"#{wf}=GEOMETRICALLY_BOUNDED_WIREFRAME_REPRESENTATION('{product_name}',(#{curve_set}),#{geo_ctx});")
    sdr = nxt()
    body.append(f"#{sdr}=SHAPE_DEFINITION_REPRESENTATION(#{pds},#{wf});")

    header = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('rebar group wireframe'),'2;1');\n"
        f"FILE_NAME('{os.path.basename(path)}','2026-06-17T00:00:00',(''),(''),'','strains','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\n"
        "ENDSEC;\n"
        "DATA;\n"
    )
    with open(path, 'w') as f:
        f.write(header)
        f.write('\n'.join(body))
        f.write("\nENDSEC;\nEND-ISO-10303-21;\n")


def main():
    names, polylines, C = parse_rebars_full(DSCAD)
    tris, info = load_triangles(STEP, scale=0.001)
    inside = points_inside(C, tris)
    g1 = [polylines[i] for i in range(len(names)) if inside[i]]
    g2 = [polylines[i] for i in range(len(names)) if not inside[i]]
    print(f"Groupe 1 (tablier) : {len(g1)} aciers / Groupe 2 (structure) : {len(g2)}")

    p1 = os.path.join(OUT_DIR, 'GROUPE1_tablier.stp')
    p2 = os.path.join(OUT_DIR, 'GROUPE2_structure.stp')
    write_step_wireframe(p1, g1, 'GROUPE1_tablier')
    print(f"-> {p1}  ({os.path.getsize(p1)//1024} Ko)")
    write_step_wireframe(p2, g2, 'GROUPE2_structure')
    print(f"-> {p2}  ({os.path.getsize(p2)//1024} Ko)")


if __name__ == '__main__':
    main()
