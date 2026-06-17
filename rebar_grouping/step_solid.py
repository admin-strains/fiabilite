"""Parseur STEP minimal : extrait un solide polyedrique (faces PLANES) en triangles.

Cible : bounding_box_acier_tablier.stp (1 MANIFOLD_SOLID_BREP, 38 faces planes).
Ne depend que de numpy. Retourne les triangles (M, 3, 3) en metres (STEP en mm -> /1000).

Approche : on resout la topologie B-rep
  ADVANCED_FACE -> FACE_*_BOUND -> EDGE_LOOP -> ORIENTED_EDGE -> EDGE_CURVE
                -> VERTEX_POINT -> CARTESIAN_POINT
Pour chaque face plane on recupere ses sommets, on les ordonne par angle dans
le plan de la face (faces convexes -> robuste), puis fan-triangulation.
"""
import re
import numpy as np


def _load_entities(path):
    """Retourne {id_int: (TYPE, raw_args_str)} en gerant les entites multi-lignes."""
    txt = open(path, encoding='utf-8', errors='replace').read()
    # isoler la section DATA
    if 'DATA;' in txt:
        txt = txt.split('DATA;', 1)[1]
    if 'ENDSEC;' in txt:
        txt = txt.split('ENDSEC;', 1)[0]
    ents = {}
    # chaque entite : #id = TYPE(args) ;  (args peut contenir des virgules/parentheses imbriquees)
    for m in re.finditer(r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;", txt, re.DOTALL):
        eid = int(m.group(1))
        etype = m.group(2)
        args = m.group(3)
        ents[eid] = (etype, args)
    return ents


def _refs(s):
    """Liste des #id references dans une chaine d'arguments."""
    return [int(x) for x in re.findall(r"#(\d+)", s)]


def load_triangles(path, scale=0.001):
    """Parse le STEP -> (tris (M,3,3) float, infos dict). Coords * scale (mm->m par defaut)."""
    ents = _load_entities(path)

    # --- Resolution des CARTESIAN_POINT ---
    cart = {}
    for eid, (etype, args) in ents.items():
        if etype == 'CARTESIAN_POINT':
            nums = re.findall(r"\(([^()]*)\)", args)
            if nums:
                coords = [float(v) for v in nums[-1].split(',')]
                if len(coords) == 3:
                    cart[eid] = np.array(coords) * scale

    # --- VERTEX_POINT -> CARTESIAN_POINT ---
    vertex = {}
    for eid, (etype, args) in ents.items():
        if etype == 'VERTEX_POINT':
            r = _refs(args)
            for ref in r:
                if ref in cart:
                    vertex[eid] = cart[ref]
                    break

    # --- EDGE_CURVE -> (v_start, v_end) ---
    edge_curve = {}
    for eid, (etype, args) in ents.items():
        if etype == 'EDGE_CURVE':
            r = _refs(args)
            vs = [x for x in r if x in vertex]
            if len(vs) >= 2:
                edge_curve[eid] = (vs[0], vs[1])

    # --- ORIENTED_EDGE -> (edge_curve ref, orientation bool) ---
    oriented_edge = {}
    for eid, (etype, args) in ents.items():
        if etype == 'ORIENTED_EDGE':
            r = _refs(args)
            ec = next((ref for ref in r if ref in edge_curve), None)
            if ec is not None:
                orient = args.strip().endswith('.T.')   # .T. = meme sens que l'edge_curve
                oriented_edge[eid] = (ec, orient)

    # --- EDGE_LOOP -> liste d'oriented_edge (dans l'ordre) ---
    edge_loop = {}
    for eid, (etype, args) in ents.items():
        if etype == 'EDGE_LOOP':
            edge_loop[eid] = [x for x in _refs(args) if x in oriented_edge]

    # --- FACE_*_BOUND -> edge_loop ---
    face_bound = {}
    for eid, (etype, args) in ents.items():
        if etype in ('FACE_OUTER_BOUND', 'FACE_BOUND'):
            for ref in _refs(args):
                if ref in edge_loop:
                    face_bound[eid] = ref
                    break

    def _ordered_loop_vertices(loop):
        """Chaine les oriented edges d'un loop -> liste ordonnee d'IDs de sommets."""
        # directed edges (v_start, v_end) selon l'orientation
        dedges = []
        for oe in loop:
            ec, orient = oriented_edge[oe]
            v0, v1 = edge_curve[ec]
            dedges.append((v0, v1) if orient else (v1, v0))
        if not dedges:
            return []
        # chainage : end de l'un = start du suivant
        chain = [dedges[0][0], dedges[0][1]]
        used = {0}
        for _ in range(len(dedges) - 1):
            cur = chain[-1]
            nxt = None
            for j, (s, e) in enumerate(dedges):
                if j in used:
                    continue
                if s == cur:
                    nxt = e; used.add(j); break
                if e == cur:
                    nxt = s; used.add(j); break
            if nxt is None:
                break
            chain.append(nxt)
        # retirer le dernier s'il reboucle sur le premier
        if len(chain) > 1 and chain[-1] == chain[0]:
            chain = chain[:-1]
        return chain

    # --- ADVANCED_FACE -> boundary ordonne -> triangles (par ID de sommet) ---
    tri_ids = []     # triplets d'IDs de sommets (pour etancheite exacte)
    n_faces = 0
    for eid, (etype, args) in ents.items():
        if etype != 'ADVANCED_FACE':
            continue
        n_faces += 1
        for ref in _refs(args):
            if ref not in face_bound:
                continue
            loop = edge_loop[face_bound[ref]]
            vids = _ordered_loop_vertices(loop)
            vids = [v for v in vids if v in vertex]
            if len(vids) < 3:
                continue
            # fan triangulation de la boundary ordonnee (faces convexes)
            for i in range(1, len(vids) - 1):
                tri_ids.append((vids[0], vids[i], vids[i + 1]))

    # coords
    tris = np.array([[vertex[i], vertex[j], vertex[k]] for (i, j, k) in tri_ids])
    # rapport d'etancheite par ID (arete = paire d'IDs)
    from collections import Counter
    ec_count = Counter()
    for (i, j, k) in tri_ids:
        for a, b in [(i, j), (j, k), (k, i)]:
            ec_count[tuple(sorted((a, b)))] += 1
    n_non2 = sum(1 for v in ec_count.values() if v != 2)
    info = {
        'n_advanced_face': n_faces,
        'n_triangles': len(tris),
        'n_cartesian_point': len(cart),
        'n_vertex': len(vertex),
        'n_edges_non_manifold': n_non2,   # 0 = etanche
        'watertight': n_non2 == 0,
        'bbox_min': tris.reshape(-1, 3).min(axis=0).tolist() if len(tris) else None,
        'bbox_max': tris.reshape(-1, 3).max(axis=0).tolist() if len(tris) else None,
    }
    return tris, info


def points_inside(points, tris, ray_dir=None):
    """Test point-dans-solide par ray-casting (Moller-Trumbore vectorise).

    points : (N,3), tris : (M,3,3). Retourne masque booleen (N,) True=interieur.
    Compte les intersections d'un rayon (direction generique) avec les triangles :
    nombre impair => interieur.
    """
    points = np.asarray(points, dtype=float)
    if ray_dir is None:
        # direction generique pour eviter de taper aretes/sommets exactement
        ray_dir = np.array([0.5773, 0.5774, 0.5775])
    d = np.asarray(ray_dir, dtype=float)
    d = d / np.linalg.norm(d)

    v0 = tris[:, 0, :]            # (M,3)
    e1 = tris[:, 1, :] - v0       # (M,3)
    e2 = tris[:, 2, :] - v0       # (M,3)
    pvec = np.cross(d[None, :], e2)          # (M,3)
    det = np.einsum('mj,mj->m', e1, pvec)    # (M,)
    eps = 1e-12
    valid_det = np.abs(det) > eps
    inv_det = np.where(valid_det, 1.0 / np.where(valid_det, det, 1.0), 0.0)  # (M,)

    # Vectorisation complete N x M (broadcast) -- pas de boucle Python
    tvec = points[:, None, :] - v0[None, :, :]          # (N,M,3)
    u = np.einsum('nmj,mj->nm', tvec, pvec) * inv_det[None, :]   # (N,M)
    qvec = np.cross(tvec, e1[None, :, :])               # (N,M,3)
    v = np.einsum('j,nmj->nm', d, qvec) * inv_det[None, :]       # (N,M)
    t = np.einsum('mj,nmj->nm', e2, qvec) * inv_det[None, :]     # (N,M)
    hit = (valid_det[None, :] & (u >= 0) & (u <= 1) &
           (v >= 0) & (u + v <= 1) & (t > 1e-9))        # (N,M)
    crossings = hit.sum(axis=1)                          # (N,)
    return (crossings % 2) == 1


def _orient_outward(tris):
    """Oriente chaque triangle pour que sa normale pointe vers l'exterieur
    (loin du centre du solide). Valide pour un solide etoile/convexe-ish.
    Indispensable pour le winding number sur une triangulation imparfaite."""
    center = tris.reshape(-1, 3).mean(axis=0)
    a, b, c = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]
    normal = np.cross(b - a, c - a)
    tri_center = (a + b + c) / 3.0
    outward = np.einsum('mj,mj->m', normal, tri_center - center)
    flip = outward < 0
    out = tris.copy()
    out[flip, 1, :], out[flip, 2, :] = tris[flip, 2, :], tris[flip, 1, :]
    return out


def points_inside_winding(points, tris, thresh=0.5):
    """Test point-dans-solide par GENERALIZED WINDING NUMBER (vectorise N x M).

    Robuste aux triangulations non etanches (trous) : la somme des angles solides
    sous-tendus par les triangles vaut ~4*pi (winding=1) a l'interieur, ~0 a l'exterieur.
    Formule d'angle solide de Van Oosterom-Strackee. Necessite une orientation
    coherente -> _orient_outward (normales vers l'exterieur).
    """
    points = np.asarray(points, dtype=float)
    T = _orient_outward(tris)
    A = T[:, 0, :]; B = T[:, 1, :]; C = T[:, 2, :]   # (M,3)

    # vecteurs du point vers chaque sommet : (N,M,3)
    a = A[None, :, :] - points[:, None, :]
    b = B[None, :, :] - points[:, None, :]
    c = C[None, :, :] - points[:, None, :]
    la = np.linalg.norm(a, axis=2)   # (N,M)
    lb = np.linalg.norm(b, axis=2)
    lc = np.linalg.norm(c, axis=2)
    # numerateur = det([a,b,c]) = a . (b x c)
    bxc = np.cross(b, c)                              # (N,M,3)
    num = np.einsum('nmj,nmj->nm', a, bxc)           # (N,M)
    ab = np.einsum('nmj,nmj->nm', a, b)
    bc = np.einsum('nmj,nmj->nm', b, c)
    ca = np.einsum('nmj,nmj->nm', c, a)
    den = la * lb * lc + ab * lc + bc * la + ca * lb
    omega = 2.0 * np.arctan2(num, den)               # angle solide signe (N,M)
    winding = omega.sum(axis=1) / (4.0 * np.pi)      # (N,)
    return np.abs(winding) > thresh
