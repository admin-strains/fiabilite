"""
Projected Polyhedron algorithm (Sherbrooke & Patrikalakis 1993)
pour trouver toutes les solutions isolees de {f1=0, f2=0}
ou f1, f2 sont des polynomes bivaries en forme de Bernstein tensorielle.

Reference :
  Sherbrooke & Patrikalakis, "Computation of the solutions of nonlinear
  polynomial systems", CAGD 10(5), 1993.
  MIT Hyperbook, Patrikalakis-Maekawa-Cho, Ch. Nonlinear Solvers.
"""

import numpy as np
from dataclasses import dataclass
from scipy.spatial import ConvexHull


# ============================================================
# Structures de donnees
# ============================================================

@dataclass
class BernsteinPatch:
    """Polynome bivarié en base de Bernstein tensorielle sur un domaine."""
    coeffs: np.ndarray   # shape (n1+1, n2+1)
    domain: tuple         # ((a1, b1), (a2, b2))

    @property
    def degrees(self):
        return (self.coeffs.shape[0] - 1, self.coeffs.shape[1] - 1)

    def box_size(self):
        (a1, b1), (a2, b2) = self.domain
        return max(b1 - a1, b2 - a2)


# ============================================================
# De Casteljau 1D : subdivision
# ============================================================

def decasteljau_split_1d(coeffs, t):
    """
    Subdivise un polynome de Bernstein 1D au parametre t in [0,1].
    Retourne (left_coeffs, right_coeffs).
    """
    n = len(coeffs) - 1
    # Triangle de de Casteljau
    tri = [coeffs.copy().astype(float)]
    for r in range(1, n + 1):
        prev = tri[r - 1]
        cur = (1.0 - t) * prev[:-1] + t * prev[1:]
        tri.append(cur)

    left = np.array([tri[r][0] for r in range(n + 1)])
    right = np.array([tri[n - r][r] for r in range(n + 1)])
    return left, right


# ============================================================
# Subdivision tensorielle
# ============================================================

def subdivide_patch(patch, dim, t=0.5):
    """
    Subdivise un BernsteinPatch le long de la dimension dim (0 ou 1)
    au parametre t in [0,1].
    Retourne (patch_left, patch_right).
    """
    C = patch.coeffs
    (a1, b1), (a2, b2) = patch.domain

    if dim == 0:
        left_C = np.zeros_like(C, dtype=float)
        right_C = np.zeros_like(C, dtype=float)
        for j in range(C.shape[1]):
            left_C[:, j], right_C[:, j] = decasteljau_split_1d(C[:, j], t)
        mid = a1 + t * (b1 - a1)
        return (BernsteinPatch(left_C, ((a1, mid), (a2, b2))),
                BernsteinPatch(right_C, ((mid, b1), (a2, b2))))
    else:
        left_C = np.zeros_like(C, dtype=float)
        right_C = np.zeros_like(C, dtype=float)
        for i in range(C.shape[0]):
            left_C[i, :], right_C[i, :] = decasteljau_split_1d(C[i, :], t)
        mid = a2 + t * (b2 - a2)
        return (BernsteinPatch(left_C, ((a1, b1), (a2, mid))),
                BernsteinPatch(right_C, ((a1, b1), (mid, b2))))


# ============================================================
# Extraction du sous-patch (reparametrisation sur un sous-domaine)
# ============================================================

def extract_subpatch(patch, new_domain):
    """
    Re-parametrise le patch sur un sous-domaine.
    new_domain est en coordonnees absolues, inclus dans patch.domain.
    """
    (a1, b1), (a2, b2) = patch.domain
    (na1, nb1), (na2, nb2) = new_domain

    # Convertir en parametres locaux [0,1]
    if b1 - a1 < 1e-15 or b2 - a2 < 1e-15:
        return patch

    t1_lo = (na1 - a1) / (b1 - a1)
    t1_hi = (nb1 - a1) / (b1 - a1)
    t2_lo = (na2 - a2) / (b2 - a2)
    t2_hi = (nb2 - a2) / (b2 - a2)

    # Clamper
    t1_lo = max(0.0, min(1.0, t1_lo))
    t1_hi = max(0.0, min(1.0, t1_hi))
    t2_lo = max(0.0, min(1.0, t2_lo))
    t2_hi = max(0.0, min(1.0, t2_hi))

    # Subdivision en u1 : garder [t1_lo, t1_hi]
    if t1_hi < 1.0:
        left, _ = subdivide_patch(patch, 0, t1_hi)
        cur = left
    else:
        cur = patch

    if t1_lo > 0.0:
        # Recalculer le parametre local
        if t1_hi > 1e-15:
            t_local = t1_lo / t1_hi
        else:
            t_local = 0.0
        _, cur = subdivide_patch(cur, 0, t_local)

    # Subdivision en u2 : garder [t2_lo, t2_hi]
    if t2_hi < 1.0:
        left, _ = subdivide_patch(cur, 1, t2_hi)
        cur = left

    if t2_lo > 0.0:
        if t2_hi > 1e-15:
            t_local = t2_lo / t2_hi
        else:
            t_local = 0.0
        _, cur = subdivide_patch(cur, 1, t_local)

    # Mettre a jour le domaine
    cur = BernsteinPatch(cur.coeffs.copy(), new_domain)
    return cur


# ============================================================
# Convex hull zero-crossing
# ============================================================

def convex_hull_zero_crossing(points):
    """
    Etant donne des points 2D {(t_k, f_k)}, calcule l'enveloppe convexe
    et retourne l'intervalle [t_min, t_max] ou le hull croise f=0.
    Retourne None si pas d'intersection.
    """
    f_vals = points[:, 1]

    # Test rapide : tous du meme signe
    if np.all(f_vals >= 0) or np.all(f_vals <= 0):
        return None

    # Cas degenere : < 3 points
    if len(points) < 3:
        crossings = []
        if len(points) == 2:
            p1, p2 = points[0], points[1]
            if p1[1] * p2[1] <= 0:
                if abs(p2[1] - p1[1]) > 1e-30:
                    tc = p1[0] + (-p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                    crossings.append(tc)
        if crossings:
            return (min(crossings), max(crossings))
        return None

    # Eliminer les doublons pour ConvexHull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Points colineaires ou degeneres
        crossings = []
        for k in range(len(points) - 1):
            p1, p2 = points[k], points[k + 1]
            if p1[1] * p2[1] <= 0 and abs(p2[1] - p1[1]) > 1e-30:
                tc = p1[0] + (-p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                crossings.append(tc)
        if crossings:
            return (min(crossings), max(crossings))
        return None

    # Parcourir les aretes du hull
    verts = hull.vertices
    hull_pts = points[verts]
    n_v = len(hull_pts)
    crossings = []

    for k in range(n_v):
        p1 = hull_pts[k]
        p2 = hull_pts[(k + 1) % n_v]
        t1, f1 = p1
        t2, f2 = p2

        if f1 * f2 <= 0:
            if abs(f2 - f1) < 1e-30:
                crossings.append(t1)
                crossings.append(t2)
            else:
                tc = t1 + (-f1) * (t2 - t1) / (f2 - f1)
                crossings.append(tc)

    if not crossings:
        return None
    return (min(crossings), max(crossings))


# ============================================================
# Etape de reduction
# ============================================================

def _project_and_bound(coeffs, dim):
    """
    Projete les coefficients Bernstein sur le plan (u_dim, f).
    coeffs : shape (n1+1, n2+1).
    dim : 0 pour projeter sur (u1, f), 1 pour (u2, f).
    Retourne l'intervalle [lo, hi] en parametre [0,1] ou f peut etre 0, ou None.
    """
    n = coeffs.shape[dim] - 1
    if n < 0:
        return None

    # Construire les points projetes
    pts = []
    if dim == 0:
        for i in range(coeffs.shape[0]):
            t_i = i / n if n > 0 else 0.5
            for j in range(coeffs.shape[1]):
                pts.append([t_i, coeffs[i, j]])
    else:
        n2 = coeffs.shape[1] - 1
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                t_j = j / n2 if n2 > 0 else 0.5
                pts.append([t_j, coeffs[i, j]])

    pts = np.array(pts)
    return convex_hull_zero_crossing(pts)


def reduce_box(F1, F2):
    """
    Reduction par Projected Polyhedron.
    Retourne le nouveau domaine ((a1,b1),(a2,b2)) ou None si pas de racine.
    """
    (a1, b1), (a2, b2) = F1.domain

    intervals = []  # [(lo_abs, hi_abs)] pour chaque dimension

    for dim in range(2):
        bounds = []
        for F in (F1, F2):
            iv = _project_and_bound(F.coeffs, dim)
            if iv is None:
                return None
            bounds.append(iv)

        # Intersecter les intervalles (en parametre [0,1])
        lo = max(bounds[0][0], bounds[1][0])
        hi = min(bounds[0][1], bounds[1][1])
        if lo > hi + 1e-15:
            return None

        lo = max(0.0, lo)
        hi = min(1.0, hi)

        # Convertir en coordonnees absolues
        if dim == 0:
            a, b = a1, b1
        else:
            a, b = a2, b2
        intervals.append((a + lo * (b - a), a + hi * (b - a)))

    return (intervals[0], intervals[1])


# ============================================================
# Raffinement Newton
# ============================================================

def newton_refine(f1_eval, f2_eval, u0, max_iter=10, tol=1e-12):
    """
    Raffine une solution approchee par Newton sur le systeme {f1=0, f2=0}.
    f1_eval, f2_eval : callables (u1, u2) -> float.
    """
    u = np.array(u0, dtype=float)
    h = 1e-7

    for _ in range(max_iter):
        v1 = f1_eval(u[0], u[1])
        v2 = f2_eval(u[0], u[1])

        if abs(v1) + abs(v2) < tol:
            break

        # Jacobien par differences finies
        J = np.zeros((2, 2))
        for d in range(2):
            uh = u.copy()
            uh[d] += h
            J[0, d] = (f1_eval(uh[0], uh[1]) - v1) / h
            J[1, d] = (f2_eval(uh[0], uh[1]) - v2) / h

        try:
            delta = np.linalg.solve(J, np.array([v1, v2]))
            u -= delta
        except np.linalg.LinAlgError:
            break

    return u


# ============================================================
# Algorithme principal
# ============================================================

def pp_solve(patches_f1, patches_f2, f1_eval=None, f2_eval=None,
             epsilon=1e-8, max_depth=50, verbose=False):
    """
    Trouve toutes les solutions isolees de {f1=0, f2=0}.

    Parameters
    ----------
    patches_f1 : list of BernsteinPatch
        Patches Bernstein de f1 (un par cellule du decoupage en noeuds).
    patches_f2 : list of BernsteinPatch
        Patches Bernstein de f2 (meme decoupage).
    f1_eval, f2_eval : callable(u1, u2) -> float, optional
        Pour le raffinement Newton.
    epsilon : float
        Tolerance sur la taille de la boite.
    max_depth : int
        Profondeur maximale de subdivision.
    verbose : bool
        Afficher la progression.

    Returns
    -------
    roots : list of np.ndarray, shape (2,)
        Toutes les solutions trouvees.
    """
    # Construire la liste des sous-problemes initiaux
    # Indexer les patches par domaine pour un appariement efficace
    stack = []

    # Construire un index spatial pour patches_f2
    f2_by_cell = {}
    for p2 in patches_f2:
        key = (round(p2.domain[0][0], 10), round(p2.domain[0][1], 10),
               round(p2.domain[1][0], 10), round(p2.domain[1][1], 10))
        f2_by_cell[key] = p2

    for p1 in patches_f1:
        (a1, b1), (a2, b2) = p1.domain

        # Chercher le patch f2 sur le meme domaine exact
        key = (round(a1, 10), round(b1, 10), round(a2, 10), round(b2, 10))
        if key in f2_by_cell:
            stack.append((p1, f2_by_cell[key], 0))
            continue

        # Sinon chercher les chevauchements
        for p2 in patches_f2:
            (c1, d1), (c2, d2) = p2.domain
            lo1, hi1 = max(a1, c1), min(b1, d1)
            lo2, hi2 = max(a2, c2), min(b2, d2)
            if lo1 < hi1 - 1e-15 and lo2 < hi2 - 1e-15:
                overlap = ((lo1, hi1), (lo2, hi2))
                ep1 = extract_subpatch(p1, overlap)
                ep2 = extract_subpatch(p2, overlap)
                stack.append((ep1, ep2, 0))

    roots = []
    n_prune = 0
    n_found = 0

    while stack:
        F1, F2, depth = stack.pop()

        # Reduction
        new_dom = reduce_box(F1, F2)
        if new_dom is None:
            n_prune += 1
            continue

        # Taille de la boite reduite
        (na1, nb1), (na2, nb2) = new_dom
        box_size = max(nb1 - na1, nb2 - na2)

        # Si une dimension est collapee a un point, c'est converge
        if (nb1 - na1) < epsilon and (nb2 - na2) < epsilon:
            center = np.array([(na1 + nb1) / 2.0, (na2 + nb2) / 2.0])
            roots.append(center)
            n_found += 1
            if verbose:
                print(f"  Root #{n_found}: u=({center[0]:.6f}, {center[1]:.6f}) [collapsed]")
            continue

        if box_size < epsilon:
            center = np.array([(na1 + nb1) / 2.0, (na2 + nb2) / 2.0])
            roots.append(center)
            n_found += 1
            if verbose:
                print(f"  Root #{n_found}: u=({center[0]:.6f}, {center[1]:.6f})")
            continue

        if depth >= max_depth:
            center = np.array([(na1 + nb1) / 2.0, (na2 + nb2) / 2.0])
            roots.append(center)
            n_found += 1
            continue

        # Reparametriser sur le domaine reduit
        # Elargir legerement les dimensions collapsees pour eviter la degenerescence
        pad = epsilon * 0.1
        if nb1 - na1 < pad:
            mid1 = (na1 + nb1) / 2.0
            na1, nb1 = mid1 - pad, mid1 + pad
        if nb2 - na2 < pad:
            mid2 = (na2 + nb2) / 2.0
            na2, nb2 = mid2 - pad, mid2 + pad
        new_dom = ((na1, nb1), (na2, nb2))

        F1_new = extract_subpatch(F1, new_dom)
        F2_new = extract_subpatch(F2, new_dom)

        # Verifier la qualite de la reduction
        old_size = F1.box_size()
        ratio = box_size / old_size if old_size > 1e-15 else 1.0

        if ratio < 0.8:
            # Bonne reduction : continuer a reduire
            stack.append((F1_new, F2_new, depth + 1))
        else:
            # Mauvaise reduction : subdiviser
            # Choisir la dimension avec le plus grand ecart
            sizes = [nb1 - na1, nb2 - na2]
            dim = int(np.argmax(sizes))
            F1a, F1b = subdivide_patch(F1_new, dim, 0.5)
            F2a, F2b = subdivide_patch(F2_new, dim, 0.5)
            stack.append((F1a, F2a, depth + 1))
            stack.append((F1b, F2b, depth + 1))

    if verbose:
        print(f"PP: {n_found} racines, {n_prune} branches elaguees")

    # Raffinement Newton
    if f1_eval is not None and f2_eval is not None:
        refined = []
        for r in roots:
            rr = newton_refine(f1_eval, f2_eval, r)
            refined.append(rr)
        roots = refined

    # Fusionner les doublons
    roots = _merge_roots(roots, 10 * epsilon)
    return roots


def _merge_roots(roots, tol):
    """Fusionne les racines a distance < tol."""
    if not roots:
        return []
    merged = [roots[0]]
    for r in roots[1:]:
        is_dup = False
        for m in merged:
            if np.linalg.norm(r - m) < tol:
                is_dup = True
                break
        if not is_dup:
            merged.append(r)
    return merged


# ============================================================
# Conversion B-spline tensorielle → patches Bernstein
# ============================================================

def _power_to_bernstein_1d(power_coeffs):
    """
    Convertit les coefficients en base puissance [a0, a1, ..., an]
    (p(t) = a0 + a1*t + ... + an*t^n sur [0,1])
    en coefficients Bernstein.
    """
    from math import comb
    n = len(power_coeffs) - 1
    bern = np.zeros(n + 1)
    for j in range(n + 1):
        s = 0.0
        for k in range(j + 1):
            s += comb(j, k) / comb(n, k) * power_coeffs[k]
        bern[j] = s
    return bern


def ndspline_to_bernstein_patches(spl):
    """
    Convertit une NDSpline 2D (ndsplines) en liste de BernsteinPatch.

    Strategie : extraire les polynomes par morceau sur chaque cellule
    du maillage de noeuds, puis convertir power → Bernstein.
    """
    from scipy.interpolate import NdBSpline

    knots = spl.knots
    coeffs = spl.coefficients
    degrees = spl.degrees

    if len(degrees) != 2:
        raise ValueError("Seules les splines 2D sont supportees")

    d1, d2 = int(degrees[0]), int(degrees[1])
    t1, t2 = knots[0], knots[1]

    # Noeuds interieurs uniques
    u1_breaks = np.unique(t1[d1:-d1])
    u2_breaks = np.unique(t2[d2:-d2])

    patches = []

    for i in range(len(u1_breaks) - 1):
        for j in range(len(u2_breaks) - 1):
            a1, b1 = u1_breaks[i], u1_breaks[i + 1]
            a2, b2 = u2_breaks[j], u2_breaks[j + 1]

            if b1 - a1 < 1e-15 or b2 - a2 < 1e-15:
                continue

            # Evaluer le polynome sur une grille de Chebyshev
            # et retrouver les coefficients par interpolation
            # Alternative plus robuste : evaluer directement la spline
            # aux noeuds de Bernstein
            n1, n2 = d1, d2
            bern_coeffs = np.zeros((n1 + 1, n2 + 1))

            # Points de Bernstein en parametres locaux
            for ki in range(n1 + 1):
                for kj in range(n2 + 1):
                    u = a1 + ki / n1 * (b1 - a1) if n1 > 0 else (a1 + b1) / 2
                    v = a2 + kj / n2 * (b2 - a2) if n2 > 0 else (a2 + b2) / 2
                    pt = np.array([[u, v]])
                    bern_coeffs[ki, kj] = spl(pt).ravel()[0]

            # Les valeurs aux points de Bernstein ne sont PAS les coefficients
            # Bernstein. Il faut inverser la relation.
            # Pour degre d, la matrice M telle que vals = M @ bern_coeffs est :
            # M[i,j] = b_{j,d}(i/d) = C(d,j) * (i/d)^j * (1-i/d)^{d-j}
            from math import comb

            # Direction u1
            M1 = np.zeros((n1 + 1, n1 + 1))
            for ki in range(n1 + 1):
                t_val = ki / n1 if n1 > 0 else 0.5
                for kj in range(n1 + 1):
                    M1[ki, kj] = comb(n1, kj) * t_val**kj * (1 - t_val)**(n1 - kj)

            # Direction u2
            M2 = np.zeros((n2 + 1, n2 + 1))
            for ki in range(n2 + 1):
                t_val = ki / n2 if n2 > 0 else 0.5
                for kj in range(n2 + 1):
                    M2[ki, kj] = comb(n2, kj) * t_val**kj * (1 - t_val)**(n2 - kj)

            # bern_coeffs_grid = M1 @ true_bern @ M2.T
            # => true_bern = M1^{-1} @ bern_coeffs_grid @ M2^{-T}
            true_bern = np.linalg.solve(M1, bern_coeffs) @ np.linalg.inv(M2).T

            patches.append(BernsteinPatch(true_bern, ((a1, b1), (a2, b2))))

    return patches
