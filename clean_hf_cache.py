"""Post-process le cache HF apres un run avec valeurs garbage.

Convention λ=0 :
  Quand le solveur SOCP diverge (cas materiaux trop faibles, ex fy=10 MPa),
  on impose par convention `lambda = pObj = 0` (le coefficient d'amplification
  de LM1 est nul, la structure n'a aucune marge).
  → g = pObj - 1 = -1

Detection : |g_raw| > GARBAGE_THRESHOLD (defaut 10) = considere divergent.

Usage :
    python clean_hf_cache.py path/to/_calc_LM1_PRESSURE.log

Output :
    - Affiche le cache nettoye (copier dans AC_moulin_blanc*.py ligne hf_2d_grid_fixed)
    - Liste les points divergents detectes
    - Optionnel: genere un PNG comparant courbe rouge brute vs nettoyee
"""
import re
import ast
import sys
import os
from pathlib import Path

GARBAGE_THRESHOLD = 10.0      # |g| > 10 = divergent (g physique typ. dans [-2, +10])
LAMBDA_NULL_VALUE = -1.0      # convention : lambda=0 → g = pObj-1 = -1


def extract_hf_cache_from_log(log_path):
    """Parse le log pour extraire le dict hf_2d_grid_fixed = {...} imprime."""
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # On cherche la ligne 'hf_2d_grid_fixed = {' (la derniere si plusieurs)
    pattern = re.compile(r"hf_2d_grid_fixed\s*=\s*(\{.*?\})\s*\n", re.DOTALL)
    matches = pattern.findall(content)
    if not matches:
        raise ValueError(f"Aucun 'hf_2d_grid_fixed = {{...}}' trouve dans {log_path}")
    # Prendre le dernier match (cache final apres calcul complet)
    last_match = matches[-1]
    # Parse via ast.literal_eval (safe vs eval)
    cache = ast.literal_eval(last_match)
    return cache


def clean_cache(cache, threshold=GARBAGE_THRESHOLD, replace=LAMBDA_NULL_VALUE):
    """Remplace les valeurs garbage par la valeur de convention lambda=0."""
    Z = cache["Z"]
    n_total = 0
    n_garbage = 0
    garbage_points = []
    Z_clean = []
    params = cache["params"]
    u1_vals = _linspace(params["u1_min"], params["u1_max"], params["n_grid_hf"])
    u2_vals = _linspace(params["u2_min"], params["u2_max"], params["n_grid_hf"])
    for i, row in enumerate(Z):
        row_clean = []
        for j, val in enumerate(row):
            n_total += 1
            if abs(val) > threshold:
                n_garbage += 1
                garbage_points.append({
                    "i": i, "j": j,
                    "u1": u1_vals[j], "u2": u2_vals[i],
                    "g_raw": val, "g_clean": replace,
                })
                row_clean.append(replace)
            else:
                row_clean.append(val)
        Z_clean.append(row_clean)
    cache_clean = {"params": dict(params), "Z": Z_clean}
    return cache_clean, garbage_points, n_total, n_garbage


def _linspace(lo, hi, n):
    if n <= 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + k * step for k in range(n)]


def format_cache_for_python(cache):
    """Formate le cache nettoye en code Python pret a copier-coller."""
    params = cache["params"]
    Z = cache["Z"]
    lines = ["hf_2d_grid_fixed = {"]
    lines.append(f"    'params': {{'u1_min': {params['u1_min']}, 'u1_max': {params['u1_max']}, "
                 f"'u2_min': {params['u2_min']}, 'u2_max': {params['u2_max']}, "
                 f"'n_grid_hf': {params['n_grid_hf']}}},")
    lines.append("    'Z': [")
    for row in Z:
        vals_str = ", ".join(f"{v}" for v in row)
        lines.append(f"        [{vals_str}],")
    lines.append("    ]")
    lines.append("}")
    return "\n".join(lines)


def main(log_path):
    print(f"=== Cache HF post-process (convention lambda=0) ===")
    print(f"Log : {log_path}")
    print(f"Threshold garbage : |g| > {GARBAGE_THRESHOLD}")
    print(f"Valeur de remplacement : g = {LAMBDA_NULL_VALUE} (lambda=0)")
    print()

    cache_raw = extract_hf_cache_from_log(log_path)
    cache_clean, garbage_points, n_total, n_garbage = clean_cache(cache_raw)

    print(f"Total points HF    : {n_total}")
    print(f"Points divergents  : {n_garbage}  ({100*n_garbage/n_total:.1f}%)")
    if garbage_points:
        print(f"\nDetail des points divergents :")
        for gp in garbage_points:
            print(f"  [{gp['i']},{gp['j']}] u=({gp['u1']:+.2f}, {gp['u2']:+.2f})  "
                  f"g_raw={gp['g_raw']:+.3e}  -> g_clean={gp['g_clean']}")

    # Sauve la version Python copier-collable
    out_path = Path(log_path).parent / "hf_cache_clean.py"
    code_str = format_cache_for_python(cache_clean)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Cache HF nettoye (convention lambda=0)\n")
        f.write(f"# Source : {log_path}\n")
        f.write(f"# Threshold garbage : |g| > {GARBAGE_THRESHOLD}\n")
        f.write(f"# Convention : g_replace = {LAMBDA_NULL_VALUE} (= lambda=0)\n")
        f.write(f"# {n_garbage}/{n_total} points cleanes\n\n")
        f.write(code_str)
        f.write("\n")
    print(f"\nCache nettoye sauve : {out_path}")
    print(f"  -> Copier le contenu dans AC_moulin_blanc*.py ligne 'hf_2d_grid_fixed = ...'")

    return cache_clean


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Defaut : log LM1 PRESSURE le plus recent
        default_log = r"C:\workspace\fiabilite\_calc_LM1_PRESSURE.log"
        if os.path.exists(default_log):
            log_path = default_log
            print(f"[INFO] Aucun argument, utilisation log par defaut : {log_path}")
        else:
            print("Usage : python clean_hf_cache.py path/to/calc.log")
            sys.exit(1)
    else:
        log_path = sys.argv[1]
    main(log_path)
