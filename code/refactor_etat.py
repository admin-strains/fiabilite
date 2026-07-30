"""
Script de refactoring : remplace les variables d'etat dans AC3_fiabilite.py
par des references vers etat.py (etat.var).

3 operations :
  1. Supprime les declarations de variables d'etat (elles sont dans etat.py)
  2. Supprime les 'global var' pour ces variables
  3. Remplace chaque reference par etat.var (sauf dans les strings)
"""
import re

FILE = r"C:\_workingDir\dir_fiabilite\code\AC3_fiabilite.py"

# Toutes les variables d'etat a prefixer par etat.
ETAT_VARS = [
    # Ordre du plus long au plus court pour eviter les remplacements partiels
    # (ex: hf_2d_grid_fixed_final avant hf_2d_grid_fixed)
    'hf_2d_grid_fixed_final',
    '_eff_history_beta_IS',
    '_eff_history_theta',
    '_eff_history_EFF',
    '_eff_history_Pf',
    '_eff_history_BB',
    '_eff_history_BS',
    '_socp_call_counter',
    '_gepck_pce_label',
    '_hf_grid_full_axes',
    '_hf_custom_result',
    '_round_sizes_prev',
    '_restart_xt_eff',
    '_point_log_phase',
    '_point_log_round',
    'hf_2d_grid_fixed',
    '_enrich_round',
    '_run_HF_count',
    '_fosm_u0_cache',
    '_hf_grid_full',
    'slice_def_final',
    'out_dir_eff',
    '_gepck_loo',
    'timestamp',
]

# Lignes a supprimer entierement (declarations d'etat dans le main)
# On les identifie par leur contenu exact (stripped)
DECLARATIONS_TO_DELETE = [
    '_gepck_pce_label = ""',
    "_gepck_pce_label = ''",
    '_gepck_loo       = None',
    '_gepck_loo = None',
    '_eff_history_EFF   = []',
    '_eff_history_EFF = []',
    '_eff_history_BB    = []',
    '_eff_history_BB = []',
    '_eff_history_BS    = []',
    '_eff_history_BS = []',
    '_eff_history_theta = []',
    '_eff_history_Pf    = []',
    '_eff_history_Pf = []',
    '_fosm_u0_cache     = [None]',
    '_fosm_u0_cache = [None]',
    '_point_log_phase   = ["?"]',
    '_point_log_phase = ["?"]',
    '_point_log_round   = [0]',
    '_point_log_round = [0]',
    '_eff_history_beta_IS = []',
    '_enrich_round     = 0',
    '_enrich_round = 0',
    '_round_sizes_prev = []',
    '_restart_xt_eff   = []',
    '_restart_xt_eff = []',
    '_socp_call_counter = [0]',
    '_run_HF_count = [0]',
    'hf_2d_grid_fixed = None',
    'hf_2d_grid_fixed_final = None',
    '_hf_grid_full = [None]',
    '_hf_grid_full_axes = [None]',
    '_hf_custom_result = [None]',
    'slice_def_final = None',
]

def is_inside_string(line, start, end):
    """Verifie si la position start..end est a l'interieur d'une string."""
    # Approche simple : compte les guillemets avant la position
    prefix = line[:start]
    # Si nombre impair de guillemets simples ou doubles avant la position, on est dans une string
    in_single = prefix.count("'") % 2 == 1
    in_double = prefix.count('"') % 2 == 1
    return in_single or in_double

def process_line(line):
    """Traite une seule ligne : remplace les noms de variables par etat.nom."""
    # Ne pas toucher les commentaires purs
    stripped = line.strip()
    if stripped.startswith('#'):
        return line

    # Ne pas toucher les lignes d'import
    if 'import etat' in line:
        return line

    # Pour chaque variable d'etat, remplacer word-boundary matches
    for var in ETAT_VARS:
        pattern = r'(?<!etat\.)(?<!\w)' + re.escape(var) + r'(?!\w)'
        new_line = line
        for m in reversed(list(re.finditer(pattern, line))):
            # Verifier qu'on n'est pas dans une string
            if not is_inside_string(line, m.start(), m.end()):
                new_line = new_line[:m.start()] + 'etat.' + var + new_line[m.end():]
        line = new_line

    return line

def main():
    with open(FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    deleted = 0
    globals_removed = 0
    replacements = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Operation 1 : supprimer les declarations
        # (on compare le stripped sans les commentaires de fin de ligne)
        stripped_no_comment = stripped.split('#')[0].strip()
        if stripped_no_comment in DECLARATIONS_TO_DELETE:
            # Garder les commentaires associes (ligne precedente si c'est un commentaire)
            deleted += 1
            print(f"  SUPPRIME ligne {i+1}: {stripped[:80]}")
            continue

        # Operation 2 : supprimer les 'global' pour les variables d'etat
        if stripped.startswith('global '):
            global_vars = [v.strip() for v in stripped[7:].split(',')]
            remaining = [v for v in global_vars if v not in ETAT_VARS]
            if len(remaining) < len(global_vars):
                globals_removed += len(global_vars) - len(remaining)
                if remaining:
                    # Garder les globals qui ne sont pas dans etat
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(indent + 'global ' + ', '.join(remaining) + '\n')
                    print(f"  GLOBAL PARTIEL ligne {i+1}: garde {remaining}, supprime {[v for v in global_vars if v in ETAT_VARS]}")
                else:
                    print(f"  GLOBAL SUPPRIME ligne {i+1}: {stripped}")
                continue

        # Operation 3 : prefixer par etat.
        original = line
        line = process_line(line)
        if line != original:
            # Compter les remplacements
            n = sum(1 for a, b in zip(original, line) if a != b)
            replacements += 1
            if replacements <= 30:  # limiter l'output
                print(f"  REMPLACE ligne {i+1}: {original.strip()[:80]}")
                print(f"        -> {line.strip()[:80]}")

        new_lines.append(line)

    print(f"\nResume: {deleted} declarations supprimees, {globals_removed} globals supprimes, {replacements} lignes modifiees")

    # Ecrire le resultat
    with open(FILE, 'w') as f:
        f.writelines(new_lines)
    print(f"Fichier ecrit: {FILE}")

if __name__ == '__main__':
    main()