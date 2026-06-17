"""Patche le dsCad d'un projet fiabilite 2-fy : 2 nuances d'acier par groupe.

Pour chaque REBAR : GRADE=fyd -> GRADE=fyd1 (groupe 1) ou fyd2 (groupe 2).
+ ajoute fy1/fy2/fyd1/fyd2 dans l'en-tete, fixe fc=20 (moyenne, plus de variable
beton), corrige le chemin EXTERNAL_FILE vers le STP local du projet.
"""
import os, re, sys

MB = r"C:\workspace\storage\admin\Moulin_Blanc"
RG = r"C:\workspace\fiabilite\rebar_grouping"

# (projet .ds, fichier noms groupe1, fichier noms groupe2)
CASES = [
    ("Calcul_fiabilite_13k_2fy_membrure_inf_tablier",
     os.path.join(RG, "cas1_membrure_inf_dans_tablier", "noms_groupe1_tablier_membrureinf.txt"),
     os.path.join(RG, "cas1_membrure_inf_dans_tablier", "noms_groupe2_structure.txt")),
    ("Calcul_fiabilite_13k_2fy_membrure_inf_diagonal",
     os.path.join(RG, "cas2_membrure_inf_dans_treillis", "noms_groupe1_tablier.txt"),
     os.path.join(RG, "cas2_membrure_inf_dans_treillis", "noms_groupe2_structure_membrureinf.txt")),
]


def load_names(path):
    return set(l.strip() for l in open(path) if l.strip())


def patch(project, f_g1, f_g2):
    g1 = load_names(f_g1); g2 = load_names(f_g2)
    ds = os.path.join(MB, project + ".ds", "dsCad.txt")
    txt = open(ds, encoding='utf-8', errors='replace').read()

    # 1) en-tete : fc fixe a 20 (moyenne)
    txt = re.sub(r"(?m)^fc\s*=.*$", "fc    = 20.0000000000", txt, count=1)
    # 2) ajouter fy1/fy2/fyd1/fyd2 apres la ligne 'fyd = fy/gamma_s'
    if "fy1   =" not in txt:
        txt = txt.replace(
            "fyd = fy/gamma_s",
            "fyd = fy/gamma_s\n"
            "fy1   = 235.0000000000   # groupe 1 (variable fiabilite)\n"
            "fy2   = 235.0000000000   # groupe 2 (variable fiabilite)\n"
            "fyd1 = fy1/gamma_s\n"
            "fyd2 = fy2/gamma_s",
            1)
    # 3) chemin STP -> copie locale du projet
    txt = re.sub(
        r'(EXTERNAL_FILE\("External_file0",")[^"]+(/pont_complet\.stp")',
        r'\g<1>C:/workspace/storage/admin/Moulin_Blanc/' + project + r'.ds\g<2>',
        txt)

    # 4) REBAR : GRADE=fyd -> fyd1/fyd2 par groupe
    lines = txt.split('\n')
    n1 = n2 = nunk = 0
    for i, line in enumerate(lines):
        m = re.match(r"\s*REBAR\('([^']+)'", line)
        if m and "GRADE=fyd," in line:
            name = m.group(1)
            if name in g1:
                lines[i] = line.replace("GRADE=fyd,", "GRADE=fyd1,"); n1 += 1
            elif name in g2:
                lines[i] = line.replace("GRADE=fyd,", "GRADE=fyd2,"); n2 += 1
            else:
                nunk += 1
    txt = '\n'.join(lines)
    open(ds, 'w', encoding='utf-8').write(txt)

    print(f"[{project}]")
    print(f"  groupe1 (fyd1) : {n1}   groupe2 (fyd2) : {n2}   inconnus : {nunk}")
    print(f"  attendu : g1={len(g1)}  g2={len(g2)}  ->  match g1={n1==len(g1)} g2={n2==len(g2)}")
    # verif header
    for key in ("fc    =", "fy1   =", "fy2   =", "fyd1 =", "fyd2 ="):
        ln = next((l for l in txt.split('\n') if l.startswith(key)), None)
        print(f"    {ln}")
    ext = next((l for l in txt.split('\n') if l.startswith("EXTERNAL_FILE")), "")
    print(f"    STP -> ...{ext[-60:]}")


if __name__ == '__main__':
    for proj, fg1, fg2 in CASES:
        patch(proj, fg1, fg2)
        print()
