"""Nettoie l'en-tete dsCad (retire fc/fy/fyd, inline fcd=20) et passe le poids
propre en LIVE LOAD (amplifie par lambda) dans le dsLoad, pour les 2 projets 2-fy.
"""
import os

MB = r"C:\workspace\storage\admin\Moulin_Blanc"
PROJECTS = [
    "Calcul_fiabilite_13k_2fy_membrure_inf_tablier",
    "Calcul_fiabilite_13k_2fy_membrure_inf_diagonal",
]

# --- en-tete dsCad : bloc actuel -> bloc nettoye ---
OLD_HEADER = (
    "fc    = 20.0000000000\n"
    "fy    = 235.0000000000\n"
    "gamma_c = 1.0\n"
    "gamma_s = 1.0\n"
    "fcd = fc/gamma_c\n"
    "fyd = fy/gamma_s\n"
    "fy1   = 235.0000000000   # groupe 1 (variable fiabilite)\n"
    "fy2   = 235.0000000000   # groupe 2 (variable fiabilite)\n"
    "fyd1 = fy1/gamma_s\n"
    "fyd2 = fy2/gamma_s"
)
NEW_HEADER = (
    "gamma_c = 1.0\n"
    "gamma_s = 1.0\n"
    "fcd = 20.0/gamma_c   # beton fixe (fcm=20 MPa, plus de variable)\n"
    "fy1   = 235.0000000000   # groupe 1 (variable fiabilite)\n"
    "fy2   = 235.0000000000   # groupe 2 (variable fiabilite)\n"
    "fyd1 = fy1/gamma_s\n"
    "fyd2 = fy2/gamma_s"
)

# --- dsLoad : poids propre DEAD -> LIVE (amplifie par lambda) ---
OLD_YIELD = (
    "DEAD_LOAD_CASES=[('LC_poids', '1')],\n"
    "               LIVE_LOAD_CASES=[('LC_LM1_trafic', '1')],"
)
NEW_YIELD = (
    "DEAD_LOAD_CASES=[],\n"
    "               LIVE_LOAD_CASES=[('LC_poids', '1'), ('LC_LM1_trafic', '1')],"
)

for proj in PROJECTS:
    dscad = os.path.join(MB, proj + ".ds", "dsCad.txt")
    dsload = os.path.join(MB, proj + ".ds", "dsLoad.txt")

    t = open(dscad, encoding='utf-8', errors='replace').read()
    assert OLD_HEADER in t, f"header introuvable dans {proj}"
    t = t.replace(OLD_HEADER, NEW_HEADER, 1)
    open(dscad, 'w', encoding='utf-8').write(t)

    l = open(dsload, encoding='utf-8', errors='replace').read()
    assert OLD_YIELD in l, f"YIELD_ANALYSIS introuvable dans {proj}"
    l = l.replace(OLD_YIELD, NEW_YIELD, 1)
    open(dsload, 'w', encoding='utf-8').write(l)

    print(f"[{proj}] OK")
    print("  header dsCad nettoye (fc/fy/fyd retires, fcd=20 inline)")
    print("  dsLoad : LC_poids -> LIVE (DEAD vide, lambda amplifie poids+trafic)")

print("\n=== Verif header tablier ===")
t = open(os.path.join(MB, PROJECTS[0] + ".ds", "dsCad.txt"), encoding='utf-8').read()
for line in t.split('\n')[:9]:
    print("  " + line)
print("=== Verif YIELD tablier ===")
l = open(os.path.join(MB, PROJECTS[0] + ".ds", "dsLoad.txt"), encoding='utf-8').read()
import re
m = re.search(r"YIELD_ANALYSIS\(.*?\)\)", l, re.DOTALL)
print("  " + m.group(0).replace('\n', '\n  ') if m else "  (introuvable)")
