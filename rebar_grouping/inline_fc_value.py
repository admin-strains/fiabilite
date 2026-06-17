import os
MB = r"C:\workspace\storage\admin\Moulin_Blanc"
PROJECTS = ["Calcul_fiabilite_13k_2fy_membrure_inf_tablier",
            "Calcul_fiabilite_13k_2fy_membrure_inf_diagonal"]
for proj in PROJECTS:
    ds = os.path.join(MB, proj + ".ds", "dsCad.txt")
    t = open(ds, encoding='utf-8', errors='replace').read()
    # 1) valeur directe dans l'IMPORT
    assert "COMPRESSIVE_STRENGTH=str(fcd)" in t, f"IMPORT fcd introuvable {proj}"
    t = t.replace("COMPRESSIVE_STRENGTH=str(fcd)", "COMPRESSIVE_STRENGTH='20.0'", 1)
    # 2) retirer gamma_c et fcd
    t = t.replace("gamma_c = 1.0\n", "", 1)
    t = t.replace("fcd = 20.0/gamma_c   # beton fixe (fcm=20 MPa, plus de variable)\n", "", 1)
    open(ds, 'w', encoding='utf-8').write(t)
    print(f"[{proj}] OK")

# verif
print("\n=== Header + IMPORT tablier ===")
t = open(os.path.join(MB, PROJECTS[0]+".ds", "dsCad.txt"), encoding='utf-8').read().split('\n')
for l in t[:8]: print("  " + l)
print("  ...")
print("  " + next(l for l in t if l.startswith("IMPORT"))[:130] + " ...")
print("\n  reste-t-il 'fc'/'fcd'/'gamma_c' ? ",
      any(l.strip().startswith(('fc ','fc=','fcd','gamma_c')) for l in t))
