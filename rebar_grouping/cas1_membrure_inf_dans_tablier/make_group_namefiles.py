"""Genere les fichiers de noms d'aciers par groupe depuis rebar_groups.json.

Produit (scenario membrure inf dans le tablier) :
  - groupes_membrure_inf_dans_tablier.json  (2 listes + meta)
  - noms_groupe1_tablier_membrureinf.txt     (1 nom/ligne)
  - noms_groupe2_structure.txt
A lancer apres classify_rebars_in_box.py.
"""
import json, os, re
from collections import Counter

OUT = os.path.dirname(os.path.abspath(__file__))
src = json.load(open(os.path.join(OUT, "rebar_groups.json")))
g1 = src['group1_in_box']
g2 = src['group2_rest']

data = {
    'scenario': 'membrure_inf_dans_tablier',
    'description': ('Groupe 1 = aciers du tablier (membrure inferieure incluse). '
                    'Les HA_8 hors box aux coins du tablier sont forces en groupe 1.'),
    'criterion': 'centre de gravite dans bounding_box_acier_tablier.stp + override HA_8',
    'n_groupe1': len(g1), 'n_groupe2': len(g2),
    'groupe1_tablier_avec_membrure_inf': g1,
    'groupe2_structure': g2,
}
json.dump(data, open(os.path.join(OUT, "groupes_membrure_inf_dans_tablier.json"), 'w'), indent=1)
open(os.path.join(OUT, "noms_groupe1_tablier_membrureinf.txt"), 'w').write('\n'.join(g1) + '\n')
open(os.path.join(OUT, "noms_groupe2_structure.txt"), 'w').write('\n'.join(g2) + '\n')

def pref(n):
    m = re.match(r"(HA_\d+)_", n); return m.group(1) if m else n
print(f"Groupe 1 : {len(g1)}  |  Groupe 2 : {len(g2)}")
print(f"Diametres groupe 2 : {dict(sorted(Counter(pref(n) for n in g2).items()))}")
print("-> groupes_membrure_inf_dans_tablier.json, noms_groupe1/2*.txt")
