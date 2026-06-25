# `_tools/` — scripts utilitaires (rangés le 2026-06-22)

Scripts d'appoint (non versionnés) pour visualiser, régénérer, comparer. Lancés avec `C:\python3\python.exe`.
Le code métier (`AC_*.py`, `launcher_*.py`, `InitSolver.py`, `clean_hf_cache.py`) reste à la **racine**.
La bibliothèque GEPCK (`branche*`) est dans `../_lib/` (voir son README). Les logs sont dans `../_logs/`.

## `viz/` — rendus / visualisations
- **_pyvista_geom_loads.py** — géométrie CAO + chargement LM1 (UDL/TS par voie, scalés) + appuis. Légende anglaise.
- **_pyvista_mesh.py** / **_pyvista_mesh_s21.py** — vues du maillage remaillé (gmsh multicolore / qualité / gris). `_s21` = modes 1 & 4 du stop 21 pts.
- **_pyvista_rebars.py** — 2 groupes d'aciers en tubes 2 couleurs (`_RSCALE`).
- **visualize_3methods.py**, **visualize_polyfit_vs_tps.py** — comparaison visuelle des méthodes de courbe rouge.

## `regen/` — régénération de graphes/PNG depuis dump ou logs
- **_regen_stop_diagonal.py** — visuGEPCK + EFF_graphs + export + log pour un arrêt à N pts EFF (`_KEEP`, def 21). Importe `branche1` → ajoute `../_lib` au path.
- **_regen_visu_from_dump.py**, **_regen_eff_graphs_from_dump.py** — régénèrent une planche depuis `restart_state.json`. (`_regen_visu` importe branche1.)
- **regen_all_pngs_LM1.py**, **regen_pngs_nan_from_log.py** — re-génèrent les PNG EFF depuis les logs.

## `redcurve/` — courbe rouge / surface limite g=0 (méthodes, contour, tests)
- **predict_g0_data.py** — données partagées (importé par les 4 suivants → restent groupés ici).
- **predict_g0_polyfit.py**, **predict_g0_rbf.py**, **predict_g0_gepck_reuse.py** — méthodes de reconstruction de la courbe rouge (polyfit / RBF-TPS / réutilisation GEPCK).
- **test_polyfit_red_curve.py**, **test_logs_red_curve.py**, **test_patch_RBF.py** — tests des méthodes.
- **test_contour_algorithm.py**, **test_contour_all_positive.py**, **test_contour_borderline.py**, **test_contour_nan_row.py**, **test_filter_tightened.py** — tests de l'algo de contour/filtrage (cas limites : tout positif, ligne NaN, λ=0).
- **postprocess_nan_redcurve.py**, **compare_lambda0_vs_nan.py** — post-traitement / comparaison convention λ=0 vs NaN sur points divergents.

## `verify/` — vérifications
- **validate_hf_at_ustar.py** — vérifie le HF au point de conception u*.
- **verify_sigma_fc.py**, **verify_sigma_fy.py** — vérifient les écarts-types fc / fy.

## racine `_tools/`
- **hf_cache_clean.py** — nettoyage du cache HF (courbe rouge).

## `_images/` — PNG de tests (cas limites contour / borderline λ0 / NaN).
