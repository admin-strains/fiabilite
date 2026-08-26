"""
Le depot sait-il ce que son code ecrit ?

Un run de fiabilite ecrit a DEUX endroits, et l'un des deux surprend :

  * le dossier de SORTIE de l'etude -- figures, journal, configuration ;
  * le MODELE lui-meme, le `.ds` -- caches, dump de reprise, journal des
    points, et surtout les fichiers de sortie du solveur, REECRITS a chaque
    appel. Sur la flexion pure, ils pesent 42 Mo pour 0,01 Mo de modele.

`tools/inventaire_sorties.py` les recense et signale ce qu'il ne connait pas.
En le lancant pour la premiere fois, il a trouve DIX fichiers non recenses --
ce qui est exactement son role.

Ces tests ne demandent ni Digital Structure ni OpenTURNS : ils travaillent sur
une arborescence fabriquee.
"""

import os
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "_config")):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("tomli", reason="lecture TOML") if sys.version_info < (3, 11) else None

import inventaire_sorties as inv                               # noqa: E402

#: ce qu'un run laisse derriere lui, releve sur un vrai `.ds` de flexion pure
SORTIES_REELLES = [
    "dsCad.txt", "dsLoad.txt", "dsNote.txt", "coupe.txt",
    "doe_cache.json", "restart_state.json", "points_log.jsonl",
    "Yield_analysis0.dscad", "Yield_analysis0.dsload",
    "Yield_analysis0_0_kine.dsmed", "Yield_analysis0_0_kine.dslog",
    "Yield_analysis0_0_kine.dsmetares", "Yield_analysis0_0_kine.dslogloc",
    "Yield_analysis0_0_kine_MeshQuality.txt",
    "Yield_analysis0_0_stat.dsmed",
    "Yield_analysis0_0_mesh.dslog", "Yield_analysis0_0_mesh.dsmetares",
    "Yield_analysis0_0_PL_cin_out.msh", "Yield_analysis0_0_PL_cin_out.pos",
    "Yield_analysis0_0_Reordered.msh", "Yield_analysis0_0_UV.msh",
    "Yield_analysis0_0_quad.msh", "Yield_analysis0_0_surf.msh",
    "Yield_analysis0_0_vol.msh", "Yield_analysis0_0_MeshRatios.msh",
    "Yield_analysis0_0_cadSurf.mesh", "Yield_analysis0_0_surf.mesh",
    "Yield_analysis0_0_tetra.mesh", "Yield_analysis0_0_vol.mesh",
    "Yield_analysis0_0_tetra.sol",
    "hf_grid_cache.json", "hf_grid_cache_final.json",
    "hf_grid_full_cache.json", "hf_custom_cache.json",
    "solve_one.json", "pont_complet.stp",
    "SOCP_history", "_doe_workers", "_hf_workers",
    "png EFF", "configuration.json", "globalplanche_EFF_2608_1022.png",
]


@pytest.mark.parametrize("nom", SORTIES_REELLES)
def test_chaque_sortie_reelle_est_recensee(nom):
    """Un fichier que la chaine ecrit et que le depot ne sait pas nommer est
    un fichier dont personne ne sait s'il est un resultat, un cache ou un
    dechet -- et donc s'il faut le sauvegarder ou le purger."""
    categorie, quoi = inv._classer(nom)
    assert categorie != "inconnu", (
        "%s n'est pas recense dans inventaire_sorties.CONNUS.\n"
        "Dire ce que c'est : modele, cache, reprise, trace, resultat, "
        "travail ou archive." % nom)
    assert quoi, "%s est classe %s mais sans explication" % (nom, categorie)


def test_un_fichier_inconnu_est_bien_signale():
    """Le pendant : l'outil ne doit pas classer par defaut ce qu'il ignore,
    sinon il cesse de servir a quoi que ce soit."""
    categorie, _ = inv._classer("un_truc_jamais_vu.xyz")
    assert categorie == "inconnu"


def test_les_categories_sont_celles_annoncees():
    attendues = {"modele", "cache", "reprise", "trace", "resultat",
                 "travail", "archive"}
    obtenues = {categorie for _, categorie, _ in inv.CONNUS}
    assert obtenues <= attendues, "categorie(s) inventee(s) : %s" % (obtenues - attendues)


def test_les_fichiers_du_solveur_sont_du_travail_pas_des_resultats():
    """Distinction qui decide de ce qu'on sauvegarde. `.dsmetares` porte
    alpha et le statut : c'est LE resultat. Les `.dsmed` et les maillages
    sont reecrits a chaque appel -- les archiver coute 424 Mo par point sur le
    Moulin Blanc."""
    assert inv._classer("Yield_analysis0_0_kine.dsmetares")[0] == "resultat"
    for nom in ("Yield_analysis0_0_kine.dsmed", "Yield_analysis0_0_stat.dsmed",
                "Yield_analysis0_0_vol.msh", "Yield_analysis0.dscad"):
        assert inv._classer(nom)[0] == "travail", nom


def test_le_dsCad_est_signale_comme_reecrit():
    """Le piege le plus couteux du depot : `patch_params` reecrit `dsCad.txt`
    EN PLACE, dans le modele de l'utilisateur, a chaque evaluation. Qui lance
    un run sans sauvegarde perd son modele d'origine."""
    categorie, quoi = inv._classer("dsCad.txt")
    assert categorie == "modele"
    assert "REECRITE" in quoi or "reecrit" in quoi


def test_l_inventaire_tourne_sur_une_arborescence_fabriquee(tmp_path, capsys):
    """Verifie que l'outil ne plante pas et rend des totaux coherents."""
    d = tmp_path / "modele.ds"
    d.mkdir()
    (d / "dsCad.txt").write_bytes(b"x" * 1000)
    (d / "Yield_analysis0_0_kine.dsmed").write_bytes(b"y" * 4000)
    (d / "Yield_analysis0_0_kine.dsmetares").write_text('{"info": {}}')
    categories = inv.inventorier(str(d), "essai")
    assert categories["travail"] == 4000
    assert categories["modele"] == 1000
    assert categories["resultat"] == 12
    sortie = capsys.readouterr().out
    assert "NON RECENSES" not in sortie


def test_un_dossier_absent_ne_fait_pas_planter(tmp_path, capsys):
    assert inv.inventorier(str(tmp_path / "nulle_part"), "essai") == {}
    assert "ABSENT" in capsys.readouterr().out
