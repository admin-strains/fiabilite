r"""Lance un run, filtre son journal a la volee, et RAPPORTE COMMENT IL FINIT.

    python tools/executer_filtre.py --journal C:\tmp\run.log -- \
        C:\tmp\venv_fiab310\Scripts\python.exe -u launcher.py Moulinblanc\AC3_moulinblanc.py

POURQUOI CET OUTIL EXISTE -- 26/08/2026
----------------------------------------
Deux problemes distincts, decouverts le meme jour, et le second est le pire.

**1. Le volume.** `DS_PROF|` represente 99,9 % de la sortie de Digital
Structure : ~75 Mo par point evalue contre ~0,1 Mo une fois filtre. Un run de
592 appels ecrirait 44 Go de journal. Mesure d'un run reel : 7 804 lignes
gardees, 13 546 433 lignes ecartees.

`findstr /V` fait ce filtrage mais BUFFERISE : sur un run de plusieurs heures
le journal reste vide et on ne suit rien. Ici chaque ligne gardee est ecrite
puis flushee.

**2. Le silence sur l'echec.** Le montage precedent etait

    python launcher.py ... 2>&1 | filtre >> journal

Dans un pipeline `cmd`, `%ERRORLEVEL%` rend le code du DERNIER maillon -- le
filtre -- jamais celui de python. Quand Digital Structure a termine le
processus en pleine iteration IPM, apres deux heures de calcul, le filtre a vu
son entree se fermer, a rendu 0, et le `.bat` a tranquillement ecrit « FIN ».
**Un run mort etait indiscernable d'un run reussi.**

C'est pour cela que cet outil LANCE la commande au lieu de la subir : il tient
le code de retour de l'enfant, l'ecrit en clair dans le journal, et le rend au
systeme.

Ce que le journal porte desormais a la fin :

    ==== FIN : succes (code 0) ====
    ==== FIN : ECHEC (code 3221225477) ====        <- 0xC0000005, violation d'acces
    ==== FIN : ECHEC (tue par le signal 9) ====

Ne demande ni Digital Structure, ni OpenTURNS, ni rien d'installe.
"""

import argparse
import os
import subprocess
import sys
import time

#: prefixe des lignes de profilage de Digital Structure
PREFIXE_ECARTE = b"DS_PROF|"


def _diagnostic(code):
    """Traduit un code de retour en phrase, y compris les crashs Windows."""
    if code == 0:
        return "succes (code 0)"
    if code < 0:                                     # POSIX : tue par un signal
        return "ECHEC (tue par le signal %d)" % (-code)
    connus = {
        3221225477: "0xC0000005 -- violation d'acces (crash natif)",
        3221225725: "0xC00000FD -- debordement de pile",
        3221226505: "0xC0000409 -- corruption de pile detectee",
        1: "erreur generique",
    }
    detail = connus.get(code)
    return "ECHEC (code %d%s)" % (code, " : " + detail if detail else "")


def executer(commande, journal, prefixe=PREFIXE_ECARTE, cwd=None):
    """Lance `commande`, filtre sa sortie vers `journal`, rend son code."""
    gardees = ecartees = 0
    t0 = time.time()

    with open(journal, "ab", buffering=0) as sortie:
        def dire(texte):
            sortie.write(texte.encode("utf-8", "replace") + b"\r\n")

        dire("=" * 64)
        dire("DEBUT   %s" % time.strftime("%d/%m/%Y %H:%M:%S"))
        dire("COMMANDE %s" % " ".join(commande))
        dire("=" * 64)

        # bufsize par defaut (BufferedReader) + `readline` explicite.
        #
        # PAS `for ligne in proc.stdout` : l'iterateur d'un objet fichier fait
        # de la LECTURE ANTICIPEE et ne rend les lignes que par blocs -- soit
        # exactement le defaut de `findstr` qu'on cherche a eliminer, le
        # journal restant muet pendant des minutes. `readline()` d'un
        # BufferedReader rend des qu'une fin de ligne est disponible.
        #
        # PAS `bufsize=0` non plus : le `readline()` d'un flux brut lit octet
        # par octet, ce qui coute cher sur 13,5 millions de lignes.
        proc = subprocess.Popen(commande, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=cwd)
        try:
            for ligne in iter(proc.stdout.readline, b""):
                if ligne.startswith(prefixe):
                    ecartees += 1
                    continue
                gardees += 1
                sortie.write(ligne)
        except KeyboardInterrupt:
            proc.terminate()
            dire("")
            dire("==== INTERROMPU AU CLAVIER ====")
            proc.wait()
            raise
        finally:
            if proc.stdout is not None:
                proc.stdout.close()

        code = proc.wait()
        duree = time.time() - t0
        dire("")
        dire("[filtre] %d lignes gardees, %d lignes %s ecartees"
             % (gardees, ecartees, prefixe.decode("ascii", "replace").rstrip("|")))
        dire("==== FIN : %s ====" % _diagnostic(code))
        dire("        duree %.1f min   fin %s"
             % (duree / 60.0, time.strftime("%d/%m/%Y %H:%M:%S")))
        dire("=" * 64)
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print(__doc__.splitlines()[0], file=sys.stderr)
        print("\nIl faut separer les options de la commande par ' -- '.",
              file=sys.stderr)
        return 2
    coupe = argv.index("--")
    options, commande = argv[:coupe], argv[coupe + 1:]
    if not commande:
        print("aucune commande apres ' -- '", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--journal", required=True, help="fichier de journal")
    ap.add_argument("--prefixe", default=PREFIXE_ECARTE.decode("ascii"),
                    help="prefixe des lignes a ecarter (defaut : DS_PROF|)")
    ap.add_argument("--cwd", default=None, help="repertoire de travail")
    args = ap.parse_args(options)

    code = executer(commande, args.journal,
                    prefixe=args.prefixe.encode("utf-8"), cwd=args.cwd)
    if code != 0:
        print("ECHEC : %s -- voir %s"
              % (_diagnostic(code), os.path.abspath(args.journal)),
              file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
