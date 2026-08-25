@echo off
REM ===================================================================
REM  Harness de non-regression fiabilite -- lancement en un clic.
REM
REM  N'a besoin NI de STRAINS, NI d'OpenTURNS, NI de l'environnement
REM  conda de production : seulement numpy + scipy + pytest.
REM  Duree typique : ~1 min.
REM
REM  Surcharger l'interpreteur :  set FIAB_PYTHON=C:\chemin\python.exe
REM ===================================================================
setlocal
if "%FIAB_PYTHON%"=="" set FIAB_PYTHON=C:\python3\python.exe
set PYTHONPATH=
set MPLBACKEND=Agg
echo Interpreteur : %FIAB_PYTHON%
"%FIAB_PYTHON%" -m pytest %*
set RC=%ERRORLEVEL%
if %RC% NEQ 0 echo.& echo === ECHEC (code %RC%) ===
exit /b %RC%
