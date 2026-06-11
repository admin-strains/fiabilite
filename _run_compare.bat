@echo off
cd /d C:\workspace\front
call STRAINS\rupt\core\set_runtime_path.bat >/dev/null
set PYTHONPATH=
C:\python3\python.exe C:\workspace\fiabilite\_test_compare_dsviewres.py
