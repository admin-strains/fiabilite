@echo off
set "NoDefaultCurrentDirectoryInExePath="
cd /d C:\workspace\fiabilite
set "PYTHONPATH=C:\workspace\front;C:\workspace\fiabilite"
C:\python3\python.exe _tools\verify\validate_hf_at_ustar.py
