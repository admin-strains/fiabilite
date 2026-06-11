$TS = Get-Date -Format "HHmmss"
Write-Host "=== Deploy CmnVISU.dll ==="
Move-Item -Force "C:\workspace\front\STRAINS\common\Dll\CmnVISU.dll" "C:\workspace\front\STRAINS\common\Dll\CmnVISU.dll.OLD_$TS" -ErrorAction SilentlyContinue
Copy-Item -Force "C:\workspace\build-sln-mohamad\front\02_COMMON\15_CmnVISU\CmnVISU.dll" "C:\workspace\front\STRAINS\common\Dll\CmnVISU.dll"
Move-Item -Force "C:\workspace\front_mohamad\STRAINS\common\Dll\CmnVISU.dll" "C:\workspace\front_mohamad\STRAINS\common\Dll\CmnVISU.dll.OLD_$TS" -ErrorAction SilentlyContinue
Copy-Item -Force "C:\workspace\build-sln-mohamad\front\02_COMMON\15_CmnVISU\CmnVISU.dll" "C:\workspace\front_mohamad\STRAINS\common\Dll\CmnVISU.dll"
Write-Host "Deploy OK"

Write-Host "=== Stop Django ==="
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

Write-Host "=== Restart Django ==="
Start-Process cmd.exe -ArgumentList '/c', 'C:\workspace\launch_ds.bat' -WindowStyle Hidden -RedirectStandardOutput 'C:\workspace\fiabilite\_django4.log' | Out-Null
Start-Sleep -Seconds 6

Write-Host "=== Delete cached dsviewres for L-shape ==="
Get-ChildItem "C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\" -Filter "Yield_analysis0_0_*.dsviewres" | Remove-Item -Force
Write-Host "dsviewres cleared"

Write-Host "=== Relance L-shape ==="
Start-Process cmd.exe -ArgumentList '/c', 'C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\_run.bat' -WindowStyle Hidden -RedirectStandardOutput 'C:\workspace\fiabilite\_lshape4.log' | Out-Null
Write-Host "L-shape launched"
