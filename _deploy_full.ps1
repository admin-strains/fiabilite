$TS = Get-Date -Format "HHmmss"
Write-Host "=== Deploy CmnVISU.dll + CetVISU.pyd ==="
# IMPORTANT: set_runtime_path.bat met STRAINS/rupt/core en tete du PATH, donc le
# loader Windows prend la copie LA-BAS en priorite sur common/Dll. Toujours copier
# CmnVISU.dll dans common/Dll ET dans rupt/core (sinon un vieux fichier eclipse le nouveau).
foreach ($pair in @(
    @("C:\workspace\build-sln-mohamad\front\02_COMMON\15_CmnVISU\CmnVISU.dll", "C:\workspace\front\STRAINS\common\Dll\CmnVISU.dll"),
    @("C:\workspace\build-sln-mohamad\front\02_COMMON\15_CmnVISU\CmnVISU.dll", "C:\workspace\front_mohamad\STRAINS\common\Dll\CmnVISU.dll"),
    @("C:\workspace\build-sln-mohamad\front\02_COMMON\15_CmnVISU\CmnVISU.dll", "C:\workspace\front\STRAINS\rupt\core\CmnVISU.dll"),
    @("C:\workspace\build-sln-mohamad\front\02_COMMON\15_CmnVISU\CmnVISU.dll", "C:\workspace\front_mohamad\STRAINS\rupt\core\CmnVISU.dll"),
    @("C:\workspace\build-sln-mohamad\front\03_CETAUTOMATIX\CetVISU\CetVISU.pyd", "C:\workspace\front\STRAINS\rupt\core\CetVISU.pyd"),
    @("C:\workspace\build-sln-mohamad\front\03_CETAUTOMATIX\CetVISU\CetVISU.pyd", "C:\workspace\front_mohamad\STRAINS\rupt\core\CetVISU.pyd")
)) {
    $src, $dst = $pair[0], $pair[1]
    if (Test-Path $dst) { Move-Item -Force $dst "$dst.OLD_$TS" -EA SilentlyContinue }
    Copy-Item -Force $src $dst
}
Write-Host "Deploy OK"

Write-Host "=== Stop Django ==="
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

Write-Host "=== Restart Django ==="
Start-Process cmd.exe -ArgumentList '/c', 'C:\workspace\launch_ds.bat' -WindowStyle Hidden -RedirectStandardOutput 'C:\workspace\fiabilite\_django5.log' | Out-Null
Start-Sleep -Seconds 6

Write-Host "=== Delete cached dsviewres for L-shape ==="
Get-ChildItem "C:\workspace\storage\admin\Moulin_Blanc\Test_Lshape_NORMAL.ds\" -Filter "Yield_analysis0_0_*.dsviewres" | Remove-Item -Force
Write-Host "dsviewres cleared"
