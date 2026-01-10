$scriptName = "cs336_basics/train_bpe.py"

Write-Host "Step 1: Starting uv run..." -ForegroundColor Cyan
Start-Process uv -ArgumentList "run", "python", $scriptName

Write-Host "Step 2: Waiting for Python to warm up..." -ForegroundColor Cyan
Start-Sleep -Seconds 3

$latestPy = Get-Process python -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1

if ($null -eq $latestPy) {
    Write-Host "Error: Could not find any running Python process." -ForegroundColor Red
    exit
}

$targetPid = $latestPy.Id.ToString()
Write-Host "Step 3: Target PID found -> $targetPid" -ForegroundColor Green

$spyArgs = "record -o profile.svg --pid $targetPid"

Write-Host "Step 4: Requesting Admin privileges for py-spy..." -ForegroundColor Yellow

try {
    Start-Process py-spy -ArgumentList $spyArgs -Verb RunAs
    Write-Host "Success: py-spy is now recording in a new window." -ForegroundColor Green
    Write-Host "The result will be saved as profile.svg when the script finishes." -ForegroundColor Cyan
} catch {
    Write-Host "Error: Failed to start py-spy. Did you click 'Yes' on the Admin prompt?" -ForegroundColor Red
}