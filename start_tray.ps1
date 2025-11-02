# PowerShell startup script for Tray mode
Write-Host "============================================================" -ForegroundColor Green
Write-Host "AI Pose Match - System Tray Startup" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[1/3] Python Environment: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found!" -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Check if virtual environment exists and use it
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "[2/3] Using virtual environment..." -ForegroundColor Green
    $pythonExe = ".\.venv\Scripts\python.exe"
} else {
    Write-Host "[2/3] Virtual environment not found, using system Python" -ForegroundColor Yellow
    Write-Host "      Warning: pystray may not be available!" -ForegroundColor Yellow
    $pythonExe = "python"
}

Write-Host "[3/3] Starting tray application..." -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Run the program using the selected Python
& $pythonExe run_with_tray.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Program exited" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
