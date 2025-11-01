# PowerShell一键启动脚本
Write-Host "============================================================" -ForegroundColor Green
Write-Host "AI Pose Match - RVM一键启动程序" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[1/3] Python环境: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[错误] 未找到Python，请先安装Python 3.9或更高版本" -ForegroundColor Red
    Write-Host "按任意键退出..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[2/3] 激活虚拟环境..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
} else {
    Write-Host "[2/3] 未找到虚拟环境，使用系统Python" -ForegroundColor Yellow
}

Write-Host "[3/3] 启动程序..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Run the program
python run_with_rvm.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "程序已退出" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

