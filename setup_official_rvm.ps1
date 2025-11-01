# PowerShell script to setup official RVM (Robust Video Matting)
Write-Host "========================================" -ForegroundColor Green
Write-Host "Official RVM Setup Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

$rvmPath = "RobustVideoMatting"

# Check if git is available
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: git is not installed. Please install git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Check if RVM repository already exists
if (Test-Path $rvmPath) {
    Write-Host "⚠ RobustVideoMatting directory already exists" -ForegroundColor Yellow
    $response = Read-Host "Do you want to remove it and re-clone? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Remove-Item -Recurse -Force $rvmPath
        Write-Host "✓ Removed existing directory" -ForegroundColor Green
    } else {
        Write-Host "Keeping existing directory" -ForegroundColor Yellow
        exit 0
    }
}

# Clone the RVM repository
Write-Host ""
Write-Host "Cloning RobustVideoMatting repository..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

try {
    git clone https://github.com/PeterL1n/RobustVideoMatting.git
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Repository cloned successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to clone repository" -ForegroundColor Red
        Write-Host ""
        Write-Host "Possible solutions:" -ForegroundColor Yellow
        Write-Host "1. Check your internet connection" -ForegroundColor White
        Write-Host "2. Try again later" -ForegroundColor White
        Write-Host "3. Manually download from: https://github.com/PeterL1n/RobustVideoMatting" -ForegroundColor White
        exit 1
    }
} catch {
    Write-Host "✗ Error cloning repository: $_" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow

if (Test-Path "$rvmPath\model\matting_network.py") {
    Write-Host "✓ Official RVM model files found" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: Expected files not found" -ForegroundColor Yellow
}

# Check for model files
Write-Host ""
Write-Host "Checking for model files..." -ForegroundColor Yellow

$modelsPath = "models"
if (Test-Path $modelsPath) {
    $mobilenetv3 = "$modelsPath\rvm_mobilenetv3.pth"
    $resnet50 = "$modelsPath\rvm_resnet50.pth"
    
    if (Test-Path $mobilenetv3) {
        $size = (Get-Item $mobilenetv3).Length / 1MB
        Write-Host "✓ rvm_mobilenetv3.pth found ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
    } else {
        Write-Host "⚠ rvm_mobilenetv3.pth not found" -ForegroundColor Yellow
    }
    
    if (Test-Path $resnet50) {
        $size = (Get-Item $resnet50).Length / 1MB
        Write-Host "✓ rvm_resnet50.pth found ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
    } else {
        Write-Host "⚠ rvm_resnet50.pth not found" -ForegroundColor Yellow
    }
    
    if (-not (Test-Path $mobilenetv3) -and -not (Test-Path $resnet50)) {
        Write-Host ""
        Write-Host "Model files are missing. Download them using:" -ForegroundColor Yellow
        Write-Host "  python download_rvm_model.py" -ForegroundColor Cyan
    }
} else {
    Write-Host "⚠ models directory not found" -ForegroundColor Yellow
    Write-Host "Create it and download models using: python download_rvm_model.py" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. If model files are missing, run: python download_rvm_model.py" -ForegroundColor White
Write-Host "2. Run your application: python run_with_rvm.py" -ForegroundColor White
Write-Host "3. The official RVM will be used automatically if available" -ForegroundColor White
Write-Host ""

