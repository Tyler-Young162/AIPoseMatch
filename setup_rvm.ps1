# PowerShell script to setup RVM (Robust Video Matting)
Write-Host "Setting up RVM (Robust Video Matting)" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""

# Check if git is available
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: git is not installed. Please install git first." -ForegroundColor Red
    exit 1
}

# Clone the RVM repository if it doesn't exist
if (-not (Test-Path "RobustVideoMatting")) {
    Write-Host "Cloning RVM repository..." -ForegroundColor Yellow
    git clone https://github.com/PeterL1n/RobustVideoMatting.git
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to clone RVM repository" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Repository cloned successfully" -ForegroundColor Green
} else {
    Write-Host "RVM repository already exists, skipping clone" -ForegroundColor Yellow
}

# Create models directory
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
    Write-Host "✓ Created models directory" -ForegroundColor Green
}

# Check if model files exist
Write-Host ""
Write-Host "Checking for model files..." -ForegroundColor Yellow

if (Test-Path "models\rvm_mobilenetv3.pth") {
    Write-Host "✓ rvm_mobilenetv3.pth found" -ForegroundColor Green
} else {
    Write-Host "⚠ rvm_mobilenetv3.pth not found" -ForegroundColor Yellow
    Write-Host "Please download it from:" -ForegroundColor White
    Write-Host "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_mobilenetv3.pth" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or run the download script:" -ForegroundColor White
    Write-Host "python download_rvm_model.py" -ForegroundColor Cyan
}

if (Test-Path "models\rvm_resnet50.pth") {
    Write-Host "✓ rvm_resnet50.pth found" -ForegroundColor Green
} else {
    Write-Host "⚠ rvm_resnet50.pth not found" -ForegroundColor Yellow
    Write-Host "Please download it from:" -ForegroundColor White
    Write-Host "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_resnet50.pth" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "If you need to use the official RVM, you can import it from RobustVideoMatting directory" -ForegroundColor White

