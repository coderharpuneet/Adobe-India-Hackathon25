# PowerShell script to test the Docker container on Windows

Write-Host "Building Docker image..." -ForegroundColor Green
docker build --platform linux/amd64 -t pdf-outline-extractor .

Write-Host "`nTesting with sample data..." -ForegroundColor Green

# Create test directories
New-Item -ItemType Directory -Force -Path "test_input", "test_output" | Out-Null

# Copy a sample PDF to test input
Copy-Item "sample_dataset\pdfs\file01.pdf" "test_input\"

Write-Host "Running container..." -ForegroundColor Green
docker run --rm `
  -v "$(Get-Location)\test_input:/app/input:ro" `
  -v "$(Get-Location)\test_output:/app/output" `
  --network none `
  pdf-outline-extractor

Write-Host "`nOutput should be in test_output\file01.json" -ForegroundColor Yellow
Write-Host "Contents:" -ForegroundColor Yellow
Get-Content "test_output\file01.json"
