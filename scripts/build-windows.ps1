$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$frontendRoot = Join-Path $repoRoot 'frontend'
$python = $env:FRAME_PYTHON
$pythonArgs = @()

if (-not $python) {
    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        $python = $venvPython
    } else {
        $python = 'py'
        $pythonArgs = @('-3.11')
    }
}

Write-Host 'Building the Python backend sidecar with PyInstaller (Windows x64)...'
Push-Location $backendRoot
try {
    & $python @pythonArgs -m pip install -r requirements-build.txt
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install backend build requirements.' }

    & $python @pythonArgs -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --name FrameBackend `
        --distpath (Join-Path $backendRoot 'dist') `
        --workpath (Join-Path $backendRoot 'build') `
        --specpath (Join-Path $backendRoot 'build') `
        --paths $backendRoot `
        --collect-all cv2 `
        --collect-all onnxruntime `
        --collect-all uvicorn `
        --collect-all fastapi `
        --hidden-import app.main `
        --hidden-import app.api.routes `
        --hidden-import app.camera.websocket `
        launcher.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed to build FrameBackend.' }
} finally {
    Pop-Location
}

$sidecar = Join-Path $backendRoot 'dist\FrameBackend\FrameBackend.exe'
if (-not (Test-Path $sidecar)) {
    throw "PyInstaller output not found: $sidecar"
}

Write-Host 'Verifying the packaged sidecar launch, authenticated health check and graceful shutdown...'
& $python @pythonArgs (Join-Path $repoRoot 'scripts\smoke-test-sidecar.py') --executable $sidecar
if ($LASTEXITCODE -ne 0) { throw 'The packaged Python sidecar smoke test failed.' }

Write-Host 'Building React frontend and Electron main/preload processes...'
Push-Location $frontendRoot
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend/Electron build failed.' }
    Write-Host 'Creating the Windows NSIS installer...'
    npm run package:win
    if ($LASTEXITCODE -ne 0) { throw 'electron-builder failed to create the NSIS installer.' }
} finally {
    Pop-Location
}

Write-Host "Windows installer output: $(Join-Path $frontendRoot 'release')"
