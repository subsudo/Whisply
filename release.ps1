param(
    [ValidateSet("all", "portable", "installer")]
    [string]$Target = "all",
    [switch]$Clean,
    [switch]$SkipCompile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Assert-CommandAvailable {
    param(
        [string]$CommandName,
        [string]$Hint
    )
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "$CommandName wurde nicht gefunden. $Hint"
    }
}

function Assert-FileExists {
    param([string]$PathValue)
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "Datei fehlt: $PathValue"
    }
}

function Get-IsccPath {
    $fromPath = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }
    $fromPathLower = Get-Command "iscc" -ErrorAction SilentlyContinue
    if ($fromPathLower) { return $fromPathLower.Source }

    $candidates = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

function Invoke-BuildPy {
    param([string]$BuildArg)
    & $python "build.py" $BuildArg
    if ($LASTEXITCODE -ne 0) {
        throw "Build fehlgeschlagen (build.py ExitCode=$LASTEXITCODE)."
    }
}

function Invoke-InstallerBuildWithFallback {
    param([string]$IsccPath)

    $dist = Join-Path $projectRoot "dist"
    $primaryIss = Join-Path $projectRoot "installer.iss"
    $fallbackDir = Join-Path $projectRoot "dist_installer"
    $fallbackIss = Join-Path $projectRoot "installer.dist_installer.iss"
    $fallbackInstaller = Join-Path $fallbackDir "Whisply-Installer.exe"
    $finalInstaller = Join-Path $dist "Whisply-Installer.exe"

    Assert-FileExists -PathValue (Join-Path $dist "Whisply.exe")

    try {
        & $IsccPath $primaryIss
        if ($LASTEXITCODE -ne 0) {
            throw "ISCC ExitCode=$LASTEXITCODE"
        }
        return
    }
    catch {
        Write-Host "Normaler Inno-Lauf fehlgeschlagen. Versuche Fallback ueber dist_installer ..." -ForegroundColor Yellow
    }

    if (Test-Path -LiteralPath $fallbackInstaller) {
        Remove-Item -LiteralPath $fallbackInstaller -Force -ErrorAction SilentlyContinue
    }

    $issContent = Get-Content -LiteralPath $primaryIss -Raw -Encoding UTF8
    $patched = $issContent -replace 'OutputDir=dist', 'OutputDir=dist_installer'
    Set-Content -LiteralPath $fallbackIss -Value $patched -Encoding UTF8

    try {
        & $IsccPath $fallbackIss
        if ($LASTEXITCODE -ne 0) {
            throw "ISCC Fallback ExitCode=$LASTEXITCODE"
        }
        Assert-FileExists -PathValue $fallbackInstaller
        Copy-Item -LiteralPath $fallbackInstaller -Destination $finalInstaller -Force
    }
    finally {
        if (Test-Path -LiteralPath $fallbackIss) {
            Remove-Item -LiteralPath $fallbackIss -Force -ErrorAction SilentlyContinue
        }
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Assert-FileExists -PathValue $python

if ($Clean) {
    Write-Step "Cleanup alter Build-Artefakte"
    foreach ($dir in @("build", "dist", "dist_installer")) {
        $full = Join-Path $projectRoot $dir
        if (Test-Path -LiteralPath $full) {
            Remove-Item -LiteralPath $full -Recurse -Force
            Write-Host "Entfernt: $full"
        }
    }
}

if (-not $SkipCompile) {
    Write-Step "Python Compile-Check"
    $rootPy = Get-ChildItem -Path $projectRoot -File -Filter *.py | ForEach-Object { $_.FullName }
    $backendPy = Get-ChildItem -Path (Join-Path $projectRoot "backends") -File -Filter *.py | ForEach-Object { $_.FullName }
    $allPy = @($rootPy + $backendPy)
    if ($allPy.Count -eq 0) {
        throw "Keine Python-Dateien fuer Compile-Check gefunden."
    }
    & $python -m py_compile @allPy
    Write-Host "Compile-Check erfolgreich ($($allPy.Count) Dateien)."
}

if ($Target -in @("all", "portable", "installer")) {
    Write-Step "Tool-Check: PyInstaller"
    & $python -m PyInstaller --version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller ist in der .venv nicht verfuegbar."
    }
    Write-Host "PyInstaller gefunden."
}

if ($Target -in @("all", "installer")) {
    Write-Step "Tool-Check: Inno Setup Compiler"
    $isccPath = Get-IsccPath
    if (-not $isccPath) {
        throw "ISCC.exe wurde nicht gefunden. Installiere Inno Setup 6."
    }
    Write-Host "ISCC gefunden: $isccPath"
}

Write-Step "Build starten ($Target)"
switch ($Target) {
    "portable" {
        Invoke-BuildPy "--portable"
    }
    "installer" {
        $portablePath = Join-Path $projectRoot "dist\Whisply.exe"
        if (-not (Test-Path -LiteralPath $portablePath)) {
            Invoke-BuildPy "--portable"
        }
        Invoke-InstallerBuildWithFallback $isccPath
    }
    "all" {
        Invoke-BuildPy "--portable"
        Invoke-InstallerBuildWithFallback $isccPath
    }
}

Write-Step "Smoke-Checks der Artefakte"
$dist = Join-Path $projectRoot "dist"
Assert-FileExists -PathValue $dist

$expected = @()
if ($Target -in @("all", "portable")) {
    $expected += (Join-Path $dist "Whisply.exe")
}
if ($Target -in @("all", "installer")) {
    $expected += (Join-Path $dist "Whisply-Installer.exe")
}

foreach ($artifact in $expected) {
    Assert-FileExists -PathValue $artifact
    $item = Get-Item -LiteralPath $artifact
    if ($item.Length -le 0) {
        throw "Artefakt ist leer: $artifact"
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $artifact).Hash
    $sizeMb = [Math]::Round($item.Length / 1MB, 2)
    Write-Host ("OK  {0}  ({1} MB)" -f $item.Name, $sizeMb)
    Write-Host ("SHA256 {0}" -f $hash)
}

Write-Step "Release-Lauf erfolgreich abgeschlossen"
Write-Host "Fertig. Artefakte liegen in: $dist" -ForegroundColor Green
