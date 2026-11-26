$ErrorActionPreference = 'Stop'
$Version = '26.09.0-0'
$Url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v$Version/Release-$Version.zip"
$ExpectedSha256 = '7a6f256a0ddf7536182246a5733331bf4677cbcc34f4663774947ad34556c8d0'
$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolsDir = Join-Path $AppDir 'tools\poppler'
$TempDir = Join-Path $env:TEMP "ffcs-poppler-$Version"
$ZipPath = Join-Path $env:TEMP "ffcs-poppler-$Version.zip"

Write-Host 'Poppler PDF text engine is being downloaded...'
Invoke-WebRequest -Uri $Url -OutFile $ZipPath
$ActualSha256 = (Get-FileHash -Path $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualSha256 -ne $ExpectedSha256) {
    Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
    throw "Poppler checksum mismatch. Expected $ExpectedSha256, got $ActualSha256"
}

if (Test-Path $TempDir) { Remove-Item $TempDir -Recurse -Force }
New-Item -ItemType Directory -Path $TempDir | Out-Null
Expand-Archive -Path $ZipPath -DestinationPath $TempDir -Force
$LibraryDir = Join-Path $TempDir 'poppler-26.09.0\Library'
if (-not (Test-Path (Join-Path $LibraryDir 'bin\pdftotext.exe'))) {
    $candidate = Get-ChildItem -Path $TempDir -Filter pdftotext.exe -Recurse | Select-Object -First 1
    if (-not $candidate) { throw 'pdftotext.exe was not found in the downloaded package.' }
    $LibraryDir = Split-Path -Parent (Split-Path -Parent $candidate.FullName)
}
New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null
if (Test-Path $ToolsDir) { Get-ChildItem $ToolsDir -Force | Remove-Item -Recurse -Force }
Copy-Item -Path $LibraryDir -Destination $ToolsDir -Recurse -Force

$Installed = Join-Path $ToolsDir 'Library\bin\pdftotext.exe'
if (-not (Test-Path $Installed)) {
    $Installed = Get-ChildItem -Path $ToolsDir -Filter pdftotext.exe -Recurse | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Installed) { throw 'Poppler installation did not produce pdftotext.exe.' }
Write-Host "Poppler installed successfully: $Installed" -ForegroundColor Green
Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
