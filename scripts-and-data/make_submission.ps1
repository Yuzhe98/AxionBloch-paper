$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$texDir = Join-Path $root "tex"
$buildDir = Join-Path $texDir "build"
$stageDir = Join-Path $buildDir "MDPI-submission"
$zipPath = Join-Path $buildDir "MDPI-submission.zip"

if (-not (Test-Path -LiteralPath $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

if (Test-Path -LiteralPath $stageDir) {
    Remove-Item -LiteralPath $stageDir -Recurse -Force
}

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

New-Item -ItemType Directory -Path $stageDir | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageDir "figures") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageDir "Definitions") | Out-Null

Copy-Item (Join-Path $texDir "paper_in_MDPI_template.tex") $stageDir
Copy-Item (Join-Path $texDir "body.tex") $stageDir
Copy-Item (Join-Path $texDir "references.bib") $stageDir
Copy-Item (Join-Path $texDir "figures\*.pdf") (Join-Path $stageDir "figures")
Copy-Item (Join-Path $texDir "Definitions\*") (Join-Path $stageDir "Definitions")

Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath

Write-Host "Created submission archive: $zipPath"
