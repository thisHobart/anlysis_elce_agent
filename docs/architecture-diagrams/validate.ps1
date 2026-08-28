$ErrorActionPreference = "Stop"
$diagramRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDirectory = Join-Path $diagramRoot "sources"
$outputDirectory = Join-Path $diagramRoot "rendered"
$indexPath = Join-Path $diagramRoot "README.md"

$expected = @(
    "A00-system-context",
    "A01-module-overview",
    "M01-desktop-flow",
    "M02-application-flow",
    "M03-graph-flow",
    "M04-decision-planning-flow",
    "M05-data-tools-flow",
    "M06-evaluation-reporting-flow",
    "M07-state-contracts-flow",
    "ST01-research-session-state",
    "D01-research-loop-state-contract",
    "S01-question-to-plan",
    "S02-approval-to-report",
    "S03-bounded-revision",
    "S04-restart-recovery",
    "S05-result-followup"
)

$index = Get-Content -LiteralPath $indexPath -Raw -Encoding UTF8
[void][Reflection.Assembly]::LoadWithPartialName("System.Drawing")

foreach ($baseName in $expected) {
    $source = Join-Path $sourceDirectory "$baseName.mmd"
    $svg = Join-Path $outputDirectory "$baseName.svg"
    $png = Join-Path $outputDirectory "$baseName.png"
    foreach ($path in @($source, $svg, $png)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Missing architecture diagram artifact: $path"
        }
        if ((Get-Item -LiteralPath $path).Length -lt 100) {
            throw "Architecture diagram artifact is unexpectedly small: $path"
        }
    }
    $sourceText = Get-Content -LiteralPath $source -Raw -Encoding UTF8
    if ($sourceText -notmatch "(?m)^(flowchart|sequenceDiagram|stateDiagram-v2|classDiagram)") {
        throw "Unsupported or empty Mermaid source: $source"
    }
    if ($sourceText -match "<<[^>]*\s+[^>]*>>") {
        throw "Mermaid stereotype names must not contain whitespace: $source"
    }
    if ($index -notmatch [regex]::Escape("sources/$baseName.mmd")) {
        throw "README index does not link source: $baseName"
    }
    if ($index -notmatch [regex]::Escape("rendered/$baseName.svg")) {
        throw "README index does not link SVG: $baseName"
    }
    if ($index -notmatch [regex]::Escape("rendered/$baseName.png")) {
        throw "README index does not link PNG: $baseName"
    }
    $image = [Drawing.Image]::FromFile($png)
    try {
        if ($image.Width -lt 300 -or $image.Height -lt 200) {
            throw "Rendered PNG is too small to review: $png ($($image.Width)x$($image.Height))"
        }
    }
    finally {
        $image.Dispose()
    }
}

$actualSources = @(Get-ChildItem -LiteralPath $sourceDirectory -Filter "*.mmd")
$actualSvg = @(Get-ChildItem -LiteralPath $outputDirectory -Filter "*.svg")
$actualPng = @(Get-ChildItem -LiteralPath $outputDirectory -Filter "*.png")
if ($actualSources.Count -ne $expected.Count -or $actualSvg.Count -ne $expected.Count -or $actualPng.Count -ne $expected.Count) {
    throw "Expected $($expected.Count) sources, SVGs and PNGs; found $($actualSources.Count), $($actualSvg.Count), $($actualPng.Count)"
}

Write-Output "Validated $($expected.Count) architecture diagrams, README links, SVGs and PNG dimensions"
