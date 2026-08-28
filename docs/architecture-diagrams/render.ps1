param(
    [string]$MermaidCliVersion = "11.12.0",
    [int]$PngScale = 2
)

$ErrorActionPreference = "Stop"
$diagramRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDirectory = Join-Path $diagramRoot "sources"
$outputDirectory = Join-Path $diagramRoot "rendered"

$browserCandidates = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Google\Chrome\Application\chrome.exe"
)
$browser = $browserCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($browser) {
    $env:PUPPETEER_EXECUTABLE_PATH = $browser
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$sources = Get-ChildItem -LiteralPath $sourceDirectory -Filter "*.mmd" | Sort-Object Name
if (-not $sources) {
    throw "No Mermaid sources found in $sourceDirectory"
}

foreach ($source in $sources) {
    $baseName = [IO.Path]::GetFileNameWithoutExtension($source.Name)
    foreach ($format in @("svg", "png")) {
        $arguments = @(
            "-y",
            "@mermaid-js/mermaid-cli@$MermaidCliVersion",
            "-i", $source.FullName,
            "-o", (Join-Path $outputDirectory "$baseName.$format"),
            "-b", "white"
        )
        if ($format -eq "png") {
            $arguments += @("-s", $PngScale)
        }
        & npx @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to render $($source.Name) as $format"
        }
    }
}

Write-Output "Rendered $($sources.Count) Mermaid diagrams to SVG and PNG in $outputDirectory"

