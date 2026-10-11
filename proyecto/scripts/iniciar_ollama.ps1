$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $PSScriptRoot
$portableExe = Join-Path $projectDir '.ollama-bin\ollama.exe'
$globalCommand = Get-Command ollama -ErrorAction SilentlyContinue

if (Test-Path -LiteralPath $portableExe) {
    $ollamaExe = $portableExe
} elseif ($globalCommand) {
    $ollamaExe = $globalCommand.Source
} else {
    throw 'No se encontro Ollama. Instalalo desde https://ollama.com/download/windows o ubica la version portable en proyecto\.ollama-bin\ollama.exe.'
}

try {
    $version = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2
    Write-Output "Ollama ya esta activo. Version: $($version.version)"
    exit 0
} catch {
    # El servidor todavia no esta iniciado.
}

$process = Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden -PassThru

for ($attempt = 1; $attempt -le 20; $attempt++) {
    try {
        $version = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2
        Write-Output "Ollama iniciado. PID: $($process.Id). Version: $($version.version)"
        exit 0
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

throw 'Ollama se inicio, pero no respondio en http://127.0.0.1:11434.'
