$scriptPath = Join-Path $PSScriptRoot "launcher\run.ps1"
& $scriptPath @args
exit $LASTEXITCODE

