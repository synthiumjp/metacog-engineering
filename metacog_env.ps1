$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
$env:MIOPEN_DISABLE_CACHE = "1"
& D:\metacog\.venv_metacog\Scripts\Activate.ps1
Write-Host "metacog env active. torch: $(python -c 'import torch; print(torch.__version__)')"
