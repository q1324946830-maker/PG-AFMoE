@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON=python"
set "RELATED_MODELS=%~dp0..\Related-Models"
set "DATASETS=ISIC17,ISIC18,PH2"
set "ARCHS=UNet,UNetPlusPlus,UNet3Plus,MALUNet,EGEUNet"
set "EPOCHS=400"
set "BATCH_SIZE=8"
set "NUM_WORKERS=4"
set "SEEDS=41"
set "VARIANTS=baseline,full"
set "EXTRA_ARGS=--skip_existing --continue_on_error"

where "%PYTHON%" >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python was not found on PATH: %PYTHON%
  exit /b 1
)
if not exist "%RELATED_MODELS%" (
  echo ERROR: External-model root not found: %RELATED_MODELS%
  exit /b 1
)

if /I "%~1"=="smoke" goto smoke
if /I "%~1"=="main" goto run
if /I "%~1"=="ablation" (
  set "VARIANTS=all"
  goto run
)
if /I "%~1"=="multiseed" (
  set "DATASETS=ISIC17"
  set "SEEDS=41,42,43"
  set "VARIANTS=baseline,full"
  goto run
)
if /I "%~1"=="dryrun" (
  set "EXTRA_ARGS=--dry_run"
  goto run
)
goto usage

:smoke
"%PYTHON%" run_external_backbone_experiments.py ^
  --architectures "%ARCHS%" ^
  --variants baseline,full ^
  --related_models_root "%RELATED_MODELS%" ^
  --input_size 256 ^
  --smoke_test
exit /b %ERRORLEVEL%

:run
"%PYTHON%" run_external_backbone_experiments.py ^
  --python "%PYTHON%" ^
  --datasets "%DATASETS%" ^
  --architectures "%ARCHS%" ^
  --variants "%VARIANTS%" ^
  --seeds "%SEEDS%" ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --num_workers %NUM_WORKERS% ^
  --input_size 256 ^
  --loss BCEDiceLoss ^
  --base_channels 32 ^
  --related_models_root "%RELATED_MODELS%" ^
  --output_dir experiment_results_rtx5090 ^
  %EXTRA_ARGS%
exit /b %ERRORLEVEL%

:usage
echo Usage:
echo   run_external_backbones_rtx5090.bat smoke
echo   run_external_backbones_rtx5090.bat dryrun
echo   run_external_backbones_rtx5090.bat main
echo   run_external_backbones_rtx5090.bat ablation
echo   run_external_backbones_rtx5090.bat multiseed
echo.
echo main      = 3 datasets x 5 backbones x baseline/full = 30 jobs
echo ablation  = 3 datasets x 5 backbones x 8 variants = 120 jobs
echo multiseed = ISIC17 x 5 backbones x baseline/full x seeds 41/42/43 = 30 jobs
exit /b 2
