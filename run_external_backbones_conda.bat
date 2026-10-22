@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if /I not "%CONDA_DEFAULT_ENV%"=="set" (
    echo ERROR: Anaconda environment "set" is not active.
    echo Activate it first with: conda activate set
    exit /b 1
)

if not exist "train_GBC.py" (
    echo ERROR: train_GBC.py was not found in %CD%.
    exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python was not found in the active Anaconda environment.
    exit /b 1
)

set "VARIANT=%~1"
if not defined VARIANT set "VARIANT=all"
set "EPOCHS=%~2"
if not defined EPOCHS set "EPOCHS=400"
set "BATCH_SIZE=%~3"
if not defined BATCH_SIZE set "BATCH_SIZE=1"
set "NUM_WORKERS=%~4"
if not defined NUM_WORKERS set "NUM_WORKERS=0"
set "DATA_SEED=%~5"
if not defined DATA_SEED set "DATA_SEED=41"
set "BASE_CHANNELS=%~6"
if not defined BASE_CHANNELS set "BASE_CHANNELS=32"
set "DRY_RUN=0"
if /I "%~7"=="dryrun" set "DRY_RUN=1"
if /I "%~7"=="--dry-run" set "DRY_RUN=1"

set "RELATED_MODELS_ROOT=%~dp0..\Related-Models"
if not exist "!RELATED_MODELS_ROOT!" (
    echo ERROR: Related model directory was not found: !RELATED_MODELS_ROOT!
    exit /b 1
)
set "VALID_VARIANT=0"
for %%V in (baseline pg_afmoe sfga aspp pg_afmoe_sfga pg_afmoe_aspp full all) do if /I "!VARIANT!"=="%%V" set "VALID_VARIANT=1"
if "!VALID_VARIANT!"=="0" goto :usage

if /I "!VARIANT!"=="all" (
    set "VARIANT_LIST=baseline pg_afmoe sfga aspp pg_afmoe_sfga pg_afmoe_aspp full"
) else (
    set "VARIANT_LIST=!VARIANT!"
)

for %%D in (ISIC17 ISIC18 PH2) do (
    for %%A in (UNet UNetPlusPlus UNet3Plus MALUNet EGEUNet) do (
        for %%V in (!VARIANT_LIST!) do (
            call :run_one "%%D" "%%A" "%%V"
            if errorlevel 1 exit /b 1
        )
    )
)

echo.
echo All requested training jobs completed.
exit /b 0

:run_one
set "DATASET=%~1"
set "ARCH=%~2"
set "CURRENT_VARIANT=%~3"
set "NAME=!DATASET!_!ARCH!_!CURRENT_VARIANT!"
set "MODULE_ARGS=--frequency_mode add --frequency_stages none --frequency_gamma 0.0 --use_align false --align_stages none --context_module none"

if /I "!CURRENT_VARIANT!"=="pg_afmoe" set "MODULE_ARGS=--frequency_mode afmoe --frequency_stages t3 --frequency_gamma 0.15 --frequency_experts add,dwt,gabor,dct --moe_top_k 2 --moe_prior add:0.05,dwt:0.60,gabor:0.25,dct:0.10 --moe_balance_weight 0.003 --moe_entropy_weight 0.0003 --use_align false --align_stages none --context_module none"
if /I "!CURRENT_VARIANT!"=="sfga" set "MODULE_ARGS=--frequency_mode add --frequency_stages none --frequency_gamma 0.0 --use_align true --align_stages t3 --align_max_flow 4.0 --context_module none"
if /I "!CURRENT_VARIANT!"=="aspp" set "MODULE_ARGS=--frequency_mode add --frequency_stages none --frequency_gamma 0.0 --use_align false --align_stages none --context_module aspp --context_rates 1,2,3 --context_dropout 0.0"
if /I "!CURRENT_VARIANT!"=="pg_afmoe_sfga" set "MODULE_ARGS=--frequency_mode afmoe --frequency_stages t3 --frequency_gamma 0.15 --frequency_experts add,dwt,gabor,dct --moe_top_k 2 --moe_prior add:0.05,dwt:0.60,gabor:0.25,dct:0.10 --moe_balance_weight 0.003 --moe_entropy_weight 0.0003 --use_align true --align_stages t3 --align_max_flow 4.0 --context_module none"
if /I "!CURRENT_VARIANT!"=="pg_afmoe_aspp" set "MODULE_ARGS=--frequency_mode afmoe --frequency_stages t3 --frequency_gamma 0.15 --frequency_experts add,dwt,gabor,dct --moe_top_k 2 --moe_prior add:0.05,dwt:0.60,gabor:0.25,dct:0.10 --moe_balance_weight 0.003 --moe_entropy_weight 0.0003 --use_align false --align_stages none --context_module aspp --context_rates 1,2,3 --context_dropout 0.0"
if /I "!CURRENT_VARIANT!"=="full" set "MODULE_ARGS=--frequency_mode afmoe --frequency_stages t3 --frequency_gamma 0.15 --frequency_experts add,dwt,gabor,dct --moe_top_k 2 --moe_prior add:0.05,dwt:0.60,gabor:0.25,dct:0.10 --moe_balance_weight 0.003 --moe_entropy_weight 0.0003 --use_align true --align_stages t3 --align_max_flow 4.0 --context_module aspp --context_rates 1,2,3 --context_dropout 0.0"

set "TRAIN_ARGS=--dataset !DATASET! --arch !ARCH! --name !NAME! --epochs !EPOCHS! --batch_size !BATCH_SIZE! --num_workers !NUM_WORKERS! --dataseed !DATA_SEED! --loss BCEDiceLoss --deep_supervision false --base_channels !BASE_CHANNELS! --related_models_root "!RELATED_MODELS_ROOT!" !MODULE_ARGS!"
echo [!DATASET!][!ARCH!][!CURRENT_VARIANT!]
echo   python train_GBC.py !TRAIN_ARGS!

if "!DRY_RUN!"=="1" exit /b 0
python train_GBC.py !TRAIN_ARGS!
if errorlevel 1 (
    echo ERROR: Training failed for !NAME!.
    exit /b 1
)
exit /b 0

:usage
echo Usage: call run_external_backbones_conda.bat [variant] [epochs] [batch_size] [num_workers] [data_seed] [base_channels] [dryrun]
echo Variants: baseline, pg_afmoe, sfga, aspp, pg_afmoe_sfga, pg_afmoe_aspp, full, all
exit /b 1
