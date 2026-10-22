param(
    [ValidateSet("baseline", "pg_afmoe", "sfga", "aspp", "pg_afmoe_sfga", "pg_afmoe_aspp", "full", "all")]
    [string]$Variant = "all",
    [string]$Python = "python",
    [string]$RelatedModelsRoot = "../Related-Models",
    [int]$Epochs = 400,
    [int]$BatchSize = 8,
    [int]$NumWorkers = 4,
    [int]$DataSeed = 41,
    [int]$BaseChannels = 32,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$datasets = @("ISIC17", "ISIC18", "PH2")
$architectures = @("UNet", "UNetPlusPlus", "UNet3Plus", "MALUNet", "EGEUNet")
$variants = if ($Variant -eq "all") {
    @("baseline", "pg_afmoe", "sfga", "aspp", "pg_afmoe_sfga", "pg_afmoe_aspp", "full")
} else {
    @($Variant)
}

$pythonCommand = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python executable not found: $Python"
}
$Python = $pythonCommand.Source
if (-not (Test-Path -LiteralPath $RelatedModelsRoot)) {
    throw "Related model directory not found: $RelatedModelsRoot"
}

function Get-VariantArguments {
    param([string]$Name)

    $args = @(
        "--frequency_mode", "add",
        "--frequency_stages", "none",
        "--frequency_gamma", "0.0",
        "--use_align", "false",
        "--align_stages", "none",
        "--context_module", "none"
    )

    if ($Name -eq "pg_afmoe" -or $Name -eq "pg_afmoe_sfga" -or $Name -eq "pg_afmoe_aspp" -or $Name -eq "full") {
        $args += @(
            "--frequency_mode", "afmoe",
            "--frequency_stages", "t3",
            "--frequency_gamma", "0.15",
            "--frequency_experts", "add,dwt,gabor,dct",
            "--moe_top_k", "2",
            "--moe_prior", "add:0.05,dwt:0.60,gabor:0.25,dct:0.10",
            "--moe_balance_weight", "0.003",
            "--moe_entropy_weight", "0.0003"
        )
    }
    if ($Name -eq "sfga" -or $Name -eq "pg_afmoe_sfga" -or $Name -eq "full") {
        $args += @(
            "--use_align", "true",
            "--align_stages", "t3",
            "--align_max_flow", "4.0"
        )
    }
    if ($Name -eq "aspp" -or $Name -eq "pg_afmoe_aspp" -or $Name -eq "full") {
        $args += @(
            "--context_module", "aspp",
            "--context_rates", "1,2,3",
            "--context_dropout", "0.0"
        )
    }
    return $args
}

Push-Location $projectRoot
try {
    foreach ($dataset in $datasets) {
        foreach ($architecture in $architectures) {
            foreach ($currentVariant in $variants) {
                $name = "{0}_{1}_{2}" -f $dataset.ToLower(), $architecture, $currentVariant
                $trainArgs = @(
                    "train_GBC.py",
                    "--dataset", $dataset,
                    "--arch", $architecture,
                    "--name", $name,
                    "--epochs", $Epochs,
                    "--batch_size", $BatchSize,
                    "--num_workers", $NumWorkers,
                    "--dataseed", $DataSeed,
                    "--loss", "BCEDiceLoss",
                    "--deep_supervision", "false",
                    "--base_channels", $BaseChannels,
                    "--related_models_root", $RelatedModelsRoot
                )
                $trainArgs += Get-VariantArguments $currentVariant

                Write-Host "[$dataset][$architecture][$currentVariant]"
                Write-Host "  $Python $($trainArgs -join ' ')"
                if (-not $DryRun) {
                    & $Python @trainArgs
                    if ($LASTEXITCODE -ne 0) {
                        throw "Training failed for $name with exit code $LASTEXITCODE"
                    }
                }
            }
        }
    }
}
finally {
    Pop-Location
}
