# PG-AFMoE

PyTorch implementation and experiment scripts for PG-AFMoE, a prior-guided adaptive frequency-domain mixture-of-experts framework for medical image segmentation. The implementation combines spatial and frequency-domain skip fusion with optional semantic feature alignment, lightweight ASPP context, and AD-GBC region-level refinement.

This repository contains source code and experiment runners. Dataset files, trained checkpoints, prediction images, and training logs are not distributed. Download datasets from their official sources and follow their terms of use.

## Requirements

Python 3.10 or newer is recommended. Install the PyTorch build appropriate for your CUDA setup, then install the remaining project dependencies:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. External backbones are optional for the built-in GBC models. To use the external architectures, download their source repositories with:

```bash
python setup_external_backbones.py --root ../Related-Models
```

The root can also be set with the `RELATED_MODELS_ROOT` environment variable or the `--related_models_root` command-line option.

## Datasets

Download each dataset from its official project page. The code expects images and masks in this structure:

```text
inputs/
├── ISIC17/images/       ISIC17/masks/0/
├── ISIC18/images/       ISIC18/masks/0/
├── PH2/images/          PH2/masks/0/
├── BUSI/images/         BUSI/masks/0/
├── CVC-ClinicDB/images/ CVC-ClinicDB/masks/0/
└── GlaS/images/         GlaS/masks/0/
```

Image and mask filenames must follow the suffix rules in `dataset_config.py`. The supported datasets are ISIC 2016, ISIC 2017, ISIC 2018, PH2, BUSI, CVC-ClinicDB, and GlaS.

## Train and Evaluate

Example PG-AFMoE + SFGA + ASPP training run on ISIC 2017:

```bash
python train_GBC.py \
  --dataset ISIC17 \
  --name isic17_pg_afmoe_sfga_aspp \
  --arch GBC_Rolling_Unet_L \
  --loss BCEDiceWithGeometryLoss \
  --div_weight 0.01 \
  --scale_weight 0.1 \
  --dataseed 41 \
  --frequency_mode afmoe \
  --frequency_stages t3 \
  --frequency_gamma 0.15 \
  --frequency_experts add,dwt,gabor,dct \
  --moe_top_k 2 \
  --use_align true \
  --align_stages t3 \
  --context_module aspp \
  --context_rates 1,2,3
```

Evaluate a trained run with:

```bash
python val_GBC.py --name isic17_pg_afmoe_sfga_aspp
```

Each dataset is trained independently using a fixed within-dataset split. The validation partition is used for checkpoint selection and metric reporting; it is not an independent test set. See the saved configuration and training logs for the exact settings of a run.

Additional scripts support frequency ablations and comparisons with U-Net, U-Net++, U-Net 3+, MALUNet, and EGE-UNet. External architecture source projects are listed in `setup_external_backbones.py` and should be used according to their respective licenses.

## Tests

Run the lightweight tests with:

```bash
python -m pytest tests
```

## License and Attribution

This project is distributed under the MIT License. The AD-GBC implementation builds on [Rolling-UNet](https://github.com/Jiaoyang45/Rolling-Unet). Optional external backbone sources are [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), [UNet-Version](https://github.com/ZJUGiveLab/UNet-Version), [MALUNet](https://github.com/JCruan519/MALUNet), and [EGE-UNet](https://github.com/JCruan519/EGE-UNet). Check each upstream project for its license and citation requirements before use.
