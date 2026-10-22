"""Shared dataset registry for AD-GBC experiments."""

import os
import re
from copy import deepcopy

__all__ = [
    "DATASET_SPECS",
    "get_dataset_paths",
    "list_supported_datasets",
    "resolve_dataset_spec",
]


def _normalize_dataset_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


DATASET_SPECS = {
    "busi": {
        "name": "BUSI",
        "folder": "BUSI",
        "img_ext": ".png",
        "mask_ext": "_mask.png",
        "num_classes": 1,
        "aliases": ["busi"],
    },
    "cvc-clinicdb": {
        "name": "CVC-ClinicDB",
        "folder": "CVC-ClinicDB",
        "img_ext": ".png",
        "mask_ext": ".png",
        "num_classes": 1,
        "aliases": ["cvc", "clinicdb"],
    },
    "glas": {
        "name": "GlaS",
        "folder": "GlaS",
        "img_ext": ".bmp",
        "mask_ext": "_anno.bmp",
        "num_classes": 1,
        "aliases": ["glas"],
    },
    "isic16": {
        "name": "ISIC16",
        "folder": "ISIC16",
        "img_ext": ".jpg",
        "mask_ext": "_Segmentation.png",
        "num_classes": 1,
        "aliases": ["isic16"],
    },
    "isic17": {
        "name": "ISIC17",
        "folder": "ISIC17",
        "img_ext": ".jpg",
        "mask_ext": "_segmentation.png",
        "num_classes": 1,
        "aliases": ["isic17", "isic"],
    },
    "isic18": {
        "name": "ISIC18",
        "folder": "ISIC18",
        "img_ext": ".jpg",
        "mask_ext": "_segmentation.png",
        "num_classes": 1,
        "aliases": ["isic18"],
    },
    "ph2": {
        "name": "PH2",
        "folder": "PH2",
        "img_ext": ".bmp",
        "mask_ext": "_lesion.bmp",
        "num_classes": 1,
        "aliases": ["ph2"],
    },
}


_ALIASES = {}
for key, spec in DATASET_SPECS.items():
    tokens = {key, spec["name"], spec["folder"], *spec.get("aliases", [])}
    for token in tokens:
        _ALIASES[_normalize_dataset_name(token)] = key


def list_supported_datasets():
    return [spec["name"] for spec in DATASET_SPECS.values()]


def resolve_dataset_spec(dataset):
    normalized = _normalize_dataset_name(dataset)
    key = _ALIASES.get(normalized)
    if key is None:
        supported = ", ".join(list_supported_datasets())
        raise ValueError(
            "Unsupported dataset '{}'. Supported datasets: {}.".format(dataset, supported)
        )

    spec = deepcopy(DATASET_SPECS[key])
    spec["key"] = key
    spec["dataset_input"] = dataset
    return spec


def get_dataset_paths(dataset, root="inputs"):
    spec = resolve_dataset_spec(dataset)
    dataset_root = os.path.join(root, spec["folder"])
    images_dir = os.path.join(dataset_root, "images")
    masks_dir = os.path.join(dataset_root, "masks")

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            "Dataset folder not found: {}. Supported datasets: {}".format(
                dataset_root, ", ".join(list_supported_datasets())
            )
        )
    if not os.path.isdir(images_dir):
        raise FileNotFoundError("Image folder not found: {}".format(images_dir))
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError("Mask folder not found: {}".format(masks_dir))

    return spec, images_dir, masks_dir
