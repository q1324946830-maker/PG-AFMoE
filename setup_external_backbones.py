"""Download the external backbone source trees expected by external_models.py."""

import argparse
from pathlib import Path
import subprocess


REPOSITORIES = {
    "UNet": (
        "https://github.com/milesial/Pytorch-UNet.git",
        "Pytorch-UNet-master",
        "unet/__init__.py",
    ),
    "UNet3Plus": (
        "https://github.com/ZJUGiveLab/UNet-Version.git",
        "UNetPlusPlusPlus-master",
        "models/UNet_3Plus.py",
    ),
    "MALUNet": (
        "https://github.com/JCruan519/MALUNet.git",
        "MALUNet-main",
        "models/malunet.py",
    ),
    "EGEUNet": (
        "https://github.com/JCruan519/EGE-UNet.git",
        "EGE-UNet-main",
        "models/egeunet.py",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="../Related-Models")
    parser.add_argument("--architectures", default=",".join(REPOSITORIES))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    selected = [item.strip() for item in args.architectures.split(",") if item.strip()]
    unknown = [name for name in selected if name not in REPOSITORIES]
    if unknown:
        parser.error("unsupported architectures: {}".format(", ".join(unknown)))

    root.mkdir(parents=True, exist_ok=True)
    failures = []
    for architecture in selected:
        url, folder, marker = REPOSITORIES[architecture]
        destination = root / folder
        marker_path = destination / marker
        if marker_path.is_file():
            print("READY: {} -> {}".format(architecture, marker_path))
            continue
        if destination.exists():
            failures.append("{} destination exists but marker is missing: {}".format(architecture, marker_path))
            continue
        command = ["git", "clone", "--depth", "1", url, str(destination)]
        print("COMMAND:", subprocess.list2cmdline(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command)
        if completed.returncode != 0 or not marker_path.is_file():
            failures.append("{} clone/verification failed: {}".format(architecture, marker_path))
        else:
            print("READY: {} -> {}".format(architecture, marker_path))

    if failures:
        print("\nSetup failures:")
        for failure in failures:
            print("  - " + failure)
        return 1
    if args.dry_run:
        print("\nDry run complete. No repositories were downloaded.")
    else:
        print("\nExternal backbones are ready under {}".format(root))
    print("UNetPlusPlus is implemented locally and needs no external repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
