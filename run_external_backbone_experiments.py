"""Train/evaluate PG-AFMoE, SFGA and ASPP on external backbones."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ARCHITECTURES = ("UNet", "UNetPlusPlus", "UNet3Plus", "MALUNet", "EGEUNet")
VARIANTS = (
    "baseline",
    "pg_afmoe",
    "sfga",
    "aspp",
    "pg_afmoe_sfga",
    "pg_afmoe_aspp",
    "sfga_aspp",
    "full",
)
ALL_ABLATIONS = VARIANTS
METRIC_NAMES = ("iou", "dice", "accuracy", "hd", "hd95", "recall", "specificity", "precision")
REQUIRED_REPOSITORIES = {
    "UNet": ("Pytorch-UNet-master/unet/__init__.py",),
    "UNetPlusPlus": (),
    "UNet3Plus": ("UNetPlusPlusPlus-master/models/UNet_3Plus.py",),
    "MALUNet": ("MALUNet-main/models/malunet.py",),
    "EGEUNet": ("EGE-UNet-main/models/egeunet.py",),
}


def variant_arguments(variant):
    if variant not in VARIANTS:
        raise ValueError("Unsupported variant: {}".format(variant))

    use_pg = variant in {"pg_afmoe", "pg_afmoe_sfga", "pg_afmoe_aspp", "full"}
    use_sfga = variant in {"sfga", "pg_afmoe_sfga", "sfga_aspp", "full"}
    use_aspp = variant in {"aspp", "pg_afmoe_aspp", "sfga_aspp", "full"}

    args = [
        "--frequency_mode", "afmoe" if use_pg else "add",
        "--frequency_stages", "t3" if use_pg else "none",
        "--frequency_gamma", "0.15" if use_pg else "0.0",
        "--frequency_experts", "add,dwt,gabor,dct",
        "--moe_top_k", "2",
        "--moe_temperature", "1.0",
        "--moe_prior", "add:0.05,dwt:0.60,gabor:0.25,dct:0.10",
        "--moe_balance_weight", "0.003" if use_pg else "0.0",
        "--moe_entropy_weight", "0.0003" if use_pg else "0.0",
        "--use_align", "true" if use_sfga else "false",
        "--align_stages", "t3" if use_sfga else "none",
        "--align_max_flow", "4.0",
        "--context_module", "aspp" if use_aspp else "none",
        "--context_rates", "1,2,3",
        "--context_dropout", "0.0",
    ]
    return args


def experiment_name(dataset, architecture, variant, seed):
    return "ext_{}_{}_{}_s{}".format(dataset, architecture, variant, seed)


def required_external_paths(architecture):
    return list(REQUIRED_REPOSITORIES[architecture])


def parse_validation_metrics(text):
    result = {}
    for name in METRIC_NAMES:
        match = re.search(r"(?im)^{}\s*:\s*([-+0-9.eE]+)\s*$".format(re.escape(name)), text)
        if match:
            result[name] = float(match.group(1))
    return result


def parse_csv_list(value, allowed=None):
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if allowed is not None:
        unknown = [item for item in items if item not in allowed]
        if unknown:
            raise ValueError("Unsupported values: {}".format(", ".join(unknown)))
    return items


def validate_prerequisites(
    project_root,
    related_models_root,
    architectures,
    dry_run=False,
    allow_missing_datasets=False,
):
    errors = []
    if not (project_root / "train_GBC.py").is_file():
        errors.append("train_GBC.py not found in {}".format(project_root))
    if not (project_root / "val_GBC.py").is_file():
        errors.append("val_GBC.py not found in {}".format(project_root))

    related_root = Path(related_models_root)
    for architecture in architectures:
        for relative in required_external_paths(architecture):
            candidate = related_root / relative
            if not candidate.is_file():
                errors.append("{} requires missing file: {}".format(architecture, candidate))

    # Dry-run/smoke-test remain useful before datasets are copied.
    if not dry_run and not allow_missing_datasets and not (project_root / "inputs").is_dir():
        errors.append("Dataset root not found: {}".format(project_root / "inputs"))
    return errors


def run_command(command, cwd, log_path=None):
    print("COMMAND:", subprocess.list2cmdline(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines = []
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    return_code = process.wait()
    text = "".join(lines)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text, encoding="utf-8")
    return return_code, text


def training_command(args, dataset, architecture, variant, seed, name):
    command = [
        args.python,
        "train_GBC.py",
        "--dataset", dataset,
        "--arch", architecture,
        "--name", name,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--dataseed", str(seed),
        "--input_h", str(args.input_size),
        "--input_w", str(args.input_size),
        "--loss", args.loss,
        "--deep_supervision", "false",
        "--base_channels", str(args.base_channels),
        "--related_models_root", args.related_models_root,
    ]
    command.extend(variant_arguments(variant))
    return command


def validation_command(args, name):
    return [
        args.python,
        "val_GBC.py",
        "--name", name,
        "--tta", "true" if args.tta else "false",
        "--tta_scales", args.tta_scales,
        "--tta_flip", "true" if args.tta_flip else "false",
    ]


def merge_results(existing_rows, new_rows):
    by_name = {row["name"]: dict(row) for row in existing_rows}
    order = [row["name"] for row in existing_rows]
    for row in new_rows:
        if row["name"] not in by_name:
            order.append(row["name"])
        by_name[row["name"]] = dict(row)
    return [by_name[name] for name in order]


def load_saved_results(output_dir):
    json_path = output_dir / "external_backbone_results.json"
    if not json_path.is_file():
        return []
    try:
        value = json.loads(json_path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, ValueError):
        return []


def save_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "external_backbone_results.json"
    csv_path = output_dir / "external_backbone_results.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "dataset", "architecture", "variant", "seed", "name", "status",
        *METRIC_NAMES, "model_path", "train_log", "validation_log",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train, validate and summarize external-backbone grafting experiments."
    )
    parser.add_argument("--datasets", default="ISIC17")
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--variants", default="baseline,full",
                        help="comma list, 'all', or baseline,full for the main comparison")
    parser.add_argument("--seeds", default="41")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--related_models_root", default="../Related-Models")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--loss", default="BCEDiceLoss",
                        help="shared segmentation loss for fair backbone comparisons")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta_scales", default="0.75,1.0,1.25")
    parser.add_argument("--tta_flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke_test", action="store_true",
                        help="build each selected model and run one CPU forward pass")
    parser.add_argument("--output_dir", default="experiment_results")
    parser.add_argument("--reset_results", action="store_true",
                        help="replace the consolidated results instead of merging by job name")
    return parser.parse_args()


def smoke_test(project_root, related_models_root, architectures, variants, input_size):
    import torch
    from model_factory import build_model

    for architecture in architectures:
        for variant in variants:
            kwargs = dict(
                num_classes=1,
                input_channels=3,
                deep_supervision=False,
                related_models_root=related_models_root,
                base_channels=16,
            )
            variant_args = variant_arguments(variant)
            parsed = {variant_args[i][2:]: variant_args[i + 1] for i in range(0, len(variant_args), 2)}
            kwargs.update(
                frequency_mode=parsed["frequency_mode"],
                frequency_stages=parsed["frequency_stages"],
                frequency_gamma=float(parsed["frequency_gamma"]),
                frequency_experts=parsed["frequency_experts"],
                moe_top_k=int(parsed["moe_top_k"]),
                moe_temperature=float(parsed["moe_temperature"]),
                moe_prior=parsed["moe_prior"],
                moe_balance_weight=float(parsed["moe_balance_weight"]),
                moe_entropy_weight=float(parsed["moe_entropy_weight"]),
                use_align=parsed["use_align"] == "true",
                align_stages=parsed["align_stages"],
                align_max_flow=float(parsed["align_max_flow"]),
                context_module=parsed["context_module"],
                context_rates=parsed["context_rates"],
                context_dropout=float(parsed["context_dropout"]),
            )
            model = build_model(architecture, **kwargs).eval()
            with torch.no_grad():
                output = model(torch.randn(1, 3, input_size, input_size))
            if isinstance(output, (tuple, list)):
                output = output[0]
            expected = (1, 1, input_size, input_size)
            if tuple(output.shape) != expected or not torch.isfinite(output).all():
                raise RuntimeError("Smoke test failed for {} {}: {}".format(
                    architecture, variant, tuple(output.shape)))
            print("SMOKE PASS: {} {} -> {}".format(architecture, variant, expected))


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    architectures = parse_csv_list(args.architectures, ARCHITECTURES)
    variants = list(ALL_ABLATIONS) if args.variants.strip().lower() == "all" else parse_csv_list(args.variants, VARIANTS)
    datasets = parse_csv_list(args.datasets)
    seeds = [int(value) for value in parse_csv_list(args.seeds)]

    errors = validate_prerequisites(
        project_root,
        args.related_models_root,
        architectures,
        dry_run=args.dry_run and not args.smoke_test,
        allow_missing_datasets=args.smoke_test,
    )
    if errors:
        print("Prerequisite check failed:", file=sys.stderr)
        for error in errors:
            print("  - " + error, file=sys.stderr)
        return 2

    if args.smoke_test:
        smoke_test(project_root, args.related_models_root, architectures, variants, args.input_size)
        return 0

    output_dir = project_root / args.output_dir
    logs_dir = output_dir / "logs"
    existing_rows = [] if args.reset_results else load_saved_results(output_dir)
    rows = []

    def persist():
        combined = rows if args.reset_results else merge_results(existing_rows, rows)
        return save_results(combined, output_dir)

    for dataset in datasets:
        for architecture in architectures:
            for variant in variants:
                for seed in seeds:
                    name = experiment_name(dataset, architecture, variant, seed)
                    model_path = project_root / "models" / name / "model.pth"
                    row = {
                        "dataset": dataset,
                        "architecture": architecture,
                        "variant": variant,
                        "seed": seed,
                        "name": name,
                        "status": "pending",
                        "model_path": str(model_path),
                        "train_log": str(logs_dir / (name + "_train.txt")),
                        "validation_log": str(logs_dir / (name + "_val.txt")),
                    }
                    train_cmd = training_command(args, dataset, architecture, variant, seed, name)
                    val_cmd = validation_command(args, name)
                    print("\n=== {} ===".format(name))
                    print("TRAIN:", subprocess.list2cmdline(train_cmd))
                    print("VALID:", subprocess.list2cmdline(val_cmd))
                    if args.dry_run:
                        row["status"] = "dry_run"
                        rows.append(row)
                        continue

                    if not args.validate_only:
                        if args.skip_existing and model_path.is_file():
                            print("SKIP TRAIN: existing model {}".format(model_path))
                        else:
                            code, _ = run_command(train_cmd, project_root, Path(row["train_log"]))
                            if code != 0:
                                row["status"] = "train_failed"
                                rows.append(row)
                                persist()
                                if args.continue_on_error:
                                    continue
                                return code

                    if args.train_only:
                        row["status"] = "trained" if model_path.is_file() else "train_finished_no_model"
                    else:
                        if not model_path.is_file():
                            row["status"] = "missing_model"
                            rows.append(row)
                            persist()
                            if args.continue_on_error:
                                continue
                            return 3
                        code, text = run_command(val_cmd, project_root, Path(row["validation_log"]))
                        row.update(parse_validation_metrics(text))
                        row["status"] = "completed" if code == 0 else "validation_failed"
                        if code != 0 and not args.continue_on_error:
                            rows.append(row)
                            persist()
                            return code
                    rows.append(row)
                    persist()

    csv_path, json_path = persist()
    print("\nCompleted {} jobs in this invocation.".format(len(rows)))
    print("CSV :", csv_path)
    print("JSON:", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
