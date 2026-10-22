import argparse
import os
import subprocess
import sys


EXPERIMENTS = {
    "baseline": [
        ("baseline_add", "add", "none"),
    ],
    "methods": [
        ("baseline_add", "add", "none"),
        ("fft_t3", "fft", "t3"),
        ("dct_t3", "dct", "t3"),
        ("dwt_t3", "dwt", "t3"),
        ("laplacian_t3", "laplacian", "t3"),
        ("gabor_t3", "gabor", "t3"),
    ],
    "dwt_ablation": [
        ("dwt_t3", "dwt", "t3"),
        ("dwt_t3_t2", "dwt", "t3,t2"),
        ("dwt_t3_t2_t1", "dwt", "t3,t2,t1"),
        ("dwt_all", "dwt", "all"),
    ],
    "afmoe": [
        ("pg_afmoe_t3", "afmoe", "t3"),
    ],
    "align_context": [
        ("pg_afmoe_t3", "afmoe", "t3"),
        ("align_pg_afmoe_t3", "afmoe", "t3"),
        ("align_aspp_pg_afmoe_t3", "afmoe", "t3"),
        ("align_aspp_focal_pg_afmoe_t3", "afmoe", "t3"),
    ],
    "afmoe_ablation": [
        ("afmoe_add_dwt_t3", "afmoe", "t3"),
        ("afmoe_add_dwt_gabor_t3", "afmoe", "t3"),
        ("pg_afmoe_t3", "afmoe", "t3"),
        ("pg_afmoe_t3_t2", "afmoe", "t3,t2"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="methods", choices=EXPERIMENTS.keys())
    parser.add_argument("--dataset", default="ISIC17")
    parser.add_argument("--arch", default="GBC_Rolling_Unet_L")
    parser.add_argument(
        "--related_models_root",
        default=os.environ.get("RELATED_MODELS_ROOT", "../Related-Models"),
    )
    parser.add_argument("--base_channels", default=32, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--loss", default="BCEDiceWithGeometryLoss")
    parser.add_argument("--div_weight", default=0.01, type=float)
    parser.add_argument("--scale_weight", default=0.1, type=float)
    parser.add_argument("--frequency_gamma", default=0.1, type=float)
    parser.add_argument("--frequency_high_ratio", default=0.25, type=float)
    parser.add_argument("--frequency_experts", default="add,dwt,gabor,dct")
    parser.add_argument("--moe_top_k", default=2, type=int)
    parser.add_argument("--moe_temperature", default=1.0, type=float)
    parser.add_argument("--moe_prior", default="auto")
    parser.add_argument("--moe_balance_weight", default=0.01, type=float)
    parser.add_argument("--moe_entropy_weight", default=0.001, type=float)
    parser.add_argument("--use_align", default="false", choices=["true", "false"])
    parser.add_argument("--align_stages", default="t3")
    parser.add_argument("--align_max_flow", default=4.0, type=float)
    parser.add_argument("--context_module", default="aspp", choices=["none", "aspp", "light_aspp"])
    parser.add_argument("--context_rates", default="1,2,3")
    parser.add_argument("--context_dropout", default=0.0, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--focal_gamma", default=2.0, type=float)
    parser.add_argument("--hard_pixel_ratio", default=0.1, type=float)
    parser.add_argument("--strong_aug", default="false", choices=["true", "false"])
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for suffix, mode, stages in EXPERIMENTS[args.suite]:
        name = "%s_%s_%s" % (args.dataset.lower(), args.arch, suffix)
        frequency_experts = args.frequency_experts
        if suffix == "afmoe_add_dwt_t3":
            frequency_experts = "add,dwt"
        elif suffix == "afmoe_add_dwt_gabor_t3":
            frequency_experts = "add,dwt,gabor"
        loss_name = args.loss
        use_align = args.use_align
        context_module = "none"
        strong_aug = args.strong_aug
        if suffix.startswith("align_"):
            use_align = "true"
        if "aspp" in suffix:
            context_module = args.context_module
            strong_aug = "true"
        if "focal" in suffix:
            loss_name = "FocalBCEDiceWithGeometryLoss"
        cmd = [
            sys.executable, "train_GBC.py",
            "--dataset", args.dataset,
            "--name", name,
            "--arch", args.arch,
            "--related_models_root", args.related_models_root,
            "--base_channels", str(args.base_channels),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--loss", loss_name,
            "--div_weight", str(args.div_weight),
            "--scale_weight", str(args.scale_weight),
            "--focal_alpha", str(args.focal_alpha),
            "--focal_gamma", str(args.focal_gamma),
            "--hard_pixel_ratio", str(args.hard_pixel_ratio),
            "--frequency_mode", mode,
            "--frequency_stages", stages,
            "--frequency_gamma", str(args.frequency_gamma),
            "--frequency_high_ratio", str(args.frequency_high_ratio),
            "--frequency_experts", frequency_experts,
            "--moe_top_k", str(args.moe_top_k),
            "--moe_temperature", str(args.moe_temperature),
            "--moe_prior", args.moe_prior,
            "--moe_balance_weight", str(args.moe_balance_weight),
            "--moe_entropy_weight", str(args.moe_entropy_weight),
            "--use_align", use_align,
            "--align_stages", args.align_stages,
            "--align_max_flow", str(args.align_max_flow),
            "--context_module", context_module,
            "--context_rates", args.context_rates,
            "--context_dropout", str(args.context_dropout),
            "--strong_aug", strong_aug,
        ]
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
