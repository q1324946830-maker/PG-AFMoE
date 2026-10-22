import argparse
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "run_external_backbone_experiments.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("external_runner", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExternalBackboneRunnerTests(unittest.TestCase):
    def test_baseline_disables_all_three_grafts(self):
        runner = load_runner()
        args = runner.variant_arguments("baseline")
        self.assertEqual(args[args.index("--frequency_mode") + 1], "add")
        self.assertEqual(args[args.index("--frequency_stages") + 1], "none")
        self.assertEqual(args[args.index("--use_align") + 1], "false")
        self.assertEqual(args[args.index("--context_module") + 1], "none")

    def test_full_enables_pg_afmoe_sfga_and_aspp(self):
        runner = load_runner()
        args = runner.variant_arguments("full")
        self.assertEqual(args[args.index("--frequency_mode") + 1], "afmoe")
        self.assertEqual(args[args.index("--frequency_stages") + 1], "t3")
        self.assertEqual(args[args.index("--use_align") + 1], "true")
        self.assertEqual(args[args.index("--align_stages") + 1], "t3")
        self.assertEqual(args[args.index("--context_module") + 1], "aspp")

    def test_validation_output_is_parsed_into_numeric_metrics(self):
        runner = load_runner()
        text = """IoU: 0.8543
Dice: 0.9205
Accuracy: 0.9698
Hd: 21.9439
Hd95: 1.7070
Recall: 0.9220
Specificity: 0.9809
Precision: 0.9218
"""
        metrics = runner.parse_validation_metrics(text)
        self.assertEqual(metrics["iou"], 0.8543)
        self.assertEqual(metrics["dice"], 0.9205)
        self.assertEqual(metrics["hd95"], 1.7070)
        self.assertEqual(metrics["precision"], 0.9218)

    def test_experiment_name_contains_dataset_arch_variant_and_seed(self):
        runner = load_runner()
        self.assertEqual(
            runner.experiment_name("ISIC17", "UNet3Plus", "full", 41),
            "ext_ISIC17_UNet3Plus_full_s41",
        )

    def test_local_unetplusplus_requires_no_external_repository(self):
        runner = load_runner()
        self.assertEqual(runner.required_external_paths("UNetPlusPlus"), [])

    def test_smoke_prerequisites_do_not_require_dataset_directory(self):
        runner = load_runner()
        errors = runner.validate_prerequisites(
            ROOT,
            ROOT / "missing-related-models",
            ["UNetPlusPlus"],
            allow_missing_datasets=True,
        )
        self.assertEqual(errors, [])

    def test_result_merge_replaces_same_job_and_preserves_other_jobs(self):
        runner = load_runner()
        existing = [
            {"name": "job-a", "status": "old"},
            {"name": "job-b", "status": "completed"},
        ]
        merged = runner.merge_results(existing, [{"name": "job-a", "status": "completed"}])
        self.assertEqual(merged, [
            {"name": "job-a", "status": "completed"},
            {"name": "job-b", "status": "completed"},
        ])

    def test_training_command_uses_selected_loss(self):
        runner = load_runner()
        args = argparse.Namespace(
            python="python",
            epochs=400,
            batch_size=8,
            num_workers=4,
            input_size=256,
            loss="BCEWithLogitsLoss",
            base_channels=32,
            related_models_root="../Related-Models",
        )
        command = runner.training_command(args, "ISIC17", "UNet", "baseline", 41, "job")
        self.assertEqual(command[command.index("--loss") + 1], "BCEWithLogitsLoss")


if __name__ == "__main__":
    unittest.main()
