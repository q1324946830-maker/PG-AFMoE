import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class EpochConsoleOutputTests(unittest.TestCase):
    def test_training_has_no_per_batch_tqdm_output(self):
        source = (ROOT / "train_GBC.py").read_text(encoding="utf-8")
        self.assertNotIn("from tqdm import tqdm", source)
        self.assertNotIn("pbar", source)

    def test_standalone_validation_has_no_per_batch_tqdm_output(self):
        source = (ROOT / "val_GBC.py").read_text(encoding="utf-8")
        self.assertNotIn("from tqdm import tqdm", source)
        self.assertNotIn("tqdm(", source)

    def test_epoch_counter_is_one_based(self):
        source = (ROOT / "train_GBC.py").read_text(encoding="utf-8")
        self.assertIn("epoch + 1, config['epochs']", source)


if __name__ == "__main__":
    unittest.main()
