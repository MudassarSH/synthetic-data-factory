import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from ml_service.training.dataset import DocumentDefectDataset, load_clean_image
from ml_service.training.dataset import fixed_sample_seed
from ml_service.training.synthesis import synthesize
from ml_service.training.synthesis.compose import GENERATORS
from ml_service.training.split import split_clean_pages
from ml_service.training.stats import summarize
from ml_service.training.preview import export_samples


@contextmanager
def test_directory():
    root = Path(__file__).parent / "temp" / uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        image = Image.new("RGB", (96, 128), "white")
        draw = ImageDraw.Draw(image)
        for y in range(15, 120, 12):
            draw.text((8, y), "Document text 123", fill="black")
        self.clean = np.asarray(image, dtype=np.float32) / 255

    def test_seed_and_clean_sample(self):
        first = synthesize(self.clean, seed=123, count=3)
        second = synthesize(self.clean, seed=123, count=3)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        clean, mask, metadata = synthesize(self.clean, seed=123, count=0)
        np.testing.assert_array_equal(clean, self.clean)
        self.assertFalse(mask.any())
        self.assertEqual(metadata["defects"], [])

    def test_each_generator_has_bounded_aligned_mask(self):
        for kind in GENERATORS:
            with self.subTest(kind=kind):
                degraded, mask, metadata = synthesize(self.clean, seed=73, defect_types=[kind], count=1)
                self.assertEqual(degraded.shape, self.clean.shape)
                self.assertEqual(mask.shape, self.clean.shape[:2])
                self.assertTrue(np.isfinite(degraded).all())
                self.assertTrue(np.isfinite(mask).all())
                self.assertGreater(mask.max(), 0)
                self.assertGreaterEqual(mask.min(), 0)
                self.assertLessEqual(mask.max(), 1)
                self.assertGreaterEqual(degraded.min(), 0)
                self.assertLessEqual(degraded.max(), 1)
                self.assertEqual(metadata["defects"][0]["type"], kind)
                self.assertAlmostEqual(metadata["defects"][0]["coverage_at_0.05"], float(np.mean(mask >= 0.05)))
                self.assertTrue(np.all(degraded <= self.clean + 1e-6))

    def test_loader_padding_and_epoch_seeding(self):
        with test_directory() as directory:
            path = directory / "page.png"
            Image.fromarray(np.uint8(self.clean * 255)).save(path)
            loaded = load_clean_image(path, (128, 128))
            self.assertEqual(loaded.shape, (128, 128, 3))
            self.assertTrue(np.all(loaded[:, :15] == 1))
            dataset = DocumentDefectDataset([path], size=(128, 128), seed=42, dynamic=True)
            first = dataset[0]
            self.assertEqual(first[2], dataset[0][2])
            dataset.set_epoch(1)
            self.assertNotEqual(first[2]["seed"], dataset[0][2]["seed"])
            fixed = DocumentDefectDataset([path], size=(128, 128), seed=42, dynamic=False)
            before = fixed[0][2]
            fixed.set_epoch(5)
            self.assertEqual(before, fixed[0][2])

    def test_fixed_seed_uses_source_and_variant(self):
        self.assertEqual(fixed_sample_seed(12, "page-a.png", 2), fixed_sample_seed(12, "page-a.png", 2))
        self.assertNotEqual(fixed_sample_seed(12, "page-a.png", 2), fixed_sample_seed(12, "page-a.png", 3))
        with test_directory() as root:
            for name in ("a.png", "c.png"):
                Image.fromarray(np.uint8(self.clean * 255)).save(root / name)
            original = DocumentDefectDataset([root / "a.png", root / "c.png"], seed=12, dynamic=False, source_root=root)
            original_seed = original[1][2]["seed"]
            Image.fromarray(np.uint8(self.clean * 255)).save(root / "b.png")
            expanded = DocumentDefectDataset([root / "a.png", root / "b.png", root / "c.png"], seed=12, dynamic=False, source_root=root)
            self.assertEqual(original_seed, expanded[2][2]["seed"])

    def test_split_keeps_identical_pages_together(self):
        with test_directory() as root:
            source, output = root / "unsplit", root / "split"
            source.mkdir()
            for name, color in (("a.png", "white"), ("duplicate.png", "white"), ("b.png", "gray"), ("c.png", "black")):
                Image.new("RGB", (20, 20), color).save(source / name)
            manifest = split_clean_pages(source, output, seed=7)
            assigned = {row["source"]: row["split"] for row in manifest["pages"]}
            self.assertEqual(assigned["a.png"], assigned["duplicate.png"])
            self.assertGreater(manifest["counts"]["train"], 0)
            self.assertGreater(manifest["counts"]["validation"], 0)
            self.assertEqual(len(list((output / "train").rglob("*.png"))) + len(list((output / "validation").rglob("*.png"))), 4)

    def test_strength_and_statistics(self):
        for kind in GENERATORS:
            with self.subTest(kind=kind):
                degraded, mask, _ = synthesize(self.clean, seed=101, defect_types=[kind], count=1)
                white = (self.clean.min(axis=2) > 0.99) & (mask > 0.02)
                if white.any():
                    attenuation = (1 - degraded[white].mean(axis=1)) / mask[white]
                    self.assertGreater(float(np.median(attenuation)), 0.84)
                    self.assertLessEqual(float(np.median(attenuation)), 1.01)
        with test_directory() as directory:
            path = directory / "page.png"
            Image.fromarray(np.uint8(self.clean * 255)).save(path)
            report = summarize(directory, count=20, seed=4, size=(96, 128))
            self.assertEqual(report["sample_count"], 20)
            self.assertAlmostEqual(sum(report["coverage_buckets_percent"].values()), 100)
            self.assertAlmostEqual(sum(report["defect_count_percent"].values()), 100)
            self.assertEqual(set(report["coverage_by_threshold_percent"]), {"0.05", "0.15", "0.30", "0.50"})
            means = [report["coverage_by_threshold_percent"][threshold]["mean"] for threshold in ("0.05", "0.15", "0.30", "0.50")]
            self.assertEqual(means, sorted(means, reverse=True))
            self.assertEqual(set(report["per_defect_coverage_percent"]), set(GENERATORS))
            for kind, values in report["per_defect_coverage_percent"].items():
                self.assertEqual(values["sample_count"], round(report["type_percent"][kind] * 20 / 100))
                if values["sample_count"]:
                    self.assertTrue(0 <= values["mean"] <= 100)
                    self.assertTrue(0 <= values["p50"] <= 100)
                    self.assertTrue(0 <= values["p95"] <= 100)

    def test_split_root_is_rejected_before_export_or_stats(self):
        with test_directory() as root:
            (root / "train").mkdir()
            (root / "validation").mkdir()
            output = root / "export"
            with self.assertRaisesRegex(ValueError, "Input appears to be a split root"):
                summarize(root, count=1)
            with self.assertRaisesRegex(ValueError, "Input appears to be a split root"):
                export_samples(root, output, count=1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
