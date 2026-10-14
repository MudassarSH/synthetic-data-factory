import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ml_service.training.dataset import (
    DocumentDefectDataset, collate_document_samples, find_images, load_clean_page,
)
from ml_service.training.preview import export_samples
from ml_service.training.split import split_clean_pages
from ml_service.training.stats import summarize
from ml_service.training.synthesis import synthesize
from ml_service.training.synthesis.compose import GENERATORS


class OperationalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.page = self.source / "page.png"
        Image.new("RGB", (32, 48), "white").save(self.page)

    def test_numpy_collation_preserves_ragged_metadata_and_large_seed(self):
        image = np.ones((8, 8, 3), np.float32)
        mask = np.zeros((8, 8), np.float32)
        metadata = [{"seed": 2**64 - 1, "defects": []},
                    {"seed": 1, "defects": [{"type": "stain", "tint": [1, .9, .8]}]}]
        images, masks, records = collate_document_samples([(image, mask, item) for item in metadata])
        self.assertEqual(images.shape, (2, 8, 8, 3))
        self.assertEqual(masks.shape, (2, 8, 8))
        self.assertIs(records[0], metadata[0])
        self.assertEqual(records, metadata)
        with self.assertRaises(ValueError):
            collate_document_samples([])

    def test_defects_respect_page_bounds_and_padding_is_clean(self):
        weights = {kind: float(kind == "edge_shadow") for kind in GENERATORS}
        dataset = DocumentDefectDataset([self.page], size=(64, 64), seed=1,
                                        samples_per_epoch=20, defect_weights=weights)
        affected = False
        for i in range(len(dataset)):
            image, mask, metadata = dataset[i]
            x, y, w, h = metadata["page_bounds"]
            inside = np.zeros(mask.shape, dtype=bool)
            inside[y:y+h, x:x+w] = True
            self.assertTrue(np.all(image[~inside] == 1))
            self.assertTrue(np.all(mask[~inside] == 0))
            if metadata["defects"]:
                affected = True
                self.assertGreater(mask[y:y+h, x:x+w].max(), .1)
                self.assertGreater(mask[y:y+h, x].max(), 0)
        self.assertTrue(affected)

    def test_exif_orientation_precedes_page_resize(self):
        original = Image.new("RGB", (20, 40), "red")
        original.paste("blue", (0, 20, 20, 40))
        exif = Image.Exif()
        exif[274] = 6
        path = self.source / "oriented.png"
        original.save(path, exif=exif)
        page = load_clean_page(path, (64, 64))
        self.assertEqual(page.shape, (20, 40, 3))
        np.testing.assert_array_equal(page[10, 5], [0, 0, 1])
        np.testing.assert_array_equal(page[10, 35], [1, 0, 0])

    def test_export_refuses_existing_output_and_replacement_removes_stale_files(self):
        output = self.root / "export"
        export_samples(self.source, output, count=3, size=(64, 64))
        original = (output / "export_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            export_samples(self.source, output, count=1, size=(64, 64))
        self.assertEqual((output / "export_manifest.json").read_bytes(), original)
        export_samples(self.source, output, count=1, size=(64, 64), grid=False, replace=True,
                       mask_mode="attenuation")
        self.assertEqual({p.name for p in output.iterdir()}, {"sample_00000", "export_manifest.json"})
        manifest = json.loads((output / "export_manifest.json").read_text())
        self.assertEqual(manifest["sample_count"], 1)
        self.assertEqual(manifest["mask_mode"], "attenuation")
        self.assertEqual(manifest["sources"][0]["sha256"], hashlib.sha256(self.page.read_bytes()).hexdigest())
        metadata = json.loads((output / "sample_00000/metadata.json").read_text())
        self.assertEqual(manifest["samples"][0]["seed"], metadata["seed"])

    def test_failed_replacement_preserves_old_export(self):
        output = self.root / "export"
        export_samples(self.source, output, count=1, size=(64, 64), grid=False)
        original = (output / "sample_00000/input.png").read_bytes()
        with patch("ml_service.training.dataset.synthesize", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                export_samples(self.source, output, count=2, size=(64, 64), replace=True)
        self.assertEqual((output / "sample_00000/input.png").read_bytes(), original)
        self.assertEqual(len(list(output.glob("sample_*"))), 1)

    def test_invalid_export_options_have_no_output_side_effects(self):
        for options in ({"count": 0}, {"size": (0, 64)}, {"mask_mode": "unknown"},
                        {"defect_weights": {"haze": -1}}):
            with self.subTest(options=options):
                output = self.root / "new-parent" / "export"
                with self.assertRaises(ValueError):
                    export_samples(self.source, output, **options)
                self.assertFalse(output.parent.exists())

    def test_replacement_protects_unmanaged_data_and_source(self):
        output = self.root / "export"
        output.mkdir()
        (output / "notes.txt").write_text("keep me")
        with self.assertRaises(ValueError):
            export_samples(self.source, output, count=1, replace=True)
        self.assertTrue((output / "notes.txt").exists())
        with self.assertRaises(ValueError):
            export_samples(self.source, self.source, count=1, replace=True)
        managed = self.root / "managed"
        export_samples(self.source, managed, count=1, size=(64, 64), grid=False)
        (managed / "sample_00000/notes.txt").write_text("keep me")
        with self.assertRaises(ValueError):
            export_samples(self.source, managed, count=1, replace=True)
        self.assertTrue((managed / "sample_00000/notes.txt").exists())

    def test_document_sampling_balances_documents_and_is_order_independent(self):
        for group, count in (("large", 20), ("small", 1)):
            folder = self.source / group
            folder.mkdir()
            for i in range(count):
                Image.new("RGB", (8, 12), "white").save(folder / f"{i}.png")
        self.page.unlink()
        dataset = DocumentDefectDataset(find_images(self.source), seed=42, size=(16, 16),
                                        samples_per_epoch=400, sampling="document", source_root=self.source)
        counts = Counter(dataset[i][2]["source"].split("/")[0] for i in range(len(dataset)))
        self.assertTrue(160 <= counts["small"] <= 240, counts)
        first = dataset[20]
        dataset[4]
        np.testing.assert_array_equal(first[0], dataset[20][0])
        self.assertEqual(first[2], dataset[20][2])
        with self.assertRaises(ValueError):
            DocumentDefectDataset([self.page], dynamic=False, sampling="document")

    def test_defect_weights_restrict_selection_and_reject_invalid_values(self):
        weights = {kind: float(kind == "ring") for kind in GENERATORS}
        image = np.ones((32, 32, 3), np.float32)
        for seed in range(12):
            _, _, metadata = synthesize(image, seed, count=1, defect_weights=weights)
            self.assertEqual([d["type"] for d in metadata["defects"]], ["ring"])
        for weights in ({"haze": -1}, {"haze": float("nan")}, {"unknown": 1},
                        {kind: 0 for kind in GENERATORS}, [1, 2]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                synthesize(image, 0, defect_weights=weights)

    def test_attenuation_mask_accounts_for_overlap_tint_and_black_text(self):
        def neutral(image, rng):
            mask = np.full(image.shape[:2], .3, np.float32)
            return image * .7, mask, {"type": "haze"}

        with patch.dict(GENERATORS, {"haze": neutral, "stain": neutral}):
            image = np.ones((8, 8, 3), np.float32)
            image[0] = 0
            output, mask, _ = synthesize(image, 0, defect_types=["haze", "stain"], count=2,
                                         mask_mode="attenuation")
            np.testing.assert_allclose(mask, .51, atol=1e-6)
            np.testing.assert_allclose(output[1:], .49, atol=1e-6)
            _, maximum, _ = synthesize(image, 0, defect_types=["haze", "stain"], count=2)
            np.testing.assert_allclose(maximum, .3)
        white = np.ones((64, 64, 3), np.float32)
        for kind in GENERATORS:
            output, mask, _ = synthesize(white, 3, count=1, defect_types=[kind], mask_mode="attenuation")
            np.testing.assert_allclose(mask, (1 - output).mean(axis=2), atol=1e-6)

    def test_weighted_sampling_changes_frequency_without_changing_target_mode(self):
        white = np.ones((8, 8, 3), np.float32)
        def haze(image, rng):
            return image, np.zeros(image.shape[:2], np.float32), {"type": "haze"}
        def ring(image, rng):
            return image, np.zeros(image.shape[:2], np.float32), {"type": "ring"}
        with patch.dict(GENERATORS, {"haze": haze, "ring": ring}):
            haze_count = 0
            for seed in range(200):
                _, _, metadata = synthesize(white, seed, count=1, defect_types=["haze", "ring"],
                                             defect_weights={"haze": 9, "ring": 1})
                haze_count += metadata["defects"][0]["type"] == "haze"
                self.assertEqual(metadata["mask_mode"], "max")
            self.assertGreater(haze_count, 160)

    def test_split_groups_multiple_different_pages_and_duplicate_bridges(self):
        self.page.unlink()
        for group, colors in (("one", ["white", "gray"]), ("two", ["gray", "black"]),
                              ("three", ["red", "blue"])):
            folder = self.source / group
            folder.mkdir()
            for i, color in enumerate(colors):
                Image.new("RGB", (8, 8), color).save(folder / f"{i}.png")
        manifest = split_clean_pages(self.source, self.root / "split", seed=3)
        assigned = {row["source"]: row["split"] for row in manifest["pages"]}
        self.assertEqual(len({assigned[f"{group}/{i}.png"] for group in ("one", "two") for i in range(2)}), 1)
        self.assertEqual(assigned["three/0.png"], assigned["three/1.png"])
        self.assertNotEqual(assigned["one/0.png"], assigned["three/0.png"])

    def test_statistics_report_effective_defects_and_pixel_balance(self):
        weights = {kind: float(kind == "bleed") for kind in GENERATORS}
        report = summarize(self.source, count=20, size=(64, 64), defect_weights=weights)
        self.assertEqual(report["effective_defect_count_percent"]["0"], 100)
        self.assertLess(report["clean_sample_ratio"], 1)
        self.assertEqual(sum(report["per_defect_positive_pixel_share_percent"].values()), 0)
        self.assertEqual(sum(report["document_sample_percent"].values()), 100)
        self.assertEqual(report["coverage_region"], "physical_page_excluding_padding")


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency not installed")
class TorchIntegrationTests(unittest.TestCase):
    def test_tensor_collation_preserves_unsigned_seed(self):
        import torch

        metadata = [{"seed": 2**64 - 1, "defects": []}, {"seed": 0, "defects": [{"type": "haze"}]}]
        batch = [(torch.ones(3, 8, 8), torch.zeros(1, 8, 8), item) for item in metadata]
        images, masks, records = collate_document_samples(batch)
        self.assertEqual(tuple(images.shape), (2, 3, 8, 8))
        self.assertEqual(tuple(masks.shape), (2, 1, 8, 8))
        self.assertEqual(records, metadata)

    def test_dataloader_reproducibility_across_workers_and_epochs(self):
        import torch
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for group in ("one", "two"):
                (root / group).mkdir()
                Image.new("RGB", (24, 32), "white").save(root / group / "page.png")
            for dynamic, sampling in ((True, "page"), (True, "document"), (False, "page")):
                dataset = DocumentDefectDataset(find_images(root), seed=2026, samples_per_epoch=8,
                                                size=(32, 32), dynamic=dynamic, sampling=sampling,
                                                source_root=root, as_torch=True)
                previous = None
                for epoch in (0, 1):
                    dataset.set_epoch(epoch)
                    serial = list(DataLoader(dataset, batch_size=3, collate_fn=collate_document_samples))
                    parallel = list(DataLoader(dataset, batch_size=3, num_workers=2,
                                               multiprocessing_context="spawn", collate_fn=collate_document_samples))
                    for a, b in zip(serial, parallel):
                        self.assertTrue(torch.equal(a[0], b[0]))
                        self.assertTrue(torch.equal(a[1], b[1]))
                        self.assertEqual(a[2], b[2])
                    if previous is not None:
                        self.assertEqual(previous == serial[0][2], not dynamic)
                    previous = serial[0][2]


if __name__ == "__main__":
    unittest.main()
