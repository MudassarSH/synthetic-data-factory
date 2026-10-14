import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


HAS_TRAINING = all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "tqdm"))


def fixed_samples(root, size=64):
    root.mkdir()
    for i in range(2):
        sample = root / f"sample_{i:05d}"
        sample.mkdir()
        mask = np.zeros((size, size), dtype=np.uint8)
        if i:
            mask[:, :size // 2] = 255
        Image.fromarray(np.repeat(mask[..., None], 3, axis=2)).save(sample / "input.png")
        Image.fromarray(mask).save(sample / "mask.png")
        metadata = {"source": f"doc_validation/page{i}.png", "seed": 2**64 - 1,
                    "defects": [{"type": "stain"}] if i else []}
        (sample / "metadata.json").write_text(json.dumps(metadata))


@unittest.skipUnless(HAS_TRAINING, "optional training dependencies not installed")
class TrainingStackTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_model_full_resolution_and_backward(self):
        import torch
        from ml_service.training.model import DocumentDefectModel
        from ml_service.training.losses import BCEDiceLoss

        model = DocumentDefectModel(pretrained=False).eval()
        with torch.no_grad():
            logits = model(torch.rand(1, 3, 512, 512))
        self.assertEqual(tuple(logits.shape), (1, 1, 512, 512))
        self.assertTrue(torch.isfinite(logits).all())
        model.train()
        logits = model(torch.rand(2, 3, 64, 96))
        self.assertEqual(tuple(logits.shape), (2, 1, 64, 96))
        BCEDiceLoss()(logits, torch.zeros_like(logits)).backward()
        self.assertIsNotNone(model.encoder[0][0].weight.grad)
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)

    def test_loss_handles_clean_pages_and_large_half_precision_reductions(self):
        import torch
        from ml_service.training.losses import BCEDiceLoss, segmentation_targets

        criterion = BCEDiceLoss()
        targets = torch.zeros(2, 1, 512, 512)
        targets[1, :, :256] = 1
        logits = torch.zeros_like(targets, dtype=torch.float16, requires_grad=True)
        loss = criterion(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        perfect = torch.where(targets.bool(), 20., -20.)
        self.assertLess(float(criterion(perfect, targets)), float(loss.detach()))
        masks = torch.tensor([0., .049, .05, .1])
        self.assertEqual(segmentation_targets(masks).tolist(), [0., 0., 1., 1.])

    def test_fixed_evaluation_metrics_and_clean_false_positives(self):
        import torch
        from torch.utils.data import DataLoader
        from ml_service.training.dataset import collate_document_samples
        from ml_service.training.evaluate import FixedExportDataset, evaluate, prediction_indices

        validation_dir = self.root / "validation"
        fixed_samples(validation_dir)
        dataset = FixedExportDataset(validation_dir, size=(64, 64))
        original_fingerprint = dataset.fingerprint
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_document_samples)

        class Perfect(torch.nn.Module):
            def forward(self, images):
                return torch.logit(images[:, :1].clamp(.0001, .9999))

        model = Perfect().train()
        with patch("ml_service.training.dataset.synthesize", side_effect=AssertionError("validation regenerated")):
            metrics = evaluate(model, loader, torch.device("cpu"), preview_indices=prediction_indices(dataset, 2),
                               preview_path=self.root / "predictions.png")
        self.assertTrue(model.training)
        self.assertEqual(metrics["dice"], 1)
        self.assertEqual(metrics["iou"], 1)
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["clean_sample_count"], 1)
        self.assertLess(metrics["clean_mean_predicted_probability"], .001)
        self.assertEqual(metrics["clean_predicted_coverage_gt_0_5"], 0)
        with Image.open(self.root / "predictions.png") as sheet:
            self.assertEqual(sheet.size, (768, 604))

        class AllDefect(torch.nn.Module):
            def forward(self, images):
                return torch.full_like(images[:, :1], 10)

        bad = evaluate(AllDefect(), loader, torch.device("cpu"))
        self.assertGreater(bad["clean_mean_predicted_probability"], .99)
        self.assertEqual(bad["clean_predicted_coverage_gt_0_5"], 1)
        Image.new("RGB", (64, 64), "gray").save(validation_dir / "sample_00000/input.png")
        changed = FixedExportDataset(validation_dir, size=(64, 64))
        self.assertNotEqual(original_fingerprint, changed.fingerprint)

    def test_fixed_validation_rejects_wrong_size_or_mask_mode(self):
        from ml_service.training.evaluate import FixedExportDataset

        validation = self.root / "validation"
        fixed_samples(validation)
        with self.assertRaisesRegex(ValueError, "expected"):
            FixedExportDataset(validation)
        with self.assertRaisesRegex(ValueError, "mask mode mismatch"):
            FixedExportDataset(validation, size=(64, 64), mask_mode="attenuation")

    def test_training_evaluation_and_resume_cli(self):
        import torch

        train = self.root / "train" / "doc_train"
        train.mkdir(parents=True)
        Image.new("RGB", (48, 64), "white").save(train / "page.png")
        validation = self.root / "validation"
        fixed_samples(validation)
        output = self.root / "run"
        command = [sys.executable, "-m", "ml_service.training.train", str(train.parent), str(validation),
                   "--output-dir", str(output), "--epochs", "1", "--samples-per-epoch", "4",
                   "--batch-size", "2", "--size", "64", "--device", "cpu", "--cpu-threads", "2",
                   "--preview-count", "2", "--no-pretrained", "--no-progress"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((output / "best.pt").is_file())
        self.assertTrue((output / "previews/epoch_001.png").is_file())
        checkpoint = torch.load(output / "last.pt", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 0)
        self.assertEqual(checkpoint["metrics"]["validation"]["sample_count"], 2)
        self.assertTrue(checkpoint["optimizer"]["state"])
        evaluation = subprocess.run([sys.executable, "-m", "ml_service.training.evaluate",
                                     str(output / "best.pt"), str(validation), "--output-dir", str(self.root / "evaluation"),
                                     "--device", "cpu", "--preview-count", "2"],
                                    capture_output=True, text=True, timeout=120,
                                    env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)
        self.assertTrue((self.root / "evaluation/metrics.json").is_file())
        command[command.index("--epochs") + 1] = "2"
        resume = subprocess.run(command + ["--resume", str(output / "last.pt")],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(resume.returncode, 0, resume.stdout + resume.stderr)
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([record["epoch"] for record in records], [1, 2])
        self.assertTrue((output / "previews/epoch_002.png").is_file())
        # Fixed sample bytes are never rewritten during either epoch.
        fingerprint = torch.load(output / "last.pt", weights_only=True)["config"]["validation_fingerprint"]
        self.assertEqual(fingerprint, checkpoint["config"]["validation_fingerprint"])


if __name__ == "__main__":
    unittest.main()
