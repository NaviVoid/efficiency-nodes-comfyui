import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "py" / "latent_noise.py"
SPEC = importlib.util.spec_from_file_location("efficiency_latent_noise", MODULE_PATH)
latent_noise = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latent_noise)


class LatentNoiseTests(unittest.TestCase):
    def test_missing_script_cache_value_is_none(self):
        script = None
        cache_value = script.get("latent_noise") if script else None
        self.assertIsNone(cache_value)

    def test_same_seed_is_reproducible(self):
        samples = torch.zeros((2, 4, 8, 12))
        first = latent_noise.transform_samples(
            samples, 1234, 0.35, "random", "perlin", True
        )
        second = latent_noise.transform_samples(
            samples, 1234, 0.35, "random", "perlin", True
        )
        self.assertTrue(torch.equal(first, second))

    def test_different_seed_changes_noise(self):
        samples = torch.zeros((1, 4, 8, 8))
        first = latent_noise.transform_samples(samples, 1, 0.5, "none", "gaussian", False)
        second = latent_noise.transform_samples(samples, 2, 0.5, "none", "gaussian", False)
        self.assertFalse(torch.equal(first, second))

    def test_geometry_preserves_latent_shape(self):
        samples = torch.randn((1, 4, 8, 12))
        for transform in latent_noise.GEOMETRY_TRANSFORMS:
            result = latent_noise.transform_samples(samples, 0, 0.0, transform, "gaussian", False)
            self.assertEqual(result.shape, samples.shape)

    def test_anima_latent_shape_is_supported(self):
        samples = torch.randn((1, 16, 1, 8, 12))
        for transform in latent_noise.GEOMETRY_TRANSFORMS:
            result = latent_noise.transform_samples(samples, 0, 0.2, transform, "perlin", True)
            self.assertEqual(result.shape, samples.shape)

    def test_latent_metadata_is_preserved(self):
        samples = torch.zeros((1, 4, 4, 4))
        latent = {"samples": samples, "noise_mask": torch.ones((1, 4, 4))}
        result = latent_noise.transform_latent(latent, 0, 0.1, "none", "uniform", False)
        self.assertIs(result["noise_mask"], latent["noise_mask"])
        self.assertIsNot(result["samples"], samples)


if __name__ == "__main__":
    unittest.main()
