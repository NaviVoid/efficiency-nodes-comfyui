import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


COMFY_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_DIR))

MODULE_PATH = Path(__file__).resolve().parents[1] / "py" / "save_image.py"
SPEC = importlib.util.spec_from_file_location("efficiency_save_image", MODULE_PATH)
save_image = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(save_image)
SaveImageEfficient = save_image.SaveImageEfficient
extract_metadata = save_image.extract_metadata
format_filename = save_image.format_filename
format_metadata = save_image.format_metadata


class DummyTensor:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def workflow():
    return {
        "1": {
            "class_type": "Efficient Loader",
            "inputs": {
                "ckpt_name": "models/demo.safetensors",
                "positive": "bright landscape",
                "negative": "blurry",
                "lora_name": "styles/detail.safetensors",
                "lora_model_strength": 0.7,
            },
        },
        "2": {
            "class_type": "KSampler (Efficient)",
            "inputs": {
                "model": ["1", 0],
                "positive": ["1", 1],
                "negative": ["1", 2],
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "karras",
            },
        },
        "3": {
            "class_type": "Save Image (Efficient)",
            "inputs": {"images": ["2", 5]},
        },
    }


class SaveImageTests(unittest.TestCase):
    def test_extracts_efficiency_metadata_and_formats_filename(self):
        metadata = extract_metadata(workflow(), "3", 1024, 768)

        self.assertEqual(
            metadata,
            {
                "size": "1024x768",
                "seed": 42,
                "steps": 20,
                "cfg_scale": 7.0,
                "sampler": "euler_ancestral",
                "scheduler": "karras",
                "prompt": "bright landscape",
                "negative_prompt": "blurry",
                "checkpoint": "models/demo.safetensors",
                "loras": "<lora:detail:0.7>",
            },
        )
        self.assertEqual(
            format_filename(
                "%model%_%seed%_%width%x%height%_%date:yyyyMMdd%",
                metadata,
                datetime(2026, 7, 13, 12, 34, 56),
            ),
            "demo_42_1024x768_20260713",
        )

    def test_saves_png_jpeg_and_webp_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch("folder_paths.get_output_directory", return_value=directory), patch(
                "folder_paths.get_save_image_path",
                return_value=(directory, "sample", 1, "", "sample"),
            ):
                image = DummyTensor(np.zeros((2, 2, 3), dtype=np.float32))
                node = SaveImageEfficient()

                for file_format, suffix in (
                    ("png", ".png"),
                    ("jpeg", ".jpg"),
                    ("webp", ".webp"),
                ):
                    result = node.process_image(
                        image,
                        file_format=file_format,
                        unique_id="3",
                        prompt=workflow(),
                        embed_workflow=True,
                        extra_pnginfo={"workflow": {"nodes": []}},
                    )
                    output_path = output_dir / f"sample_00001_{suffix}"
                    self.assertEqual(
                        result["ui"]["images"][0]["filename"], output_path.name
                    )

                    with Image.open(output_path) as saved:
                        if file_format == "png":
                            self.assertIn("Seed: 42", saved.info["parameters"])
                            self.assertEqual(
                                json.loads(saved.info["workflow"]), {"nodes": []}
                            )
                        else:
                            user_comment = saved.getexif()[0x9286]
                            self.assertTrue(user_comment.startswith(b"UNICODE\0"))
                            self.assertIn(
                                "Seed: 42",
                                user_comment[len(b"UNICODE\0") :].decode("utf-16be"),
                            )
                            if file_format == "webp":
                                self.assertTrue(
                                    saved.getexif()[0x010E].startswith("Workflow:")
                                )

    def test_writes_checkpoint_and_lora_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "demo.safetensors"
            lora_path = Path(directory) / "detail.safetensors"
            checkpoint_path.write_bytes(b"checkpoint")
            lora_path.write_bytes(b"lora")

            def get_full_path(folder_name, filename):
                paths = {
                    ("checkpoints", "models/demo.safetensors"): checkpoint_path,
                    ("loras", "styles/detail.safetensors"): lora_path,
                }
                path = paths.get((folder_name, filename))
                return str(path) if path else None

            with patch("folder_paths.get_full_path", side_effect=get_full_path):
                metadata = extract_metadata(workflow(), "3", 1024, 768)

            checkpoint_hash = hashlib.sha256(b"checkpoint").hexdigest()
            lora_hash = hashlib.sha256(b"lora").hexdigest()
            self.assertEqual(metadata["checkpoint_hash"], checkpoint_hash)
            self.assertEqual(metadata["lora_hashes"], {"detail": lora_hash})

            parameters = format_metadata(metadata)
            self.assertIn(f"Model hash: {checkpoint_hash[:10]}", parameters)
            self.assertIn(f'Lora hashes: "detail: {lora_hash[:10]}"', parameters)


if __name__ == "__main__":
    unittest.main()
