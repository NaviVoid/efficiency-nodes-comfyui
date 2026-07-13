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
RUNTIME_MODULE_PATH = Path(__file__).resolve().parents[1] / "py" / "runtime_metadata.py"
RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "efficiency_runtime_metadata", RUNTIME_MODULE_PATH
)
runtime_metadata = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(runtime_metadata)
sys.modules["efficiency_runtime_metadata"] = runtime_metadata
SPEC = importlib.util.spec_from_file_location("efficiency_save_image", MODULE_PATH)
save_image = importlib.util.module_from_spec(SPEC)
save_image.__package__ = "efficiency_save_image"
sys.modules["efficiency_save_image.runtime_metadata"] = runtime_metadata
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

    def test_normalizes_prompts_and_hashes_inline_loras(self):
        with tempfile.TemporaryDirectory() as directory:
            lora_paths = {
                "tool-front_lighting-illu.safetensors": Path(directory) / "front.safetensors",
                "tool-Dramatic Lighting Slider-illu.safetensors": Path(directory)
                / "dramatic.safetensors",
            }
            for index, path in enumerate(lora_paths.values(), 1):
                path.write_bytes(f"lora-{index}".encode())

            prompt = (
                "1girl, solo,  \n\n meion, masterpiece, "
                "<lora:tool-front_lighting-illu:0.80> "
                "<lora:tool-Dramatic Lighting Slider-illu:4.00>"
            )
            current_workflow = workflow()
            current_workflow["1"]["inputs"].pop("lora_name")
            current_workflow["1"]["inputs"]["positive"] = prompt
            current_workflow["1"]["inputs"]["negative"] = "bad quality,\n\n watermark"

            def get_full_path(folder_name, filename):
                path = lora_paths.get(filename) if folder_name == "loras" else None
                return str(path) if path else None

            with patch("folder_paths.get_full_path", side_effect=get_full_path), patch(
                "folder_paths.get_filename_list",
                return_value=list(lora_paths),
            ):
                metadata = extract_metadata(current_workflow, "3", 1280, 1920)

            self.assertNotIn("\n", metadata["prompt"])
            self.assertEqual(metadata["negative_prompt"], "bad quality, watermark")
            self.assertEqual(metadata["loras"], "")
            self.assertEqual(set(metadata["lora_hashes"]), {
                "tool-front_lighting-illu",
                "tool-Dramatic Lighting Slider-illu",
            })

            parameters = format_metadata(metadata)
            self.assertEqual(len(parameters.splitlines()), 3)
            self.assertIn('Lora hashes: "tool-front_lighting-illu:', parameters)

    def test_prompt_trace_does_not_mix_positive_and_negative_branches(self):
        prompt = {
            "positive": {"inputs": {"text": "positive text"}},
            "negative": {"inputs": {"text": "negative text"}},
            "pipe": {
                "inputs": {
                    "pos": ["positive", 0],
                    "neg": ["negative", 0],
                }
            },
        }

        self.assertEqual(
            save_image._find_prompt_text(prompt, ["pipe", 1], "negative"),
            "negative text",
        )

    def test_runtime_pipe_metadata_overrides_static_prompt_trace(self):
        runtime_metadata._reset_for_tests()
        pipe = {
            "loader_settings": {
                "positive": "runtime positive",
                "negative": "runtime negative",
                "ckpt_name": "demo.safetensors",
                "lora_stack": [("lighting.safetensors", 0.8, 0.8)],
            }
        }
        runtime_metadata.record_runtime_inputs(
            "prompt-1", "pipe", "pipeOut", {"pipe": [pipe]}
        )
        runtime_metadata.record_runtime_inputs(
            "prompt-1",
            "lora",
            "LoraStackerLM",
            {"text": ["<lora:lighting:0.80>"]},
        )
        runtime_data = runtime_metadata.get_runtime_metadata({"pipe", "lora"})

        with patch("folder_paths.get_full_path", return_value=None), patch(
            "folder_paths.get_filename_list", return_value=[]
        ):
            metadata = extract_metadata(
                workflow(), "3", 1024, 768, runtime_data=runtime_data
            )

        self.assertEqual(metadata["prompt"], "runtime positive")
        self.assertEqual(metadata["negative_prompt"], "runtime negative")
        self.assertEqual(metadata["checkpoint"], "demo.safetensors")
        self.assertEqual(metadata["loras"], "<lora:lighting:0.8>")


if __name__ == "__main__":
    unittest.main()
