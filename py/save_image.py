from __future__ import annotations

import hashlib
import json
import os
import re
from collections import deque
from datetime import datetime
from functools import lru_cache
from typing import Any

import folder_paths
import numpy as np
from PIL import Image, PngImagePlugin

from .runtime_metadata import get_runtime_metadata

_PATTERN = re.compile(r"(%[^%]+%)")
_PARAMETER_KEYS = {
    "Steps",
    "Sampler",
    "CFG scale",
    "Seed",
    "Size",
    "Model hash",
    "Model",
    "Hires upscale",
    "Hires resize",
    "Hires upscaler",
}
_SAMPLER_NAMES = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "heun": "Heun",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "lms": "LMS",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "ddim": "DDIM",
}
_SCHEDULER_NAMES = {
    "normal": "Simple",
    "karras": "Karras",
    "exponential": "Exponential",
    "sgm_uniform": "SGM Uniform",
    "sgm_quadratic": "SGM Quadratic",
}


def _model_stem(filename: str) -> str:
    normalized = os.path.normpath(filename)
    stem, extension = os.path.splitext(normalized)
    return stem if extension.casefold() in folder_paths.supported_pt_extensions else normalized


def _single_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _as_first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else ""
    return value


def _to_string(value: Any) -> str:
    value = _as_first(value)
    return "" if value is None else str(value)


def _get_node(prompt: dict, node_id: Any) -> dict | None:
    return prompt.get(node_id) or prompt.get(str(node_id))


def _link_node_id(prompt: dict, value: Any) -> Any | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and _get_node(prompt, value[0]) is not None
    ):
        return value[0]
    return None


def _upstream_items(prompt: dict | None, node_id: Any) -> list[tuple[str, dict]]:
    if not isinstance(prompt, dict):
        return []

    node = _get_node(prompt, node_id)
    if node is None:
        return [(str(key), value) for key, value in prompt.items()]

    queue = deque(
        linked_id
        for value in node.get("inputs", {}).values()
        if (linked_id := _link_node_id(prompt, value)) is not None
    )
    seen = set()
    result = []
    while queue:
        current_id = queue.popleft()
        current_key = str(current_id)
        if current_key in seen:
            continue
        seen.add(current_key)

        current = _get_node(prompt, current_id)
        if current is None:
            continue
        result.append((current_key, current))
        queue.extend(
            linked_id
            for value in current.get("inputs", {}).values()
            if (linked_id := _link_node_id(prompt, value)) is not None
        )
    return result


def _upstream_nodes(prompt: dict | None, node_id: Any) -> list[dict]:
    return [node for _node_id, node in _upstream_items(prompt, node_id)]


def _find_prompt_text(prompt: dict, value: Any, polarity: str) -> str:
    linked_id = _link_node_id(prompt, value)
    if linked_id is None:
        return value if isinstance(value, str) else ""

    queue = deque([linked_id])
    seen = set()
    texts = []
    while queue:
        current_id = queue.popleft()
        current_key = str(current_id)
        if current_key in seen:
            continue
        seen.add(current_key)

        node = _get_node(prompt, current_id)
        if node is None:
            continue
        inputs = node.get("inputs", {})

        followed_polarity = False
        short_name = "pos" if polarity == "positive" else "neg"
        prompt_keys = [polarity, f"base_{polarity}", short_name]
        if polarity == "positive":
            prompt_keys.append("base_prompt")
        for key in prompt_keys:
            direct_text = inputs.get(key)
            if isinstance(direct_text, str):
                texts.append(direct_text)
                followed_polarity = True
                break
            if (next_id := _link_node_id(prompt, direct_text)) is not None:
                queue.append(next_id)
                followed_polarity = True
                break
        if followed_polarity:
            continue
        if isinstance(inputs.get("text"), str):
            texts.append(inputs["text"])
            continue

        flux_texts = [inputs.get("clip_l"), inputs.get("t5xxl")]
        flux_texts = [text for text in flux_texts if isinstance(text, str) and text]
        if flux_texts:
            texts.append(" ".join(dict.fromkeys(flux_texts)))
            continue

        queue.extend(
            next_id
            for input_value in inputs.values()
            if (next_id := _link_node_id(prompt, input_value)) is not None
        )

    return _single_line(" ".join(dict.fromkeys(text for text in texts if text)))


def _split_a1111_parameters(line: str) -> list[str]:
    parts = []
    current = []
    quote = ""
    escaped = False

    for char in line:
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue

        if char in {'"', "'"}:
            quote = char
            current.append(char)
        elif char == ",":
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(char)

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _unquote_parameter(value: str) -> str:
    value = value.strip()
    if len(value) < 2 or value[0] not in {'"', "'"} or value[-1] != value[0]:
        return value
    if value[0] == '"':
        try:
            loaded = json.loads(value)
            return str(loaded)
        except json.JSONDecodeError:
            pass
    return value[1:-1]


def _quote_parameter(value: Any) -> str:
    text = _single_line(value)
    if any(char in text for char in {",", '"', "\n"}):
        return json.dumps(text, ensure_ascii=False)
    return text


def _parse_parameters_line(line: str) -> dict[str, str]:
    parameters = {}
    for part in _split_a1111_parameters(line):
        key, separator, value = part.partition(":")
        if separator:
            parameters[key.strip()] = _unquote_parameter(value)
    return parameters


def _replace_parameters(line: str, replacements: dict[str, Any]) -> str:
    parts = _split_a1111_parameters(line)
    replaced = set()
    updated = []

    for part in parts:
        key, separator, _value = part.partition(":")
        key = key.strip()
        if separator and key in replacements:
            updated.append(f"{key}: {_quote_parameter(replacements[key])}")
            replaced.add(key)
        else:
            updated.append(part)

    for key, value in replacements.items():
        if key not in replaced:
            updated.append(f"{key}: {_quote_parameter(value)}")
    return ", ".join(updated)


def _parameters_line_index(lines: list[str]) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if _PARAMETER_KEYS.intersection(_parse_parameters_line(lines[index])):
            return index
    return None


def parse_metadata_text(metadata: Any) -> dict[str, Any]:
    metadata = _to_string(metadata).strip()
    lines = metadata.splitlines()
    parameters_index = _parameters_line_index(lines)
    parameter_values = (
        _parse_parameters_line(lines[parameters_index])
        if parameters_index is not None
        else {}
    )

    prompt_lines = lines[:parameters_index] if parameters_index is not None else lines
    negative_index = next(
        (
            index
            for index, line in enumerate(prompt_lines)
            if line.startswith("Negative prompt:")
        ),
        None,
    )
    if negative_index is None:
        prompt = " ".join(prompt_lines)
        negative_prompt = ""
    else:
        prompt = " ".join(prompt_lines[:negative_index])
        first_negative = prompt_lines[negative_index][len("Negative prompt:") :]
        negative_prompt = " ".join([first_negative] + prompt_lines[negative_index + 1 :])

    result: dict[str, Any] = {
        "prompt": _single_line(prompt),
        "negative_prompt": _single_line(negative_prompt),
    }
    if seed := parameter_values.get("Seed"):
        result["seed"] = seed
    if size := parameter_values.get("Size"):
        result["size"] = size
    if model := parameter_values.get("Model"):
        result["checkpoint"] = model
    if hires_upscaler := parameter_values.get("Hires upscaler"):
        result["hires_upscaler"] = hires_upscaler
    if hires_upscale := parameter_values.get("Hires upscale"):
        result["hires_upscale"] = hires_upscale
    if hires_resize := parameter_values.get("Hires resize"):
        result["hires_resize"] = hires_resize
    return {key: value for key, value in result.items() if value}


def _format_scale(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _parse_size(value: Any) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\s*x\s*(\d+)", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _infer_hires_resize(
    metadata: dict[str, Any], width: int, height: int, upscale_by: float = 0.0
) -> tuple[str, str] | None:
    if upscale_by and upscale_by > 0:
        return "Hires upscale", _format_scale(float(upscale_by))

    source_size = _parse_size(metadata.get("size"))
    if source_size is None:
        return None
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        return None

    scale_width = width / source_width
    scale_height = height / source_height
    if scale_width <= 1 and scale_height <= 1:
        return None
    if abs(scale_width - scale_height) <= 0.01:
        return "Hires upscale", _format_scale((scale_width + scale_height) / 2)
    return "Hires resize", f"{width}x{height}"


def _model_display_name(filename: str) -> str:
    return os.path.basename(_model_stem(filename))


def _valid_upscale_name(value: Any) -> bool:
    return isinstance(value, str) and value not in {
        "",
        "None",
        "(auto)",
        "(use same)",
    }


def _upscale_model_from_nodes(nodes: list[dict]) -> str:
    for node in nodes:
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})

        for key in ("upscale_model_name", "upscaler_name", "pixel_upscaler"):
            value = inputs.get(key)
            if _valid_upscale_name(value):
                return _model_display_name(value)

        value = inputs.get("model_name")
        if (
            _valid_upscale_name(value)
            and "upscale" in class_type.casefold()
        ):
            return _model_display_name(value)
    return ""


def _upscale_model_from_prompt(prompt: dict | None, node_id: Any) -> str:
    return _upscale_model_from_nodes(_upstream_nodes(prompt, node_id))


def append_hires_metadata(
    metadata: Any,
    width: int,
    height: int,
    prompt: dict | None = None,
    unique_id: Any = None,
    upscale_model_name: str = "(auto)",
    upscale_by: float = 0.0,
    update_size: bool = True,
) -> str:
    metadata_text = _to_string(metadata).strip()
    metadata_dict = parse_metadata_text(metadata_text)
    additions = []

    resize_field = _infer_hires_resize(metadata_dict, width, height, upscale_by)
    if resize_field is not None:
        additions.append(resize_field)

    upscale_model = _to_string(upscale_model_name).strip()
    if upscale_model == "(auto)":
        upscale_model = _upscale_model_from_prompt(prompt, unique_id)
    if upscale_model and upscale_model not in {"None", "(auto)"}:
        additions.append(("Hires upscaler", _model_display_name(upscale_model)))

    lines = metadata_text.splitlines()
    parameters_index = _parameters_line_index(lines)
    if parameters_index is None:
        existing = {}
        parameter_line = ""
    else:
        parameter_line = lines[parameters_index]
        existing = _parse_parameters_line(parameter_line)

    replacements = {}
    output_size = f"{width}x{height}"
    if update_size and existing.get("Size") != output_size:
        replacements["Size"] = output_size
    for key, value in additions:
        if key not in existing:
            replacements[key] = value

    if not replacements:
        return metadata_text

    parameter_line = _replace_parameters(parameter_line, replacements)

    if parameters_index is None:
        if lines:
            lines.append(parameter_line)
        else:
            lines = [parameter_line]
    else:
        lines[parameters_index] = parameter_line
    return "\n".join(lines)


@lru_cache(maxsize=128)
def _sha256(path: str, _size: int, _mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_hash(folder_name: str, filename: str) -> str | None:
    path = filename if os.path.isfile(filename) else folder_paths.get_full_path(
        folder_name, filename
    )
    if path is None:
        requested_stem = _model_stem(filename).casefold()
        candidates = folder_paths.get_filename_list(folder_name)
        match = next(
            (
                candidate
                for candidate in candidates
                if _model_stem(candidate).casefold() == requested_stem
            ),
            None,
        )
        if match is None:
            requested_stem = os.path.basename(requested_stem)
            matches = [
                candidate
                for candidate in candidates
                if os.path.basename(_model_stem(candidate)).casefold() == requested_stem
            ]
            match = matches[0] if len(matches) == 1 else None
        if match is not None:
            path = folder_paths.get_full_path(folder_name, match)
    if path is None:
        return None
    stat = os.stat(path)
    return _sha256(path, stat.st_size, stat.st_mtime_ns)


def _collect_loras(
    nodes: list[dict],
    prompt_text: str,
    runtime_loras: list[tuple[str, Any, Any]] | None = None,
    runtime_lora_text: str = "",
) -> tuple[str, dict[str, str]]:
    graph_loras = []
    for node in nodes:
        inputs = node.get("inputs", {})
        lora_name = inputs.get("lora_name")
        if isinstance(lora_name, str) and lora_name != "None":
            strength = inputs.get(
                "strength_model", inputs.get("lora_model_strength", 1.0)
            )
            graph_loras.append((lora_name, strength))

        mode = inputs.get("input_mode", "simple")
        lora_count = inputs.get("lora_count", 0)
        lora_count = lora_count if isinstance(lora_count, int) else 0
        for index in range(1, lora_count + 1):
            lora_name = inputs.get(f"lora_name_{index}")
            if not isinstance(lora_name, str) or lora_name == "None":
                continue
            strength_key = f"lora_wt_{index}" if mode == "simple" else f"model_str_{index}"
            graph_loras.append((lora_name, inputs.get(strength_key, 1.0)))

    pattern = r"<lora:([^:>]+):([^:>]+)(?::[^>]+)?>"
    prompt_loras = re.findall(
        pattern,
        prompt_text,
        re.IGNORECASE,
    )
    runtime_text_loras = re.findall(pattern, runtime_lora_text, re.IGNORECASE)
    runtime_loras = [
        (name, strength) for name, strength, _clip_strength in runtime_loras or []
    ]
    applied_loras = runtime_loras or graph_loras
    prompt_names = {
        os.path.basename(_model_stem(name)).casefold()
        for name, _strength in prompt_loras
    }

    unique = []
    seen_names = set()
    for filename, strength in applied_loras + runtime_text_loras + prompt_loras:
        name_key = os.path.basename(_model_stem(filename)).casefold()
        if name_key not in seen_names:
            seen_names.add(name_key)
            unique.append((filename, strength))
    lora_text = []
    hashes = {}
    for filename, strength in unique:
        name = os.path.basename(_model_stem(filename))
        if (
            (filename, strength) in applied_loras + runtime_text_loras
            and name.casefold() not in prompt_names
        ):
            lora_text.append(f"<lora:{name}:{strength}>")
        if hash_value := _model_hash("loras", filename):
            hashes[name] = hash_value
    return " ".join(lora_text), hashes


def extract_metadata(
    prompt: dict | None,
    node_id: Any,
    width: int,
    height: int,
    runtime_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"size": f"{width}x{height}"}
    upstream_items = _upstream_items(prompt, node_id)
    nodes = [node for _upstream_id, node in upstream_items]
    runtime_data = runtime_data or get_runtime_metadata(
        {upstream_id for upstream_id, _node in upstream_items}
    )

    sampler = next(
        (
            node
            for node in nodes
            if "steps" in node.get("inputs", {})
            and (
                "seed" in node.get("inputs", {})
                or "noise_seed" in node.get("inputs", {})
            )
        ),
        None,
    )
    if sampler is not None:
        inputs = sampler.get("inputs", {})
        metadata.update(
            {
                "seed": inputs.get("seed", inputs.get("noise_seed")),
                "steps": inputs.get("steps"),
                "cfg_scale": inputs.get("cfg"),
                "sampler": inputs.get("sampler_name"),
                "scheduler": inputs.get("scheduler"),
            }
        )
        if isinstance(prompt, dict):
            metadata["prompt"] = _find_prompt_text(
                prompt, inputs.get("positive"), "positive"
            )
            metadata["negative_prompt"] = _find_prompt_text(
                prompt, inputs.get("negative"), "negative"
            )

    if not metadata.get("prompt"):
        loader = next(
            (
                node
                for node in nodes
                if isinstance(node.get("inputs", {}).get("positive"), str)
            ),
            None,
        )
        if loader is not None:
            metadata["prompt"] = loader["inputs"].get("positive", "")
            metadata["negative_prompt"] = loader["inputs"].get("negative", "")

    if runtime_data.get("prompt"):
        metadata["prompt"] = runtime_data["prompt"]
    if runtime_data.get("negative_prompt"):
        metadata["negative_prompt"] = runtime_data["negative_prompt"]

    for key in ("seed", "steps", "cfg_scale", "sampler", "scheduler"):
        if key in runtime_data:
            metadata[key] = runtime_data[key]

    anima_artist_chain = runtime_data.get("anima_artist_chain")
    if not isinstance(anima_artist_chain, str):
        anima_pack = next(
            (
                node
                for node in nodes
                if node.get("class_type") == "AnimaArtistPack"
                and isinstance(node.get("inputs", {}).get("artist_chain"), str)
            ),
            None,
        )
        if anima_pack is not None:
            anima_artist_chain = anima_pack["inputs"]["artist_chain"]
    if isinstance(anima_artist_chain, str) and anima_artist_chain.strip():
        metadata["anima_artist_chain"] = _single_line(anima_artist_chain)

    metadata["prompt"] = _single_line(metadata.get("prompt", ""))
    metadata["negative_prompt"] = _single_line(
        metadata.get("negative_prompt", "")
    )

    checkpoint_folder = "checkpoints"
    for node in nodes:
        inputs = node.get("inputs", {})
        checkpoint = next(
            (
                (inputs[key], folder_name)
                for key, folder_name in (
                    ("ckpt_name", "checkpoints"),
                    ("base_ckpt_name", "checkpoints"),
                    ("unet_name", "unet"),
                )
                if isinstance(inputs.get(key), str) and inputs[key]
            ),
            None,
        )
        if checkpoint:
            checkpoint_name, folder_name = checkpoint
            metadata["checkpoint"] = checkpoint_name
            checkpoint_folder = folder_name
            break

    if runtime_data.get("checkpoint"):
        metadata["checkpoint"] = runtime_data["checkpoint"]
        checkpoint_folder = runtime_data.get("checkpoint_folder", "checkpoints")
    if checkpoint_name := metadata.get("checkpoint"):
        if hash_value := _model_hash(checkpoint_folder, checkpoint_name):
            metadata["checkpoint_hash"] = hash_value

    metadata["loras"], lora_hashes = _collect_loras(
        nodes,
        metadata["prompt"],
        runtime_data.get("lora_stack"),
        runtime_data.get("lora_text", ""),
    )
    if lora_hashes:
        metadata["lora_hashes"] = lora_hashes
    return {key: value for key, value in metadata.items() if value is not None}


def format_metadata(metadata: dict[str, Any]) -> str:
    prompt = _single_line(metadata.get("prompt", ""))
    loras = _single_line(metadata.get("loras", ""))
    positive = f"{prompt.rstrip(', ')}, {loras}" if prompt and loras else prompt or loras
    parts = [positive]

    negative_prompt = _single_line(metadata.get("negative_prompt", ""))
    if negative_prompt:
        parts.append(f"Negative prompt: {negative_prompt}")

    parameters = []
    if "steps" in metadata:
        parameters.append(f"Steps: {metadata['steps']}")

    sampler = metadata.get("sampler")
    if sampler:
        sampler = _SAMPLER_NAMES.get(sampler, sampler)
        scheduler = metadata.get("scheduler")
        if scheduler:
            sampler += f" {_SCHEDULER_NAMES.get(scheduler, scheduler)}"
        parameters.append(f"Sampler: {sampler}")

    cfg = metadata.get("guidance", metadata.get("cfg_scale", metadata.get("cfg")))
    if cfg is not None:
        parameters.append(f"CFG scale: {cfg}")
    if "seed" in metadata:
        parameters.append(f"Seed: {metadata['seed']}")
    if "size" in metadata:
        parameters.append(f"Size: {metadata['size']}")

    checkpoint = metadata.get("checkpoint")
    if checkpoint:
        model_name = os.path.splitext(os.path.basename(str(checkpoint)))[0]
        checkpoint_hash = metadata.get("checkpoint_hash")
        if checkpoint_hash:
            parameters.append(
                f"Model hash: {checkpoint_hash[:10]}, Model: {model_name}"
            )
        else:
            parameters.append(f"Model: {model_name}")

    lora_hashes = metadata.get("lora_hashes", {})
    if lora_hashes:
        hashes = ", ".join(
            f"{name}: {hash_value[:10]}"
            for name, hash_value in lora_hashes.items()
        )
        parameters.append(f'Lora hashes: "{hashes}"')
    anima_artist_chain = _single_line(metadata.get("anima_artist_chain", ""))
    if anima_artist_chain:
        parameters.append(
            f"Anima artist chain: {json.dumps(anima_artist_chain, ensure_ascii=False)}"
        )
    if parameters:
        parts.append(", ".join(parameters))
    return "\n".join(part for part in parts if part)


def format_filename(
    filename: str, metadata: dict[str, Any], now: datetime | None = None
) -> str:
    for segment in _PATTERN.findall(filename):
        key, _, option = segment[1:-1].partition(":")
        replacement = None

        if key == "seed" and "seed" in metadata:
            replacement = str(metadata["seed"])
        elif key in {"width", "height"} and "size" in metadata:
            size = str(metadata["size"]).split("x", 1)
            if len(size) == 2:
                replacement = size[key == "height"]
        elif key in {"pprompt", "nprompt"}:
            metadata_key = "prompt" if key == "pprompt" else "negative_prompt"
            replacement = _single_line(metadata.get(metadata_key, ""))
            if option.isdigit():
                replacement = replacement[: int(option)]
        elif key == "model":
            checkpoint = metadata.get("checkpoint")
            replacement = (
                os.path.splitext(os.path.basename(str(checkpoint)))[0]
                if checkpoint
                else "model_unavailable"
            )
            if option.isdigit():
                replacement = replacement[: int(option)]
        elif key == "date":
            current = now or datetime.now()
            replacement = option or "yyyyMMddhhmmss"
            for token, value in (
                ("yyyy", f"{current.year:04d}"),
                ("yy", f"{current.year % 100:02d}"),
                ("MM", f"{current.month:02d}"),
                ("dd", f"{current.day:02d}"),
                ("hh", f"{current.hour:02d}"),
                ("mm", f"{current.minute:02d}"),
                ("ss", f"{current.second:02d}"),
            ):
                replacement = replacement.replace(token, value)

        if replacement is not None:
            filename = filename.replace(segment, replacement)
    return filename


def _workflow(extra_pnginfo: dict | None) -> Any | None:
    return extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None


def _exif(metadata: str, workflow: Any = None) -> Image.Exif:
    exif = Image.Exif()
    if metadata:
        exif[0x9286] = b"UNICODE\0" + metadata.encode("utf-16be")
    if workflow is not None:
        exif[0x010E] = "Workflow:" + json.dumps(workflow)
    return exif


def _pil_from_tensor(image: Any) -> Image.Image:
    array = 255.0 * image.detach().cpu().numpy()
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


class SaveImageEfficient:
    CATEGORY = "Efficiency Nodes/utils"
    DESCRIPTION = "Save images with generation metadata and dynamic filenames"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process_image"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "Supports %seed%, %width%, %height%, %pprompt:N%, %nprompt:N%, %model:N%, and %date:FORMAT%.",
                    },
                ),
                "file_format": (["png", "jpeg", "webp"],),
            },
            "optional": {
                "lossless_webp": ("BOOLEAN", {"default": False}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100}),
                "embed_workflow": ("BOOLEAN", {"default": False}),
                "save_with_metadata": ("BOOLEAN", {"default": True}),
                "add_counter_to_filename": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def process_image(
        self,
        images,
        filename_prefix="ComfyUI",
        file_format="png",
        unique_id=None,
        prompt=None,
        extra_pnginfo=None,
        lossless_webp=False,
        quality=100,
        embed_workflow=False,
        save_with_metadata=True,
        add_counter_to_filename=True,
    ):
        original_images = images
        if hasattr(images, "shape") and len(images.shape) == 3:
            image_batch = [images]
        else:
            image_batch = list(images)
        if not image_batch:
            return {"result": (original_images,), "ui": {"images": []}}

        height, width = image_batch[0].shape[:2]
        metadata_dict = extract_metadata(prompt, unique_id, width, height)
        metadata = format_metadata(metadata_dict)
        filename_prefix = format_filename(filename_prefix, metadata_dict)
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, width, height
            )
        )
        os.makedirs(full_output_folder, exist_ok=True)

        results = []
        workflow = _workflow(extra_pnginfo) if embed_workflow else None
        for index, image in enumerate(image_batch):
            array = 255.0 * image.detach().cpu().numpy()
            pil_image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            base_filename = filename
            if add_counter_to_filename:
                base_filename += f"_{counter + index:05}_"

            if file_format == "png":
                output_name = base_filename + ".png"
                pnginfo = PngImagePlugin.PngInfo()
                if save_with_metadata and metadata:
                    pnginfo.add_text("parameters", metadata)
                if workflow is not None:
                    pnginfo.add_text("workflow", json.dumps(workflow))
                pil_image.save(
                    os.path.join(full_output_folder, output_name),
                    format="PNG",
                    pnginfo=pnginfo,
                    compress_level=self.compress_level,
                )
            elif file_format == "jpeg":
                output_name = base_filename + ".jpg"
                save_kwargs = {"quality": quality, "optimize": True}
                if save_with_metadata and metadata:
                    save_kwargs["exif"] = _exif(metadata)
                pil_image.convert("RGB").save(
                    os.path.join(full_output_folder, output_name),
                    format="JPEG",
                    **save_kwargs,
                )
            elif file_format == "webp":
                output_name = base_filename + ".webp"
                save_kwargs = {
                    "quality": quality,
                    "lossless": lossless_webp,
                    "method": 0,
                }
                exif = _exif(metadata if save_with_metadata else "", workflow)
                if len(exif):
                    save_kwargs["exif"] = exif
                pil_image.save(
                    os.path.join(full_output_folder, output_name),
                    format="WEBP",
                    **save_kwargs,
                )
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            results.append(
                {"filename": output_name, "subfolder": subfolder, "type": self.type}
            )

        return {"result": (original_images,), "ui": {"images": results}}


class SaveImageWithMetadata:
    CATEGORY = "Efficiency Nodes/utils"
    DESCRIPTION = "Save a single PNG with existing A1111 metadata and dynamic filename handling"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "Supports %seed%, %width%, %height%, %pprompt:N%, %nprompt:N%, %model:N%, and %date:FORMAT%.",
                    },
                ),
                "metadata": ("STRING", {"default": ""}),
            },
            "optional": {
                "upscale_model_name": (
                    ["(auto)", "None"] + folder_paths.get_filename_list("upscale_models"),
                    {
                        "tooltip": "Written as A1111 Hires upscaler. Auto reads an upstream upscale model loader when possible.",
                    },
                ),
                "upscale_by": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": "Written as A1111 Hires upscale when greater than 0. Otherwise inferred from metadata Size and output image size.",
                    },
                ),
                "add_counter_to_filename": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    def save(
        self,
        image,
        filename_prefix="ComfyUI",
        metadata="",
        upscale_model_name="(auto)",
        upscale_by=0.0,
        add_counter_to_filename=True,
        unique_id=None,
        prompt=None,
    ):
        original_image = image
        if isinstance(image, list):
            image = image[0]
        elif hasattr(image, "shape") and len(image.shape) == 4:
            image = image[0]

        height, width = image.shape[:2]
        metadata = append_hires_metadata(
            metadata,
            width,
            height,
            prompt=prompt,
            unique_id=unique_id,
            upscale_model_name=upscale_model_name,
            upscale_by=upscale_by,
        )

        filename_metadata = parse_metadata_text(metadata)
        filename_metadata["size"] = f"{width}x{height}"
        filename_prefix = format_filename(
            filename_prefix + self.prefix_append,
            filename_metadata,
        )
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, width, height
            )
        )
        os.makedirs(full_output_folder, exist_ok=True)

        pil_image = _pil_from_tensor(image)
        pnginfo = PngImagePlugin.PngInfo()
        if metadata:
            pnginfo.add_text("parameters", metadata)

        base_filename = filename
        if add_counter_to_filename:
            base_filename += f"_{counter:05}_"
        output_name = base_filename + ".png"
        pil_image.save(
            os.path.join(full_output_folder, output_name),
            format="PNG",
            pnginfo=pnginfo,
            compress_level=self.compress_level,
        )

        return {
            "result": (original_image,),
            "ui": {
                "images": [
                    {"filename": output_name, "subfolder": subfolder, "type": self.type}
                ]
            },
        }
