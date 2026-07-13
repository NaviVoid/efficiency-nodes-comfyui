from __future__ import annotations

from collections import OrderedDict
from typing import Any


_records: OrderedDict[str, OrderedDict[str, dict[str, Any]]] = OrderedDict()
_node_cache: dict[str, dict[str, Any]] = {}
_current_prompt_id: str | None = None
_installed = False


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _lora_stack(value: Any) -> list[tuple[str, Any, Any]]:
    if isinstance(value, dict) and "__value__" in value:
        value = value["__value__"]
    if not isinstance(value, (list, tuple)):
        return []

    stack = []
    for item in value:
        if isinstance(item, dict):
            if not item.get("active", True):
                continue
            name = item.get("name")
            strength = item.get("strength", 1.0)
            clip_strength = item.get("clipStrength", strength)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name = item[0]
            strength = item[1]
            clip_strength = item[2] if len(item) >= 3 else strength
        else:
            continue
        if isinstance(name, str) and name and name != "None":
            stack.append((name, strength, clip_strength))
    return stack


def _pipe_metadata(pipe: Any) -> dict[str, Any]:
    if not isinstance(pipe, dict):
        return {}
    settings = pipe.get("loader_settings")
    if not isinstance(settings, dict):
        return {}
    return {
        key: settings[key]
        for key in ("positive", "negative", "ckpt_name", "lora_stack")
        if key in settings
    }


def _extract_metadata(class_type: str, input_data_all: dict) -> dict[str, Any]:
    inputs = {key: _first(value) for key, value in input_data_all.items()}
    data = _pipe_metadata(inputs.get("pipe"))

    for key in ("positive", "negative", "ckpt_name", "base_ckpt_name", "unet_name"):
        value = inputs.get(key)
        if isinstance(value, (str, int, float)):
            data[key] = value

    stack = []
    for key in ("lora_stack", "optional_lora_stack", "loras"):
        stack.extend(_lora_stack(inputs.get(key)))
    stack.extend(_lora_stack(data.get("lora_stack")))

    lora_name = inputs.get("lora_name")
    if isinstance(lora_name, str) and lora_name != "None":
        strength = inputs.get(
            "strength_model", inputs.get("lora_model_strength", 1.0)
        )
        stack.append((lora_name, strength, inputs.get("lora_clip_strength", strength)))
    if stack:
        data["lora_stack"] = []
        for item in stack:
            if item not in data["lora_stack"]:
                data["lora_stack"].append(item)

    text = inputs.get("text")
    if "LoraStacker" in class_type and isinstance(text, str):
        data["lora_text"] = text
    return data


def record_runtime_inputs(
    prompt_id: Any,
    node_id: Any,
    class_type: str,
    input_data_all: dict,
) -> None:
    global _current_prompt_id

    prompt_key = str(prompt_id)
    if prompt_key != _current_prompt_id:
        _current_prompt_id = prompt_key
        _records.setdefault(prompt_key, OrderedDict())
        _records.move_to_end(prompt_key)
        while len(_records) > 3:
            _records.popitem(last=False)

    data = _extract_metadata(class_type, input_data_all)
    if not data:
        return
    record = {"class_type": class_type, **data}
    node_key = str(node_id)
    _records[prompt_key][node_key] = record
    _node_cache[node_key] = record


def get_runtime_metadata(node_ids: set[str] | None = None) -> dict[str, Any]:
    current = _records.get(_current_prompt_id or "", {})
    candidates = []
    seen = set()

    for node_id, record in current.items():
        if node_ids is None or node_id in node_ids:
            candidates.append(record)
            seen.add(node_id)
    if node_ids is not None:
        candidates.extend(
            _node_cache[node_id]
            for node_id in node_ids
            if node_id not in seen and node_id in _node_cache
        )

    result: dict[str, Any] = {}
    for record in candidates:
        if isinstance(record.get("positive"), str):
            result["prompt"] = record["positive"]
        if isinstance(record.get("negative"), str):
            result["negative_prompt"] = record["negative"]
        for key in ("ckpt_name", "base_ckpt_name", "unet_name"):
            if isinstance(record.get(key), str):
                result["checkpoint"] = record[key]
                result["checkpoint_folder"] = (
                    "unet" if key == "unet_name" else "checkpoints"
                )
        if record.get("lora_stack"):
            result["lora_stack"] = record["lora_stack"]
        if record.get("lora_text"):
            result["lora_text"] = record["lora_text"]
    return result


def install_runtime_metadata_hook() -> None:
    global _installed
    if _installed:
        return

    try:
        import execution
    except ImportError:
        return

    original = getattr(execution, "_async_map_node_over_list", None)
    if original is None:
        return
    if getattr(original, "_efficiency_metadata_hook", False):
        _installed = True
        return

    async def map_node_over_list_with_metadata(
        prompt_id,
        unique_id,
        obj,
        input_data_all,
        func,
        allow_interrupt=False,
        execution_block_cb=None,
        pre_execute_cb=None,
        v3_data=None,
    ):
        if func == obj.FUNCTION:
            record_runtime_inputs(
                prompt_id, unique_id, obj.__class__.__name__, input_data_all
            )
        return await original(
            prompt_id,
            unique_id,
            obj,
            input_data_all,
            func,
            allow_interrupt,
            execution_block_cb,
            pre_execute_cb,
            v3_data=v3_data,
        )

    map_node_over_list_with_metadata._efficiency_metadata_hook = True
    execution._async_map_node_over_list = map_node_over_list_with_metadata
    _installed = True


def _reset_for_tests() -> None:
    global _current_prompt_id
    _records.clear()
    _node_cache.clear()
    _current_prompt_id = None
