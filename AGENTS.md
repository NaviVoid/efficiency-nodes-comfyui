# Repository Guidelines

## Project Structure & Module Organization

- `efficiency_nodes.py` contains the main ComfyUI node implementations and `NODE_CLASS_MAPPINGS` registration table.
- `py/` holds focused helpers, sampler integrations, runtime metadata collection, image saving, and required latent-upscaler model assets.
- `js/` contains browser-side ComfyUI extensions and node menu behavior.
- `tests/` contains executable Python `unittest` coverage. Add tests beside `tests/test_save_image.py` or create another `test_*.py` file.
- `images/` and `workflows/` are documentation and example assets; they are not runtime dependencies.
- `node_settings.json` defines model-cache defaults and must remain valid JSON.

## Build, Test, and Development Commands

Install dependencies using the same Python environment that runs ComfyUI:

```bash
python -m pip install -r requirements.txt
```

Run the current test suite:

```bash
python tests/test_save_image.py -v
```

The local ROCm container can run the same check:

```bash
podman exec -w /home/navi/ComfyUI/custom_nodes/efficiency-nodes-comfyui \
  Rocm7.2 /opt/venv/bin/python tests/test_save_image.py -v
```

Use `python -m py_compile path/to/file.py` for a quick syntax check. Restart ComfyUI after Python or JavaScript changes so node mappings and runtime hooks reload.

## Coding Style & Naming Conventions

Use four-space indentation for Python and follow the surrounding module style. Prefer `snake_case` for functions and variables, `PascalCase` for classes, and descriptive ComfyUI display names such as `Save Image (Efficient)`. Register new nodes once in `NODE_CLASS_MAPPINGS`; avoid parallel registries or speculative abstractions. Keep comments short and only for non-obvious behavior. No formatter is enforced, so keep diffs focused and run `git diff --check` before submission.

## Testing Guidelines

Tests use the standard-library `unittest` framework with temporary directories and mocks for ComfyUI paths. Every parser, metadata branch, or filename rule should have a focused regression test. Tests must not require GPU execution, network access, or real model files. There is no numeric coverage threshold; changed behavior must be exercised directly.

## Commit & Pull Request Guidelines

History uses short, imperative subjects, for example `add Save Image (Efficient)` or `fix list unwrap`. Keep commits scoped to one behavior. Pull requests should explain the user-visible change, list tests run, and note affected ComfyUI workflows. Include a screenshot or sample workflow for UI-facing node changes. Do not commit generated images, output files, caches, credentials, or local settings.
