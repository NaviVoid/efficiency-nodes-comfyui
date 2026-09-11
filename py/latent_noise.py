"""Deterministic latent transforms used by the latent noise script."""

import torch
import torch.nn.functional as F


GEOMETRY_TRANSFORMS = (
    "none",
    "horizontal_flip",
    "vertical_flip",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "transpose",
    "random",
)
NOISE_TYPES = ("gaussian", "uniform", "perlin", "salt_pepper")


def _generator(seed):
    return torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)


def _resize_spatial(tensor, height, width):
    if tensor.shape[-2:] == (height, width):
        return tensor
    if tensor.ndim == 4:
        return F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    if tensor.ndim == 5:
        batch, channels, frames = tensor.shape[:3]
        flattened = tensor.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, tensor.shape[-2], tensor.shape[-1]
        )
        flattened = F.interpolate(
            flattened, size=(height, width), mode="bilinear", align_corners=False
        )
        return flattened.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    raise ValueError(f"Expected a 4D or 5D latent tensor, got shape {tuple(tensor.shape)}")


def _apply_geometry(tensor, transform, generator):
    if transform == "random":
        transform = GEOMETRY_TRANSFORMS[1 + int(torch.randint(0, 7, (), generator=generator))]

    height, width = tensor.shape[-2:]
    if transform == "horizontal_flip":
        return tensor.flip(-1)
    if transform == "vertical_flip":
        return tensor.flip(-2)
    if transform == "rotate_180":
        return tensor.rot90(2, (-2, -1))
    if transform in ("rotate_90", "rotate_270"):
        turns = 1 if transform == "rotate_90" else 3
        return _resize_spatial(tensor.rot90(turns, (-2, -1)), height, width)
    if transform == "transpose":
        return _resize_spatial(tensor.transpose(-2, -1), height, width)
    return tensor


def _make_noise(shape, noise_type, generator):
    if noise_type == "uniform":
        noise = torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
    elif noise_type == "perlin":
        low_height = max(1, (shape[-2] + 7) // 8)
        low_width = max(1, (shape[-1] + 7) // 8)
        low = torch.randn(
            (*shape[:-2], low_height, low_width),
            generator=generator,
            dtype=torch.float32,
        )
        noise = _resize_spatial(low, shape[-2], shape[-1])
    elif noise_type == "salt_pepper":
        values = torch.rand(shape, generator=generator, dtype=torch.float32)
        noise = torch.where(values < 0.5, -torch.ones_like(values), torch.ones_like(values))
    else:
        noise = torch.randn(shape, generator=generator, dtype=torch.float32)

    # Keep different distributions at a comparable strength.
    std = noise.std(unbiased=False)
    if float(std) > 1e-6:
        noise = noise / std
    return noise


def transform_samples(
    samples,
    seed,
    injection_ratio,
    geometry_transform="none",
    noise_type="gaussian",
    channel_shuffle=False,
):
    """Apply one deterministic transform to a BCHW or BCTHW latent tensor."""
    if samples.ndim not in (4, 5):
        raise ValueError(
            f"Expected a BCHW or BCTHW latent tensor, got shape {tuple(samples.shape)}"
        )

    ratio = min(max(float(injection_ratio), 0.0), 1.0)
    generator = _generator(seed)
    working = samples.clone()
    working = _apply_geometry(working, geometry_transform, generator)

    if channel_shuffle and working.shape[1] > 1:
        permutation = torch.randperm(working.shape[1], generator=generator)
        working = working[:, permutation.to(working.device)]

    if ratio == 0.0:
        return working

    noise = _make_noise(tuple(working.shape), noise_type, generator)
    noise = noise.to(device=working.device, dtype=working.dtype)
    return working * (1.0 - ratio) + noise * ratio


def transform_latent(latent, seed, injection_ratio, geometry_transform, noise_type, channel_shuffle):
    """Return a latent copy while preserving ComfyUI metadata fields."""
    output = latent.copy()
    output["samples"] = transform_samples(
        latent["samples"],
        seed,
        injection_ratio,
        geometry_transform,
        noise_type,
        channel_shuffle,
    )
    return output
