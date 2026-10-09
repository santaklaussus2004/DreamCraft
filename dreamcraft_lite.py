"""DreamCraft-inspired Minecraft text-to-voxel generation.

Real mode: SDS with Stable Diffusion 1.5 -> differentiable voxel renderer ->
coordinate MLPs -> Sponge v2 .schem file.

--smoke-test tests the local pipeline WITHOUT downloading a diffusion model.
It fits a procedural castle target rather than performing text-to-3D.

This is an independent, simplified research implementation, NOT the official
DreamCraft implementation or a faithful reproduction of its reported results.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Names are valid for modern Minecraft Java and WorldEdit/Sponge v2 schematics.
PALETTE = {
    "minecraft:stone_bricks": (0.53, 0.53, 0.53),
    "minecraft:cobblestone": (0.47, 0.47, 0.47),
    "minecraft:oak_planks": (0.65, 0.49, 0.29),
    "minecraft:spruce_planks": (0.39, 0.27, 0.15),
    "minecraft:dark_oak_planks": (0.25, 0.18, 0.11),
    "minecraft:bricks": (0.60, 0.34, 0.28),
    "minecraft:white_concrete": (0.82, 0.83, 0.82),
    "minecraft:deepslate_tiles": (0.20, 0.20, 0.23),
    "minecraft:gold_block": (0.91, 0.73, 0.15),
    "minecraft:mossy_stone_bricks": (0.44, 0.49, 0.40),
    "minecraft:red_wool": (0.64, 0.18, 0.17),
}
BLOCKS = list(PALETTE)


def coordinate_grid(size: int, device: torch.device) -> torch.Tensor:
    """Flattened grid coordinates in (x,y,z) order, storage (z,y,x)."""
    z, y, x = torch.meshgrid(
        *[torch.linspace(-1 + 1 / size, 1 - 1 / size, size, device=device)] * 3,
        indexing="ij",
    )
    return torch.stack((x, y, z), dim=-1).reshape(-1, 3)


class FourierEncoding(nn.Module):
    def __init__(self, frequencies: int = 5):
        super().__init__()
        self.register_buffer("freq", 2 ** torch.arange(frequencies).float() * math.pi)
        self.out_features = 3 + 3 * frequencies * 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        phases = coords[..., None] * self.freq
        return torch.cat([coords, phases.sin().flatten(-2), phases.cos().flatten(-2)], dim=-1)


def small_mlp(inp: int, out: int, hidden: int = 96) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, out))


class BlockScene(nn.Module):
    def __init__(self, size: int, device: torch.device):
        super().__init__()
        self.size = size
        self.encoder = FourierEncoding()
        self.material_net = small_mlp(self.encoder.out_features, len(BLOCKS))
        self.occupancy_net = small_mlp(self.encoder.out_features, 1)
        self.register_buffer("coords", coordinate_grid(size, device))
        self.register_buffer("colors", torch.tensor(list(PALETTE.values()), dtype=torch.float32))
        # Initially sparse, but not completely empty.
        nn.init.constant_(self.occupancy_net[-1].bias, -0.9)
        self.to(device)

    def predict(self) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(self.coords)
        n = self.size
        occ_logits = self.occupancy_net(features).reshape(n, n, n)
        mat_logits = self.material_net(features).reshape(n, n, n, len(BLOCKS))
        return occ_logits, mat_logits

    def field(self, tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        occ_logits, mat_logits = self.predict()
        occupancy = torch.sigmoid(occ_logits)   # Soft air, important for trainability.
        # Hard in forward pass, continuous surrogate gradient in backward pass.
        materials = F.gumbel_softmax(mat_logits, tau=tau, hard=True, dim=-1)
        colors = torch.einsum("zyxm,mc->zyxc", materials, self.colors)
        return occupancy, colors, materials

    @torch.no_grad()
    def discrete(self) -> np.ndarray:
        occ, mats = self.predict()
        ids = mats.argmax(dim=-1).cpu().numpy().astype(np.int16) + 1
        ids[occ.sigmoid().cpu().numpy() < 0.5] = 0
        return ids  # (z,y,x), 0=air, 1..M = material index +1.


def render_voxels(
    occupancy: torch.Tensor,
    colors: torch.Tensor,
    angle: float = 35,
    elevation: float = 25,
    resolution: int = 64,
    samples: int = 64,
) -> torch.Tensor:
    """Differentiable orthographic volume renderer, returns RGB (3,H,W).

    Voxel occupancy and block colors are sampled with nearest-neighbor
    sampling; gradients flow through the sampled field into both MLPs.
    """
    device = occupancy.device
    dtype = occupancy.dtype
    az = math.radians(angle)
    el = math.radians(elevation)
    eye = torch.tensor([math.sin(az) * math.cos(el), math.sin(el),
                        math.cos(az) * math.cos(el)], device=device, dtype=dtype)
    forward = -eye
    up_world = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
    right = F.normalize(torch.cross(forward, up_world, dim=0), dim=0)
    up = F.normalize(torch.cross(right, forward, dim=0), dim=0)
    uv = torch.linspace(-1.3, 1.3, resolution, device=device, dtype=dtype)
    v, u = torch.meshgrid(uv.flip(0), uv, indexing="ij")
    # Orthographic camera at distance 2.5, looking at the origin.
    origins = eye * 2.5 + u[..., None] * right + v[..., None] * up
    distances = torch.linspace(0., 5., samples, device=device, dtype=dtype)
    positions = origins[None] + distances[:, None, None, None] * forward
    # F.grid_sample expects (x,y,z) as the final dimension, storage (z,y,x).
    sampling_grid = positions[None]  # (1, S, H, W, 3)
    volume = torch.cat((occupancy[..., None], colors), dim=-1)
    volume = volume.permute(3, 0, 1, 2)[None].contiguous()
    sampled = F.grid_sample(volume, sampling_grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)[0]
    density = sampled[0].clamp(0, 1)  # S,H,W
    rgb = sampled[1:].permute(1, 2, 3, 0).clamp(0, 1)  # S,H,W,3
    step_length = float(distances[1] - distances[0])
    # A filled block is approximately opaque after one voxel's thickness.
    strength = occupancy.shape[0] * 1.8
    alpha = 1.0 - torch.exp(-density * strength * step_length)
    transmittance = torch.cumprod(
        torch.cat((torch.ones_like(alpha[:1]), 1 - alpha + 1e-7), dim=0), dim=0
    )[:-1]
    weights = alpha * transmittance
    sky = torch.tensor([0.70, 0.82, 0.96], device=device, dtype=dtype)
    output = (weights[..., None] * rgb).sum(dim=0)
    output = output + transmittance[-1, ..., None] * (1 - alpha[-1, ..., None]) * sky
    return output.clamp(0, 1).permute(2, 0, 1)


def save_preview(rgb: torch.Tensor, path: Path) -> None:
    arr = rgb.detach().permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


# Minimal NBT writer; avoids third-party dependencies for .schem exports.
def _named_tag(tag_type: int, name: str, payload: bytes) -> bytes:
    raw = name.encode("utf-8")
    return bytes((tag_type,)) + struct.pack(">H", len(raw)) + raw + payload


def _integer(name: str, value: int, short: bool = False) -> bytes:
    return _named_tag(2 if short else 3, name,
                      struct.pack(">h" if short else ">i", value))


def _varint(number: int) -> bytes:
    result = bytearray()
    while True:
        b = number & 0x7F
        number >>= 7
        if number:
            result.append(b | 0x80)
        else:
            result.append(b)
            return bytes(result)


def export_sponge_v2(blocks: np.ndarray, path: Path) -> None:
    """Write Minecraft Java Sponge Schematic v2 (WorldEdit-compatible)."""
    if blocks.ndim != 3:
        raise ValueError("Expected (z,y,x) voxel array")
    depth, height, width = blocks.shape
    if min(width, height, depth) <= 0 or max(width, height, depth) > 32767:
        raise ValueError("Invalid schematic dimensions")
    ids = ["minecraft:air"] + BLOCKS
    palette_tags = b"".join(_integer(name, idx) for idx, name in enumerate(ids))
    # Sponge block order: X fastest, then Z, then Y.
    flattened = blocks.transpose(1, 0, 2).reshape(-1)
    encoded = b"".join(_varint(int(i)) for i in flattened)
    offset = struct.pack(">i", 3) + struct.pack(">iii", 0, 0, 0)
    root = b"".join((
        _integer("Version", 2),
        _integer("DataVersion", 3465),  # Minecraft Java 1.20.1
        _integer("Width", width, short=True),
        _integer("Height", height, short=True),
        _integer("Length", depth, short=True),
        _named_tag(11, "Offset", offset),
        _integer("PaletteMax", len(ids)),
        _named_tag(10, "Palette", palette_tags + b"\x00"),
        _named_tag(7, "BlockData", struct.pack(">i", len(encoded)) + encoded),
        _named_tag(10, "Metadata", b"\x00"),
    ))
    with gzip.open(path, "wb") as f:
        f.write(_named_tag(10, "Schematic", root + b"\x00"))


class DiffusionSDS:
    """SD 1.5 noise-prediction SDS; UNet is frozen, gradients reach 3D scene."""
    def __init__(self, prompt: str, device: torch.device,
                 model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 guidance: float = 50.0):
        if device.type != "cuda":
            raise RuntimeError("SDS generation requires CUDA. Use --smoke-test for CPU.")
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements: pip install -r requirements.txt") from exc
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        self.pipe.unet.requires_grad_(False).eval()
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.text_encoder.requires_grad_(False).eval()
        with torch.no_grad():
            positive, negative = self.pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="blurry, photorealistic, text, watermark",
            )
        self.embeds = torch.cat((negative, positive), dim=0)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
        self.guidance = guidance
        self.device = device
        self.scheduler = self.pipe.scheduler

    def loss(self, rgb: torch.Tensor) -> torch.Tensor:
        # Keep VAE gradients: VAE weights frozen, graph connects latents to RGB.
        img = F.interpolate(rgb[None], size=(512, 512), mode="bilinear",
                            align_corners=False)
        img = img.mul(2).sub(1).to(self.pipe.vae.dtype)
        latents = self.pipe.vae.encode(img).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor
        total_t = self.scheduler.config.num_train_timesteps
        t = torch.randint(int(0.15 * total_t), int(0.85 * total_t),
                          (1,), device=self.device)
        noise = torch.randn_like(latents)
        noisy = self.scheduler.add_noise(latents.detach(), noise, t)
        with torch.no_grad():
            model_input = torch.cat((noisy, noisy))
            time_input = torch.cat((t, t))
            output = self.pipe.unet(model_input, time_input,
                                    encoder_hidden_states=self.embeds).sample
            uncond, cond = output.chunk(2)
            guided = uncond + self.guidance * (cond - uncond)
            weight = 1 - self.alphas[t].view(1, 1, 1, 1)
            grad = weight * (guided.float() - noise.float())
            grad = torch.nan_to_num(grad).clamp(-1, 1)
        # Gradient of this loss w.r.t. latents equals the SDS estimate.
        target = (latents.float() - grad).detach()
        return 0.5 * F.mse_loss(latents.float(), target, reduction="sum")


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return ((x[1:] - x[:-1]).abs().mean() +
            (x[:, 1:] - x[:, :-1]).abs().mean() +
            (x[:, :, 1:] - x[:, :, :-1]).abs().mean()) / 3


def toy_castle(size: int, device: torch.device) -> torch.Tensor:
    """Synthetic 3D target used ONLY to verify the offline smoke-test."""
    if size < 10:
        raise ValueError("--smoke-test requires --size >= 10")
    b = torch.zeros((size, size, size), dtype=torch.long, device=device)
    low, high = size // 4, size - size // 4
    ground = size // 7
    wall_top = size * 3 // 5
    b[low:high, ground:wall_top, low:high] = 1  # outer masonry body
    b[low + 1:high - 1, ground + 2:wall_top, low + 1:high - 1] = 0  # hollow
    turret_r = max(2, size // 7)
    for zc in (low, high - 1):
        for xc in (low, high - 1):
            za, zb = max(0, zc - turret_r // 2), min(size, zc + turret_r)
            xa, xb = max(0, xc - turret_r // 2), min(size, xc + turret_r)
            b[za:zb, ground:size - 1, xa:xb] = 2
            b[za + 1:zb - 1, ground + 2:size - 1, xa + 1:xb - 1] = 0
    # Door in the front wall (z=low)
    b[low, ground:ground + max(2, size // 5), size // 2 - 1:size // 2 + 1] = 0
    return b


def smoke_train(scene: BlockScene, args: argparse.Namespace) -> list[float]:
    """Trains against a KNOWN target, not a text prompt; tests MLP training."""
    target = toy_castle(scene.size, scene.coords.device)
    optimizer = torch.optim.Adam(scene.parameters(), lr=0.007)
    losses = []
    for i in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        occupancy_logits, block_logits = scene.predict()
        loss_occ = F.binary_cross_entropy_with_logits(
            occupancy_logits, (target > 0).float())
        if (target > 0).any():
            loss_mat = F.cross_entropy(block_logits[target > 0], target[target > 0] - 1)
        else:
            loss_mat = loss_occ * 0
        loss = loss_occ + 0.3 * loss_mat
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i == 0 or (i + 1) % 20 == 0 or i == args.steps - 1:
            print(f"[SMOKE] step={i + 1}/{args.steps}, supervised_loss={loss.item():.4f}")
    return losses


def text_train(scene: BlockScene, args: argparse.Namespace) -> list[float]:
    device = scene.coords.device
    prompt = f"A Minecraft block-built structure: {args.prompt}, voxel art, centered, full view"
    print("Loading Stable Diffusion model (first run downloads weights)...")
    teacher = DiffusionSDS(prompt, device, args.sd_model, args.guidance)
    optimizer = torch.optim.Adam(scene.parameters(), lr=args.lr)
    losses = []
    for i in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        # Anneal Gumbel temperature while keeping soft air for training.
        tau = max(0.55, 1.5 - i / max(1, args.steps) * 0.95)
        occ, colors, materials = scene.field(tau=tau)
        rgb = render_voxels(occ, colors, angle=random.uniform(0, 360),
                            elevation=random.uniform(10, 40),
                            resolution=args.render_size, samples=args.samples)
        sds = teacher.loss(rgb)
        regularizer = args.sparsity * occ.mean() + args.smoothness * tv_loss(occ)
        if args.target_material:
            material_id = BLOCKS.index(args.target_material)
            proportion = (materials[..., material_id] * occ).sum() / occ.sum().clamp_min(1)
            regularizer = regularizer + args.material_weight * (proportion - args.target_ratio).square()
        loss = args.sds_weight * sds + regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scene.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if i == 0 or (i + 1) % 25 == 0 or i == args.steps - 1:
            print(f"step={i + 1}/{args.steps} loss={loss.item():.4f} "
                  f"sds={sds.item():.4f} occupied={occ.mean().item():.3f}")
        if (i + 1) % args.save_every == 0:
            with torch.no_grad():
                occ, colors, _ = scene.field(tau=0.6)
                snap = render_voxels(occ, colors, resolution=args.render_size,
                                     samples=args.samples)
                save_preview(snap, args.output / f"step_{i + 1:05d}.png")
                torch.save(scene.state_dict(), args.output / "checkpoint.pt")
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="a small medieval stone castle with four towers")
    parser.add_argument("--smoke-test", action="store_true", help="Offline supervised architecture test; NO text guidance")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--size", type=int, default=24, help="Voxel side length")
    parser.add_argument("--render-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--guidance", type=float, default=50)
    parser.add_argument("--sds-weight", type=float, default=1.0)
    parser.add_argument("--sparsity", type=float, default=0.03)
    parser.add_argument("--smoothness", type=float, default=0.02)
    parser.add_argument("--sd-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--target-material", default="", choices=[""] + BLOCKS)
    parser.add_argument("--target-ratio", type=float, default=0.6)
    parser.add_argument("--material-weight", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume scene weights from model.pt or checkpoint.pt")
    args = parser.parse_args()
    if args.steps < 1 or args.size < 10 or args.samples < 2 or args.render_size < 8 or args.save_every < 1:
        parser.error("Use steps>=1, size>=10, samples>=2, render-size>=8, save-every>=1")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke_test and device.type != "cuda":
        parser.error("Real text-to-3D SDS requires an NVIDIA CUDA GPU; try --smoke-test on CPU")
    print(f"device={device}, mode={'smoke-test (NO text guidance)' if args.smoke_test else 'text-to-3D SDS'}")
    scene = BlockScene(args.size, device)
    if args.resume:
        scene.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Loaded weights from {args.resume} (optimizer state is reset).")
    losses = smoke_train(scene, args) if args.smoke_test else text_train(scene, args)
    with torch.no_grad():
        occ, colors, _ = scene.field(tau=0.5)
        rgb = render_voxels(occ, colors, resolution=max(64, args.render_size),
                            samples=args.samples)
        save_preview(rgb, args.output / "preview.png")
    blocks = scene.discrete()
    export_sponge_v2(blocks, args.output / "generated.schem")
    torch.save(scene.state_dict(), args.output / "model.pt")
    meta = {"prompt": args.prompt, "smoke_test": args.smoke_test,
            "size": args.size, "steps": args.steps, "non_air_blocks": int(np.count_nonzero(blocks)),
            "palette": ["minecraft:air"] + BLOCKS,
            "first_loss": losses[0], "last_loss": losses[-1]}
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output / 'generated.schem'}, preview.png, model.pt, metadata.json")
    if args.smoke_test:
        print("CAUTION: This was an OFFLINE SUPERVISED test of a pre-programmed target castle.")
        print("Run WITHOUT --smoke-test on CUDA for actual text-conditioned generation.")


if __name__ == "__main__":
    main()
