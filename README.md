# DreamCraft Lite — research-inspired text-to-Minecraft

Independent, simplified implementation inspired by **DreamCraft** (Earle et al., 2024): https://arxiv.org/abs/2404.15538

**Not** the official DreamCraft code and **not** a verified reproduction of the paper's image quality.

## Architecture

Text prompt -> frozen Stable Diffusion 1.5 teacher (SDS) -> differentiable orthographic 3D volume renderer -> coordinate-encoded MLPs -> soft air/straight-through Gumbel material predictions -> Sponge v2 `.schem`.

Differences from the article:
- Stable Diffusion 1.5 instead of preliminary Emu.
- Simplified fixed block palette and solid colors instead of native Minecraft 16x16 voxelized textures.
- Compact differentiable orthographic renderer instead of the authors' renderer.
- Simple sparsity/TV loss (plus optional block proportion) instead of the complete functional constraint suite.
- No guarantee of good Minecraft objects without experimentation; SDS often produces artifacts.

## Installation

Python 3.10+; Windows/Linux. NVIDIA CUDA card with **~12 GB VRAM or more recommended** for SDS training. VRAM usage has **not been benchmarked**. First use downloads the pretrained Stable Diffusion weights from Hugging Face (model access/network required).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
# Install a CUDA-compatible build of PyTorch first if using CUDA.
pip install -r requirements.txt
```

### 1. Offline smoke-test (runs on CPU, NOT text generation)

```bash
python dreamcraft_lite.py --smoke-test --size 12 --steps 40 --render-size 32 --samples 24 --output test_output
```

This tests coordinate networks, backprop, rendering, PNG preview, and .schem output by fitting a **hard-coded castle target**.

### 2. Actual text-guided SDS generation (requires CUDA)

```bash
python dreamcraft_lite.py --prompt "a small medieval stone castle with four towers" --size 24 --steps 500 --output castle
```

Optional: force approximately 60% stone bricks among non-air voxels:

```bash
python dreamcraft_lite.py --prompt "a stone castle" --size 24 --steps 500 --target-material minecraft:stone_bricks --target-ratio 0.6 --output castle
```

Check `castle/preview.png`, `castle/metadata.json`, and `castle/generated.schem`. Load `.schem` with **WorldEdit for Minecraft Java**, e.g. put it in the WorldEdit schematics directory, then `/schem load generated` and `//paste` (syntax may vary with your version).

**Important:** A completed train run does not guarantee a recognizable castle. The image teacher was trained mostly on natural images, and learning attractive 3D discrete voxel structures from SDS is challenging. Adjust prompts, timesteps, resolution, learning rates and palette. For stronger fidelity to the paper, replace simplified colors with textured block voxel grids and add distribution/adjacency losses.

## Files

- `dreamcraft_lite.py`: runnable training/export implementation.
- `requirements.txt`: optional diffusion dependencies are only imported for text-to-3D mode.
- `test_dreamcraft_lite.py`: quick structural tests.

## Resume a model

```bash
python dreamcraft_lite.py --prompt "a stone castle" --size 24 --resume castle/checkpoint.pt --steps 300 --output castle_continued
```

Model weights are restored; optimizer moments and iteration count are not restored.
