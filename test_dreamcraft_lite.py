"""Quick tests; run: python -m unittest -v test_dreamcraft_lite.py"""
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dreamcraft_lite import BlockScene, export_sponge_v2, render_voxels, toy_castle


class BasicTest(unittest.TestCase):
    def test_shapes_and_differentiable_renderer(self):
        scene = BlockScene(10, torch.device('cpu'))
        occ, col, materials = scene.field()
        self.assertEqual(tuple(occ.shape), (10,10,10))
        self.assertEqual(tuple(col.shape), (10,10,10,3))
        self.assertEqual(tuple(materials.shape[:3]), (10,10,10))
        img = render_voxels(occ, col, resolution=12, samples=16)
        self.assertEqual(tuple(img.shape), (3,12,12))
        self.assertTrue(torch.isfinite(img).all())
        img.mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in scene.parameters()))

    def test_export_nbt(self):
        vox = np.zeros((10,10,10), dtype=np.int16)
        vox[1,2,3] = 1
        vox[2,3,4] = 8
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.schem'
            export_sponge_v2(vox, path)
            raw = gzip.decompress(path.read_bytes())
            self.assertEqual(raw[0], 10) # root TAG_Compound
            self.assertIn(b'Schematic', raw)
            self.assertIn(b'BlockData', raw)
            self.assertIn(b'minecraft:stone_bricks', raw)
            self.assertGreater(path.stat().st_size, 100)

    def test_toy(self):
        vox = toy_castle(12, torch.device('cpu'))
        self.assertGreater((vox > 0).sum().item(), 1)
        self.assertGreater((vox == 0).sum().item(), 1)


if __name__ == '__main__':
    unittest.main()
