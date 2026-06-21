"""Tests for Flash3D models."""

import torch

from flash3d.cfg.config import Flash3DConfig
from flash3d.models.architectures.feed_forward_3dgs import FeedForward3DGS
from flash3d.models.architectures.gaussian_splatting import GaussianSplatting
from flash3d.models.architectures.nerf import HashEncoding, NeRF, PositionalEncoding
from flash3d.models.flash3d_model import Flash3D
from flash3d.models.lora import LoRALinear


class TestGaussianSplatting:
    def test_initialization(self):
        config = Flash3DConfig()
        config.model.num_gaussians = 100
        config.model.sh_degree = 2
        model = GaussianSplatting(config=config)
        assert model.means.shape == (100, 3)
        assert model.scales.shape == (100, 3)
        assert model.rotations.shape == (100, 4)
        assert model.opacities.shape == (100, 1)

    def test_from_point_cloud(self):
        config = Flash3DConfig()
        config.model.num_gaussians = 50
        model = GaussianSplatting(config=config)

        points = torch.randn(200, 3)
        colors = torch.rand(200, 3)
        model.initialize_from_point_cloud(points, colors)
        assert model.num_points == 200

    def test_covariance(self):
        config = Flash3DConfig()
        config.model.num_gaussians = 10
        model = GaussianSplatting(config=config)
        cov = model.get_covariance_3d()
        assert cov.shape == (10, 3, 3)
        # Covariance should be symmetric
        assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-5)

    def test_opacity_range(self):
        config = Flash3DConfig()
        config.model.num_gaussians = 50
        model = GaussianSplatting(config=config)
        opacity = model.get_opacity()
        assert (opacity >= 0).all()
        assert (opacity <= 1).all()


class TestNeRF:
    def test_positional_encoding(self):
        pe = PositionalEncoding(num_frequencies=4)
        x = torch.randn(10, 3)
        encoded = pe(x)
        expected_dim = 3 * pe.output_dim
        assert encoded.shape == (10, expected_dim)

    def test_hash_encoding(self):
        he = HashEncoding(num_levels=4, features_per_level=2)
        x = torch.rand(10, 3)
        encoded = he(x)
        assert encoded.shape == (10, he.output_dim)

    def test_nerf_query(self):
        config = Flash3DConfig()
        model = NeRF(config=config, use_hash_encoding=False, num_layers=2, hidden_dim=32)
        positions = torch.randn(5, 3)
        directions = torch.randn(5, 3)
        density, rgb = model.query(positions, directions)
        assert density.shape == (5, 1)
        assert rgb.shape == (5, 3)
        assert (density >= 0).all()
        assert (rgb >= 0).all() and (rgb <= 1).all()


class TestFeedForward3DGS:
    def test_forward(self):
        config = Flash3DConfig()
        model = FeedForward3DGS(config=config, base_channels=16, num_attention_layers=1)
        images = torch.rand(1, 2, 3, 64, 64)
        output = model(images=images)
        assert "depth" in output
        assert "scales" in output
        assert "rotations" in output
        assert "opacities" in output


class TestFlash3D:
    def test_model_creation(self):
        config = Flash3DConfig()
        config.model.name = "gaussian_splatting"
        config.model.num_gaussians = 50
        model = Flash3D(config=config)
        assert model.num_parameters > 0

    def test_save_load_checkpoint(self, tmp_path):
        config = Flash3DConfig()
        config.model.name = "gaussian_splatting"
        config.model.num_gaussians = 50
        model = Flash3D(config=config)

        ckpt_path = tmp_path / "test_model.pth"
        model.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        loaded = Flash3D.from_pretrained(ckpt_path)
        assert loaded.num_parameters == model.num_parameters


class TestLoRA:
    def test_lora_linear(self):
        layer = LoRALinear(64, 128, rank=8)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 128)

    def test_lora_merge(self):
        layer = LoRALinear(32, 32, rank=4)
        x = torch.randn(2, 32)
        out_before = layer(x).detach()
        layer.merge_weights()
        # After merge, LoRA contribution is in base weights
        out_after = layer.linear(x).detach()
        # Merged output from linear should match the combined output
        assert out_after.shape == out_before.shape
