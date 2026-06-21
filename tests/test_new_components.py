"""Tests for new Flash3D P0 components."""

import numpy as np
import torch

from flash3d.cfg.config import Flash3DConfig


class TestHashEncoding:
    def test_multi_resolution_hash_encoding_shape(self):
        from flash3d.models.encodings.hash_encoding import MultiResolutionHashEncoding

        enc = MultiResolutionHashEncoding(num_levels=4, features_per_level=2)
        x = torch.rand(100, 3)
        out = enc(x)
        assert out.shape == (100, 4 * 2)

    def test_hash_encoding_batch(self):
        from flash3d.models.encodings.hash_encoding import MultiResolutionHashEncoding

        enc = MultiResolutionHashEncoding(num_levels=4, features_per_level=2)
        x = torch.rand(2, 50, 3)
        out = enc(x)
        assert out.shape == (2, 50, 8)

    def test_instant_ngp_encoding(self):
        from flash3d.models.encodings.hash_encoding import InstantNGPHashEncoding

        enc = InstantNGPHashEncoding(
            num_levels=4, features_per_level=2, hidden_dim=32, num_layers=2
        )
        pos = torch.rand(10, 3)
        dirs = torch.randn(10, 3)
        result = enc(pos, dirs)
        assert "density" in result
        assert "rgb" in result
        assert result["density"].shape == (10, 1)
        assert result["rgb"].shape == (10, 3)
        assert (result["density"] >= 0).all()
        assert (result["rgb"] >= 0).all() and (result["rgb"] <= 1).all()

    def test_instant_ngp_registered(self):
        from flash3d.registry import MODELS

        assert "instant_ngp_encoding" in MODELS


class TestCOLMAPLoader:
    def test_qvec_to_rotation(self):
        from flash3d.data.colmap_loader import qvec_to_rotation_matrix

        R = qvec_to_rotation_matrix(np.array([1, 0, 0, 0]))
        assert np.allclose(R, np.eye(3), atol=1e-6)

    def test_camera_params_pinhole(self):
        from flash3d.data.colmap_loader import camera_params_to_intrinsics

        params = np.array([500.0, 500.0, 320.0, 240.0])
        K = camera_params_to_intrinsics("PINHOLE", params, 640, 480)
        assert K[0, 0] == 500.0
        assert K[1, 1] == 500.0
        assert K[0, 2] == 320.0
        assert K[1, 2] == 240.0


class TestDepthAnythingV2:
    def test_fallback_creation(self):
        from flash3d.geometry.depth_anything import DepthAnythingV2

        estimator = DepthAnythingV2(variant="small", device="cpu")
        assert estimator._model is not None

    def test_forward_shape(self):
        from flash3d.geometry.depth_anything import DepthAnythingV2

        estimator = DepthAnythingV2(variant="small", device="cpu")
        img = torch.rand(1, 3, 64, 64)
        depth = estimator(img)
        assert depth.shape[0] == 1
        assert depth.shape[1] == 1


class TestMipNeRF360:
    def test_mip_nerf_registered(self):
        from flash3d.registry import MODELS

        assert "mip_nerf_360" in MODELS

    def test_ipe_shape(self):
        from flash3d.models.architectures.mip_nerf import IntegratedPositionalEncoding

        ipe = IntegratedPositionalEncoding(num_frequencies=4)
        means = torch.randn(10, 3)
        covs = torch.rand(10, 3).abs()
        out = ipe(means, covs)
        assert out.shape == (10, ipe.output_dim)

    def test_scene_contraction(self):
        from flash3d.models.architectures.mip_nerf import SceneContraction

        sc = SceneContraction()
        inside = torch.tensor([[0.5, 0.3, 0.1]])
        outside = torch.tensor([[5.0, 3.0, 1.0]])

        c_in = sc(inside)
        assert torch.allclose(c_in, inside, atol=1e-6)

        c_out = sc(outside)
        assert c_out.norm(dim=-1).item() < outside.norm(dim=-1).item()

    def test_distortion_loss(self):
        from flash3d.models.architectures.mip_nerf import MipNeRF360

        config = Flash3DConfig()
        model = MipNeRF360(config=config, hidden_dim=32, num_layers=2, num_pos_frequencies=4)
        weights = torch.rand(1, 10)
        t_vals = torch.linspace(0.1, 10, 11).unsqueeze(0)
        loss = model._distortion_loss(weights, t_vals)
        assert loss.dim() == 0


class TestAntiAliasing:
    def test_rasterize_with_antialias(self):
        from flash3d.rendering.rasterizer import _compute_cov2d

        means = torch.randn(5, 3)
        scales = torch.ones(5, 3) * 0.1
        rots = torch.zeros(5, 4)
        rots[:, 0] = 1.0
        viewmat = torch.eye(4)
        projmat = torch.eye(4)
        projmat[0, 0] = 2.0
        projmat[1, 1] = 2.0

        cov2d_no_aa = _compute_cov2d(
            means, scales, rots, viewmat, projmat, 800, 800, antialias=False
        )
        cov2d_aa = _compute_cov2d(
            means, scales, rots, viewmat, projmat, 800, 800, antialias=True, mip_filter_size=0.3
        )

        assert cov2d_no_aa.shape == (5, 2, 2)
        assert cov2d_aa.shape == (5, 2, 2)


class TestPointNetPP:
    def test_pointnet_pp_backbone(self):
        from flash3d.models.point_cloud.pointnet_pp import PointNetPP

        model = PointNetPP(in_channel=3, use_msg=False)
        xyz = torch.randn(2, 1024, 3)
        features = torch.randn(2, 1024, 3)
        global_feat, _, intermediates = model(xyz, features)
        assert global_feat.shape == (2, 1024)
        assert len(intermediates) == 4

    def test_pointnet_pp_classifier(self):
        from flash3d.models.point_cloud.pointnet_pp import PointNetPPClassifier

        model = PointNetPPClassifier(num_classes=10, in_channel=3, use_msg=False)
        xyz = torch.randn(2, 1024, 3)
        features = torch.randn(2, 1024, 3)
        logits = model(xyz, features)
        assert logits.shape == (2, 10)

    def test_farthest_point_sample(self):
        from flash3d.models.point_cloud.pointnet_pp import farthest_point_sample

        xyz = torch.randn(1, 100, 3)
        indices = farthest_point_sample(xyz, 10)
        assert indices.shape == (1, 10)
        assert indices.unique().shape[0] == 10

    def test_pointnet_pp_registered(self):
        from flash3d.registry import MODELS

        assert "pointnet_pp" in MODELS
        assert "pointnet_pp_cls" in MODELS
        assert "pointnet_pp_seg" in MODELS


class TestGaussianSplatting4D:
    def test_4d_gs_registered(self):
        from flash3d.registry import MODELS

        assert "gaussian_splatting_4d" in MODELS

    def test_deformation_network(self):
        from flash3d.models.architectures.gaussian_splatting_4d import DeformationNetwork

        net = DeformationNetwork(hidden_dim=32, num_layers=3)
        pos = torch.randn(50, 3)
        t = torch.tensor(0.5)
        out = net(pos, t)
        assert out["delta_pos"].shape == (50, 3)
        assert out["delta_rot"].shape == (50, 4)
        assert out["delta_scale"].shape == (50, 3)

    def test_deform_gaussians(self):
        from flash3d.models.architectures.gaussian_splatting_4d import GaussianSplatting4D

        config = Flash3DConfig()
        config.model.num_gaussians = 20
        config.model.sh_degree = 1
        model = GaussianSplatting4D(
            config=config, deformation_hidden_dim=32, deformation_num_layers=3
        )
        deformed = model.deform_gaussians(0.5)
        assert deformed["means"].shape == (20, 3)
        assert deformed["rotations"].shape == (20, 4)
        assert deformed["scales"].shape == (20, 3)

    def test_temporal_smoothness_loss(self):
        from flash3d.models.architectures.gaussian_splatting_4d import GaussianSplatting4D

        config = Flash3DConfig()
        config.model.num_gaussians = 10
        config.model.sh_degree = 1
        model = GaussianSplatting4D(
            config=config, deformation_hidden_dim=32, deformation_num_layers=3
        )
        loss = model.temporal_smoothness_loss(0.5)
        assert loss.dim() == 0


class TestTexture:
    def test_spherical_projection(self):
        from flash3d.geometry.texture import UVMapper

        vertices = np.random.randn(100, 3)
        uvs = UVMapper.spherical_projection(vertices)
        assert uvs.shape == (100, 2)
        assert (uvs >= 0).all() and (uvs <= 1).all()

    def test_cylindrical_projection(self):
        from flash3d.geometry.texture import UVMapper

        vertices = np.random.randn(50, 3)
        uvs = UVMapper.cylindrical_projection(vertices)
        assert uvs.shape == (50, 2)

    def test_planar_projection(self):
        from flash3d.geometry.texture import UVMapper

        vertices = np.random.randn(50, 3)
        uvs = UVMapper.planar_projection(vertices, plane="xy")
        assert uvs.shape == (50, 2)

    def test_texture_atlas(self):
        from flash3d.geometry.texture import TextureAtlas

        atlas = TextureAtlas(atlas_size=256)
        chart = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        ox, oy = atlas.pack_chart(chart)
        assert ox >= 0 and oy >= 0
        result = atlas.get_atlas()
        assert result.shape == (256, 256, 3)
