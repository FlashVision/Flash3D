"""Comprehensive tests for Flash3D covering all architectures, modules, and pipelines."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from flash3d.cfg.config import Flash3DConfig


# ---------------------------------------------------------------------------
# 1. Gaussian Splatting
# ---------------------------------------------------------------------------
class TestGaussianSplattingComprehensive:
    def _cfg(self, n=20, sh=1):
        cfg = Flash3DConfig()
        cfg.model.num_gaussians = n
        cfg.model.sh_degree = sh
        return cfg

    def test_init_shapes(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(30, 2))
        assert m.means.shape == (30, 3)
        assert m.scales.shape == (30, 3)
        assert m.rotations.shape == (30, 4)
        assert m.opacities.shape == (30, 1)
        assert m.sh_coeffs.shape[0] == 30

    def test_num_points_property(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(15))
        assert m.num_points == 15

    def test_covariance_symmetric_positive(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(10))
        cov = m.get_covariance_3d()
        assert cov.shape == (10, 3, 3)
        assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-5)

    def test_opacity_sigmoid(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(20))
        o = m.get_opacity()
        assert (o >= 0).all() and (o <= 1).all()

    def test_get_scales_positive(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(10))
        s = m.get_scales()
        assert (s > 0).all()

    def test_initialize_from_point_cloud(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(5))
        pts = torch.randn(50, 3)
        cols = torch.rand(50, 3)
        m.initialize_from_point_cloud(pts, cols)
        assert m.num_points == 50
        assert m.sh_coeffs.shape[0] == 50

    def test_init_from_pc_no_colors(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(5))
        m.initialize_from_point_cloud(torch.randn(10, 3))
        assert m.num_points == 10

    def test_forward_no_cameras(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(10))
        out = m(cameras=None)
        assert "means" in out
        assert "num_gaussians" in out

    def test_quaternion_to_matrix(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        R = GaussianSplatting._quaternion_to_matrix(q)
        assert torch.allclose(R.squeeze(), torch.eye(3), atol=1e-5)

    def test_clone_gaussians(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(10))
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:3] = True
        m._clone_gaussians(mask)
        assert m.num_points == 13

    def test_prune_gaussians(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting

        m = GaussianSplatting(config=self._cfg(10))
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:2] = True
        m._prune_gaussians(mask)
        assert m.num_points == 8


# ---------------------------------------------------------------------------
# 2. NeRF
# ---------------------------------------------------------------------------
class TestNeRFComprehensive:
    def test_positional_encoding_shape(self):
        from flash3d.models.architectures.nerf import PositionalEncoding

        pe = PositionalEncoding(num_frequencies=4)
        x = torch.randn(5, 3)
        out = pe(x)
        assert out.shape == (5, 3 * pe.output_dim)

    def test_positional_encoding_include_input(self):
        from flash3d.models.architectures.nerf import PositionalEncoding

        pe = PositionalEncoding(num_frequencies=2, include_input=True)
        assert pe.output_dim == 2 * 2 + 1

    def test_positional_encoding_no_input(self):
        from flash3d.models.architectures.nerf import PositionalEncoding

        pe = PositionalEncoding(num_frequencies=2, include_input=False)
        assert pe.output_dim == 2 * 2

    def test_hash_encoding_shape(self):
        from flash3d.models.architectures.nerf import HashEncoding

        he = HashEncoding(num_levels=4, features_per_level=2)
        x = torch.rand(8, 3)
        out = he(x)
        assert out.shape == (8, he.output_dim)

    def test_hash_encoding_output_dim(self):
        from flash3d.models.architectures.nerf import HashEncoding

        he = HashEncoding(num_levels=8, features_per_level=3)
        assert he.output_dim == 24

    def test_nerf_mlp_forward(self):
        from flash3d.models.architectures.nerf import NeRFMLP

        mlp = NeRFMLP(input_dim=32, dir_dim=16, hidden_dim=32, num_layers=4, skip_connections=(2,))
        pos = torch.randn(5, 32)
        d = torch.randn(5, 16)
        density, rgb = mlp(pos, d)
        assert density.shape == (5, 1)
        assert rgb.shape == (5, 3)
        assert (density >= 0).all()
        assert (rgb >= 0).all() and (rgb <= 1).all()

    def test_nerf_query(self):
        cfg = Flash3DConfig()
        from flash3d.models.architectures.nerf import NeRF

        m = NeRF(config=cfg, use_hash_encoding=False, num_layers=2, hidden_dim=32)
        density, rgb = m.query(torch.randn(3, 3), torch.randn(3, 3))
        assert density.shape == (3, 1)
        assert rgb.shape == (3, 3)

    def test_nerf_query_with_hash(self):
        from flash3d.models.architectures.nerf import NeRF

        m = NeRF(config=Flash3DConfig(), use_hash_encoding=True, num_layers=2, hidden_dim=32)
        density, rgb = m.query(torch.rand(4, 3), torch.randn(4, 3))
        assert density.shape == (4, 1)

    def test_nerf_forward_no_cameras(self):
        from flash3d.models.architectures.nerf import NeRF

        m = NeRF(config=Flash3DConfig(), num_layers=2, hidden_dim=32)
        out = m(cameras=None)
        assert "model" in out

    def test_nerf_num_parameters(self):
        from flash3d.models.architectures.nerf import NeRF

        m = NeRF(config=Flash3DConfig(), num_layers=2, hidden_dim=32)
        assert m.num_parameters > 0


# ---------------------------------------------------------------------------
# 3. Feed-Forward 3DGS
# ---------------------------------------------------------------------------
class TestFeedForward3DGSComprehensive:
    def test_image_encoder(self):
        from flash3d.models.architectures.feed_forward_3dgs import ImageEncoder

        enc = ImageEncoder(in_channels=3, base_channels=16)
        x = torch.rand(2, 3, 32, 32)
        out = enc(x)
        assert out.shape[0] == 2
        assert out.shape[1] == enc.output_dim

    def test_cross_view_attention(self):
        from flash3d.models.architectures.feed_forward_3dgs import CrossViewAttention

        attn = CrossViewAttention(embed_dim=32, num_heads=4)
        q = torch.randn(1, 16, 32)
        ctx = torch.randn(1, 16, 32)
        out = attn(q, ctx)
        assert out.shape == q.shape

    def test_gaussian_head(self):
        from flash3d.models.architectures.feed_forward_3dgs import GaussianHead

        head = GaussianHead(in_channels=32, sh_degree=1)
        feat = torch.randn(1, 32, 8, 8)
        out = head(feat)
        assert "depth" in out
        assert "scales" in out
        assert "rotations" in out
        assert "opacities" in out
        assert "sh_coeffs" in out
        assert (out["opacities"] >= 0).all() and (out["opacities"] <= 1).all()

    def test_ff3dgs_forward(self):
        from flash3d.models.architectures.feed_forward_3dgs import FeedForward3DGS

        cfg = Flash3DConfig()
        m = FeedForward3DGS(config=cfg, base_channels=16, num_attention_layers=1)
        imgs = torch.rand(1, 2, 3, 64, 64)
        out = m(images=imgs)
        assert "depth" in out and "scales" in out

    def test_ff3dgs_single_view(self):
        from flash3d.models.architectures.feed_forward_3dgs import FeedForward3DGS

        m = FeedForward3DGS(config=Flash3DConfig(), base_channels=16, num_attention_layers=1)
        imgs = torch.rand(1, 3, 64, 64)
        out = m(images=imgs)
        assert "depth" in out

    def test_ff3dgs_no_images(self):
        from flash3d.models.architectures.feed_forward_3dgs import FeedForward3DGS

        m = FeedForward3DGS(config=Flash3DConfig(), base_channels=16, num_attention_layers=1)
        out = m(images=None)
        assert "model" in out


# ---------------------------------------------------------------------------
# 4. Mip-NeRF 360
# ---------------------------------------------------------------------------
class TestMipNeRF360Comprehensive:
    def test_ipe_shape(self):
        from flash3d.models.architectures.mip_nerf import IntegratedPositionalEncoding

        ipe = IntegratedPositionalEncoding(num_frequencies=4)
        means = torch.randn(10, 3)
        covs = torch.rand(10, 3).abs()
        out = ipe(means, covs)
        assert out.shape == (10, ipe.output_dim)

    def test_ipe_output_dim(self):
        from flash3d.models.architectures.mip_nerf import IntegratedPositionalEncoding

        ipe = IntegratedPositionalEncoding(num_frequencies=4, include_input=True)
        assert ipe.output_dim == (4 * 2 + 1) * 3

    def test_scene_contraction_inside(self):
        from flash3d.models.architectures.mip_nerf import SceneContraction

        sc = SceneContraction()
        inside = torch.tensor([[0.3, 0.2, 0.1]])
        assert torch.allclose(sc(inside), inside, atol=1e-6)

    def test_scene_contraction_outside(self):
        from flash3d.models.architectures.mip_nerf import SceneContraction

        sc = SceneContraction()
        outside = torch.tensor([[5.0, 3.0, 1.0]])
        contracted = sc(outside)
        assert contracted.norm(dim=-1).item() < outside.norm(dim=-1).item()

    def test_contract_covariance(self):
        from flash3d.models.architectures.mip_nerf import SceneContraction

        sc = SceneContraction()
        x = torch.tensor([[5.0, 0.0, 0.0]])
        cov = torch.tensor([[1.0, 1.0, 1.0]])
        out = sc.contract_covariance(x, cov)
        assert out.shape == cov.shape

    def test_proposal_network(self):
        from flash3d.models.architectures.mip_nerf import ProposalNetwork

        pn = ProposalNetwork(input_dim=16, hidden_dim=32, num_layers=2)
        out = pn(torch.randn(5, 16))
        assert out.shape == (5, 1)
        assert (out >= 0).all()

    def test_mip_nerf_mlp(self):
        from flash3d.models.architectures.mip_nerf import MipNeRFMLP

        mlp = MipNeRFMLP(input_dim=32, dir_dim=16, hidden_dim=32, num_layers=4)
        density, rgb = mlp(torch.randn(5, 32), torch.randn(5, 16))
        assert density.shape == (5, 1) and rgb.shape == (5, 3)

    def test_distortion_loss_scalar(self):
        from flash3d.models.architectures.mip_nerf import MipNeRF360

        cfg = Flash3DConfig()
        m = MipNeRF360(config=cfg, hidden_dim=32, num_layers=2, num_pos_frequencies=4)
        weights = torch.rand(1, 10)
        t_vals = torch.linspace(0.1, 10, 11).unsqueeze(0)
        loss = m._distortion_loss(weights, t_vals)
        assert loss.dim() == 0

    def test_mip_nerf_forward_no_cameras(self):
        from flash3d.models.architectures.mip_nerf import MipNeRF360

        m = MipNeRF360(config=Flash3DConfig(), hidden_dim=32, num_layers=2, num_pos_frequencies=4)
        out = m(cameras=None)
        assert out["model"] == "mip_nerf_360"

    def test_mip_nerf_num_params(self):
        from flash3d.models.architectures.mip_nerf import MipNeRF360

        m = MipNeRF360(config=Flash3DConfig(), hidden_dim=32, num_layers=2, num_pos_frequencies=4)
        assert m.num_parameters > 0


# ---------------------------------------------------------------------------
# 5. 4D Gaussian Splatting
# ---------------------------------------------------------------------------
class TestGaussianSplatting4DComprehensive:
    def _cfg(self, n=10):
        cfg = Flash3DConfig()
        cfg.model.num_gaussians = n
        cfg.model.sh_degree = 1
        return cfg

    def test_temporal_encoding(self):
        from flash3d.models.architectures.gaussian_splatting_4d import TemporalEncoding

        te = TemporalEncoding(num_frequencies=4)
        t = torch.tensor(0.5)
        out = te(t)
        assert out.shape[-1] == te.output_dim

    def test_temporal_encoding_batch(self):
        from flash3d.models.architectures.gaussian_splatting_4d import TemporalEncoding

        te = TemporalEncoding(num_frequencies=4)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = te(t)
        assert out.shape[0] == 3

    def test_deformation_network(self):
        from flash3d.models.architectures.gaussian_splatting_4d import DeformationNetwork

        net = DeformationNetwork(hidden_dim=32, num_layers=3)
        pos = torch.randn(20, 3)
        out = net(pos, torch.tensor(0.5))
        assert out["delta_pos"].shape == (20, 3)
        assert out["delta_rot"].shape == (20, 4)
        assert out["delta_scale"].shape == (20, 3)

    def test_deform_gaussians(self):
        from flash3d.models.architectures.gaussian_splatting_4d import GaussianSplatting4D

        m = GaussianSplatting4D(
            config=self._cfg(10), deformation_hidden_dim=32, deformation_num_layers=3
        )
        d = m.deform_gaussians(0.5)
        assert d["means"].shape == (10, 3)
        assert d["rotations"].shape == (10, 4)

    def test_quaternion_multiply(self):
        from flash3d.models.architectures.gaussian_splatting_4d import GaussianSplatting4D

        q_id = torch.tensor([[1.0, 0, 0, 0]])
        q = torch.tensor([[0.7071, 0.7071, 0, 0]])
        result = GaussianSplatting4D._quaternion_multiply(q_id, q)
        assert torch.allclose(result, q, atol=1e-4)

    def test_temporal_smoothness(self):
        from flash3d.models.architectures.gaussian_splatting_4d import GaussianSplatting4D

        m = GaussianSplatting4D(
            config=self._cfg(10), deformation_hidden_dim=32, deformation_num_layers=3
        )
        loss = m.temporal_smoothness_loss(0.5)
        assert loss.dim() == 0

    def test_forward_no_cameras(self):
        from flash3d.models.architectures.gaussian_splatting_4d import GaussianSplatting4D

        m = GaussianSplatting4D(
            config=self._cfg(10), deformation_hidden_dim=32, deformation_num_layers=3
        )
        out = m(cameras=None)
        assert "model" in out

    def test_init_from_point_cloud(self):
        from flash3d.models.architectures.gaussian_splatting_4d import GaussianSplatting4D

        m = GaussianSplatting4D(
            config=self._cfg(5), deformation_hidden_dim=32, deformation_num_layers=3
        )
        m.initialize_from_point_cloud(torch.randn(20, 3))
        assert m.num_points == 20


# ---------------------------------------------------------------------------
# 6. Hash Encoding (multi-resolution)
# ---------------------------------------------------------------------------
class TestHashEncodingComprehensive:
    def test_multi_res_shape(self):
        from flash3d.models.encodings.hash_encoding import MultiResolutionHashEncoding

        enc = MultiResolutionHashEncoding(num_levels=4, features_per_level=2)
        x = torch.rand(50, 3)
        out = enc(x)
        assert out.shape == (50, 8)

    def test_multi_res_batch(self):
        from flash3d.models.encodings.hash_encoding import MultiResolutionHashEncoding

        enc = MultiResolutionHashEncoding(num_levels=4, features_per_level=2)
        x = torch.rand(2, 25, 3)
        out = enc(x)
        assert out.shape == (2, 25, 8)

    def test_multi_res_clamping(self):
        from flash3d.models.encodings.hash_encoding import MultiResolutionHashEncoding

        enc = MultiResolutionHashEncoding(num_levels=2, features_per_level=2)
        x = torch.tensor([[2.0, -1.0, 0.5]])
        out = enc(x)
        assert out.shape == (1, 4)

    def test_instant_ngp_density_color(self):
        from flash3d.models.encodings.hash_encoding import InstantNGPHashEncoding

        enc = InstantNGPHashEncoding(
            num_levels=4, features_per_level=2, hidden_dim=32, num_layers=2
        )
        result = enc(torch.rand(5, 3), torch.randn(5, 3))
        assert result["density"].shape == (5, 1)
        assert result["rgb"].shape == (5, 3)
        assert (result["density"] >= 0).all()
        assert (result["rgb"] >= 0).all() and (result["rgb"] <= 1).all()

    def test_instant_ngp_density_only(self):
        from flash3d.models.encodings.hash_encoding import InstantNGPHashEncoding

        enc = InstantNGPHashEncoding(
            num_levels=4, features_per_level=2, hidden_dim=32, num_layers=2
        )
        result = enc(torch.rand(5, 3))
        assert "density" in result
        assert "rgb" not in result

    def test_instant_ngp_registered(self):
        from flash3d.registry import MODELS

        assert "instant_ngp_encoding" in MODELS


# ---------------------------------------------------------------------------
# 7. COLMAP Loader
# ---------------------------------------------------------------------------
class TestCOLMAPLoaderComprehensive:
    def test_qvec_identity(self):
        from flash3d.data.colmap_loader import qvec_to_rotation_matrix

        R = qvec_to_rotation_matrix(np.array([1, 0, 0, 0]))
        assert np.allclose(R, np.eye(3), atol=1e-6)

    def test_qvec_orthogonal(self):
        from flash3d.data.colmap_loader import qvec_to_rotation_matrix

        s = 1.0 / np.sqrt(2.0)
        q = np.array([s, s, 0, 0])
        R = qvec_to_rotation_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)

    def test_pinhole_intrinsics(self):
        from flash3d.data.colmap_loader import camera_params_to_intrinsics

        K = camera_params_to_intrinsics("PINHOLE", np.array([500, 500, 320, 240]), 640, 480)
        assert K[0, 0] == 500.0 and K[1, 1] == 500.0
        assert K[0, 2] == 320.0 and K[1, 2] == 240.0

    def test_simple_pinhole(self):
        from flash3d.data.colmap_loader import camera_params_to_intrinsics

        K = camera_params_to_intrinsics("SIMPLE_PINHOLE", np.array([500, 320, 240]), 640, 480)
        assert K[0, 0] == K[1, 1] == 500.0

    def test_simple_radial(self):
        from flash3d.data.colmap_loader import camera_params_to_intrinsics

        K = camera_params_to_intrinsics("SIMPLE_RADIAL", np.array([500, 320, 240, 0.01]), 640, 480)
        assert K[0, 0] == 500.0

    def test_opencv_model(self):
        from flash3d.data.colmap_loader import camera_params_to_intrinsics

        K = camera_params_to_intrinsics(
            "OPENCV", np.array([500, 500, 320, 240, 0.1, -0.1, 0, 0]), 640, 480
        )
        assert K[0, 0] == 500.0

    def test_get_distortion_params(self):
        from flash3d.data.colmap_loader import get_distortion_params

        d = get_distortion_params("SIMPLE_RADIAL", np.array([500, 320, 240, 0.05]))
        assert d[0] == pytest.approx(0.05)

    def test_colmap_scene_empty(self, tmp_path):
        from flash3d.data.colmap_loader import COLMAPScene

        sparse = tmp_path / "sparse" / "0"
        sparse.mkdir(parents=True)
        scene = COLMAPScene(str(tmp_path))
        assert scene.num_images == 0
        assert scene.num_cameras == 0

    def test_scene_bounds_empty(self, tmp_path):
        from flash3d.data.colmap_loader import COLMAPScene

        sparse = tmp_path / "sparse" / "0"
        sparse.mkdir(parents=True)
        scene = COLMAPScene(str(tmp_path))
        bmin, bmax = scene.compute_scene_bounds()
        assert bmin.shape == (3,)

    def test_make_projection_matrix(self):
        from flash3d.data.colmap_loader import _make_projection_matrix

        P = _make_projection_matrix(500, 500, 320, 240, 640, 480, 0.01, 100)
        assert P.shape == (4, 4)
        assert P[3, 2] == -1.0


# ---------------------------------------------------------------------------
# 8. Depth Anything V2 wrapper
# ---------------------------------------------------------------------------
class TestDepthAnythingV2Comprehensive:
    def test_fallback_creation(self):
        from flash3d.geometry.depth_anything import DepthAnythingV2

        est = DepthAnythingV2(variant="small", device="cpu")
        assert est._model is not None

    def test_forward_shape(self):
        from flash3d.geometry.depth_anything import DepthAnythingV2

        est = DepthAnythingV2(variant="small", device="cpu")
        img = torch.rand(1, 3, 32, 32)
        depth = est(img)
        assert depth.shape[0] == 1 and depth.shape[1] == 1

    def test_batch_predict(self):
        from flash3d.geometry.depth_anything import DepthAnythingV2

        est = DepthAnythingV2(variant="small", device="cpu")
        imgs = [torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)]
        results = est.batch_predict(imgs)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 9. PointNet++
# ---------------------------------------------------------------------------
class TestPointNetPPComprehensive:
    def test_square_distance(self):
        from flash3d.models.point_cloud.pointnet_pp import square_distance

        src = torch.randn(1, 10, 3)
        dst = torch.randn(1, 5, 3)
        d = square_distance(src, dst)
        assert d.shape == (1, 10, 5)
        assert (d >= 0).all()

    def test_farthest_point_sample(self):
        from flash3d.models.point_cloud.pointnet_pp import farthest_point_sample

        xyz = torch.randn(2, 100, 3)
        idx = farthest_point_sample(xyz, 10)
        assert idx.shape == (2, 10)

    def test_query_ball_point(self):
        from flash3d.models.point_cloud.pointnet_pp import query_ball_point

        xyz = torch.randn(1, 50, 3)
        new_xyz = torch.randn(1, 5, 3)
        idx = query_ball_point(radius=1.0, n_sample=8, xyz=xyz, new_xyz=new_xyz)
        assert idx.shape == (1, 5, 8)

    def test_index_points(self):
        from flash3d.models.point_cloud.pointnet_pp import index_points

        pts = torch.randn(2, 10, 3)
        idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
        out = index_points(pts, idx)
        assert out.shape == (2, 3, 3)

    def test_set_abstraction(self):
        from flash3d.models.point_cloud.pointnet_pp import PointNetSetAbstraction

        sa = PointNetSetAbstraction(
            n_point=16, radius=0.5, n_sample=8, in_channel=0, mlp_channels=[16, 32]
        )
        xyz = torch.randn(2, 64, 3)
        new_xyz, new_pts = sa(xyz, None)
        assert new_xyz.shape == (2, 16, 3)

    def test_set_abstraction_group_all(self):
        from flash3d.models.point_cloud.pointnet_pp import PointNetSetAbstraction

        sa = PointNetSetAbstraction(
            n_point=0, radius=0, n_sample=0, in_channel=0, mlp_channels=[16, 32], group_all=True
        )
        xyz = torch.randn(2, 64, 3)
        new_xyz, new_pts = sa(xyz, None)
        assert new_xyz.shape == (2, 1, 3)

    def test_feature_propagation(self):
        from flash3d.models.point_cloud.pointnet_pp import FeaturePropagation

        fp = FeaturePropagation(in_channel=32, mlp_channels=[16])
        xyz1 = torch.randn(2, 64, 3)
        xyz2 = torch.randn(2, 16, 3)
        pts2 = torch.randn(2, 16, 32)
        out = fp(xyz1, xyz2, None, pts2)
        assert out.shape[1] == 64

    def test_pointnet_pp_backbone(self):
        from flash3d.models.point_cloud.pointnet_pp import PointNetPP

        m = PointNetPP(in_channel=3, use_msg=False)
        xyz = torch.randn(2, 1024, 3)
        features = torch.randn(2, 1024, 3)
        global_feat, _, intermediates = m(xyz, features)
        assert global_feat.shape == (2, 1024)
        assert len(intermediates) == 4

    def test_pointnet_pp_classifier(self):
        from flash3d.models.point_cloud.pointnet_pp import PointNetPPClassifier

        m = PointNetPPClassifier(num_classes=10, in_channel=3, use_msg=False)
        xyz = torch.randn(2, 1024, 3)
        features = torch.randn(2, 1024, 3)
        logits = m(xyz, features)
        assert logits.shape == (2, 10)

    def test_pointnet_pp_segmentor(self):
        from flash3d.models.point_cloud.pointnet_pp import PointNetPPSegmentor

        m = PointNetPPSegmentor(num_classes=5, in_channel=3)
        xyz = torch.randn(2, 1024, 3)
        features = torch.randn(2, 1024, 3)
        out = m(xyz, features)
        assert out.shape == (2, 1024, 5)

    def test_registration(self):
        from flash3d.registry import MODELS

        assert "pointnet_pp" in MODELS
        assert "pointnet_pp_cls" in MODELS
        assert "pointnet_pp_seg" in MODELS


# ---------------------------------------------------------------------------
# 10. Rendering: rasterizer, cameras, volume rendering, SH
# ---------------------------------------------------------------------------
class TestRenderingComprehensive:
    def test_rasterize_empty(self):
        from flash3d.rendering.rasterizer import rasterize_gaussians

        result = rasterize_gaussians(
            means3d=torch.tensor([[0.0, 0.0, -5.0]]),
            scales=torch.ones(1, 3) * 0.1,
            rotations=torch.tensor([[1.0, 0, 0, 0]]),
            opacities=torch.tensor([[0.9]]),
            sh_coeffs=torch.zeros(1, 1, 3),
            viewmatrix=torch.eye(4),
            projmatrix=torch.eye(4),
            camera_center=torch.zeros(3),
            image_width=16,
            image_height=16,
            sh_degree=0,
        )
        assert "rgb" in result and "depth" in result and "alpha" in result

    def test_compute_cov2d_shapes(self):
        from flash3d.rendering.rasterizer import _compute_cov2d

        means = torch.randn(5, 3)
        scales = torch.ones(5, 3) * 0.1
        rots = torch.zeros(5, 4)
        rots[:, 0] = 1.0
        P = torch.eye(4)
        P[0, 0] = 2.0
        P[1, 1] = 2.0
        cov = _compute_cov2d(means, scales, rots, torch.eye(4), P, 32, 32)
        assert cov.shape == (5, 2, 2)

    def test_compute_cov2d_antialias(self):
        from flash3d.rendering.rasterizer import _compute_cov2d

        means = torch.randn(3, 3)
        scales = torch.ones(3, 3) * 0.1
        rots = torch.zeros(3, 4)
        rots[:, 0] = 1.0
        P = torch.eye(4)
        P[0, 0] = 2.0
        P[1, 1] = 2.0
        cov = _compute_cov2d(
            means, scales, rots, torch.eye(4), P, 32, 32, antialias=True, mip_filter_size=0.3
        )
        assert cov.shape == (3, 2, 2)


class TestCamerasComprehensive:
    def test_camera_dataclass(self):
        from flash3d.rendering.cameras import Camera

        cam = Camera(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
        K = cam.intrinsics
        assert K[0, 0] == 500 and K[1, 1] == 500

    def test_camera_extrinsics(self):
        from flash3d.rendering.cameras import Camera

        cam = Camera(
            fx=500, fy=500, cx=320, cy=240, width=640, height=480, R=torch.eye(3), t=torch.zeros(3)
        )
        E = cam.extrinsics
        assert torch.allclose(E, torch.eye(4))

    def test_camera_center(self):
        from flash3d.rendering.cameras import Camera

        cam = Camera(
            fx=500,
            fy=500,
            cx=320,
            cy=240,
            width=640,
            height=480,
            R=torch.eye(3),
            t=torch.tensor([1.0, 2.0, 3.0]),
        )
        c = cam.camera_center
        assert torch.allclose(c, torch.tensor([-1.0, -2.0, -3.0]))

    def test_camera_to_dict(self):
        from flash3d.rendering.cameras import Camera

        cam = Camera(
            fx=500, fy=500, cx=320, cy=240, width=640, height=480, R=torch.eye(3), t=torch.zeros(3)
        )
        d = cam.to_dict()
        assert "viewmatrix" in d and "projmatrix" in d and "intrinsics" in d

    def test_generate_rays(self):
        from flash3d.rendering.cameras import generate_rays

        K = torch.tensor([[50.0, 0, 16], [0, 50.0, 16], [0, 0, 1.0]])
        E = torch.eye(4)
        rays_o, rays_d = generate_rays(K, E, 32, 32)
        assert rays_o.shape == (32 * 32, 3)
        assert rays_d.shape == (32 * 32, 3)

    def test_perspective_projection(self):
        from flash3d.rendering.cameras import perspective_projection

        pts = torch.tensor([[0.0, 0.0, 5.0]])
        K = torch.tensor([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1.0]])
        E = torch.eye(4)
        pixels, depths = perspective_projection(pts, K, E)
        assert pixels.shape == (1, 2)
        assert depths.shape == (1,)

    def test_interpolate_cameras(self):
        from flash3d.rendering.cameras import Camera, interpolate_cameras

        cam1 = Camera(
            fx=500, fy=500, cx=320, cy=240, width=640, height=480, R=torch.eye(3), t=torch.zeros(3)
        )
        cam2 = Camera(
            fx=600,
            fy=600,
            cx=320,
            cy=240,
            width=640,
            height=480,
            R=torch.eye(3),
            t=torch.tensor([1.0, 0.0, 0.0]),
        )
        cams = interpolate_cameras(cam1, cam2, num_frames=5)
        assert len(cams) == 5
        assert cams[0].fx == 500 and cams[-1].fx == 600


class TestVolumeRenderingComprehensive:
    def test_volume_render_rays(self):
        from flash3d.rendering.ray_marching import volume_render_rays

        density = torch.rand(10, 32)
        rgb = torch.rand(10, 32, 3)
        t_vals = torch.linspace(0.1, 10, 32)
        result = volume_render_rays(density, rgb, t_vals)
        assert result["rgb"].shape == (10, 3)
        assert result["depth"].shape == (10,)
        assert result["alpha"].shape == (10,)
        assert result["weights"].shape == (10, 32)

    def test_volume_render_white_bg(self):
        from flash3d.rendering.ray_marching import volume_render_rays

        density = torch.zeros(1, 8)
        rgb = torch.zeros(1, 8, 3)
        t_vals = torch.linspace(1, 5, 8)
        result = volume_render_rays(density, rgb, t_vals, white_background=True)
        assert torch.allclose(result["rgb"], torch.ones(1, 3), atol=0.01)

    def test_hierarchical_sampling(self):
        from flash3d.rendering.ray_marching import hierarchical_sampling

        t = torch.linspace(0.1, 10, 16).unsqueeze(0)
        w = torch.rand(1, 16)
        combined = hierarchical_sampling(t, w, num_importance_samples=16, deterministic=True)
        assert combined.shape == (1, 32)

    def test_sample_along_rays(self):
        from flash3d.rendering.ray_marching import sample_along_rays

        rays_o = torch.zeros(10, 3)
        rays_d = torch.tensor([[0, 0, 1.0]]).expand(10, 3)
        pts, t_vals = sample_along_rays(rays_o, rays_d, num_samples=16, perturb=False)
        assert pts.shape == (10, 16, 3)
        assert t_vals.shape == (10, 16)

    def test_sample_along_rays_perturbed(self):
        from flash3d.rendering.ray_marching import sample_along_rays

        rays_o = torch.zeros(5, 3)
        rays_d = torch.tensor([[0, 0, 1.0]]).expand(5, 3)
        pts1, _ = sample_along_rays(rays_o, rays_d, num_samples=8, perturb=True)
        pts2, _ = sample_along_rays(rays_o, rays_d, num_samples=8, perturb=True)
        assert not torch.allclose(pts1, pts2)


class TestSHUtilsComprehensive:
    def test_eval_sh_degree0(self):
        from flash3d.rendering.sh_utils import eval_sh

        sh = torch.ones(5, 1, 3)
        dirs = F.normalize(torch.randn(5, 3), dim=-1)
        color = eval_sh(0, sh, dirs)
        assert color.shape == (5, 3)

    def test_eval_sh_degree3(self):
        from flash3d.rendering.sh_utils import eval_sh, random_sh_coeffs

        sh = random_sh_coeffs(10, degree=3)
        dirs = F.normalize(torch.randn(10, 3), dim=-1)
        color = eval_sh(3, sh, dirs)
        assert color.shape == (10, 3)

    def test_rgb_to_sh_roundtrip(self):
        from flash3d.rendering.sh_utils import rgb_to_sh, sh_to_rgb

        rgb = torch.rand(5, 3)
        sh0 = rgb_to_sh(rgb)
        recovered = sh_to_rgb(sh0)
        assert torch.allclose(rgb, recovered, atol=1e-5)

    def test_get_num_sh_coeffs(self):
        from flash3d.rendering.sh_utils import get_num_sh_coeffs

        assert get_num_sh_coeffs(0) == 1
        assert get_num_sh_coeffs(1) == 4
        assert get_num_sh_coeffs(2) == 9
        assert get_num_sh_coeffs(3) == 16


# ---------------------------------------------------------------------------
# 11. Geometry: transforms, mesh, depth, texture, point cloud
# ---------------------------------------------------------------------------
class TestGeometryTransformsComprehensive:
    def test_se3_identity(self):
        from flash3d.geometry.transforms_3d import SE3

        se3 = SE3.identity()
        assert torch.allclose(se3.matrix, torch.eye(4))

    def test_se3_inverse_compose(self):
        from flash3d.geometry.transforms_3d import SE3, rotation_matrix_from_euler

        R = rotation_matrix_from_euler(0.1, 0.2, 0.3)
        t = torch.tensor([1.0, 2.0, 3.0])
        se3 = SE3.from_rotation_translation(R, t)
        identity = se3.compose(se3.inverse())
        assert torch.allclose(identity.matrix, torch.eye(4), atol=1e-5)

    def test_quaternion_roundtrip(self):
        from flash3d.geometry.transforms_3d import (
            quaternion_to_rotation_matrix,
            rotation_matrix_from_euler,
            rotation_matrix_to_quaternion,
        )

        R = rotation_matrix_from_euler(0.5, -0.3, 1.0)
        q = rotation_matrix_to_quaternion(R)
        R2 = quaternion_to_rotation_matrix(q)
        assert torch.allclose(R, R2, atol=1e-4)

    def test_rotation_orthogonal(self):
        from flash3d.geometry.transforms_3d import rotation_matrix_from_euler

        R = rotation_matrix_from_euler(1.0, 2.0, 3.0)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)

    def test_look_at(self):
        from flash3d.geometry.transforms_3d import look_at

        mat = look_at(torch.tensor([0.0, 0, -3.0]), torch.zeros(3))
        assert mat.shape == (4, 4)


class TestMeshComprehensive:
    def test_simple_marching_cubes(self):
        from flash3d.geometry.mesh import _simple_marching_cubes

        grid = np.random.rand(8, 8, 8)
        verts, faces = _simple_marching_cubes(grid, 0.5)
        assert verts.ndim == 2

    def test_save_mesh_obj(self, tmp_path):
        from flash3d.geometry.mesh import save_mesh_obj

        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]])
        path = str(tmp_path / "test.obj")
        save_mesh_obj(verts, faces, path)
        assert (tmp_path / "test.obj").exists()

    def test_save_mesh_obj_with_colors(self, tmp_path):
        from flash3d.geometry.mesh import save_mesh_obj

        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        save_mesh_obj(verts, faces, str(tmp_path / "c.obj"), vertex_colors=colors)
        assert (tmp_path / "c.obj").exists()

    def test_extract_mesh_marching_cubes(self):
        from flash3d.geometry.mesh import extract_mesh_marching_cubes

        def sphere_sdf(x):
            return 1.0 - x.norm(dim=-1)

        verts, faces = extract_mesh_marching_cubes(sphere_sdf, resolution=8, threshold=0.0)
        assert verts.ndim == 2


class TestTextureComprehensive:
    def test_spherical_projection(self):
        from flash3d.geometry.texture import UVMapper

        uvs = UVMapper.spherical_projection(np.random.randn(50, 3))
        assert uvs.shape == (50, 2)
        assert (uvs >= 0).all() and (uvs <= 1).all()

    def test_cylindrical_projection(self):
        from flash3d.geometry.texture import UVMapper

        uvs = UVMapper.cylindrical_projection(np.random.randn(30, 3))
        assert uvs.shape == (30, 2)

    def test_planar_projection_all_planes(self):
        from flash3d.geometry.texture import UVMapper

        v = np.random.randn(20, 3)
        for plane in ["xy", "xz", "yz"]:
            uvs = UVMapper.planar_projection(v, plane=plane)
            assert uvs.shape == (20, 2)

    def test_box_projection(self):
        from flash3d.geometry.texture import UVMapper

        v = np.random.randn(20, 3)
        n = np.random.randn(20, 3)
        uvs = UVMapper.box_projection(v, n)
        assert uvs.shape == (20, 2)

    def test_texture_atlas_pack(self):
        from flash3d.geometry.texture import TextureAtlas

        atlas = TextureAtlas(atlas_size=128)
        chart = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        ox, oy = atlas.pack_chart(chart)
        assert ox >= 0 and oy >= 0
        result = atlas.get_atlas()
        assert result.shape == (128, 128, 3)

    def test_atlas_full_raises(self):
        from flash3d.geometry.texture import TextureAtlas

        atlas = TextureAtlas(atlas_size=32)
        with pytest.raises(RuntimeError):
            for _ in range(100):
                atlas.pack_chart(np.zeros((32, 32, 3), dtype=np.uint8))


class TestPointCloudComprehensive:
    def test_creation(self):
        from flash3d.geometry.point_cloud import PointCloud

        pc = PointCloud(torch.randn(100, 3))
        assert pc.num_points == 100

    def test_with_colors(self):
        from flash3d.geometry.point_cloud import PointCloud

        pc = PointCloud(torch.randn(50, 3), torch.rand(50, 3))
        assert pc.colors.shape == (50, 3)

    def test_normalize(self):
        from flash3d.geometry.point_cloud import PointCloud

        pc = PointCloud(torch.randn(100, 3) * 10 + 5)
        pc_n = pc.normalize(target_radius=1.0)
        assert pc_n.points.norm(dim=-1).max() <= 1.0 + 1e-5

    def test_random_subsample(self):
        from flash3d.geometry.point_cloud import PointCloud

        pc = PointCloud(torch.randn(500, 3))
        sub = pc.random_subsample(50)
        assert sub.num_points == 50

    def test_voxel_downsample(self):
        from flash3d.geometry.point_cloud import PointCloud

        pc = PointCloud(torch.randn(1000, 3))
        down = pc.voxel_downsample(voxel_size=1.0)
        assert down.num_points < pc.num_points

    def test_bounding_box(self):
        from flash3d.geometry.point_cloud import PointCloud

        pc = PointCloud(torch.tensor([[0.0, 0, 0], [1.0, 2, 3]]))
        bb_min, bb_max = pc.bounding_box
        assert torch.allclose(bb_min, torch.tensor([0.0, 0.0, 0.0]))

    def test_depth_to_point_cloud(self):
        from flash3d.geometry.depth import depth_to_point_cloud

        depth = torch.rand(32, 32) * 5 + 0.5
        K = torch.tensor([[50.0, 0, 16], [0, 50.0, 16], [0, 0, 1.0]])
        pts, _ = depth_to_point_cloud(depth, K)
        assert pts.shape == (32 * 32, 3)

    def test_depth_metrics(self):
        from flash3d.geometry.depth import compute_depth_metrics

        pred = torch.rand(32, 32) * 5 + 1
        target = pred * 1.1
        m = compute_depth_metrics(pred, target)
        assert "abs_rel" in m and "rmse" in m and "delta1" in m

    def test_monocular_depth_estimator(self):
        from flash3d.geometry.depth import MonocularDepthEstimator

        model = MonocularDepthEstimator(base_channels=16)
        img = torch.rand(1, 3, 32, 32)
        depth = model(img)
        assert depth.shape == (1, 1, 32, 32)
        assert (depth > 0).all()


# ---------------------------------------------------------------------------
# 12. LoRA
# ---------------------------------------------------------------------------
class TestLoRAComprehensive:
    def test_lora_linear(self):
        from flash3d.models.lora import LoRALinear

        layer = LoRALinear(64, 128, rank=8)
        out = layer(torch.randn(4, 64))
        assert out.shape == (4, 128)

    def test_lora_merge(self):
        from flash3d.models.lora import LoRALinear

        layer = LoRALinear(32, 32, rank=4)
        x = torch.randn(2, 32)
        out_before = layer(x).detach()
        layer.merge_weights()
        out_after = layer.linear(x).detach()
        assert out_after.shape == out_before.shape


# ---------------------------------------------------------------------------
# 13. Tasks
# ---------------------------------------------------------------------------
class TestTasksComprehensive:
    def test_scene_reconstruction_task(self):
        from flash3d.tasks.scene_reconstruction import SceneReconstructionTask

        task = SceneReconstructionTask()
        assert task.config is not None

    def test_depth_prediction_task(self):
        from flash3d.tasks.depth_prediction import DepthPredictionTask

        task = DepthPredictionTask()
        task.setup()
        img = torch.rand(1, 3, 32, 32)
        depth = task.predict(img)
        assert depth.shape[0] == 1

    def test_depth_prediction_3d_input(self):
        from flash3d.tasks.depth_prediction import DepthPredictionTask

        task = DepthPredictionTask()
        img = torch.rand(3, 32, 32)
        depth = task.predict(img)
        assert depth.shape[0] == 1

    def test_depth_prediction_evaluate(self):
        from flash3d.tasks.depth_prediction import DepthPredictionTask

        task = DepthPredictionTask()
        pred = torch.rand(32, 32) * 5 + 1
        target = pred * 1.1
        metrics = task.evaluate(pred, target)
        assert "abs_rel" in metrics

    def test_tasks_registered(self):
        from flash3d.registry import TASKS

        assert "scene_reconstruction" in TASKS
        assert "depth_prediction" in TASKS


# ---------------------------------------------------------------------------
# 14. CLI
# ---------------------------------------------------------------------------
class TestCLIComprehensive:
    def test_build_parser(self):
        from flash3d.cli import build_parser

        parser = build_parser()
        assert parser is not None

    def test_version_command(self, capsys):
        import argparse

        from flash3d.cli import cmd_version

        cmd_version(argparse.Namespace())
        captured = capsys.readouterr()
        assert "Flash3D" in captured.out

    def test_parser_subcommands(self):
        from flash3d.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["version"])
        assert args.command == "version"


# ---------------------------------------------------------------------------
# 15. Flash3D model wrapper
# ---------------------------------------------------------------------------
class TestFlash3DModelComprehensive:
    def test_model_creation_gs(self):
        from flash3d.models.flash3d_model import Flash3D

        cfg = Flash3DConfig()
        cfg.model.name = "gaussian_splatting"
        cfg.model.num_gaussians = 20
        m = Flash3D(config=cfg)
        assert m.num_parameters > 0

    def test_save_load(self, tmp_path):
        from flash3d.models.flash3d_model import Flash3D

        cfg = Flash3DConfig()
        cfg.model.name = "gaussian_splatting"
        cfg.model.num_gaussians = 20
        m = Flash3D(config=cfg)
        ckpt = tmp_path / "test.pth"
        m.save_checkpoint(ckpt)
        assert ckpt.exists()
        loaded = Flash3D.from_pretrained(ckpt)
        assert loaded.num_parameters == m.num_parameters


# ---------------------------------------------------------------------------
# 16. Registry
# ---------------------------------------------------------------------------
class TestRegistryComprehensive:
    def test_register_and_get(self):
        from flash3d.registry import Registry

        reg = Registry("test_comp")

        @reg.register("item_a")
        class A:
            pass

        assert "item_a" in reg
        assert reg.get("item_a") is A

    def test_build(self):
        from flash3d.registry import Registry

        reg = Registry("test_build_comp")

        @reg.register("adder")
        class Adder:
            def __init__(self, v=0):
                self.v = v

        obj = reg.build("adder", v=5)
        assert obj.v == 5

    def test_duplicate_raises(self):
        from flash3d.registry import Registry

        reg = Registry("test_dup_comp")

        @reg.register("x")
        class X:
            pass

        with pytest.raises(KeyError):

            @reg.register("x")
            class Y:
                pass

    def test_missing_raises(self):
        from flash3d.registry import Registry

        reg = Registry("test_miss_comp")
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_global_registries(self):
        from flash3d.registry import DATASETS, MODELS, RENDERERS, TASKS

        assert MODELS.name == "models"
        assert RENDERERS.name == "renderers"
        assert DATASETS.name == "datasets"
        assert TASKS.name == "tasks"

    def test_models_populated(self):
        from flash3d.registry import MODELS

        assert "gaussian_splatting" in MODELS
        assert "nerf" in MODELS
        assert "feed_forward_3dgs" in MODELS
        assert "mip_nerf_360" in MODELS
        assert "gaussian_splatting_4d" in MODELS
        assert "instant_ngp_encoding" in MODELS
        assert "pointnet_pp" in MODELS


# ---------------------------------------------------------------------------
# 17. Integration: load scene → optimize → render
# ---------------------------------------------------------------------------
class TestIntegrationComprehensive:
    def test_gs_init_render(self):
        from flash3d.models.architectures.gaussian_splatting import GaussianSplatting
        from flash3d.rendering.rasterizer import rasterize_gaussians

        cfg = Flash3DConfig()
        cfg.model.num_gaussians = 5
        cfg.model.sh_degree = 0
        m = GaussianSplatting(config=cfg)
        pts = torch.randn(10, 3)
        cols = torch.rand(10, 3)
        m.initialize_from_point_cloud(pts, cols)

        result = rasterize_gaussians(
            means3d=m.means,
            scales=m.get_scales(),
            rotations=F.normalize(m.rotations, dim=-1),
            opacities=m.get_opacity(),
            sh_coeffs=m.sh_coeffs,
            viewmatrix=torch.eye(4),
            projmatrix=torch.eye(4),
            camera_center=torch.zeros(3),
            image_width=16,
            image_height=16,
            sh_degree=0,
        )
        assert result["rgb"].shape == (3, 16, 16)

    def test_nerf_query_volume_render(self):
        from flash3d.models.architectures.nerf import NeRF
        from flash3d.rendering.ray_marching import volume_render_rays

        m = NeRF(config=Flash3DConfig(), use_hash_encoding=False, num_layers=2, hidden_dim=32)
        t_vals = torch.linspace(0.1, 5, 16)
        pts = torch.zeros(1, 3).unsqueeze(1) + torch.tensor([0, 0, 1.0]).unsqueeze(0).unsqueeze(
            0
        ) * t_vals.unsqueeze(-1)
        flat = pts.reshape(-1, 3)
        dirs = torch.tensor([[0, 0, 1.0]]).expand(flat.shape[0], 3)
        density, rgb = m.query(flat, dirs)
        density = density.reshape(1, 16)
        rgb_r = rgb.reshape(1, 16, 3)
        result = volume_render_rays(density, rgb_r, t_vals)
        assert result["rgb"].shape == (1, 3)

    def test_depth_prediction_pipeline(self):
        from flash3d.geometry.depth import MonocularDepthEstimator, depth_to_point_cloud

        model = MonocularDepthEstimator(base_channels=16)
        img = torch.rand(1, 3, 32, 32)
        depth = model(img)
        depth_2d = depth.squeeze(0).squeeze(0)
        K = torch.tensor([[50.0, 0, 16], [0, 50.0, 16], [0, 0, 1.0]])
        pts, _ = depth_to_point_cloud(depth_2d, K)
        assert pts.shape[0] == 32 * 32
