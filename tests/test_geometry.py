"""Tests for Flash3D geometry module."""

import pytest
import torch
import numpy as np

from flash3d.geometry.point_cloud import PointCloud
from flash3d.geometry.depth import MonocularDepthEstimator, depth_to_point_cloud, compute_depth_metrics
from flash3d.geometry.transforms_3d import (
    SE3,
    rotation_matrix_from_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    look_at,
)


class TestPointCloud:
    def test_creation(self):
        points = torch.randn(100, 3)
        pc = PointCloud(points)
        assert pc.num_points == 100

    def test_with_colors(self):
        points = torch.randn(50, 3)
        colors = torch.rand(50, 3)
        pc = PointCloud(points, colors)
        assert pc.colors is not None
        assert pc.colors.shape == (50, 3)

    def test_normalize(self):
        points = torch.randn(100, 3) * 10 + 5
        pc = PointCloud(points)
        pc_norm = pc.normalize(target_radius=1.0)
        assert pc_norm.points.norm(dim=-1).max() <= 1.0 + 1e-5

    def test_subsample(self):
        points = torch.randn(1000, 3)
        pc = PointCloud(points)
        pc_sub = pc.random_subsample(100)
        assert pc_sub.num_points == 100

    def test_voxel_downsample(self):
        points = torch.randn(1000, 3)
        pc = PointCloud(points)
        pc_down = pc.voxel_downsample(voxel_size=1.0)
        assert pc_down.num_points < pc.num_points

    def test_bounding_box(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        pc = PointCloud(points)
        bb_min, bb_max = pc.bounding_box
        assert torch.allclose(bb_min, torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(bb_max, torch.tensor([1.0, 2.0, 3.0]))


class TestDepth:
    def test_monocular_estimator(self):
        model = MonocularDepthEstimator(base_channels=16)
        image = torch.rand(1, 3, 64, 64)
        depth = model(image)
        assert depth.shape == (1, 1, 64, 64)
        assert (depth > 0).all()

    def test_depth_to_point_cloud(self):
        depth = torch.rand(64, 64) * 5 + 0.5
        intrinsics = torch.tensor([
            [50.0, 0.0, 32.0],
            [0.0, 50.0, 32.0],
            [0.0, 0.0, 1.0],
        ])
        points, _ = depth_to_point_cloud(depth, intrinsics)
        assert points.shape == (64 * 64, 3)

    def test_depth_metrics(self):
        pred = torch.rand(64, 64) * 5 + 1
        target = pred * 1.1
        metrics = compute_depth_metrics(pred, target)
        assert "abs_rel" in metrics
        assert "rmse" in metrics
        assert "delta1" in metrics
        assert metrics["abs_rel"] >= 0


class TestTransforms3D:
    def test_se3_identity(self):
        se3 = SE3.identity()
        assert torch.allclose(se3.matrix, torch.eye(4))

    def test_se3_inverse(self):
        R = rotation_matrix_from_euler(0.1, 0.2, 0.3)
        t = torch.tensor([1.0, 2.0, 3.0])
        se3 = SE3.from_rotation_translation(R, t)
        se3_inv = se3.inverse()
        identity = se3.compose(se3_inv)
        assert torch.allclose(identity.matrix, torch.eye(4), atol=1e-5)

    def test_se3_transform_points(self):
        se3 = SE3.identity()
        points = torch.randn(10, 3)
        transformed = se3.transform_points(points)
        assert torch.allclose(transformed, points, atol=1e-6)

    def test_quaternion_roundtrip(self):
        R = rotation_matrix_from_euler(0.5, -0.3, 1.0)
        q = rotation_matrix_to_quaternion(R)
        R_recovered = quaternion_to_rotation_matrix(q)
        assert torch.allclose(R, R_recovered, atol=1e-4)

    def test_rotation_orthogonal(self):
        R = rotation_matrix_from_euler(1.0, 2.0, 3.0)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)

    def test_look_at(self):
        eye = torch.tensor([0.0, 0.0, -3.0])
        center = torch.zeros(3)
        mat = look_at(eye, center)
        assert mat.shape == (4, 4)
