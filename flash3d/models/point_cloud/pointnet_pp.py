"""PointNet++ with set abstraction, feature propagation, and MSG grouping.

Implements PointNet++ (Qi et al., 2017) for point cloud classification and
segmentation with multi-scale grouping (MSG) and set abstraction layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash3d.registry import MODELS


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distance.

    Args:
        src: (B, N, 3) source points.
        dst: (B, M, 3) target points.

    Returns:
        (B, N, M) squared distances.
    """
    return (
        src.pow(2).sum(dim=-1, keepdim=True)
        + dst.pow(2).sum(dim=-1, keepdim=True).transpose(1, 2)
        - 2 * torch.bmm(src, dst.transpose(1, 2))
    )


def farthest_point_sample(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    """Farthest point sampling.

    Args:
        xyz: (B, N, 3) input points.
        n_points: Number of points to sample.

    Returns:
        (B, n_points) indices of sampled points.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_points, dtype=torch.long, device=device)
    distances = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(n_points):
        centroids[:, i] = farthest
        centroid_pts = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist = (xyz - centroid_pts).pow(2).sum(dim=-1)
        distances = torch.min(distances, dist)
        farthest = distances.argmax(dim=-1)

    return centroids


def query_ball_point(
    radius: float,
    n_sample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
) -> torch.Tensor:
    """Ball query: find all points within radius of query points.

    Args:
        radius: Search radius.
        n_sample: Max number of neighbors.
        xyz: (B, N, 3) all points.
        new_xyz: (B, S, 3) query points.

    Returns:
        (B, S, n_sample) indices of neighbors.
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device

    dists = square_distance(new_xyz, xyz)
    group_idx = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0).expand(B, S, N)
    group_idx = group_idx.clone()
    group_idx[dists > radius**2] = N

    group_idx = group_idx.sort(dim=-1).values[:, :, :n_sample]
    group_first = group_idx[:, :, 0:1].expand_as(group_idx)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index into points using indices.

    Args:
        points: (B, N, C) input features.
        idx: (B, ...) indices.

    Returns:
        (B, ..., C) indexed features.
    """
    B = points.shape[0]
    device = points.device
    batch_idx = torch.arange(B, device=device).view(B, *([1] * (idx.dim() - 1)))
    batch_idx = batch_idx.expand_as(idx)
    return points[batch_idx, idx]


class PointNetSetAbstraction(nn.Module):
    """Set Abstraction layer from PointNet++.

    Performs: sampling -> grouping -> per-point PointNet (shared MLPs) -> max pool.
    """

    def __init__(
        self,
        n_point: int,
        radius: float,
        n_sample: int,
        in_channel: int,
        mlp_channels: list[int],
        group_all: bool = False,
    ) -> None:
        super().__init__()
        self.n_point = n_point
        self.radius = radius
        self.n_sample = n_sample
        self.group_all = group_all

        self.mlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp_channels:
            self.mlps.append(nn.Conv2d(last_channel, out_channel, 1))
            self.bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3) coordinates.
            points: (B, N, D) features (or None).

        Returns:
            new_xyz: (B, n_point, 3) sampled centroids.
            new_points: (B, n_point, D') abstracted features.
        """
        if self.group_all:
            new_xyz, new_points = self._sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self._sample_and_group(xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlps, self.bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = new_points.max(dim=2).values
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points

    def _sample_and_group(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fps_idx = farthest_point_sample(xyz, self.n_point)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.n_sample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)

        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz

        return new_xyz, new_points

    def _sample_and_group_all(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
        grouped_xyz = xyz.unsqueeze(1)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points


class PointNetSetAbstractionMSG(nn.Module):
    """Multi-Scale Grouping (MSG) set abstraction.

    Uses multiple radii and grouping sizes for multi-scale feature extraction.
    """

    def __init__(
        self,
        n_point: int,
        radius_list: list[float],
        n_sample_list: list[int],
        in_channel: int,
        mlp_channels_list: list[list[int]],
    ) -> None:
        super().__init__()
        self.n_point = n_point
        self.radius_list = radius_list
        self.n_sample_list = n_sample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for mlp_channels in mlp_channels_list:
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_channels:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fps_idx = farthest_point_sample(xyz, self.n_point)
        new_xyz = index_points(xyz, fps_idx)

        new_points_list = []
        for i, (radius, n_sample) in enumerate(zip(self.radius_list, self.n_sample_list)):
            idx = query_ball_point(radius, n_sample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)
            for conv, bn in zip(self.conv_blocks[i], self.bn_blocks[i]):
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = grouped_points.max(dim=2).values
            new_points_list.append(new_points)

        new_points = torch.cat(new_points_list, dim=1)
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class FeaturePropagation(nn.Module):
    """Feature propagation layer for PointNet++ segmentation decoder.

    Interpolates features from subsampled points back to original resolution.
    """

    def __init__(self, in_channel: int, mlp_channels: list[int]) -> None:
        super().__init__()
        self.mlps = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_channels:
            self.mlps.append(nn.Conv1d(last_channel, out_channel, 1))
            self.bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor | None,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate features from xyz2 to xyz1 via inverse-distance interpolation.

        Args:
            xyz1: (B, N, 3) target points.
            xyz2: (B, S, 3) source points (S < N).
            points1: (B, N, D1) existing features at target (or None).
            points2: (B, S, D2) features at source to propagate.

        Returns:
            (B, N, D_out) propagated features.
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.expand(-1, N, -1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            weights = dist_recip / dist_recip.sum(dim=-1, keepdim=True)
            interpolated = (index_points(points2, idx) * weights.unsqueeze(-1)).sum(dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlps, self.bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)


@MODELS.register("pointnet_pp")
class PointNetPP(nn.Module):
    """PointNet++ backbone with MSG set abstraction layers.

    Extracts hierarchical point cloud features via progressive downsampling.
    """

    def __init__(
        self,
        in_channel: int = 3,
        use_msg: bool = True,
    ) -> None:
        super().__init__()
        self.use_msg = use_msg

        if use_msg:
            self.sa1 = PointNetSetAbstractionMSG(
                512,
                [0.1, 0.2, 0.4],
                [16, 32, 128],
                in_channel,
                [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            )
            self.sa2 = PointNetSetAbstractionMSG(
                128,
                [0.2, 0.4, 0.8],
                [32, 64, 128],
                320,
                [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
            )
            self.sa3 = PointNetSetAbstraction(
                n_point=0,
                radius=0,
                n_sample=0,
                in_channel=640,
                mlp_channels=[256, 512, 1024],
                group_all=True,
            )
        else:
            self.sa1 = PointNetSetAbstraction(
                512,
                0.2,
                32,
                in_channel,
                [64, 64, 128],
            )
            self.sa2 = PointNetSetAbstraction(
                128,
                0.4,
                64,
                128,
                [128, 128, 256],
            )
            self.sa3 = PointNetSetAbstraction(
                0,
                0,
                0,
                256,
                [256, 512, 1024],
                group_all=True,
            )

    @property
    def output_dim(self) -> int:
        return 1024

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """Extract hierarchical features.

        Args:
            xyz: (B, N, 3) input coordinates.
            features: (B, N, D) input features (optional).

        Returns:
            global_feat: (B, 1024) global feature vector.
            xyz3: (B, 1, 3) final centroid.
            intermediates: list of (xyz, features) at each level.
        """
        intermediates = [(xyz, features)]

        l1_xyz, l1_points = self.sa1(xyz, features)
        intermediates.append((l1_xyz, l1_points))

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        intermediates.append((l2_xyz, l2_points))

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        intermediates.append((l3_xyz, l3_points))

        global_feat = l3_points.squeeze(1)
        return global_feat, l3_xyz, intermediates


@MODELS.register("pointnet_pp_cls")
class PointNetPPClassifier(nn.Module):
    """PointNet++ classifier for point cloud classification."""

    def __init__(self, num_classes: int = 40, in_channel: int = 3, use_msg: bool = True) -> None:
        super().__init__()
        self.backbone = PointNetPP(in_channel=in_channel, use_msg=use_msg)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        global_feat, _, _ = self.backbone(xyz, features)
        return self.classifier(global_feat)


@MODELS.register("pointnet_pp_seg")
class PointNetPPSegmentor(nn.Module):
    """PointNet++ segmentor for point cloud semantic segmentation."""

    def __init__(self, num_classes: int = 13, in_channel: int = 3) -> None:
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(0, 0, 0, 256, [256, 512, 1024], group_all=True)

        self.fp3 = FeaturePropagation(1280, [256, 256])
        self.fp2 = FeaturePropagation(384, [256, 128])
        self.fp1 = FeaturePropagation(128 + in_channel, [128, 128, 128])

        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        l0_xyz, l0_points = xyz, features
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        if l0_points is not None:
            l0_cat = l0_points
        else:
            l0_cat = None
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_cat, l1_points)

        x = l0_points.permute(0, 2, 1)
        x = self.classifier(x)
        return x.permute(0, 2, 1)
