"""Flash3D Exporter – Export trained models to various formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from flash3d.models.flash3d_model import Flash3D


class Exporter:
    """Export trained Flash3D models to PLY, OBJ, ONNX, or .splat formats."""

    def __init__(
        self,
        model: Flash3D | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            self.model = Flash3D.from_pretrained(checkpoint_path)
        else:
            raise ValueError("Either model or checkpoint_path must be provided")

        self.device = next(self.model.parameters()).device
        self.model.eval()

    def export(
        self,
        format: str = "ply",
        output_path: str | Path = "exported/",
        **kwargs: Any,
    ) -> Path:
        """Export model to specified format.

        Args:
            format: One of 'ply', 'obj', 'onnx', 'splat'.
            output_path: Output file or directory path.

        Returns:
            Path to the exported file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        exporters = {
            "ply": self._export_ply,
            "obj": self._export_obj,
            "onnx": self._export_onnx,
            "splat": self._export_splat,
        }

        if format not in exporters:
            raise ValueError(f"Unsupported format: {format}. Options: {list(exporters.keys())}")

        return exporters[format](output_path, **kwargs)

    def _export_ply(self, output_path: Path, **kwargs: Any) -> Path:
        """Export Gaussians to PLY format."""
        from plyfile import PlyData, PlyElement

        if output_path.suffix != ".ply":
            output_path = output_path / "model.ply"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        backbone = self.model.backbone
        if not hasattr(backbone, "means"):
            raise ValueError("PLY export requires a Gaussian Splatting model")

        means = backbone.means.detach().cpu().numpy()
        scales = backbone.get_scales().detach().cpu().numpy()
        rotations = backbone.rotations.detach().cpu().numpy()
        opacities = backbone.get_opacity().detach().cpu().numpy()
        sh_coeffs = backbone.sh_coeffs.detach().cpu().numpy()

        N = means.shape[0]
        num_sh = sh_coeffs.shape[1]

        dtype_list = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
            ("opacity", "f4"),
        ]
        for i in range(min(num_sh * 3, 48)):
            dtype_list.append((f"f_rest_{i}", "f4"))

        data = np.zeros(N, dtype=dtype_list)
        data["x"] = means[:, 0]
        data["y"] = means[:, 1]
        data["z"] = means[:, 2]
        data["scale_0"] = np.log(scales[:, 0])
        data["scale_1"] = np.log(scales[:, 1])
        data["scale_2"] = np.log(scales[:, 2])
        data["rot_0"] = rotations[:, 0]
        data["rot_1"] = rotations[:, 1]
        data["rot_2"] = rotations[:, 2]
        data["rot_3"] = rotations[:, 3]
        data["opacity"] = opacities.squeeze()

        sh_flat = sh_coeffs.reshape(N, -1)
        for i in range(min(sh_flat.shape[1], 48)):
            data[f"f_rest_{i}"] = sh_flat[:, i]

        vertex = PlyElement.describe(data, "vertex")
        PlyData([vertex]).write(str(output_path))

        return output_path

    def _export_obj(self, output_path: Path, **kwargs: Any) -> Path:
        """Export as mesh (extracting iso-surface from NeRF or Gaussians)."""
        from flash3d.geometry.mesh import extract_mesh_marching_cubes, save_mesh_obj

        if output_path.suffix != ".obj":
            output_path = output_path / "model.obj"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model.backbone, "query"):

            def query_fn(pts: torch.Tensor) -> torch.Tensor:
                dirs = torch.zeros_like(pts)
                dirs[:, 2] = 1.0
                density, _ = self.model.backbone.query(pts.to(self.device), dirs.to(self.device))
                return density.cpu()
        else:

            def query_fn(pts: torch.Tensor) -> torch.Tensor:
                return torch.zeros(pts.shape[0])

        vertices, faces = extract_mesh_marching_cubes(
            query_fn,
            resolution=kwargs.get("resolution", 128),
        )

        save_mesh_obj(vertices, faces, str(output_path))
        return output_path

    def _export_onnx(self, output_path: Path, **kwargs: Any) -> Path:
        """Export feed-forward model to ONNX format."""
        if output_path.suffix != ".onnx":
            output_path = output_path / "model.onnx"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(1, 3, 256, 256, device=self.device)
        torch.onnx.export(
            self.model,
            (None, dummy_input),
            str(output_path),
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}},
            opset_version=17,
        )
        return output_path

    def _export_splat(self, output_path: Path, **kwargs: Any) -> Path:
        """Export to .splat format for web viewers."""
        if output_path.suffix != ".splat":
            output_path = output_path / "model.splat"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        backbone = self.model.backbone
        if not hasattr(backbone, "means"):
            raise ValueError("Splat export requires a Gaussian Splatting model")

        means = backbone.means.detach().cpu().numpy().astype(np.float32)
        scales = np.log(backbone.get_scales().detach().cpu().numpy()).astype(np.float32)
        rotations = backbone.rotations.detach().cpu().numpy().astype(np.float32)
        opacities = backbone.get_opacity().detach().cpu().numpy().astype(np.float32)

        # .splat binary format: [x,y,z, scale_x,scale_y,scale_z, r,g,b,a, qw,qx,qy,qz] per Gaussian
        from flash3d.rendering.sh_utils import sh_to_rgb

        sh0 = backbone.sh_coeffs[:, 0].detach().cpu()
        colors = sh_to_rgb(sh0).clamp(0, 1).numpy().astype(np.float32)

        N = means.shape[0]
        with open(output_path, "wb") as f:
            for i in range(N):
                f.write(means[i].tobytes())
                f.write(scales[i].tobytes())
                rgb_a = np.array(
                    [colors[i, 0], colors[i, 1], colors[i, 2], opacities[i, 0]], dtype=np.float32
                )
                f.write(rgb_a.tobytes())
                f.write(rotations[i].tobytes())

        return output_path
