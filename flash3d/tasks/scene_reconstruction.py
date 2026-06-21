"""Scene Reconstruction task definition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flash3d.cfg.config import Flash3DConfig
from flash3d.registry import TASKS


@TASKS.register("scene_reconstruction")
class SceneReconstructionTask:
    """Full 3D scene reconstruction from multi-view images.

    Pipeline: images -> SfM/depth -> 3D representation -> mesh/point cloud.
    """

    def __init__(self, config: Flash3DConfig | None = None) -> None:
        self.config = config or Flash3DConfig()

    def reconstruct(
        self,
        images_path: str | Path,
        method: str = "gaussian_splatting",
        output_path: str | Path = "reconstruction/",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run full reconstruction pipeline.

        Args:
            images_path: Path to input images.
            method: Reconstruction method ('gaussian_splatting', 'nerf', 'feed_forward_3dgs').
            output_path: Where to save results.

        Returns:
            Dict with paths to outputs and metrics.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.config.model.name = method

        from flash3d.engine.trainer import Trainer
        from flash3d.models.flash3d_model import Flash3D

        model = Flash3D(config=self.config)
        trainer = Trainer(config=self.config, model=model)

        metrics = trainer.train(**kwargs)

        from flash3d.engine.exporter import Exporter

        exporter = Exporter(model=model)
        ply_path = exporter.export(format="ply", output_path=output_path / "model.ply")

        return {
            "model_path": str(ply_path),
            "metrics": metrics,
            "num_gaussians": model.backbone.num_points
            if hasattr(model.backbone, "num_points")
            else 0,
        }
