"""Flash3D Command-Line Interface."""

from __future__ import annotations

import argparse
import sys

import flash3d


def cmd_version(args: argparse.Namespace) -> None:
    print(f"Flash3D v{flash3d.__version__}")


def cmd_settings(args: argparse.Namespace) -> None:
    import torch

    print(f"Flash3D v{flash3d.__version__}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")


def cmd_check(args: argparse.Namespace) -> None:
    """Verify installation and dependencies."""
    errors = []
    try:
        import torch  # noqa: F401
    except ImportError:
        errors.append("torch")
    try:
        import numpy  # noqa: F401
    except ImportError:
        errors.append("numpy")
    try:
        import cv2  # noqa: F401
    except ImportError:
        errors.append("opencv-python")
    try:
        import plyfile  # noqa: F401
    except ImportError:
        errors.append("plyfile")
    try:
        import trimesh  # noqa: F401
    except ImportError:
        errors.append("trimesh")

    if errors:
        print(f"[ERROR] Missing dependencies: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("[OK] All Flash3D dependencies are installed.")


def cmd_train(args: argparse.Namespace) -> None:
    """Launch training from a config file."""
    from flash3d.cfg.config import Flash3DConfig
    from flash3d.engine.trainer import Trainer

    config = Flash3DConfig.from_yaml(args.config)
    if args.output:
        config.output_dir = args.output
    trainer = Trainer(config)
    trainer.train()


def cmd_render(args: argparse.Namespace) -> None:
    """Render novel views from a trained model."""
    from flash3d.engine.predictor import Predictor

    predictor = Predictor.from_checkpoint(args.checkpoint)
    predictor.render_trajectory(
        output_dir=args.output or "renders/",
        num_frames=args.num_frames,
    )


def cmd_reconstruct(args: argparse.Namespace) -> None:
    """Reconstruct a 3D scene from images."""
    from flash3d.solutions.scene_reconstructor import SceneReconstructor

    reconstructor = SceneReconstructor(method=args.method)
    reconstructor.reconstruct(
        input_path=args.input,
        output_path=args.output or "reconstruction/",
    )


def cmd_depth(args: argparse.Namespace) -> None:
    """Estimate depth from images."""
    from flash3d.solutions.depth_estimator import DepthEstimator

    estimator = DepthEstimator()
    estimator.predict(
        input_path=args.input,
        output_path=args.output or "depth_output/",
    )


def cmd_export(args: argparse.Namespace) -> None:
    """Export a trained model to various formats."""
    from flash3d.engine.exporter import Exporter

    exporter = Exporter(checkpoint_path=args.checkpoint)
    exporter.export(
        format=args.format,
        output_path=args.output or "exported/",
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run benchmarks on trained models."""
    from flash3d.analytics.benchmark import Benchmark

    benchmark = Benchmark(checkpoint_path=args.checkpoint)
    results = benchmark.run(dataset_path=args.dataset)
    benchmark.print_results(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flash3d",
        description="Flash3D – 3D Vision CLI (Gaussian Splatting, NeRF, Depth, Reconstruction)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show Flash3D version")

    # settings
    subparsers.add_parser("settings", help="Show system settings and GPU info")

    # check
    subparsers.add_parser("check", help="Verify installation")

    # train
    train_p = subparsers.add_parser("train", help="Train a 3D model")
    train_p.add_argument("--config", "-c", required=True, help="Path to YAML config")
    train_p.add_argument("--output", "-o", help="Output directory")

    # render
    render_p = subparsers.add_parser("render", help="Render novel views")
    render_p.add_argument("--checkpoint", "-ckpt", required=True, help="Model checkpoint")
    render_p.add_argument("--output", "-o", help="Output directory")
    render_p.add_argument("--num-frames", type=int, default=120, help="Number of frames")

    # reconstruct
    recon_p = subparsers.add_parser("reconstruct", help="3D scene reconstruction")
    recon_p.add_argument("--input", "-i", required=True, help="Input images directory")
    recon_p.add_argument("--output", "-o", help="Output directory")
    recon_p.add_argument(
        "--method", default="gaussian_splatting",
        choices=["gaussian_splatting", "nerf", "feed_forward_3dgs"],
    )

    # depth
    depth_p = subparsers.add_parser("depth", help="Monocular depth estimation")
    depth_p.add_argument("--input", "-i", required=True, help="Input image or directory")
    depth_p.add_argument("--output", "-o", help="Output directory")

    # export
    export_p = subparsers.add_parser("export", help="Export model")
    export_p.add_argument("--checkpoint", "-ckpt", required=True, help="Model checkpoint")
    export_p.add_argument(
        "--format", "-f", default="ply",
        choices=["ply", "obj", "onnx", "splat"],
    )
    export_p.add_argument("--output", "-o", help="Output path")

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_p.add_argument("--checkpoint", "-ckpt", required=True, help="Model checkpoint")
    bench_p.add_argument("--dataset", "-d", required=True, help="Dataset path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "train": cmd_train,
        "render": cmd_render,
        "reconstruct": cmd_reconstruct,
        "depth": cmd_depth,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
