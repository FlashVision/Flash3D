"""Example: Benchmarking 3D models.

Demonstrates running benchmarks across different methods and datasets.
"""

import torch
from flash3d import Flash3D, Benchmark
from flash3d.cfg.config import Flash3DConfig
from flash3d.analytics.metrics import compute_psnr, compute_ssim, compute_chamfer_distance


def main():
    print("=" * 60)
    print("  Flash3D Benchmarking Example")
    print("=" * 60)

    # Compute image quality metrics
    print("\n[1] Image Quality Metrics:")
    pred_image = torch.rand(3, 256, 256)
    target_image = pred_image + 0.05 * torch.randn_like(pred_image)
    target_image = target_image.clamp(0, 1)

    psnr = compute_psnr(pred_image, target_image)
    ssim = compute_ssim(pred_image, target_image)
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")

    # Compute point cloud metrics
    print("\n[2] Point Cloud Metrics:")
    pc_pred = torch.randn(1000, 3)
    pc_target = pc_pred + 0.02 * torch.randn_like(pc_pred)

    chamfer = compute_chamfer_distance(pc_pred, pc_target)
    print(f"  Chamfer Distance: {chamfer:.6f}")

    # Benchmark a model
    print("\n[3] Model Benchmark:")
    config = Flash3DConfig()
    config.model.name = "gaussian_splatting"
    config.model.num_gaussians = 1000
    model = Flash3D(config=config)

    benchmark = Benchmark(model=model, device="cpu")
    results = benchmark.run(num_samples=5)
    benchmark.print_results(results)

    # Compare methods
    print("[4] Method Comparison:")
    for method in ["gaussian_splatting", "nerf"]:
        config.model.name = method
        model = Flash3D(config=config)
        print(f"  {method}: {model.num_parameters:,} parameters")


if __name__ == "__main__":
    main()
