# FAQ

## General

**Q: What GPU do I need?**
A: Flash3D works with any CUDA-capable GPU. For training 3DGS with 100K+ Gaussians, we recommend >= 8GB VRAM. NeRF training benefits from >= 12GB.

**Q: Can I run Flash3D on CPU?**
A: Yes, all operations work on CPU. Training will be slow but inference is feasible for small models.

**Q: What scene formats are supported?**
A: COLMAP (sparse/0/), ScanNet, RealEstate10K, DL3DV, and any directory of images.

## Gaussian Splatting

**Q: How many Gaussians should I use?**
A: Start with the COLMAP point count. Adaptive density control will grow/prune automatically. Final counts typically range from 500K to 5M for real scenes.

**Q: Training is slow/diverging?**
A: Check learning rates. Position LR should be ~1.6e-4. Ensure proper scene normalization (points within [-1, 1]³ or similar scale).

**Q: How do I handle large scenes?**
A: Use voxel downsampling for initialization, increase densification threshold, and consider tiling the scene.

## NeRF

**Q: Hash encoding vs positional encoding?**
A: Hash encoding (instant-NGP style) is 10-100x faster. Use positional encoding only for research/comparison.

**Q: How to set near/far planes?**
A: Match your scene bounds. For indoor scenes: near=0.1, far=10. Outdoor: near=0.01, far=100.

## Depth Estimation

**Q: Are predictions metric or relative?**
A: The built-in model predicts metric depth. For relative depth, use `predict_disparity()`.

**Q: How to improve depth quality?**
A: Use higher resolution inputs, enable test-time augmentation, or fine-tune on domain-specific data with LoRA.

## Export

**Q: What formats can I export to?**
A: PLY (point cloud), OBJ (mesh), ONNX (neural network), .splat (web viewer).

**Q: How to view .splat files?**
A: Use any WebGL Gaussian Splatting viewer (e.g., antimatter15/splat, playcanvas).
