# model-compression-study

## Goal
Build "Model compression: quantisation, pruning, distillation". Take a pretrained small vision or text model and measure accuracy vs latency vs size for post-training dynamic/static quantisation, structured pruning, and knowledge distillation into a smaller student; export to ONNX and benchmark with onnxruntime. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [ ] four techniques implemented
- [ ] results table (accuracy, latency, size) committed
- [ ] ONNX export benchmarked
- [ ] Tests pass with one command; CI workflow runs them
- [ ] README with setup, usage, architecture and results
- [ ] Self code review done
- [ ] Draft PR open

