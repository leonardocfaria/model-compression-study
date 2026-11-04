# model-compression-study Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** A controlled study of accuracy vs latency vs size for the standard CPU compression techniques on one model, with an ONNX Runtime deployment check.
- **"Pretrained" model:** downloads are blocked (DECISIONS #3), so the teacher is trained here first: a 3-conv CNN on sklearn `digits` upsampled to 32×32 (bundled data, seeded; ~2.2 M params, mostly in fc1). That is the "pretrained model" every technique starts from.
- **Techniques:**
  1. PTQ dynamic: `quantize_dynamic` on Linear layers (int8 weights, activations quantised per batch).
  2. PTQ static: eager mode with fused conv+relu / linear+relu, the x86 qconfig, calibration on 256 training images, int8 convs and linears.
  3. Structured pruning: physically remove the lowest-L1 conv filters (50 %) and the matching input channels and fc1 columns, then fine-tune 2 epochs. It is a really smaller dense model, unlike masks.
  4. Knowledge distillation: a 4× narrower student trained with KL(T=4) + CE against the teacher, compared with the same student trained from scratch.
  Plus ONNX export (legacy exporter, `onnx` + `onnxruntime` only) of teacher and student, benchmarked against PyTorch eager.
- **Measurement:** test accuracy on the 360 held-out images; median latency at batch 1 and batch 64 with 1 intra-op thread (comparable, less noisy); size = serialized bytes.
- **Non-goals:** QAT, unstructured sparsity, int8 ONNX, GPU.

## Files
- `compress/core.py`: data, `Net`, `train` (optional KD), `evaluate`, the four techniques, `export_onnx`, `bench_latency`, `size_bytes`.
- `study.py`: runs everything → `results/results.{md,json}`.
- `tests/test_compress.py`.

## Tasks (TDD)
- [x] 1. data shapes/determinism; KD loss with alpha=0 equals CE.
- [x] 2. prune ratio 0 gives identical outputs (weight surgery is right); ratio 0.5 halves conv channels.
- [x] 3. dynamic/static quant: smaller and accuracy within 3 points of float on a quickly trained model.
- [x] 4. ONNX output ≈ torch output; study end to end; README; CI.
