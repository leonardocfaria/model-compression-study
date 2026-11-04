# model-compression-study

Accuracy vs latency vs size on CPU for four compression techniques applied to one model:
- post-training **dynamic** quantisation;
- post-training **static** quantisation;
- **structured pruning**;
- **knowledge distillation** into smaller students.

It also covers **ONNX export** benchmarked with onnxruntime.

## Setup
```bash
cd ml/model-compression-study
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage
```bash
python -m pytest -q        # 5 tests, ~6 s
python study.py            # full study, ~1.5 min on 4 vCPUs → results/results.{md,json}
```

## Architecture
`compress/core.py` holds everything (≈170 lines); `study.py` runs the experiment.

| step | how |
|---|---|
| data | sklearn `digits` (bundled, nothing downloaded), upsampled 8×8 → 32×32, fixed 1437/360 split |
| "pretrained" teacher | 3 conv blocks (32/64/128 ch) + fc 256, 2.2 M params, trained 8 epochs here (Hugging Face/torchvision downloads are blocked) |
| PTQ dynamic | `quantize_dynamic` on `nn.Linear`: int8 weights, activations quantised on the fly |
| PTQ static | fuse conv+ReLU and linear+ReLU, x86 qconfig, calibrate on 256 training images, convert: int8 convs and linears |
| structured pruning | drop the 50 % lowest-L1 filters of every conv, slice the next layer's input channels and fc1's columns to match, then fine-tune 2 epochs. The result is a genuinely smaller dense model, not a mask |
| distillation | a student 4× narrower (and a tiny one 16× narrower), trained with `0.7·T²·KL(teacher‖student, T=4) + 0.3·CE`, against the same student trained from scratch |
| ONNX | legacy `torch.onnx.export` (needs only `onnx`), run with onnxruntime CPU, 1 thread |
| measurement | test accuracy (360 images); median latency over 50 runs at batch 1 and 64, **1 intra-op thread** everywhere; size = serialized bytes |

The tests check:
- pruning with ratio 0 reproduces the teacher's outputs exactly, which validates the weight surgery;
- 50 % pruning halves the channels;
- both int8 models are smaller and within 3 points of the teacher's accuracy;
- ONNX outputs match PyTorch within 1e-4;
- the KD loss reduces to cross-entropy at α=0.

## Results (4 vCPU Xeon with AVX-512 VNNI + AMX, torch 2.15, seed 0, 1 thread)
| model | runtime | test acc | size KB | ×smaller | latency b=1 ms | latency b=64 ms | ×faster (b=64) |
|---|---|---|---|---|---|---|---|
| fp32 teacher (baseline) | torch | 0.967 | 8569 | 1.0 | 2.105 | 65.43 | 1.00 |
| PTQ dynamic int8 (Linear) | torch | 0.964 | 2419 | 3.5 | 2.128 | 72.07 | 0.91 |
| PTQ static int8 (conv+linear) | torch | **0.967** | 2160 | 4.0 | 0.883 | 6.06 | **10.80** |
| pruned 50% channels, no fine-tune | torch | 0.472 | 4202 | 2.0 | 0.911 | 26.51 | 2.47 |
| pruned 50% channels + 2 ep fine-tune | torch | 0.956 | 4202 | 2.0 | 0.756 | 29.54 | 2.21 |
| student from scratch (4x narrower) | torch | 0.958 | 542 | 15.8 | 0.243 | 9.27 | 7.06 |
| student distilled from teacher | torch | 0.958 | 542 | 15.8 | 0.343 | 9.35 | 7.00 |
| tiny student from scratch (16x narrower) | torch | 0.886 | 38 | 224.0 | 0.081 | 3.14 | 20.85 |
| tiny student distilled | torch | 0.850 | 38 | 224.0 | 0.084 | 3.18 | 20.59 |
| fp32 teacher (baseline) | onnxruntime | 0.967 | 8567 | 1.0 | 0.854 | 38.64 | 1.69 |
| student distilled from teacher | onnxruntime | 0.958 | 540 | 15.9 | 0.098 | 4.83 | 13.56 |

Findings:
- **Static int8 is the clear winner here: no accuracy loss, 4× smaller, 10.8× faster at batch 64.** Part of that speed-up is this CPU's AMX int8 tile unit, which oneDNN uses for int8 convolutions. On a CPU with only AVX2 or VNNI, expect closer to 2-4×.
- **Dynamic quantisation only shrinks the model.** It quantises Linear layers, which hold 95 % of the bytes (fc1) but almost none of the FLOPs. The convolutions stay fp32, so it is not faster (0.91×, within noise). It suits Linear-heavy models such as transformers and LSTMs.
- **Structured pruning needs fine-tuning.** Removing half the filters drops accuracy to 47 %. Two epochs of fine-tuning bring it back to 95.6 % (-1.1 points) for 2× smaller and 2.2× faster. Speed-ups come from real channel removal; masks would give none.
- **Distillation did not help here.** The 4× student ties with training from scratch (0.958). The tiny 16× student does worse with KD (0.850 vs 0.886), because with α=0.7 and only 8 epochs a very small model fits the soft targets worse than the labels. KD usually pays off with more data, longer training or a tuned α/T. None of those were swept, to avoid tuning on the test set. The smaller *architecture* is what gives the 7-20× speed-ups.
- **ONNX Runtime beats PyTorch eager on the same fp32 graph**: 2.5× at batch 1 (0.85 vs 2.1 ms), 1.7× at batch 64, with identical accuracy. Distilled student + ORT is 13.6× faster than the teacher in PyTorch at -0.9 points.

Caveats: the test set has 360 images, so accuracy differences under about 1.5 points are noise. Latencies are single-thread medians on a shared VM (±10 %).

## Not done (time box)
Quantisation-aware training, int8 ONNX (onnxruntime quantisation), unstructured sparsity, an α/T sweep for KD on a validation split, a real pretrained checkpoint.
