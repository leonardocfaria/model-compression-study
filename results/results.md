| model | runtime | test acc | size KB | ×smaller | latency b=1 ms | latency b=64 ms | ×faster (b=64) |
|---|---|---|---|---|---|---|---|
| fp32 teacher (baseline) | torch | 0.967 | 8569 | 1.0 | 2.105 | 65.43 | 1.00 |
| PTQ dynamic int8 (Linear) | torch | 0.964 | 2419 | 3.5 | 2.128 | 72.07 | 0.91 |
| PTQ static int8 (conv+linear) | torch | 0.967 | 2160 | 4.0 | 0.883 | 6.06 | 10.80 |
| pruned 50% channels, no fine-tune | torch | 0.472 | 4202 | 2.0 | 0.911 | 26.51 | 2.47 |
| pruned 50% channels + 2 ep fine-tune | torch | 0.956 | 4202 | 2.0 | 0.756 | 29.54 | 2.21 |
| student from scratch (4x narrower) | torch | 0.958 | 542 | 15.8 | 0.243 | 9.27 | 7.06 |
| student distilled from teacher | torch | 0.958 | 542 | 15.8 | 0.343 | 9.35 | 7.00 |
| tiny student from scratch (16x narrower) | torch | 0.886 | 38 | 224.0 | 0.081 | 3.14 | 20.85 |
| tiny student distilled | torch | 0.850 | 38 | 224.0 | 0.084 | 3.18 | 20.59 |
| fp32 teacher (baseline) | onnxruntime | 0.967 | 8567 | 1.0 | 0.854 | 38.64 | 1.69 |
| student distilled from teacher | onnxruntime | 0.958 | 540 | 15.9 | 0.098 | 4.83 | 13.56 |
