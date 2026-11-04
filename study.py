"""Full study: train teacher, apply each technique, measure accuracy / latency / size, ONNX Runtime benchmark.

python study.py [--out results]     (~1.5 min on 4 vCPUs)
"""
import argparse
import json
import tempfile
from pathlib import Path

import torch

from compress.core import (Net, bench_latency, dynamic_quant, evaluate, export_onnx, load_data, ort_session,
                           prune_channels, size_bytes, static_quant, train)

STUDENT = dict(c1=8, c2=16, c3=32, hidden=64)  # 4x narrower than the teacher
TINY = dict(c1=2, c2=4, c3=8, hidden=16)  # 16x narrower


def main(out="results", epochs=8):
    torch.set_num_threads(1)  # single-thread latency: comparable across models, less noise
    Xtr, ytr, Xte, yte = load_data()
    torch.manual_seed(0)
    teacher = train(Net(), Xtr, ytr, epochs=epochs)

    models = {"fp32 teacher (baseline)": teacher,
              "PTQ dynamic int8 (Linear)": dynamic_quant(teacher),
              "PTQ static int8 (conv+linear)": static_quant(teacher, Xtr[:256])}
    pruned = prune_channels(teacher, 0.5)
    models["pruned 50% channels, no fine-tune"] = pruned
    models["pruned 50% channels + 2 ep fine-tune"] = train(prune_channels(teacher, 0.5), Xtr, ytr, epochs=2, lr=5e-4)
    torch.manual_seed(0)
    models["student from scratch (4x narrower)"] = train(Net(**STUDENT), Xtr, ytr, epochs=epochs)
    torch.manual_seed(0)
    models["student distilled from teacher"] = train(Net(**STUDENT), Xtr, ytr, epochs=epochs, teacher=teacher)
    torch.manual_seed(0)  # a much smaller student, where KD has more room to help
    models["tiny student from scratch (16x narrower)"] = train(Net(**TINY), Xtr, ytr, epochs=epochs)
    torch.manual_seed(0)
    models["tiny student distilled"] = train(Net(**TINY), Xtr, ytr, epochs=epochs, teacher=teacher)

    x1, x64 = Xte[:1], Xte[:64]
    rows = []
    with torch.inference_mode():
        for name, m in models.items():
            rows.append({"model": name, "runtime": "torch", "acc": evaluate(m, Xte, yte), "size_kb": size_bytes(m) / 1024,
                         "params": sum(p.numel() for p in m.parameters()) if not name.startswith("PTQ") else None,
                         "lat1_ms": bench_latency(m, x1), "lat64_ms": bench_latency(m, x64)})
    with tempfile.TemporaryDirectory() as d:
        for name in ("fp32 teacher (baseline)", "student distilled from teacher"):
            path = export_onnx(models[name], Path(d) / "m.onnx")
            sess = ort_session(path)
            run = lambda x: sess.run(None, {"x": x.numpy()})
            acc = (torch.tensor(run(Xte)[0]).argmax(1) == yte).float().mean().item()
            rows.append({"model": name, "runtime": "onnxruntime", "acc": acc, "size_kb": path.stat().st_size / 1024,
                         "params": None, "lat1_ms": bench_latency(run, x1), "lat64_ms": bench_latency(run, x64)})
    base = rows[0]
    for r in rows:
        r["speedup64"] = base["lat64_ms"] / r["lat64_ms"]
        r["compression"] = base["size_kb"] / r["size_kb"]

    out = Path(out)
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    lines = ["| model | runtime | test acc | size KB | ×smaller | latency b=1 ms | latency b=64 ms | ×faster (b=64) |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['runtime']} | {r['acc']:.3f} | {r['size_kb']:.0f} | {r['compression']:.1f} | "
              f"{r['lat1_ms']:.3f} | {r['lat64_ms']:.2f} | {r['speedup64']:.2f} |" for r in rows]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    main(p.parse_args().out)
