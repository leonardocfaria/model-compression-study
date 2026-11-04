"""Data, model, training (+KD), and the compression techniques: PTQ dynamic/static, structured pruning, ONNX."""
import copy
import io
import statistics
import time
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")  # no new dependency (torchao)
from torch.ao.quantization import DeQuantStub, QuantStub, convert, fuse_modules, get_default_qconfig, prepare, quantize_dynamic  # noqa: E402


def load_data(size: int = 32):
    """sklearn digits (bundled) upsampled to size×size, fixed 1437/360 split. Returns (Xtr, ytr, Xte, yte)."""
    from sklearn.datasets import load_digits

    d = load_digits()
    X = torch.tensor(d.images, dtype=torch.float32).unsqueeze(1) / 16
    X = F.interpolate(X, size=(size, size), mode="bilinear", align_corners=False)
    X = (X - 0.3) / 0.4
    y = torch.tensor(d.target)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(123))
    tr, te = perm[:1437], perm[1437:]
    return X[tr], y[tr], X[te], y[te]


class Net(nn.Module):
    """3 conv blocks + 2 linear. Quant/DeQuant stubs are identity until static quantisation converts them."""

    def __init__(self, c1=32, c2=64, c3=128, hidden=256):
        super().__init__()
        self.quant, self.dequant = QuantStub(), DeQuantStub()
        self.features = nn.Sequential(OrderedDict(
            conv1=nn.Conv2d(1, c1, 3, padding=1), relu1=nn.ReLU(),
            conv2=nn.Conv2d(c1, c2, 3, padding=1), relu2=nn.ReLU(), pool2=nn.MaxPool2d(2),
            conv3=nn.Conv2d(c2, c3, 3, padding=1), relu3=nn.ReLU(), pool3=nn.MaxPool2d(2)))
        self.head = nn.Sequential(OrderedDict(
            flat=nn.Flatten(), fc1=nn.Linear(c3 * 8 * 8, hidden), relu4=nn.ReLU(), fc2=nn.Linear(hidden, 10)))

    def forward(self, x):
        return self.dequant(self.head(self.features(self.quant(x))))


def kd_loss(student_logits, labels, teacher_logits=None, T: float = 4.0, alpha: float = 0.7):
    """alpha * T² * KL(teacher_T || student_T) + (1 - alpha) * CE. alpha=0 (or no teacher) is plain CE."""
    ce = F.cross_entropy(student_logits, labels)
    if teacher_logits is None or alpha == 0:
        return ce
    kl = F.kl_div(F.log_softmax(student_logits / T, 1), F.log_softmax(teacher_logits / T, 1), log_target=True, reduction="batchmean")
    return alpha * T * T * kl + (1 - alpha) * ce


def train(model, X, y, epochs=8, lr=1e-3, teacher=None, seed=0, batch=64):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(y)).split(batch):
            t_logits = None
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(X[idx])
            loss = kd_loss(model(X[idx]), y[idx], t_logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def evaluate(model, X, y) -> float:
    return (model(X).argmax(1) == y).float().mean().item()


def dynamic_quant(model):
    return quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)


def static_quant(model, calib_X):
    q = copy.deepcopy(model).eval()
    fuse_modules(q.features, [["conv1", "relu1"], ["conv2", "relu2"], ["conv3", "relu3"]], inplace=True)
    fuse_modules(q.head, [["fc1", "relu4"]], inplace=True)
    q.qconfig = get_default_qconfig("x86")
    prepare(q, inplace=True)
    with torch.no_grad():
        q(calib_X)  # observers record activation ranges
    return convert(q, inplace=True)


def prune_channels(model: Net, ratio: float) -> Net:
    """Structured pruning: keep the (1-ratio) conv filters with the largest L1 norm in each conv layer, and
    slice the next layer's input channels (and fc1's columns) to match. Returns a new, smaller dense Net."""
    f, h = model.features, model.head
    keep = []
    for conv in (f.conv1, f.conv2, f.conv3):
        n = max(1, round(conv.out_channels * (1 - ratio)))
        keep.append(conv.weight.abs().sum((1, 2, 3)).topk(n).indices.sort().values)
    small = Net(len(keep[0]), len(keep[1]), len(keep[2]), h.fc1.out_features)
    sf = small.features
    with torch.no_grad():
        prev = torch.arange(1)
        for src, dst, k in zip((f.conv1, f.conv2, f.conv3), (sf.conv1, sf.conv2, sf.conv3), keep):
            dst.weight.copy_(src.weight[k][:, prev])
            dst.bias.copy_(src.bias[k])
            prev = k
        cols = (keep[2][:, None] * 64 + torch.arange(64)).flatten()  # flatten order: channel-major, 8*8 positions
        small.head.fc1.weight.copy_(h.fc1.weight[:, cols])
        small.head.fc1.bias.copy_(h.fc1.bias)
        small.head.fc2.load_state_dict(h.fc2.state_dict())
    return small.eval()


def export_onnx(model, path, size=32):
    torch.onnx.export(model, (torch.zeros(1, 1, size, size),), str(path), dynamo=False, input_names=["x"],
                      output_names=["logits"], dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}})
    return path


def ort_session(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def bench_latency(fn, x, reps=50) -> float:
    """Median ms per call after warm-up."""
    for _ in range(5):
        fn(x)
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000


def size_bytes(model) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()
