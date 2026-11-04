import numpy as np
import pytest
import torch
import torch.nn.functional as F

from compress.core import (Net, dynamic_quant, evaluate, export_onnx, kd_loss, load_data, ort_session, prune_channels,
                           size_bytes, static_quant, train)


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def teacher(data):
    torch.manual_seed(0)
    return train(Net(8, 16, 32, 64), data[0], data[1], epochs=3)


def test_data_split_and_shapes(data):
    Xtr, ytr, Xte, yte = data
    assert Xtr.shape == (1437, 1, 32, 32) and Xte.shape == (360, 1, 32, 32)
    assert torch.equal(load_data()[1], ytr)


def test_kd_loss_reduces_to_ce_and_penalises_disagreement():
    s, y = torch.randn(4, 10), torch.tensor([1, 2, 3, 4])
    assert torch.allclose(kd_loss(s, y, s * 0 + 1, alpha=0), F.cross_entropy(s, y))
    assert kd_loss(s, y, s.clone(), alpha=1).item() == pytest.approx(0, abs=1e-6)  # identical logits: KL = 0
    assert kd_loss(s, y, -s, alpha=1) > 0


def test_prune_ratio_zero_is_exact_copy_and_half_halves_channels(teacher, data):
    x = data[2][:16]
    with torch.no_grad():
        assert torch.allclose(prune_channels(teacher, 0.0)(x), teacher(x), atol=1e-5)
    half = prune_channels(teacher, 0.5)
    assert [half.features.conv1.out_channels, half.features.conv2.out_channels, half.features.conv3.out_channels] == [4, 8, 16]
    assert sum(p.numel() for p in half.parameters()) < 0.6 * sum(p.numel() for p in teacher.parameters())


def test_quantised_models_are_smaller_and_accurate(teacher, data):
    acc = evaluate(teacher, data[2], data[3])
    assert acc > 0.85
    for q in (dynamic_quant(teacher), static_quant(teacher, data[0][:256])):
        assert size_bytes(q) < size_bytes(teacher)
        assert evaluate(q, data[2], data[3]) > acc - 0.03


def test_onnx_matches_torch(teacher, data, tmp_path):
    sess = ort_session(export_onnx(teacher, tmp_path / "m.onnx"))
    x = data[2][:8]
    with torch.no_grad():
        ref = teacher(x).numpy()
    assert np.allclose(sess.run(None, {"x": x.numpy()})[0], ref, atol=1e-4)
