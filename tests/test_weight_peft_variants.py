"""Numerical regression for the paper OFT implementation."""
import torch

from llamafactory.model.model_utils.weight_peft import WeightOFT, _cayley_R


def _indices(block):
    tri = torch.triu_indices(block, block, 1)
    return tri[0], tri[1]


def test_exact_cayley_matches_hf_right_solve():
    block = 4
    rows, cols = _indices(block)
    skew = torch.randn(3, rows.numel()) * 0.05
    got = _cayley_R(skew, block, rows, cols, torch.float32)
    q = torch.zeros(3, block, block)
    q[:, rows, cols] = skew
    q = q - q.transpose(-1, -2)
    eye = torch.eye(block).expand_as(q)
    expected = torch.linalg.solve(eye + q, eye - q, left=False)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.transpose(-1, -2) @ got, eye, atol=1e-5, rtol=1e-5)


def test_oft_projection_identity_rotation_and_gradients():
    weight = torch.randn(5, 4)
    adapter = WeightOFT(weight, block_size=2)
    assert torch.allclose(adapter(weight), weight)
    adapter.skew.data.fill_(0.05)
    rows, cols = _indices(2)
    rotation = _cayley_R(adapter.skew, 2, rows, cols, weight.dtype)
    block_rotation = torch.block_diag(*rotation.unbind())
    x = torch.randn(7, 4)
    assert torch.allclose(x @ adapter(weight).T, (x @ block_rotation) @ weight.T, atol=1e-5)
    adapter(weight).square().sum().backward()
    assert adapter.skew.grad is not None
    assert torch.isfinite(adapter.skew.grad).all()
