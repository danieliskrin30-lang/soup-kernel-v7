"""Быстрый тест нашего ядра нормализации: сверяем с обычным кодом PyTorch.

Запуск: python -m pytest solution -q
Тест пропускается, если нет видеокарты (например, ты запустил его локально на Windows).
"""

import pytest
import torch

pytest.importorskip("triton")

try:  # из корня репозитория
    from solution.fused_rmsnorm import rmsnorm
except ImportError:  # если pytest запущен из папки solution
    from fused_rmsnorm import rmsnorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_rmsnorm_forward_matches_torch():
    torch.manual_seed(0)
    rows, cols = 128, 1536
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
    w = torch.randn(cols, device="cuda", dtype=torch.float16)

    x_ref = x.double()
    w_ref = w.double()
    ref = x_ref / torch.sqrt(x_ref.pow(2).mean(-1, keepdim=True) + 1e-6) * w_ref

    out = rmsnorm(x, w, 1e-6)

    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.allclose(out.double(), ref, atol=5e-2, rtol=5e-2), \
        (out.double() - ref).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_rmsnorm_backward_matches_torch():
    torch.manual_seed(1)
    rows, cols = 128, 1536
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(cols, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(rows, cols, device="cuda", dtype=torch.float16)

    out = rmsnorm(x, w, 1e-6)
    out.backward(grad)

    x_ref = x.double().detach().requires_grad_(True)
    w_ref = w.double().detach().requires_grad_(True)
    ref = x_ref / torch.sqrt(x_ref.pow(2).mean(-1, keepdim=True) + 1e-6) * w_ref
    ref.backward(grad.double())

    assert torch.allclose(x.grad.double(), x_ref.grad, atol=1e-1, rtol=1e-1), \
        (x.grad.double() - x_ref.grad).abs().max().item()
