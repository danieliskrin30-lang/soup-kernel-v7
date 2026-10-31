"""Тест ядра выходного умножения на словарь: сверяем с обычным F.linear.

Запуск: python -m pytest solution -q
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

try:
    from solution.fused_lm_head import FusedLmHead
except ImportError:
    from fused_lm_head import FusedLmHead


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_lm_head_forward_matches_torch():
    torch.manual_seed(0)
    m, n, k = 128, 50000, 512
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)

    ours = FusedLmHead.apply(x, w)
    ref = F.linear(x, w)

    err = (ours.float() - ref.float()).abs().max().item()
    assert ours.shape == ref.shape
    assert err < 5e-1, f"максимальная ошибка {err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_lm_head_backward_matches_torch():
    torch.manual_seed(1)
    m, n, k = 96, 20480, 384
    x = torch.randn(m, k, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    go = torch.randn(m, n, device="cuda", dtype=torch.float16)

    out = FusedLmHead.apply(x, w)
    out.backward(go)
    ours = x.grad.clone()

    x2 = x.detach().clone().requires_grad_(True)
    ref = F.linear(x2, w)
    ref.backward(go)

    scale = ref.abs().max().item() + 1e-6
    err = (ours.float() - x2.grad.float()).abs().max().item()
    assert err / scale < 5e-2, f"dX: ошибка {err / scale:.3f} от масштаба"
