"""Тест ядра энтропии: сверяем с формулой PyTorch.

Запуск: python -m pytest solution -q
"""

import pytest
import torch

pytest.importorskip("triton")

try:
    from solution.fused_entropy import entropy_from_logits
except ImportError:
    from fused_entropy import entropy_from_logits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_entropy_matches_torch():
    torch.manual_seed(0)
    rows, vocab = 64, 4096 * 9 + 17          # словарь не кратен куску — проверяем край
    logits = torch.randn(rows, vocab, device="cuda", dtype=torch.float32)

    ours = entropy_from_logits(logits)

    log_probs = torch.log_softmax(logits, dim=-1)
    p = log_probs.exp()
    ref = -(p * log_probs).sum(-1)

    err = (ours - ref).abs().max().item()
    assert ours.shape == ref.shape
    assert err < 5e-2, f"максимальная ошибка {err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_entropy_3d_shape_matches():
    """В тренере логиты приходят трёхмерными [batch, seq, vocab]."""
    torch.manual_seed(1)
    b, s, vocab = 2, 7, 8192 * 3 + 5
    logits = torch.randn(b, s, vocab, device="cuda", dtype=torch.float32)

    ours = entropy_from_logits(logits)
    log_probs = torch.log_softmax(logits, dim=-1)
    ref = -(log_probs.exp() * log_probs).sum(-1)

    assert ours.shape == (b, s)
    assert (ours - ref).abs().max().item() < 5e-2
