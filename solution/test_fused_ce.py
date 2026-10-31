"""Тест нашего ядра подсчёта ошибки: сверяем с обычным PyTorch.

Запуск: python -m pytest solution -q
"""

import pytest
import torch

pytest.importorskip("triton")

try:
    from solution.fused_ce import fused_cross_entropy
except ImportError:
    from fused_ce import fused_cross_entropy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_ce_matches_torch():
    torch.manual_seed(0)
    rows, vocab = 64, 4096 * 3 + 17        # словарь не кратен куску — проверяем маску на краю
    logits = torch.randn(rows, vocab, device="cuda", dtype=torch.float16)
    labels = torch.randint(0, vocab, (rows,), device="cuda")
    labels[3] = -100                        # позиция без ответа: должна быть проигнорирована

    ours = fused_cross_entropy(logits, labels)
    ref = torch.nn.functional.cross_entropy(logits.float(), labels, ignore_index=-100,
                                            reduction="mean")
    assert torch.allclose(ours, ref, atol=2e-2, rtol=2e-3), (ours.item(), ref.item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_ce_backward_matches_torch():
    torch.manual_seed(1)
    rows, vocab = 32, 4096 + 5
    logits = torch.randn(rows, vocab, device="cuda", dtype=torch.float16, requires_grad=True)
    labels = torch.randint(0, vocab, (rows,), device="cuda")

    loss = fused_cross_entropy(logits, labels)
    loss.backward()
    ours = logits.grad.clone()

    logits2 = logits.detach().clone().requires_grad_(True)
    ref = torch.nn.functional.cross_entropy(logits2.float(), labels, reduction="mean")
    ref.backward()

    assert torch.allclose(ours.float(), logits2.grad.float(), atol=2e-2, rtol=2e-2), \
        (ours.float() - logits2.grad.float()).abs().max().item()
