"""Тест ядра внимания: сверяем с torch.nn.functional.scaled_dot_product_attention.

Запуск: python -m pytest solution -q
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

try:
    from solution.fused_attention import fused_sdpa
except ImportError:
    from fused_attention import fused_sdpa


def _make(b, h, kv, s, d, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b, h, s, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, kv, s, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, kv, s, d, device="cuda", dtype=torch.float16)
    return q, k, v


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
@pytest.mark.parametrize("b,h,kv,s,d", [(2, 4, 2, 128, 64), (2, 12, 2, 130, 128), (2, 3, 1, 97, 128)])
def test_attention_forward_matches_torch(b, h, kv, s, d):
    q, k, v = _make(b, h, kv, s, d)
    scale = d ** -0.5

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=scale)
    out = fused_sdpa(q, k, v, scale)

    err = (out.float() - ref.float()).abs().max().item()
    assert out.shape == ref.shape
    assert err < 2e-2, f"максимальная ошибка {err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_attention_backward_matches_torch():
    b, h, kv, s, d = 2, 6, 2, 96, 128
    q0, k0, v0 = _make(b, h, kv, s, d, seed=3)
    scale = d ** -0.5
    go = torch.randn_like(q0)

    qa = q0.clone().requires_grad_(True)
    ka = k0.clone().requires_grad_(True)
    va = v0.clone().requires_grad_(True)
    out = fused_sdpa(qa, ka, va, scale)
    out.backward(go)

    qb = q0.clone().requires_grad_(True)
    kb = k0.clone().requires_grad_(True)
    vb = v0.clone().requires_grad_(True)
    ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True, enable_gqa=True, scale=scale)
    ref.backward(go)

    for name, ours, theirs in (("dq", qa.grad, qb.grad), ("dk", ka.grad, kb.grad),
                               ("dv", va.grad, vb.grad)):
        scale_ref = theirs.float().abs().max().item() + 1e-6
        err = (ours.float() - theirs.float()).abs().max().item()
        assert err / scale_ref < 5e-2, f"{name}: ошибка {err / scale_ref:.3f} от масштаба"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_attention_with_mask_matches_torch():
    """Как в обучении: маска приходит готовая, длина не круглая (как у нас 235)."""
    b, h, kv, s, d = 2, 6, 2, 235, 128
    q0, k0, v0 = _make(b, h, kv, s, d, seed=7)
    scale = d ** -0.5
    causal = torch.tril(torch.ones(s, s, dtype=torch.bool, device="cuda"))
    mask4 = causal[None, None].expand(b, 1, s, s).contiguous()
    mask3 = mask4[:, 0].to(torch.uint8).contiguous()

    go = torch.randn_like(q0)
    qa, ka, va = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
    out = fused_sdpa(qa, ka, va, scale, mask3)
    out.backward(go)

    qb, kb, vb = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
    ref = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask4, enable_gqa=True, scale=scale)
    ref.backward(go)

    err = (out.float() - ref.float()).abs().max().item()
    assert err < 2e-2, f"прямой проход: {err}"

    for name, ours, theirs in (("dq", qa.grad, qb.grad), ("dk", ka.grad, kb.grad),
                               ("dv", va.grad, vb.grad)):
        scale_ref = theirs.float().abs().max().item() + 1e-6
        err = (ours.float() - theirs.float()).abs().max().item()
        assert err / scale_ref < 5e-2, f"{name}: ошибка {err / scale_ref:.3f} от масштаба"
