"""Тесты v5 2+3: пролог внимания, SDPA is_causal на cutlass, guard правого паддинга.

Запуск на T4: python -m pytest solution -q
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
pytest.importorskip("triton")
mq = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")

B, HQ, HK, D = 4, 12, 2, 128


def _rope_inputs(L, seed=0):
    torch.manual_seed(seed)
    qb = torch.randn(B, L, HQ, D, device="cuda")
    kb = torch.randn(B, L, HK, D, device="cuda")
    vb = torch.randn(B, L, HK, D, device="cuda")
    pos = torch.arange(L, device="cuda", dtype=torch.float32)
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    emb = torch.cat([pos[:, None] * inv[None], pos[:, None] * inv[None]], -1)
    cos = emb.cos()[None].expand(B, -1, -1).contiguous()
    sin = emb.sin()[None].expand(B, -1, -1).contiguous()
    return qb, kb, vb, cos, sin


def _reference(q, k, v, cos, sin):
    """База v4/transformers: RoPE fp32 -> repeat_kv fp32 -> приведение к fp16 (autocast)."""
    rq, rk = mq.apply_rotary_pos_emb(q, k, cos, sin)
    return rq.half(), mq.repeat_kv(rk, HQ // HK).half(), mq.repeat_kv(v, HQ // HK).half()


@pytest.mark.parametrize("L", [235, 334, 17])
def test_prologue_forward_bitwise(L):
    from solution.fused_attn_prologue import attn_prologue

    qb, kb, vb, cos, sin = _rope_inputs(L)
    q, k, v = qb.transpose(1, 2), kb.transpose(1, 2), vb.transpose(1, 2)   # как в Qwen2
    ref = _reference(q, k, v, cos, sin)
    ours = attn_prologue(q, k, v, cos, sin)
    for name, a, b in zip("qkv", ours, ref):
        assert a.dtype == torch.float16 and a.shape == b.shape
        assert torch.equal(a, b), (name, (a.float() - b.float()).abs().max().item())


@pytest.mark.parametrize("L", [235, 41])
def test_prologue_backward(L):
    from solution.fused_attn_prologue import attn_prologue

    qb, kb, vb, cos, sin = _rope_inputs(L, seed=1)
    g = [torch.randn(B, HQ, L, D, device="cuda").half() for _ in range(3)]

    leaves_ref = [t.clone().requires_grad_(True) for t in (qb, kb, vb)]
    out_ref = _reference(*(t.transpose(1, 2) for t in leaves_ref), cos, sin)
    torch.autograd.backward(out_ref, g)

    leaves = [t.clone().requires_grad_(True) for t in (qb, kb, vb)]
    out = attn_prologue(*(t.transpose(1, 2) for t in leaves), cos, sin)
    torch.autograd.backward(out, g)

    assert torch.equal(leaves[0].grad, leaves_ref[0].grad), "dq должен совпасть побитно"
    for i, name in ((1, "dk"), (2, "dv")):      # сумма по 6 головам — порядок может отличаться
        torch.testing.assert_close(leaves[i].grad, leaves_ref[i].grad, atol=1e-5, rtol=1e-6,
                                   msg=name)


def test_backend_is_cutlass_not_math():
    """EFFICIENT_ATTENTION (fmha_cutlass) доступен для наших форм; math-ядер (softmax/bmm) нет."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    q = torch.randn(B, HQ, 235, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                                   scale=D ** -0.5)
        out.sum().backward()
        torch.cuda.synchronize()
    names = [e.key.lower() for e in prof.key_averages()]
    assert any("fmha" in n or "efficient" in n for n in names), names
    assert not any("softmax" in n for n in names), names


def _layer(L, seed=3):
    torch.manual_seed(seed)
    cfg = mq.Qwen2Config(hidden_size=HQ * D, num_attention_heads=HQ, num_key_value_heads=HK,
                         head_dim=D, intermediate_size=8960, num_hidden_layers=1,
                         max_position_embeddings=4096)
    cfg._attn_implementation = "sdpa"
    attn = mq.Qwen2Attention(cfg, layer_idx=0).cuda().train()
    # Как в обучении: Linear4bit (+LoRA) считает в fp16 под autocast и возвращает fp32
    for name in ("q_proj", "k_proj", "v_proj"):
        lin = getattr(attn, name)
        lin.forward = (lambda x, lin=lin:
                       torch.nn.functional.linear(x, lin.weight, lin.bias).float())
    rot = mq.Qwen2RotaryEmbedding(config=cfg).cuda()
    lens = [L, L - 17, L - 50, 30]
    am = torch.zeros(B, L, dtype=torch.long, device="cuda")
    for i, n in enumerate(lens):
        am[i, :n] = 1                                         # паддинг справа, как в TRL
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device="cuda"))
    mask4d = (causal[None, None] & am.bool()[:, None, None, :])          # True = смотреть
    hs = torch.randn(B, L, HQ * D, device="cuda")
    pos = torch.arange(L, device="cuda")[None].expand(B, -1)
    return attn, rot, am, mask4d, hs, pos


@pytest.mark.parametrize("L", [235, 129])
def test_layer_matches_stock_sdpa_on_real_rows(L):
    import solution.fused_attn_prologue as fp

    attn, rot, am, mask4d, hs0, pos = _layer(L)
    original = fp._ORIGINAL_ATTN_FORWARD or mq.Qwen2Attention.forward
    fp._ORIGINAL_ATTN_FORWARD = original
    real = am.bool()[..., None].float()
    w = torch.randn_like(hs0)

    def run(fast):
        hs = hs0.clone().requires_grad_(True)
        attn.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pe = rot(hs, pos)
            if fast:
                fp.STATE.update(enabled=True, causal_ok=True, backend_error=None)
                out, _ = fp._attn_forward(attn, hs, pe, mask4d)
                fp.STATE.update(enabled=False, causal_ok=False)
            else:
                out, _ = original(attn, hs, pe, mask4d)
        (out.float() * w * real).sum().backward()
        return out.float() * real, hs.grad * real, attn.q_proj.weight.grad.clone()

    fast_before = fp.STATE["fast"]
    o_ref, g_ref, wq_ref = run(False)
    o, g, wq = run(True)
    assert fp.STATE["fast"] == fast_before + 1 and fp.STATE["backend_error"] is None

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()
    assert torch.isfinite(o).all() and torch.isfinite(g).all()
    assert rel(o, o_ref) < 2e-3, rel(o, o_ref)
    assert rel(g, g_ref) < 5e-3, rel(g, g_ref)
    assert rel(wq, wq_ref) < 5e-3, rel(wq, wq_ref)


def test_guard_right_padding():
    from solution.fused_attn_prologue import right_padded

    right = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 0]], device="cuda")
    left = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], device="cuda")
    holes = torch.tensor([[1, 0, 1, 0]], device="cuda")
    assert right_padded(right) is True
    assert right_padded(None) is True
    assert right_padded(left) is False
    assert right_padded(holes) is False
    assert right_padded(right, position_ids=torch.arange(4, device="cuda")[None]) is False
