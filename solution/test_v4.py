"""Тесты v4: побитное совпадение с оригиналами там, где мы обещаем «то же самое».

Запуск на T4: python -m pytest solution -q
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
pytest.importorskip("triton")


# ---------- NF4 ----------

@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("shape", [(96, 1536), (256, 1536), (512, 8960)])
def test_nf4_bitwise_equal_bnb(compress, shape):
    F = pytest.importorskip("bitsandbytes.functional")
    from solution.fused_nf4 import nf4_dequantize

    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)
    packed, state = F.quantize_4bit(x, blocksize=64, compress_statistics=compress,
                                    quant_type="nf4")
    ref = F.dequantize_4bit(packed, state)
    ours = nf4_dequantize(packed, state)
    assert ours is not None and ours.dtype == ref.dtype and ours.shape == ref.shape
    diff = (ours.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, diff
    # обещание v4: побитно (если упадёт только это — значит FMA/порядок, допуск выше всё равно ок)
    assert torch.equal(ours, ref), f"не побитно, max diff {diff}"


def test_nf4_forward_fallback_matches_bnb():
    F = pytest.importorskip("bitsandbytes.functional")
    ops = pytest.importorskip("bitsandbytes.backends.cuda.ops")
    from solution import fused_nf4

    torch.manual_seed(1)
    w = torch.randn(1536, 512, device="cuda", dtype=torch.float16)
    packed, st = F.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4")
    A = torch.randn(940, 512, device="cuda", dtype=torch.float16)
    original = fused_nf4._ORIGINAL_FALLBACK or ops._dequant_linear_fallback
    fused_nf4._ORIGINAL_FALLBACK = original
    args = (A, packed.view(-1, 1), st.shape, st.state2.absmax, st.blocksize, st.quant_type)
    kw = dict(absmax_8bit=st.absmax, absmax_code=st.state2.code, absmax_offset=st.offset)
    ref = original(*args, None, **kw)
    ours = fused_nf4._patched_fallback(*args, None, **kw)
    assert torch.equal(ours, ref), (ours.float() - ref.float()).abs().max().item()


# ---------- CE ----------

def test_ce_none_matches_torch():
    from solution.fused_ce import _row_losses

    torch.manual_seed(2)
    rows, vocab = 33, 151936
    logits = torch.randn(4, rows + 1, vocab, device="cuda", dtype=torch.float16)
    labels = torch.randint(0, vocab, (rows,), device="cuda")
    labels[5] = -100
    view = logits[2, :-1, :]                         # как в Soup row_losses
    _, ours = _row_losses(view, labels, -100)
    ref = torch.nn.functional.cross_entropy(view.float(), labels, reduction="none",
                                            ignore_index=-100)
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-5)


def test_causal_lm_loss_matches_transformers():
    pytest.importorskip("transformers")
    from transformers.loss.loss_utils import ForCausalLMLoss
    from solution.fused_ce import _our_for_causal_lm_loss
    import solution.fused_ce as fce

    fce._ORIGINAL_LOSS = fce._ORIGINAL_LOSS or ForCausalLMLoss
    torch.manual_seed(3)
    B, T, V = 2, 37, 151936
    logits = (torch.randn(B, T, V, device="cuda") * 3).half().requires_grad_(True)
    labels = torch.randint(0, V, (B, T), device="cuda")
    labels[:, :5] = -100
    n = (labels[:, 1:] != -100).sum()

    ours = _our_for_causal_lm_loss(logits, labels, V, num_items_in_batch=n)
    (ours * 1024).backward()
    g_ours = logits.grad.clone()
    logits.grad = None
    ref = ForCausalLMLoss(logits, labels, V, num_items_in_batch=n)
    (ref * 1024).backward()
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-5)
    assert g_ours.dtype == torch.float16
    torch.testing.assert_close(g_ours.float(), logits.grad.float(), atol=2e-3, rtol=1e-2)


# ---------- RoPE ----------

def test_rope_bitwise_equal_transformers():
    pytest.importorskip("transformers")
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as ref_fn
    import solution.fused_rope as fr

    torch.manual_seed(4)
    B, L = 4, 235
    q0 = torch.randn(B, L, 12, 128, device="cuda").transpose(1, 2)
    k0 = torch.randn(B, L, 2, 128, device="cuda").transpose(1, 2)
    pos = torch.arange(L, device="cuda", dtype=torch.float32)
    inv = 1.0 / (1e6 ** (torch.arange(0, 128, 2, device="cuda").float() / 128))
    emb = torch.cat([pos[:, None] * inv[None], pos[:, None] * inv[None]], -1)
    cos, sin = emb.cos()[None].expand(B, -1, -1).contiguous(), emb.sin()[None].expand(B, -1, -1).contiguous()

    q1, k1 = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True)
    q2, k2 = q0.clone().requires_grad_(True), k0.clone().requires_grad_(True)
    rq, rk = ref_fn(q1, k1, cos, sin)
    fr._ORIGINAL = fr._ORIGINAL or ref_fn
    oq, ok = fr._patched_apply_rotary_pos_emb(q2, k2, cos, sin)
    assert torch.equal(oq, rq) and torch.equal(ok, rk)
    gq, gk = torch.randn_like(rq), torch.randn_like(rk)
    torch.autograd.backward([rq, rk], [gq, gk])
    torch.autograd.backward([oq, ok], [gq, gk])
    assert torch.equal(q1.grad, q2.grad) and torch.equal(k1.grad, k2.grad)


# ---------- MLP fp16 ----------

def _linear4(n_in, n_out):
    bnb = pytest.importorskip("bitsandbytes")
    lin = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                            quant_type="nf4", compress_statistics=True)
    w = (torch.randn(n_out, n_in) * 0.02).half()
    lin.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4",
                                   compress_statistics=True)
    return lin.cuda()


def test_mlp_fp16_bitwise_equal():
    from solution.memory_patches import mlp_forward_fp16
    from solution.swiglu import swiglu

    torch.manual_seed(5)
    mlp = torch.nn.Module()
    mlp.gate_proj, mlp.up_proj = _linear4(1536, 8960), _linear4(1536, 8960)
    mlp.down_proj = _linear4(8960, 1536)
    x0 = torch.randn(4, 235, 1536, device="cuda")
    g = torch.randn(4, 235, 1536, device="cuda")

    x1, x2 = x0.clone().requires_grad_(True), x0.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        ref = mlp.down_proj(swiglu(mlp.gate_proj(x1), mlp.up_proj(x1)))   # путь базы/v3
        ours = mlp_forward_fp16(mlp, x2)
    assert ours is not None
    assert torch.equal(ours.float(), ref.float())
    (ref.float() + 0).backward(g)
    (ours.float() + 0).backward(g)
    assert torch.equal(x1.grad, x2.grad), (x1.grad - x2.grad).abs().max().item()
