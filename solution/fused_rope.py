"""Своё ядро поворотных позиционных эмбеддингов (RoPE) для Qwen2 на Triton.

Оригинал (transformers 5.16.1, modeling_qwen2.apply_rotary_pos_emb):
    q_embed = q * cos + rotate_half(q) * sin,   rotate_half(x) = cat(-x2, x1)
то же для k. В нашем запуске q, k, cos, sin — fp32 (остаточный поток в fp32 после
peft.prepare_model_for_kbit_training), q — транспонированное представление [B, 12, L, 128].
PyTorch на это запускает ~10 кернелов вперёд и ~15 назад на слой (срезы, neg, cat, mul, add),
каждый гоняет 5,8 МБ через память. У нас — один кернел вперёд и один назад на слой, q и k вместе.

Точность: те же операции в том же порядке, enable_fp_fusion=False (без слияния mul+add в FMA),
поэтому результат совпадает с PyTorch побитно (тест: torch.equal).

Обратный проход (R = rotate_half — линейный оператор, R^T(y) = cat(y2, -y1)):
    dq1 = g1*cos1 + g2*sin2,   dq2 = g2*cos2 - g1*sin1
"""

import torch
import triton
import triton.language as tl

ROPE_CALLS = {"fused": 0, "skipped": 0}
_ORIGINAL = None


@triton.jit(do_not_specialize=["L", "sqb", "sqh", "sql", "skb", "skh", "skl", "scb", "scl"])
def ida_rope_kernel(
    q_ptr, k_ptr,           # вход: q/k (forward) или их градиенты (backward), последний шаг 1
    cos_ptr, sin_ptr,       # [B, L, D]
    qo_ptr, ko_ptr,         # выход: плотные [B, HQ, L, D] и [B, HK, L, D]
    L,
    sqb, sqh, sql,
    skb, skh, skl,
    scb, scl,
    HQ: tl.constexpr, HK: tl.constexpr, HALF: tl.constexpr, BACKWARD: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // L
    l = pid % L
    d = tl.arange(0, HALF)
    cbase = b.to(tl.int64) * scb + l.to(tl.int64) * scl
    c1 = tl.load(cos_ptr + cbase + d).to(tl.float32)
    c2 = tl.load(cos_ptr + cbase + HALF + d).to(tl.float32)
    s1 = tl.load(sin_ptr + cbase + d).to(tl.float32)
    s2 = tl.load(sin_ptr + cbase + HALF + d).to(tl.float32)

    for h in tl.static_range(HQ + HK):
        if h < HQ:
            src = q_ptr + b.to(tl.int64) * sqb + h * sqh + l.to(tl.int64) * sql
            dst = qo_ptr + ((b.to(tl.int64) * HQ + h) * L + l) * (2 * HALF)
        else:
            src = k_ptr + b.to(tl.int64) * skb + (h - HQ) * skh + l.to(tl.int64) * skl
            dst = ko_ptr + ((b.to(tl.int64) * HK + (h - HQ)) * L + l) * (2 * HALF)
        x1 = tl.load(src + d).to(tl.float32)
        x2 = tl.load(src + HALF + d).to(tl.float32)
        if BACKWARD:
            o1 = x1 * c1 + x2 * s2
            o2 = x2 * c2 - x1 * s1
        else:
            o1 = x1 * c1 + (-x2) * s1
            o2 = x2 * c2 + x1 * s2
        tl.store(dst + d, o1.to(qo_ptr.dtype.element_ty))
        tl.store(dst + HALF + d, o2.to(qo_ptr.dtype.element_ty))


def _launch(q, k, cos, sin, backward):
    B, HQ, L, D = q.shape
    HK = k.shape[1]
    qo = torch.empty((B, HQ, L, D), dtype=q.dtype, device=q.device)
    ko = torch.empty((B, HK, L, D), dtype=k.dtype, device=k.device)
    ida_rope_kernel[(B * L,)](
        q, k, cos, sin, qo, ko, L,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        cos.stride(0), cos.stride(1),
        HQ=HQ, HK=HK, HALF=D // 2, BACKWARD=backward,
        num_warps=2, enable_fp_fusion=False,
    )
    return qo, ko


def _last_dim_dense(t):
    return t if t.stride(-1) == 1 else t.contiguous()


class FusedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        cos = _last_dim_dense(cos)
        sin = _last_dim_dense(sin)
        ctx.save_for_backward(cos, sin)
        ctx.shapes = (q.shape, k.shape, q.dtype, q.device)
        return _launch(_last_dim_dense(q), _last_dim_dense(k), cos, sin, False)

    @staticmethod
    def backward(ctx, gq, gk):
        cos, sin = ctx.saved_tensors
        qs, ks, dtype, device = ctx.shapes
        if gq is None:
            gq = torch.zeros(qs, dtype=dtype, device=device)
        if gk is None:
            gk = torch.zeros(ks, dtype=dtype, device=device)
        dq, dk = _launch(_last_dim_dense(gq), _last_dim_dense(gk), cos, sin, True)
        return dq, dk, None, None


def _fits(q, k, cos, sin, unsqueeze_dim):
    if unsqueeze_dim != 1 or not (q.is_cuda and k.is_cuda):
        return False
    if q.dim() != 4 or k.dim() != 4 or cos.dim() != 3 or sin.shape != cos.shape:
        return False
    B, HQ, L, D = q.shape
    if k.shape[0] != B or k.shape[2] != L or k.shape[3] != D or D % 2 or D > 256:
        return False
    if cos.shape != (B, L, D) and not (cos.shape[0] == 1 and cos.shape[1:] == (L, D)):
        return False
    if q.dtype != k.dtype or cos.dtype != q.dtype or sin.dtype != q.dtype:
        return False       # смешанные типы: оригинал продвинет тип, мы не повторяем
    return cos.requires_grad is False and sin.requires_grad is False


def _patched_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    if not _fits(q, k, cos, sin, unsqueeze_dim):
        ROPE_CALLS["skipped"] += 1
        return _ORIGINAL(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    if cos.shape[0] != q.shape[0]:
        cos = cos.expand(q.shape[0], -1, -1)
        sin = sin.expand(q.shape[0], -1, -1)
    ROPE_CALLS["fused"] += 1
    return FusedRoPE.apply(q, k, cos, sin)


def patch_rope():
    global _ORIGINAL
    from transformers.models.qwen2 import modeling_qwen2

    if _ORIGINAL is None:
        _ORIGINAL = modeling_qwen2.apply_rotary_pos_emb
        modeling_qwen2.apply_rotary_pos_emb = _patched_apply_rotary_pos_emb
        print("[fused_rope] перехвачен modeling_qwen2.apply_rotary_pos_emb", flush=True)


def rope_debug():
    return dict(ROPE_CALLS)
