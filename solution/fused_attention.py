"""Своё ядро внимания (attention) на Triton — прямой и обратный проход, с маской.

Что такое внимание простыми словами: каждый токен «смотрит» на предыдущие и решает, насколько
каждый из них важен. Три шага:
    S = Q @ K^T / sqrt(d)      — насколько токен i похож на токен j
    P = softmax(S + маска)     — превращаем в веса (сумма по строке = 1)
    O = P @ V                  — взвешенная сумма значений

PyTorch строит матрицу S целиком и трижды проходит по памяти. Мы идём по K/V блоками и
накапливаем результат (online softmax — тот же приём, что в FlashAttention и в нашем CE).

Особенности Qwen2: 12 голов запросов на 2 головы ключей (GQA), размер головы 128,
причинная маска (токен видит только себя и прошлое), плюс маска паддинга от transformers.

Идея (блочный online softmax, GQA, причинная маска) взята из FlashAttention (arXiv 2205.14135)
и туториала Triton по fused attention. Код написан самостоятельно; от примера из шаблона
(example_swiglu) отличается полностью — другая операция, другая структура, другой патч.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_forward_kernel(
    Q, K, V, Mask, O, L, sm_scale,
    N_CTX, H, KV_GROUP,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H
    off_hkv = off_h // KV_GROUP
    n_kv = H // KV_GROUP

    q_base = Q + (off_b * H + off_h) * N_CTX * HEAD_DIM
    k_base = K + (off_b * n_kv + off_hkv) * N_CTX * HEAD_DIM
    v_base = V + (off_b * n_kv + off_hkv) * N_CTX * HEAD_DIM
    o_base = O + (off_b * H + off_h) * N_CTX * HEAD_DIM
    m_base = Mask + off_b * N_CTX * N_CTX

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N_CTX

    q = tl.load(q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=m_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    hi = tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N)

    for start_n in range(0, hi * BLOCK_N, BLOCK_N):
        cols = start_n + offs_n
        valid = cols[None, :] < N_CTX
        k = tl.load(k_base + cols[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=(cols[:, None] < N_CTX), other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_MASK:
            m = tl.load(m_base + offs_m[:, None] * N_CTX + cols[None, :],
                        mask=m_mask[:, None] & valid, other=0)
            qk = tl.where((m != 0) & valid, qk, float("-inf"))
        else:
            qk = tl.where(offs_m[:, None] >= cols[None, :], qk, float("-inf"))
            qk = tl.where(valid, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        # Строка может быть закрыта маской целиком (паддинг). Тогда максимум = -inf, и
        # -inf - -inf даёт NaN, который расползётся по всей модели. Подставляем 0.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + cols[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=(cols[:, None] < N_CTX), other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_safe

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    tl.store(o_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=m_mask[:, None])
    tl.store(L + pid_bh * N_CTX + offs_m,
             tl.where(l_i > 0, m_i + tl.log(l_safe), 0.0), mask=m_mask)


@triton.jit
def fused_attention_backward_kernel(
    Q, K, V, Mask, dO, dQ, dK, dV, L, D, sm_scale,
    N_CTX, H, KV_GROUP,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H
    off_hkv = off_h // KV_GROUP
    n_kv = H // KV_GROUP

    q_base = Q + (off_b * H + off_h) * N_CTX * HEAD_DIM
    k_base = K + (off_b * n_kv + off_hkv) * N_CTX * HEAD_DIM
    v_base = V + (off_b * n_kv + off_hkv) * N_CTX * HEAD_DIM
    do_base = dO + (off_b * H + off_h) * N_CTX * HEAD_DIM
    dq_base = dQ + (off_b * H + off_h) * N_CTX * HEAD_DIM
    dk_base = dK + (off_b * n_kv + off_hkv) * N_CTX * HEAD_DIM
    dv_base = dV + (off_b * n_kv + off_hkv) * N_CTX * HEAD_DIM
    m_base = Mask + off_b * N_CTX * N_CTX

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N_CTX

    q = tl.load(q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=m_mask[:, None], other=0.0)
    do = tl.load(do_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                 mask=m_mask[:, None], other=0.0)
    l = tl.load(L + pid_bh * N_CTX + offs_m, mask=m_mask, other=0.0)
    delta = tl.load(D + pid_bh * N_CTX + offs_m, mask=m_mask, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    hi = tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N)

    for start_n in range(0, hi * BLOCK_N, BLOCK_N):
        cols = start_n + offs_n
        valid = cols[None, :] < N_CTX
        k = tl.load(k_base + cols[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=(cols[:, None] < N_CTX), other=0.0)
        v = tl.load(v_base + cols[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=(cols[:, None] < N_CTX), other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_MASK:
            mk = tl.load(m_base + offs_m[:, None] * N_CTX + cols[None, :],
                         mask=m_mask[:, None] & valid, other=0)
            keep = (mk != 0) & valid
        else:
            keep = (offs_m[:, None] >= cols[None, :]) & valid
        p = tl.where(keep, tl.exp(qk - l[:, None]), 0.0)

        dv = tl.dot(tl.trans(p).to(v.dtype), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale
        dq += tl.dot(ds.to(q.dtype), k)
        dk = tl.dot(tl.trans(ds).to(q.dtype), q)
        tl.atomic_add(dk_base + cols[:, None] * HEAD_DIM + offs_d[None, :],
                      dk.to(tl.float32), mask=(cols[:, None] < N_CTX))
        tl.atomic_add(dv_base + cols[:, None] * HEAD_DIM + offs_d[None, :],
                      dv.to(tl.float32), mask=(cols[:, None] < N_CTX))

    tl.store(dq_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             dq.to(dQ.dtype.element_ty), mask=m_mask[:, None])


class FusedAttention(torch.autograd.Function):
    """Внимание с причинной маской и GQA на своих ядрах."""

    @staticmethod
    def forward(ctx, query, key, value, scale, mask):
        b, h, n, d = query.shape
        kv_group = h // key.shape[1]
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        mask = mask.contiguous() if mask is not None else torch.empty(1, dtype=torch.uint8,
                                                                    device=query.device)

        out = torch.empty_like(query)
        lse = torch.empty((b * h, n), dtype=torch.float32, device=query.device)

        BLOCK_M, BLOCK_N = 64, 32
        grid = (triton.cdiv(n, BLOCK_M), b * h)
        fused_attention_forward_kernel[grid](
            query, key, value, mask, out, lse, scale, n, h, kv_group,
            HAS_MASK=mask.dim() == 3,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=d, num_warps=4, num_stages=2,
        )

        ctx.save_for_backward(query, key, value, out, lse, mask)
        ctx.scale = scale
        ctx.kv_group = kv_group
        return out

    @staticmethod
    def backward(ctx, grad_out):
        query, key, value, out, lse, mask = ctx.saved_tensors
        b, h, n, d = query.shape
        kv_group = ctx.kv_group
        grad_out = grad_out.contiguous()

        delta = (grad_out.float() * out.float()).sum(dim=-1)
        dq = torch.empty_like(query)
        dk = torch.zeros(b, key.shape[1], n, d, dtype=torch.float32, device=query.device)
        dv = torch.zeros_like(dk)

        # T4 (sm75): 64 КБ разделяемой памяти на блок. С маской обратный проход не
        # укладывается при 32×32 (нужно 68 КБ), поэтому блок по ключам — 16.
        BLOCK_M, BLOCK_N = 32, 16
        grid = (triton.cdiv(n, BLOCK_M), b * h)
        fused_attention_backward_kernel[grid](
            query, key, value, mask, grad_out, dq, dk, dv, lse,
            delta.view(b * h, n).contiguous(), ctx.scale, n, h, kv_group,
            HAS_MASK=mask.dim() == 3,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=d, num_warps=4, num_stages=1,
        )
        return dq, dk.to(key.dtype), dv.to(value.dtype), None, None


def fused_sdpa(query, key, value, scale, mask=None):
    return FusedAttention.apply(query, key, value, scale, mask)


_ORIGINAL_SDPA = None
_ORIGINAL_GET_INTERFACE = None
ATTN_CALLS = {"calls": 0, "fused": 0, "skipped": 0}
MASK_INFO = {"printed": False}
# Маска transformers: True = «смотреть можно» (проверено на реальном прогоне:
# row0_sum = 1, rowlast_sum = 176 → причинная маска без инверсии). Поэтому НЕ инвертируем.
INVERT_MASK = False
# Сколько первых слоёв считаем своим ядром (None = все слои), 0 = выключено
FUSED_LAYERS = 1
_layer_ids = set()


def _mask_to_3d(attention_mask, batch, seq, heads):
    """Маска transformers [B,1,S,S] -> uint8 [B,S,S]. 1 = смотреть можно, 0 = запрещено."""
    m = attention_mask
    if m is None:
        return None
    if m.dtype != torch.bool:
        return "unsupported"
    if m.dim() == 4:
        if m.shape[1] == 1:
            m = m[:, 0]
        elif m.shape[1] == heads:
            m = m[:, :1, :, :].squeeze(1)
        else:
            m = m[:, 0]
    elif m.dim() == 2:
        m = m[:, None, None, :].expand(batch, 1, seq, seq)
        m = m[:, 0]

    m = m.contiguous()
    if MASK_INFO["printed"] is False:
        MASK_INFO["printed"] = True
        first = m[0, 0]
        last = m[0, min(seq - 1, m.shape[-1] - 1)]
        print(f"[fused_attention] маска: shape={tuple(attention_mask.shape)} "
              f"dtype={attention_mask.dtype} mean={float(m.float().mean()):.3f} "
              f"row0_sum={int(first.sum())} rowlast_sum={int(last.sum())} "
              f"INVERT={INVERT_MASK}", flush=True)
    if INVERT_MASK:
        m = ~m
    return m.to(torch.uint8)


def _fallback(query, key, value, attention_mask, dropout, is_causal, scaling):
    """Откат на встроенное внимание с теми же аргументами (важно: GQA требует enable_gqa)."""
    gqa = torch.is_tensor(query) and torch.is_tensor(key) and query.dim() == 4 \
        and key.dim() == 4 and query.shape[1] != key.shape[1]
    return _ORIGINAL_SDPA(query, key, value, attn_mask=attention_mask, dropout_p=dropout,
                          is_causal=bool(is_causal) if is_causal is not None else False,
                          scale=scaling, enable_gqa=gqa), None


def _patched_attention_interface(module, query, key, value, attention_mask=None,
                                 scaling=None, dropout=0.0, is_causal=None, **kwargs):
    """Единая замена функции внимания transformers (и для eager, и для sdpa)."""
    ATTN_CALLS["calls"] += 1
    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    if not torch.is_tensor(query) or not query.is_cuda or query.dim() != 4:
        ATTN_CALLS["skipped"] += 1
        return _fallback(query, key, value, attention_mask, dropout, is_causal, scaling)

    b, h, s, d = query.shape

    # Сначала решаем, наш ли это слой: на остальных сразу откат, без лишних преобразований.
    if FUSED_LAYERS is not None:
        key_mod = id(module)
        if key_mod not in _layer_ids:
            if len(_layer_ids) >= FUSED_LAYERS:
                ATTN_CALLS["skipped"] += 1
                return _fallback(query, key, value, attention_mask, dropout, is_causal, scaling)
            _layer_ids.add(key_mod)

    mask3 = _mask_to_3d(attention_mask, b, s, h)
    if mask3 == "unsupported":
        ATTN_CALLS["skipped"] += 1
        return _fallback(query, key, value, attention_mask, dropout, is_causal, scaling)

    ATTN_CALLS["fused"] += 1
    out = fused_sdpa(query, key, value, float(scaling), mask3)

    # Самопроверка на первом же реальном вызове: считаем то же встроенным вниманием и печатаем
    # расхождение. Это сразу отделит «ядро считает неверно» от «неверен обратный проход».
    if ATTN_CALLS["fused"] == 1:
        share = float(mask3.float().mean()) if isinstance(mask3, torch.Tensor) else 1.0
        with torch.no_grad():
            ref = _ORIGINAL_SDPA(query, key, value, attn_mask=attention_mask, dropout_p=0.0,
                                 is_causal=False, scale=scaling,
                                 enable_gqa=query.shape[1] != key.shape[1])
            ref = ref.transpose(1, 2).contiguous()
            diff = (out.transpose(1, 2).contiguous().float() - ref.float()).abs()
            print(f"[fused_attention] сверка: q{tuple(query.shape)} k{tuple(key.shape)} "
                  f"mask_share={share:.3f} layers={FUSED_LAYERS} "
                  f"max_diff={diff.max().item():.4f} mean_diff={diff.mean().item():.6f} "
                  f"scale_ref={ref.abs().mean().item():.4f}", flush=True)
        del ref, diff

    return out.transpose(1, 2).contiguous(), None


def patch_attention():
    """Подменить выбор реализации внимания у transformers.

    Модель делает так:
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation,
                                                                   eager_attention_forward)
    Поэтому подменяем метод get_interface у объекта-реестра: на выходе всегда наша функция.
    """
    global _ORIGINAL_SDPA, _ORIGINAL_GET_INTERFACE
    import torch.nn.functional as functional

    if functional.scaled_dot_product_attention is not _patched_attention_interface:
        _ORIGINAL_SDPA = functional.scaled_dot_product_attention

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as registry
    except Exception as exc:  # noqa: BLE001
        print(f"[fused_attention] реестр не найден: {exc}", flush=True)
        return

    if _ORIGINAL_GET_INTERFACE is None:
        _ORIGINAL_GET_INTERFACE = registry.get_interface

        def get_interface(impl, default=None):
            _ORIGINAL_GET_INTERFACE(impl, default)      # как в оригинале: проверка имени
            return _patched_attention_interface

        registry.get_interface = get_interface
        print("[fused_attention] подменён ALL_ATTENTION_FUNCTIONS.get_interface", flush=True)


def attention_debug():
    return dict(ATTN_CALLS)
