"""v5, пункты 2+3: пролог внимания своим ядром + SDPA с is_causal вместо маски паддинга.

Что было (transformers 5.16.1, Qwen2Attention.forward -> sdpa_attention_forward), на слой:
    q, k, v (fp32) -> RoPE (fp32) -> repeat_kv(k), repeat_kv(v): копии fp32 на 12 голов
    -> autocast приводит q, k, v к fp16 (ещё три прохода) -> SDPA(attn_mask=маска паддинга).
    Backward: обратные приведения fp16->fp32, сумма по 6 головам группы, RoPE назад.
С маской cutlass-ядро (fmha_cutlassF/B) считает все плитки L x L — о причинности оно не знает.

Что стало:
    ida_attn_prologue_fwd_kernel: одним запуском RoPE для q и k, приведение к fp16 и
        разворот k, v на 12 голов (GQA 12/2) — сразу в формате, который ест SDPA;
    SDPA(is_causal=True, attn_mask=None), бэкенд жёстко EFFICIENT_ATTENTION (cutlass):
        если cutlass не может — исключение, а не тихий уход в math;
    ida_attn_prologue_bwd_kernel: градиенты fp16 -> fp32, сумма по группам GQA в fp32
        внутри ядра, RoPE назад — одним запуском.

Почему is_causal без маски — то же самое (guard ниже это проверяет на каждом шаге):
    TRL 0.29.1 (DataCollatorForLanguageModeling) дополняет батч СПРАВА. Значит, реальный токен
    на позиции t видит только позиции <= t, а они все реальные — маска паддинга для реальных
    токенов ничего не закрывает сверх причинной. Строки-заглушки в конце считаются иначе, но
    их градиент ровно 0 (метки -100), а в метриках TRL они умножаются на attention_mask = 0.
    Если паддинг не правый, переданы position_ids или есть кэш — штатный путь transformers.

Точность:
    forward пролога — побитно как база (те же операции RoPE в том же порядке без FMA, одно
    округление fp32->fp16, разворот — это копия); dq — побитно; dk, dv — порядок суммы по 6
    головам группы может отличаться от torch на последний бит fp32 (test_attn.py).
    SDPA с is_causal против SDPA с маской: на реальных строках расхождение на уровне
    округления fp16 (test_attn.py сверяет весь слой с допуском).
"""

import torch
import triton
import triton.language as tl

STATE = {"enabled": False, "causal_ok": False, "fast": 0, "fallback": 0, "guard_false": 0,
         "backend_error": None}
_ORIGINAL_ATTN_FORWARD = None
_ORIGINAL_MODEL_FORWARD = None


@triton.jit(do_not_specialize=["L", "sqb", "sqh", "sql", "skb", "skh", "skl",
                               "svb", "svh", "svl", "scb", "scl"])
def ida_attn_prologue_fwd_kernel(
    q_ptr, k_ptr, v_ptr,            # fp32: q [B,HQ,L,D], k/v [B,HK,L,D] (любые шаги, последний = 1)
    cos_ptr, sin_ptr,               # fp32 [B,L,D]
    qo_ptr, ko_ptr, vo_ptr,         # fp16, плотные [B,HQ,L,D]; k/v уже развёрнуты на HQ голов
    L,
    sqb, sqh, sql, skb, skh, skl, svb, svh, svl, scb, scl,
    HQ: tl.constexpr, HK: tl.constexpr, HALF: tl.constexpr,
):
    REP: tl.constexpr = HQ // HK
    pid = tl.program_id(0)
    b = (pid // L).to(tl.int64)
    l = (pid % L).to(tl.int64)
    d = tl.arange(0, HALF)
    cb = b * scb + l * scl
    c1 = tl.load(cos_ptr + cb + d)
    c2 = tl.load(cos_ptr + cb + HALF + d)
    s1 = tl.load(sin_ptr + cb + d)
    s2 = tl.load(sin_ptr + cb + HALF + d)

    for h in tl.static_range(HQ):
        src = q_ptr + b * sqb + h * sqh + l * sql
        x1 = tl.load(src + d)
        x2 = tl.load(src + HALF + d)
        o1 = x1 * c1 + (-x2) * s1
        o2 = x2 * c2 + x1 * s2
        dst = qo_ptr + ((b * HQ + h) * L + l) * (2 * HALF)
        tl.store(dst + d, o1.to(qo_ptr.dtype.element_ty))
        tl.store(dst + HALF + d, o2.to(qo_ptr.dtype.element_ty))

    for g in tl.static_range(HK):
        ks = k_ptr + b * skb + g * skh + l * skl
        x1 = tl.load(ks + d)
        x2 = tl.load(ks + HALF + d)
        k1 = (x1 * c1 + (-x2) * s1).to(ko_ptr.dtype.element_ty)
        k2 = (x2 * c2 + x1 * s2).to(ko_ptr.dtype.element_ty)
        vs = v_ptr + b * svb + g * svh + l * svl
        v1 = tl.load(vs + d).to(vo_ptr.dtype.element_ty)
        v2 = tl.load(vs + HALF + d).to(vo_ptr.dtype.element_ty)
        for r in tl.static_range(REP):
            off = ((b * HQ + g * REP + r) * L + l) * (2 * HALF)
            tl.store(ko_ptr + off + d, k1)
            tl.store(ko_ptr + off + HALF + d, k2)
            tl.store(vo_ptr + off + d, v1)
            tl.store(vo_ptr + off + HALF + d, v2)


@triton.jit(do_not_specialize=["L", "sqb", "sqh", "sql", "skb", "skh", "skl",
                               "svb", "svh", "svl", "scb", "scl"])
def ida_attn_prologue_bwd_kernel(
    gq_ptr, gk_ptr, gv_ptr,         # fp16 [B,HQ,L,D] — градиенты от SDPA (любые шаги)
    cos_ptr, sin_ptr,               # fp32 [B,L,D]
    dq_ptr, dk_ptr, dv_ptr,         # fp32, плотные: dq [B,HQ,L,D], dk/dv [B,HK,L,D]
    L,
    sqb, sqh, sql, skb, skh, skl, svb, svh, svl, scb, scl,
    HQ: tl.constexpr, HK: tl.constexpr, HALF: tl.constexpr,
):
    REP: tl.constexpr = HQ // HK
    pid = tl.program_id(0)
    b = (pid // L).to(tl.int64)
    l = (pid % L).to(tl.int64)
    d = tl.arange(0, HALF)
    cb = b * scb + l * scl
    c1 = tl.load(cos_ptr + cb + d)
    c2 = tl.load(cos_ptr + cb + HALF + d)
    s1 = tl.load(sin_ptr + cb + d)
    s2 = tl.load(sin_ptr + cb + HALF + d)

    # q: RoPE назад. R = rotate_half, R^T(y) = cat(y2, -y1)
    for h in tl.static_range(HQ):
        src = gq_ptr + b * sqb + h * sqh + l * sql
        g1 = tl.load(src + d).to(tl.float32)
        g2 = tl.load(src + HALF + d).to(tl.float32)
        dst = dq_ptr + ((b * HQ + h) * L + l) * (2 * HALF)
        tl.store(dst + d, g1 * c1 + g2 * s2)
        tl.store(dst + HALF + d, g2 * c2 - g1 * s1)

    # k, v: сумма по REP головам группы в fp32 (как repeat_kv назад), затем RoPE назад для k
    for g in tl.static_range(HK):
        a1 = tl.zeros([HALF], dtype=tl.float32)
        a2 = tl.zeros([HALF], dtype=tl.float32)
        u1 = tl.zeros([HALF], dtype=tl.float32)
        u2 = tl.zeros([HALF], dtype=tl.float32)
        for r in tl.static_range(REP):
            ks = gk_ptr + b * skb + (g * REP + r) * skh + l * skl
            vs = gv_ptr + b * svb + (g * REP + r) * svh + l * svl
            a1 += tl.load(ks + d).to(tl.float32)
            a2 += tl.load(ks + HALF + d).to(tl.float32)
            u1 += tl.load(vs + d).to(tl.float32)
            u2 += tl.load(vs + HALF + d).to(tl.float32)
        off = ((b * HK + g) * L + l) * (2 * HALF)
        tl.store(dk_ptr + off + d, a1 * c1 + a2 * s2)
        tl.store(dk_ptr + off + HALF + d, a2 * c2 - a1 * s1)
        tl.store(dv_ptr + off + d, u1)
        tl.store(dv_ptr + off + HALF + d, u2)


def _dense_last(t):
    return t if t.stride(-1) == 1 else t.contiguous()


def _strides(t):
    return t.stride(0), t.stride(1), t.stride(2)


class AttnPrologue(torch.autograd.Function):
    """(q, k, v, cos, sin) fp32 -> (q16, k16, v16) для SDPA; k16/v16 развёрнуты на HQ голов."""

    @staticmethod
    def forward(ctx, q, k, v, cos, sin):
        q, k, v = _dense_last(q), _dense_last(k), _dense_last(v)
        cos, sin = _dense_last(cos), _dense_last(sin)
        B, HQ, L, D = q.shape
        HK = k.shape[1]
        qo = torch.empty((B, HQ, L, D), dtype=torch.float16, device=q.device)
        ko = torch.empty_like(qo)
        vo = torch.empty_like(qo)
        ida_attn_prologue_fwd_kernel[(B * L,)](
            q, k, v, cos, sin, qo, ko, vo, L,
            *_strides(q), *_strides(k), *_strides(v), cos.stride(0), cos.stride(1),
            HQ=HQ, HK=HK, HALF=D // 2, num_warps=2, enable_fp_fusion=False,
        )
        ctx.save_for_backward(cos, sin)
        ctx.meta = (B, HQ, HK, L, D, q.dtype, q.device)
        return qo, ko, vo

    @staticmethod
    def backward(ctx, gq, gk, gv):
        cos, sin = ctx.saved_tensors
        B, HQ, HK, L, D, dtype, device = ctx.meta
        zeros = None
        if gq is None or gk is None or gv is None:
            zeros = torch.zeros((B, HQ, L, D), dtype=torch.float16, device=device)
        gq = _dense_last(gq) if gq is not None else zeros
        gk = _dense_last(gk) if gk is not None else zeros
        gv = _dense_last(gv) if gv is not None else zeros
        dq = torch.empty((B, HQ, L, D), dtype=dtype, device=device)
        dk = torch.empty((B, HK, L, D), dtype=dtype, device=device)
        dv = torch.empty((B, HK, L, D), dtype=dtype, device=device)
        ida_attn_prologue_bwd_kernel[(B * L,)](
            gq, gk, gv, cos, sin, dq, dk, dv, L,
            *_strides(gq), *_strides(gk), *_strides(gv), cos.stride(0), cos.stride(1),
            HQ=HQ, HK=HK, HALF=D // 2, num_warps=2, enable_fp_fusion=False,
        )
        return dq, dk, dv, None, None


def attn_prologue(q, k, v, cos, sin):
    return AttnPrologue.apply(q, k, v, cos, sin)


# ---------- guard: паддинг справа ----------

def right_padded(attention_mask, position_ids=None, past_key_values=None):
    """True, если is_causal без маски эквивалентен маске для реальных токенов.

    Одна синхронизация на шаг (.item()). Строки маски вида 1..1 0..0, первая позиция реальная.
    """
    if position_ids is not None or past_key_values is not None:
        return False
    if attention_mask is None:
        return True
    if not torch.is_tensor(attention_mask) or attention_mask.dim() != 2:
        return False
    am = attention_mask.to(torch.int8)
    ok = (am[:, 1:] <= am[:, :-1]).all() & (am[:, 0] == 1).all() & ((am == 0) | (am == 1)).all()
    return bool(ok.item())


def _model_forward(self, *args, **kwargs):
    am = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
    pos = kwargs.get("position_ids", args[2] if len(args) > 2 else None)
    past = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
    use_cache = kwargs.get("use_cache", args[5] if len(args) > 5 else None)
    if isinstance(am, dict) or use_cache:
        ok = False
    else:
        try:
            ok = right_padded(am, pos, past)
        except Exception:  # noqa: BLE001 - сомневаешься — штатный путь
            ok = False
    STATE["causal_ok"] = ok
    if not ok:
        STATE["guard_false"] += 1
    try:
        return _ORIGINAL_MODEL_FORWARD(self, *args, **kwargs)
    finally:
        STATE["causal_ok"] = False


# ---------- Qwen2Attention.forward ----------

def _usable(module, hidden_states, past_key_values, kwargs):
    if not (STATE["enabled"] and STATE["causal_ok"]) or past_key_values is not None:
        return False
    if not hidden_states.is_cuda or kwargs.get("output_attentions"):
        return False
    cfg = module.config
    if getattr(cfg, "_attn_implementation", None) != "sdpa" or module.sliding_window is not None:
        return False
    if module.training and module.attention_dropout:
        return False
    hq, hk = cfg.num_attention_heads, cfg.num_key_value_heads
    return module.head_dim % 2 == 0 and module.head_dim <= 256 and hq % hk == 0


def _attn_forward(self, hidden_states, position_embeddings, attention_mask,
                  past_key_values=None, **kwargs):
    from transformers.models.qwen2 import modeling_qwen2

    if not _usable(self, hidden_states, past_key_values, kwargs):
        STATE["fallback"] += 1
        return _ORIGINAL_ATTN_FORWARD(self, hidden_states, position_embeddings, attention_mask,
                                      past_key_values=past_key_values, **kwargs)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings

    same = (query_states.dtype == key_states.dtype == value_states.dtype == cos.dtype
            == sin.dtype == torch.float32)
    if same and cos.dim() == 3 and cos.shape[0] != query_states.shape[0] and cos.shape[0] == 1:
        cos = cos.expand(query_states.shape[0], -1, -1)
        sin = sin.expand(query_states.shape[0], -1, -1)
    if same and STATE["backend_error"] is None:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            q16, k16, v16 = attn_prologue(query_states, key_states, value_states, cos, sin)
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q16, k16, v16, attn_mask=None, dropout_p=0.0, scale=self.scaling,
                    is_causal=True)
            STATE["fast"] += 1
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None
        except RuntimeError as exc:
            # cutlass не взял задачу: больше не пробуем, этот вызов — штатным хвостом ниже
            # (проекции уже посчитаны: dropout LoRA не перезапускаем, маски совпадут с базой)
            STATE["backend_error"] = repr(exc)[:300]
            print(f"[attn_prologue] EFFICIENT_ATTENTION недоступен, откат: {exc!r}"[:400],
                  flush=True)

    STATE["fallback"] += 1
    query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(
        query_states, key_states, cos, sin)
    interface = modeling_qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modeling_qwen2.eager_attention_forward)
    attn_output, attn_weights = interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def patch_attention_prologue():
    """Включить 2+3. Откат — убрать вызов из apply()."""
    global _ORIGINAL_ATTN_FORWARD, _ORIGINAL_MODEL_FORWARD
    from transformers.models.qwen2 import modeling_qwen2

    if _ORIGINAL_ATTN_FORWARD is not None:
        return
    _ORIGINAL_ATTN_FORWARD = modeling_qwen2.Qwen2Attention.forward
    _ORIGINAL_MODEL_FORWARD = modeling_qwen2.Qwen2Model.forward
    modeling_qwen2.Qwen2Attention.forward = _attn_forward
    modeling_qwen2.Qwen2Model.forward = _model_forward
    STATE["enabled"] = True
    print("[attn_prologue] Qwen2Attention: пролог своим ядром + SDPA is_causal (cutlass)",
          flush=True)


def attn_debug():
    return dict(STATE)
