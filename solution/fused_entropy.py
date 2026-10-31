"""Своё ядро энтропии: один проход по словарю вместо log_softmax + умножения.

Зачем это нужно. После каждого шага тренер считает энтропию по логитам — насколько модель
уверена в своих предсказаниях. PyTorch делает это так:

    log_probs = log_softmax(logits, dim=-1)   # копия 940 × 151936 в fp32 ≈ 571 МБ
    p = exp(log_probs)
    entropy = -sum(p * log_probs)

То есть на каждом шаге создаётся таблица размером с сами логиты, плюс три прохода по памяти.

Мы считаем то же самое одним проходом по кускам словаря. Для строки x:
    m, l, s — текущий максимум, сумма exp(x-m) и сумма exp(x-m)*x
    lse = m + log(l)
    sum(p_i * x_i) = s / l
    entropy = lse - s / l
Таблица вероятностей не создаётся вообще.

Идея (online-обновление трёх аккумуляторов, чанки по словарю) — та же, что в нашем CE и в
FlashAttention. Код написан самостоятельно.
"""

import torch
import triton
import triton.language as tl

BLOCK_V = 4096          # кусок словаря за шаг цикла
ENTROPY_CALLS = {"calls": 0, "fused": 0, "skipped": 0}
_ORIGINAL_ENTROPY = None


@triton.jit
def fused_entropy_forward_kernel(
    logits_ptr,     # [rows, vocab], fp16/fp32
    out_ptr,        # [rows], fp32 — энтропия на строку
    vocab,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * vocab

    m = float("-inf")
    l = 0.0
    s = 0.0
    for start in range(0, vocab, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < vocab
        x = tl.load(logits_ptr + base + cols, mask=mask, other=float("-inf")).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk_max)
        # переносим аккумуляторы под новый максимум и добавляем кусок
        scale = tl.exp(m - new_m)
        e = tl.exp(x - new_m)
        # В маскированных ячейках x = -inf: экспонента даёт 0, а 0 * (-inf) = NaN.
        # Поэтому в произведение подставляем безопасный 0 вместо -inf.
        x_safe = tl.where(mask, x, 0.0)
        l = l * scale + tl.sum(e, axis=0)
        s = s * scale + tl.sum(e * x_safe, axis=0)
        m = new_m

    lse = m + tl.log(l)
    tl.store(out_ptr + row, lse - s / l)


class FusedEntropy(torch.autograd.Function):
    """Энтропия без градиента (в тренере она считается под torch.no_grad)."""

    @staticmethod
    def forward(ctx, logits):
        shape = logits.shape
        vocab = shape[-1]
        logits2d = logits.reshape(-1, vocab)
        rows = logits2d.shape[0]
        out = torch.empty(rows, dtype=torch.float32, device=logits.device)

        fused_entropy_forward_kernel[(rows,)](
            logits2d, out, vocab, BLOCK=BLOCK_V, num_warps=8,
        )
        return out.view(shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        raise RuntimeError("энтропия используется без градиента")


def entropy_from_logits(logits):
    return FusedEntropy.apply(logits)


def _patched_entropy_from_logits(logits, **kwargs):
    ENTROPY_CALLS["calls"] += 1
    if not torch.is_tensor(logits) or not logits.is_cuda or logits.dim() < 2 \
            or logits.shape[-1] < 50000:
        ENTROPY_CALLS["skipped"] += 1
        return _ORIGINAL_ENTROPY(logits, **kwargs)

    ENTROPY_CALLS["fused"] += 1
    if ENTROPY_CALLS["fused"] == 1:
        print(f"[fused_entropy] включилось: логиты {tuple(logits.shape)} {logits.dtype}",
              flush=True)
    return entropy_from_logits(logits)


def patch_entropy():
    """Подменить подсчёт энтропии в TRL (и в других местах, где она встречается)."""
    global _ORIGINAL_ENTROPY
    import sys

    if _ORIGINAL_ENTROPY is not None:
        return

    try:
        import trl.trainer.utils as trl_utils
    except Exception as exc:  # noqa: BLE001 - без TRL патчить нечего
        print(f"[fused_entropy] TRL не найден: {exc}", flush=True)
        return

    _ORIGINAL_ENTROPY = getattr(trl_utils, "entropy_from_logits", None)
    if _ORIGINAL_ENTROPY is None:
        print("[fused_entropy] entropy_from_logits не найден в TRL", flush=True)
        return

    trl_utils.entropy_from_logits = _patched_entropy_from_logits

    patched = ["trl.trainer.utils"]
    for name, module in list(sys.modules.items()):
        if name.startswith("trl.") and module is not None \
                and hasattr(module, "entropy_from_logits"):
            setattr(module, "entropy_from_logits", _patched_entropy_from_logits)
            patched.append(name)

    print(f"[fused_entropy] подменено в {len(patched)}: {', '.join(patched[:4])}", flush=True)


def entropy_debug():
    return dict(ENTROPY_CALLS)
