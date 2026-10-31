"""Своё ядро cross-entropy по кускам словаря — версия v4.

Что изменилось против v3 (и почему это важно для памяти):

1. В v3 патч ForCausalLMLoss НЕ срабатывал. В transformers 5.16.1 модель берёт функцию потерь
   через свойство `PreTrainedModel.loss_function` -> `LOSS_MAPPING["ForCausalLM"]` (словарь,
   заполненный при импорте). Подмена атрибута модуля словарь не меняет, поэтому вызывался
   оригинал: `logits = logits.float()` (копия 940 x 151936 в fp32 = 571 МБ), и уже её получал
   наш перехват F.cross_entropy. Её же мы сохраняли на backward и под неё же выделяли градиент
   в fp32 (ещё 571 МБ). Теперь подменяем запись в LOSS_MAPPING: ядро получает fp16-логиты,
   сохраняет их же (без копии) и пишет градиент сразу в fp16. Математика — fp32 в регистрах,
   как и раньше.
2. Нет синхронизаций: делитель и масштаб градиента остаются тензорами на GPU, ядро читает
   масштаб по указателю (в v3 были float(denom) и float(grad_out) на каждом шаге).
3. reduction="none" под no_grad: так Soup считает потери по строкам для rewind.jsonl
   (soup_cli/trainer/rewind_hf.py: row_losses). Раньше это шло в PyTorch (log_softmax на
   [234, 151936] четыре раза за шаг), теперь — в то же наше ядро.

Идея (чанки по словарю + online softmax + хранение только lse) — из практики fused
cross-entropy (Unsloth cross_entropy_loss.py, Liger-Kernel, arXiv 2410.10989) и online softmax
FlashAttention (arXiv 2205.14135). Код написан самостоятельно.
"""

import torch
import triton
import triton.language as tl

BLOCK_V = 4096
_APPLIED = False
CE_CALLS = {"calls": 0, "fused": 0, "fused_none": 0, "skipped": 0, "loss_fn": 0}
_ORIGINAL_CE = None
_ORIGINAL_LOSS = None
# v5, пункт 8: градиент логитов пишется в буфер самих логитов. Включает
# fused_metrics.patch_logits_memory(); действует только для потерь модели (LOSS_MAPPING).
INPLACE_GRAD = {"on": False, "used": 0}
_PATCHED = []


@triton.jit
def fused_ce_forward_kernel(
    logits_ptr,     # [rows, vocab], fp16/fp32, строки идут с шагом row_stride
    labels_ptr,     # [rows], int64
    lse_ptr,        # [rows], fp32
    loss_ptr,       # [rows], fp32 — ошибка на строку (0 для ignore_index)
    vocab,
    row_stride,
    IGNORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)

    if label == IGNORE:
        tl.store(lse_ptr + row, 0.0)
        tl.store(loss_ptr + row, 0.0)
        return

    base = row.to(tl.int64) * row_stride
    m = float("-inf")
    l = 0.0
    for start in range(0, vocab, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < vocab
        x = tl.load(logits_ptr + base + cols, mask=mask, other=float("-inf"),
                    eviction_policy="evict_last").to(tl.float32)
        new_m = tl.maximum(m, tl.max(x, axis=0))
        l = l * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m

    lse = m + tl.log(l)
    target = tl.load(logits_ptr + base + label).to(tl.float32)
    tl.store(lse_ptr + row, lse)
    tl.store(loss_ptr + row, lse - target)


@triton.jit
def fused_ce_backward_kernel(
    logits_ptr,     # [rows, vocab] — те же логиты, что на прямом проходе
    labels_ptr,     # [rows], int64
    lse_ptr,        # [rows], fp32
    grad_ptr,       # [rows, vocab] — градиент логитов (dtype логитов, плотный)
    scale_ptr,      # [1], fp32 — grad_out / делитель, лежит на GPU (без синхронизации)
    vocab,
    row_stride,
    IGNORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    label = tl.load(labels_ptr + row)

    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab
    gbase = row.to(tl.int64) * vocab

    if label == IGNORE:
        zero = tl.zeros([BLOCK], dtype=tl.float32).to(grad_ptr.dtype.element_ty)
        tl.store(grad_ptr + gbase + cols, zero, mask=mask)
        return

    scale = tl.load(scale_ptr)
    lse = tl.load(lse_ptr + row)
    x = tl.load(logits_ptr + row.to(tl.int64) * row_stride + cols, mask=mask,
                other=float("-inf"), eviction_policy="evict_first").to(tl.float32)
    p = tl.exp(x - lse)
    g = (p - tl.where(cols == label, 1.0, 0.0)) * scale
    tl.store(grad_ptr + gbase + cols, g.to(grad_ptr.dtype.element_ty), mask=mask)


def _row_losses(logits2d, labels1d, ignore_index):
    rows, vocab = logits2d.shape
    lse = torch.empty(rows, dtype=torch.float32, device=logits2d.device)
    per_row = torch.empty(rows, dtype=torch.float32, device=logits2d.device)
    fused_ce_forward_kernel[(rows,)](
        logits2d, labels1d, lse, per_row, vocab, logits2d.stride(0),
        IGNORE=ignore_index, BLOCK=BLOCK_V, num_warps=8,
    )
    return lse, per_row


def _as_rows(logits):
    vocab = logits.shape[-1]
    x = logits.reshape(-1, vocab)
    if x.stride(-1) != 1:
        x = x.contiguous()
    return x


class FusedCrossEntropy(torch.autograd.Function):
    """sum или mean по валидным токенам; denominator=None -> mean, 'sum' -> сумма."""

    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100, reduction="mean", inplace=False):
        shape = logits.shape
        logits2d = _as_rows(logits)
        labels1d = labels.reshape(-1).to(torch.int64)
        lse, per_row = _row_losses(logits2d, labels1d, ignore_index)
        total = per_row.sum()
        if reduction == "sum":
            denom = None
            out = total
        else:
            denom = (labels1d != ignore_index).sum().clamp(min=1).to(torch.float32)
            out = total / denom
        ctx.save_for_backward(logits2d, labels1d, lse)
        # на месте — только если буфер логитов плотный построчно (тот же адрес = тот же элемент)
        ctx.inplace = bool(inplace) and logits2d.is_contiguous()
        ctx.denom = denom
        ctx.ignore_index = ignore_index
        ctx.out_shape = shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        logits2d, labels1d, lse = ctx.saved_tensors
        rows, vocab = logits2d.shape
        scale = grad_out.float().reshape(1)
        if ctx.denom is not None:
            scale = scale / ctx.denom
        if ctx.inplace:
            # каждый элемент читается и перезаписывается одним и тем же потоком ядра
            grad = logits2d
            INPLACE_GRAD["used"] += 1
        else:
            grad = torch.empty((rows, vocab), dtype=logits2d.dtype, device=logits2d.device)
        fused_ce_backward_kernel[(rows, triton.cdiv(vocab, BLOCK_V))](
            logits2d, labels1d, lse, grad, scale, vocab, logits2d.stride(0),
            IGNORE=ctx.ignore_index, BLOCK=BLOCK_V, num_warps=8,
        )
        return grad.view(ctx.out_shape), None, None, None, None


def fused_cross_entropy(logits, labels, ignore_index=-100, reduction="mean", inplace=False):
    return FusedCrossEntropy.apply(logits, labels, ignore_index, reduction, inplace)


def _big_vocab(t):
    return torch.is_tensor(t) and t.is_cuda and t.dim() >= 2 and t.shape[-1] >= 50000 \
        and t.dtype in (torch.float16, torch.float32, torch.bfloat16)


def _patched_cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                           reduce=None, reduction="mean", label_smoothing=0.0):
    """Перехват F.cross_entropy: mean/sum с градиентом, none — только без градиента."""
    CE_CALLS["calls"] += 1
    plain = (input.dim() == 2 if torch.is_tensor(input) else False) and weight is None \
        and size_average is None and reduce is None and label_smoothing == 0.0 \
        and torch.is_tensor(target) and target.dim() == 1 and _big_vocab(input)
    if plain and reduction in ("mean", "sum"):
        CE_CALLS["fused"] += 1
        return fused_cross_entropy(input, target, ignore_index, reduction)
    if plain and reduction == "none" and not (torch.is_grad_enabled() and input.requires_grad):
        CE_CALLS["fused_none"] += 1
        _, per_row = _row_losses(_as_rows(input), target.to(torch.int64), ignore_index)
        return per_row
    CE_CALLS["skipped"] += 1
    return _ORIGINAL_CE(input, target, weight=weight, size_average=size_average,
                        ignore_index=ignore_index, reduce=reduce, reduction=reduction,
                        label_smoothing=label_smoothing)


def _our_for_causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=None,
                            ignore_index=-100, shift_labels=None, **kwargs):
    """ForCausalLMLoss без `logits.float()`. Семантика — как fixed_cross_entropy 5.16.1
    (лишние **kwargs оригинал тоже игнорирует)."""
    if not _big_vocab(logits):
        return _ORIGINAL_LOSS(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch,
                              ignore_index=ignore_index, shift_labels=shift_labels, **kwargs)
    CE_CALLS["loss_fn"] += 1
    if shift_labels is None:
        labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    logits2d = logits.reshape(-1, vocab_size)
    flat = shift_labels.reshape(-1).to(logits2d.device)
    inplace = INPLACE_GRAD["on"]
    if num_items_in_batch is None:
        return fused_cross_entropy(logits2d, flat, ignore_index, "mean", inplace)
    loss = fused_cross_entropy(logits2d, flat, ignore_index, "sum", inplace)
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(loss.device)
    return loss / num_items_in_batch


def patch_loss():
    """F.cross_entropy + LOSS_MAPPING["ForCausalLM"] (это и есть то, что зовёт модель)."""
    global _APPLIED, _ORIGINAL_CE, _ORIGINAL_LOSS
    if _APPLIED:
        return
    import torch.nn.functional as functional
    import transformers.loss.loss_utils as loss_utils

    _ORIGINAL_CE = functional.cross_entropy
    functional.cross_entropy = _patched_cross_entropy

    _ORIGINAL_LOSS = loss_utils.LOSS_MAPPING["ForCausalLM"]
    for key, fn in list(loss_utils.LOSS_MAPPING.items()):
        if fn is _ORIGINAL_LOSS:
            loss_utils.LOSS_MAPPING[key] = _our_for_causal_lm_loss
            _PATCHED.append(key)
    loss_utils.ForCausalLMLoss = _our_for_causal_lm_loss
    _APPLIED = True
    print(f"[fused_ce] v4: F.cross_entropy + LOSS_MAPPING {_PATCHED}", flush=True)


# ---------- C11: потери по строкам для rewind.jsonl одним блоком ----------
#
# Soup (soup_cli/trainer/rewind_hf.py, row_losses) на каждом шаге для каждой из 4 строк делает
# cross_entropy(reduction="none") + две синхронизации (float(...) и int(tokens.item())) —
# 8 остановок конвейера за шаг. Здесь те же числа: ошибка по токену — тем же нашим ядром
# fused_ce_forward_kernel (fp32 внутри), маска, сумма, деление — теми же операциями torch,
# но по всем строкам сразу и с ОДНОЙ синхронизацией (.tolist()). Отличие от Soup возможно
# только в последнем бите fp32 (сумма по оси 1 вместо суммы 1D-вектора) — проверка №5
# сравнивает потери строк с допуском 3%.

ROW_CALLS = {"batched": 0, "fallback": 0}
_ORIGINAL_ROW_LOSSES = None


def batched_row_losses(logits, labels):
    if not (_big_vocab(logits) and logits.dim() == 3 and torch.is_tensor(labels)
            and labels.dim() == 2 and tuple(labels.shape) == tuple(logits.shape[:2])
            and labels.shape[1] >= 2 and labels.device == logits.device):
        ROW_CALLS["fallback"] += 1
        return _ORIGINAL_ROW_LOSSES(logits, labels)
    ROW_CALLS["batched"] += 1
    with torch.no_grad():
        shift = labels[:, 1:].to(torch.int64)
        token_ce = torch.stack([
            _row_losses(_as_rows(logits[i, :-1, :].detach()), shift[i].contiguous(), -100)[1]
            for i in range(shift.size(0))
        ])                                                    # [B, T-1], fp32, 0 у -100
        mask = (shift != -100).to(token_ce.dtype)
        tokens = mask.sum(dim=1)
        total = (token_ce * mask).sum(dim=1)
        loss, count = torch.stack([total / tokens.clamp(min=1.0), tokens]).tolist()
    return [float(x) for x in loss], [int(x) for x in count]


def patch_rewind_rows():
    """Подменить soup_cli.trainer.rewind_hf.row_losses (зовётся из compute_loss по имени)."""
    global _ORIGINAL_ROW_LOSSES
    try:
        import soup_cli.trainer.rewind_hf as rewind_hf
    except Exception as exc:  # noqa: BLE001 - без Soup патчить нечего
        print(f"[fused_ce] rewind: пропуск ({exc!r})", flush=True)
        return
    if _ORIGINAL_ROW_LOSSES is None and hasattr(rewind_hf, "row_losses"):
        _ORIGINAL_ROW_LOSSES = rewind_hf.row_losses
        rewind_hf.row_losses = batched_row_losses
        print("[fused_ce] rewind: row_losses одним блоком", flush=True)


def debug_state():
    return dict(CE_CALLS, applied=_APPLIED, mapping=list(_PATCHED), rows=dict(ROW_CALLS))
