"""v5, пункты 7+8: логиты на пике живут в одном экземпляре.

На пиковом батче (M = 1 336 строк) логиты fp16 = 1 336 x 151 936 x 2 Б = 406 МБ, и рядом с
ними до сих пор появлялся второй буфер того же размера:
- в конце forward — копия TRL для точности: `outputs.logits[..., :-1, :].contiguous()`
  (trl/trainer/sft_trainer.py, SFTTrainer.compute_loss, 0.29.1), только ради argmax;
- в backward CE — буфер градиента логитов.
Убрать надо оба, иначе пик не сдвинется (пункт 7 без 8 и наоборот дают ~0).

7. ida_entropy_argmax_kernel: за один проход по логитам — энтропия (те же операции в том же
   порядке, что fused_entropy_forward_kernel, поэтому побитно то же) и argmax по словарю
   (первый максимум, как torch.argmax). SFTTrainer.compute_loss заменён копией штатной ветки
   TRL 0.29.1 (Apache-2.0), где энтропия и argmax берутся из этого ядра, а маски, суммы,
   gather_for_metrics и .item() — те же строки, что у TRL. Имена и смысл метрик не меняются.
   Нестандартные ветки (liger, prompt tuning, shift_labels, padding-free, aux_loss) —
   оригинальный метод TRL целиком.
8. Градиент логитов на месте: ядро CE в backward пишет dlogits в буфер самих логитов.
   Безопасно, потому что после CE-backward логиты никому не нужны: метрики TRL и потери строк
   Soup считаются в forward, lm_head сохраняет для backward вход и вес, а не выход. Включается
   только для потерь модели (LOSS_MAPPING), не для произвольных вызовов F.cross_entropy.
"""

import torch
import triton
import triton.language as tl

BLOCK_V = 4096
METRIC_CALLS = {"fast": 0, "fallback": 0}
_ORIGINAL_COMPUTE_LOSS = None


@triton.jit
def ida_entropy_argmax_kernel(
    logits_ptr,     # [rows, vocab], fp16/fp32, строки с шагом row_stride
    ent_ptr,        # [rows], fp32 — энтропия
    idx_ptr,        # [rows], int64 — argmax по словарю
    vocab,
    row_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * row_stride

    m = float("-inf")
    l = 0.0
    s = 0.0
    best = tl.full([], 0, tl.int32)
    for start in range(0, vocab, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < vocab
        x = tl.load(logits_ptr + base + cols, mask=mask, other=float("-inf")).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        # argmax: только строго больший максимум куска сдвигает индекс -> первый максимум
        blk_arg = tl.argmax(x, axis=0, tie_break_left=True)
        best = tl.where(blk_max > m, start + blk_arg, best)
        # энтропия — ровно как в fused_entropy_forward_kernel
        new_m = tl.maximum(m, blk_max)
        scale = tl.exp(m - new_m)
        e = tl.exp(x - new_m)
        x_safe = tl.where(mask, x, 0.0)
        l = l * scale + tl.sum(e, axis=0)
        s = s * scale + tl.sum(e * x_safe, axis=0)
        m = new_m

    lse = m + tl.log(l)
    tl.store(ent_ptr + row, lse - s / l)
    tl.store(idx_ptr + row, best.to(tl.int64))


def entropy_argmax(logits):
    """(энтропия fp32, argmax int64) по последней оси; формы logits.shape[:-1]. Без градиента."""
    shape = logits.shape
    vocab = shape[-1]
    x = logits.detach().reshape(-1, vocab)
    if x.stride(-1) != 1:
        x = x.contiguous()
    rows = x.shape[0]
    ent = torch.empty(rows, dtype=torch.float32, device=x.device)
    idx = torch.empty(rows, dtype=torch.int64, device=x.device)
    ida_entropy_argmax_kernel[(rows,)](x, ent, idx, vocab, x.stride(0), BLOCK=BLOCK_V,
                                       num_warps=8)
    return ent.view(shape[:-1]), idx.view(shape[:-1])


# ---------- 7: SFTTrainer.compute_loss ----------

def _fast_ok(trainer, inputs):
    return (not getattr(trainer.args, "use_liger_kernel", False)
            and getattr(trainer, "num_virtual_tokens", 0) == 0
            and not getattr(trainer, "aux_loss_enabled", False)
            and "shift_labels" not in inputs and "attention_mask" in inputs
            and "labels" in inputs)


def _entropy_and_predictions(logits, labels):
    """Наше ядро; если логиты неожиданные (forward уже прошёл) — ровно формулы TRL."""
    if (torch.is_tensor(logits) and logits.dim() == 3 and logits.is_cuda
            and logits.shape[-1] >= 50000 and tuple(labels.shape) == tuple(logits.shape[:2])):
        METRIC_CALLS["fast"] += 1
        ent, idx = entropy_argmax(logits)
        return ent, idx[..., :-1]
    METRIC_CALLS["fallback"] += 1
    from trl.trainer import sft_trainer
    ent = sft_trainer.entropy_from_logits(logits)
    return ent, logits[..., :-1, :].contiguous().argmax(dim=-1)


def _compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """Штатная ветка TRL 0.29.1 SFTTrainer.compute_loss; энтропия и argmax — одним ядром."""
    import transformers

    if not _fast_ok(self, inputs):
        METRIC_CALLS["fallback"] += 1
        return _ORIGINAL_COMPUTE_LOSS(self, model, inputs, return_outputs=return_outputs,
                                      num_items_in_batch=num_items_in_batch)

    mode = "train" if self.model.training else "eval"
    inputs.pop("_prediction_loss_only", None)
    labels = inputs["labels"]
    inputs["use_cache"] = False

    # TRL зовёт здесь super().compute_loss -> transformers.Trainer.compute_loss
    (loss, outputs) = transformers.Trainer.compute_loss(
        self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

    with torch.no_grad():
        per_token_entropy, predictions = _entropy_and_predictions(outputs.logits, labels)
        attention_mask = inputs["attention_mask"]
        entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
    self._metrics[mode]["entropy"].append(entropy)

    if mode == "train":
        num_tokens_in_batch = self.accelerator.gather_for_metrics(
            inputs["attention_mask"].sum()).sum().item()
        self._total_train_tokens += num_tokens_in_batch
    self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

    with torch.no_grad():
        # predictions == outputs.logits[..., :-1, :].contiguous().argmax(dim=-1), без копии
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100
        correct_predictions = (predictions == shift_labels) & mask
        total_tokens = mask.sum()
        correct_tokens = correct_predictions.sum()
        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

    return (loss, outputs) if return_outputs else loss


def patch_logits_memory(metrics=True, inplace_grad=True):
    """7+8 одной строкой в apply(). Флаги — только для отладки (по отдельности пик не падает)."""
    global _ORIGINAL_COMPUTE_LOSS
    from . import fused_ce

    if inplace_grad:
        fused_ce.INPLACE_GRAD["on"] = True
    if metrics and _ORIGINAL_COMPUTE_LOSS is None:
        from trl.trainer.sft_trainer import SFTTrainer

        _ORIGINAL_COMPUTE_LOSS = SFTTrainer.compute_loss
        SFTTrainer.compute_loss = _compute_loss
    print(f"[fused_metrics] 7: метрики TRL одним проходом={_ORIGINAL_COMPUTE_LOSS is not None}, "
          f"8: градиент логитов на месте={fused_ce.INPLACE_GRAD['on']}", flush=True)


def metrics_debug():
    from . import fused_ce
    return dict(METRIC_CALLS, inplace=dict(fused_ce.INPLACE_GRAD))
