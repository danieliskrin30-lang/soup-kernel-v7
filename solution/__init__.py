"""Наше решение v4: свои Triton-ядра + снятие лишних fp32-копий.

Свои ядра (в зачёт):
- fused_nf4.py     — распаковка NF4: теперь и в ПРЯМОМ проходе (bnb на T4 при M>512 идёт через
                     dequant + F.linear), масштабы двойного квантования — внутри ядра, без
                     синхронизаций; побитно совпадает с bitsandbytes;
- fused_rmsnorm.py — нормализация;
- fused_ce.py      — cross-entropy; теперь реально подменяет LOSS_MAPPING (без fp32-копии
                     логитов) и считает потери по строкам для rewind.jsonl;
- fused_entropy.py — энтропия;
- fused_rope.py    — RoPE для q и k одним кернелом вперёд и одним назад, побитно как PyTorch.

Без новых ядер (memory_patches.py): эмбеддинг/lm_head хранится в fp16, MLP в fp16,
логиты не переводятся в fp32 accelerate'ом — всё побитно эквивалентно базе.
precompile.py: компиляция наших ядер параллельно с загрузкой модели (в том же замере).
diag.py: диагностика, только при IDA_DIAG=1.

v5, пункт C: fused_nf4.configure / fused_rmsnorm.configure (выбор варианта ядра по bench_c.py),
patch_rewind_rows (потери строк rewind одним блоком), patch_empty_cache (выкл.).
v5, пункты 2+3: fused_attn_prologue.py — RoPE + fp16 + разворот GQA одним ядром, SDPA с
is_causal=True на cutlass (guard: паддинг справа, иначе штатный путь).
v5, пункты 7+8: fused_metrics.py — энтропия + argmax одним ядром вместо копии логитов TRL,
градиент логитов пишется в буфер логитов (на пике один буфер логитов вместо двух).

Выключены: fused_attention.py (точность/скорость), fused_lm_head.py (см. README).
"""

from .swiglu import patch_qwen2_mlp
from .fused_rmsnorm import patch_qwen2_rmsnorm
from .fused_ce import patch_loss, patch_rewind_rows
from .fused_entropy import patch_entropy
from .fused_nf4 import patch_nf4
from .fused_rope import patch_rope
from .fused_attn_prologue import patch_attention_prologue
from . import diag, fused_metrics, fused_nf4, fused_rmsnorm, memory_patches, precompile


def apply():
    # --- C: настройки ядер. Строго ДО precompile.start(): компилируется выбранный вариант.
    #     Значения по умолчанию = v4. Лучшие подставить из `python solution/bench_c.py`.
    fused_nf4.configure(mode="join", groups=64, num_warps=4)      # C5
    fused_rmsnorm.configure(rows=1, num_warps=4)                  # C10
    precompile.start()                     # компиляция параллельно с загрузкой модели
    diag.install()                         # IDA_DIAG=1, иначе ничего не делает
    patch_qwen2_mlp()                      # пример из шаблона (импортирует modeling_qwen2)
    memory_patches.patch_qwen2_mlp_fp16()  # поверх примера: тот же SwiGLU, но fp16
    patch_qwen2_rmsnorm()
    patch_rope()
    patch_attention_prologue()             # v5 2+3: пролог внимания + SDPA is_causal (cutlass)
    patch_nf4()
    patch_entropy()
    patch_loss()
    patch_rewind_rows()                    # C11: rewind — одна синхронизация вместо восьми
    fused_metrics.patch_logits_memory()    # v5 7+8: метрики TRL одним проходом + dlogits на месте
    memory_patches.patch_accelerate_logits()
    # memory_patches.patch_empty_cache()   # C6: выкл. по умолчанию (резерв v4 всего 171 МиБ)
    memory_patches.patch_trainer_init()    # последним: оборачивает diag (порядок логов)


def apply_attention(layers=1):
    """Экспериментальное ядро внимания (выключено)."""
    from . import fused_attention
    fused_attention.FUSED_LAYERS = layers
    fused_attention.patch_attention()
