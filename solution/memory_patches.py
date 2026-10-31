"""Снятие лишних fp32-копий. Без новых ядер, только «водопровод» типов; численно — побитно то же.

Откуда берутся копии (сверено с исходниками peft 0.20 / bitsandbytes 0.50.2 / accelerate 1.14 /
TRL 0.29.1 / Soup v0.77.0):

A. `peft.prepare_model_for_kbit_training` переводит ВСЕ не-4-битные параметры в fp32, в том
   числе общую матрицу эмбеддингов / lm_head (151936 x 1536): 933 МБ вместо 467. Под autocast
   F.linear(lm_head) каждый шаг делает её fp16-копию (467 МБ) и держит её до backward.
   Решение: хранить замороженную матрицу в fp16 (значения и так fp16: Soup грузит модель в fp16,
   fp16 -> fp32 -> fp16 без потерь), а выход эмбеддинга поднимать в fp32 хуком — остаточный
   поток остаётся fp32, как в базе. Autocast для lm_head видит уже fp16-вес и не копирует.

B. `bnb.nn.Linear4bit.forward` возвращает `.to(inp_dtype)`: вход MLP — fp32, значит gate/up
   возвращаются fp32 (940 x 8960 x 4 Б = 34 МБ каждая), SwiGLU их сохраняет в fp32 до backward —
   ~1 ГБ на 28 слоёв, плюс лишние проходы памяти туда-обратно. Решение: подать в gate/up fp16
   (ровно то приведение, которое Linear4bit делает сам), тогда выходы fp16, SwiGLU сохраняет
   fp16. Вход приводим ДВАЖДЫ (отдельно для gate и up), чтобы градиенты по входу складывались
   в fp32, как в базе.

C. accelerate (`ConvertOutputsToFp32`) переводит выходы модели в fp32: логиты 571 МБ, а TRL
   ещё делает `outputs.logits[..., :-1, :].contiguous()` для точности — второй раз 571 МБ.
   Решение: логиты (последняя ось >= 50000) оставляем fp16. Их потребители: энтропия (наше ядро,
   fp32 внутри), argmax точности (fp16 -> fp32 точное, argmax тот же), потери по строкам Soup
   (наше ядро CE reduction='none', fp32 внутри), loss (скаляр — он по-прежнему fp32).
"""

import torch

STATE = {"embedding_fp16": False, "mlp_fp16": 0, "mlp_fallback": 0, "logits_fp16": False,
         "empty_cache": False, "empty_cache_calls": 0}
_ORIG_TRAINER_INIT = None


# ---------- A. эмбеддинг / lm_head ----------

def _upcast_hook(module, inputs, output):
    return output.float() if output.dtype == torch.float16 else output


def halve_frozen_embedding(model):
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    w = getattr(emb, "weight", None)
    if w is None or w.requires_grad or w.dtype != torch.float32 or w.shape[0] < 50000:
        print(f"[memory] эмбеддинг не трогаем: {None if w is None else (w.dtype, w.requires_grad)}",
              flush=True)
        return False
    tied = out is not None and getattr(out, "weight", None) is w
    if out is not None and not tied and getattr(out.weight, "dtype", None) == torch.float32 \
            and not out.weight.requires_grad:
        out.weight.data = out.weight.data.to(torch.float16)
    w.data = w.data.to(torch.float16)
    emb.register_forward_hook(_upcast_hook)
    STATE["embedding_fp16"] = True
    print(f"[memory] эмбеддинг {tuple(w.shape)} -> fp16, lm_head общий: {tied}", flush=True)
    return True


def patch_trainer_init():
    """Перед сборкой Trainer модель уже прошла prepare_model_for_kbit_training и peft."""
    global _ORIG_TRAINER_INIT
    import transformers

    if _ORIG_TRAINER_INIT is not None:
        return
    _ORIG_TRAINER_INIT = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        if model is not None:
            try:
                halve_frozen_embedding(model)
            except Exception as exc:  # noqa: BLE001 - не ломаем обучение ради памяти
                print(f"[memory] эмбеддинг: пропуск ({exc!r})", flush=True)
        try:
            from . import precompile
            precompile.wait()
        except Exception:  # noqa: BLE001
            pass
        _ORIG_TRAINER_INIT(self, *args, **kwargs)
        if STATE["empty_cache"]:
            self.add_callback(_EmptyCacheCallback())

    transformers.Trainer.__init__ = init


# ---------- C6: вернуть свободный резерв аллокатора перед каждым шагом ----------
#
# В v4: torch_max_reserved 6 360 МиБ при max_allocated 6 189 — свободного резерва всего 171 МиБ.
# Внутришаговую фрагментацию empty_cache не трогает, так что ожидаемый выигрыш 0…~100 МиБ
# ценой ~2–4 мс на шаг. Настройки аллокатора (PYTORCH_CUDA_ALLOC_CONF) не меняются.

def _make_empty_cache_callback():
    import transformers

    class EmptyCache(transformers.TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()
            STATE["empty_cache_calls"] += 1

    return EmptyCache


_EmptyCacheCallback = None


def patch_empty_cache():
    """Включается одной строкой в apply(); требует patch_trainer_init()."""
    global _EmptyCacheCallback
    _EmptyCacheCallback = _make_empty_cache_callback()
    STATE["empty_cache"] = True


# ---------- B. MLP в fp16 ----------

def mlp_forward_fp16(mlp, x):
    """Qwen2MLP в fp16. None — условия не выполнены, надо звать оригинал."""
    import bitsandbytes as bnb

    from .swiglu import swiglu

    g, u, d = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    linear4 = bnb.nn.Linear4bit
    ok = (x.is_cuda and x.dtype == torch.float32
          and getattr(getattr(mlp, "config", None), "hidden_act", "silu") == "silu"
          and isinstance(g, linear4) and isinstance(u, linear4) and isinstance(d, linear4)
          and g.compute_dtype == torch.float16 and u.compute_dtype == torch.float16
          and d.compute_dtype == torch.float16)
    if not ok:
        return None
    gate = g(x.to(torch.float16))          # Linear4bit сам сделал бы это же приведение
    up = u(x.to(torch.float16))            # отдельное приведение: dX суммируется в fp32
    return d(swiglu(gate, up))             # fp16; + к fp32-остатку даёт тот же fp32


def patch_qwen2_mlp_fp16():
    """Qwen2MLP: gate/up/down в fp16, SwiGLU — ядро примера (в зачёт не идёт, но работает)."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        out = mlp_forward_fp16(self, x)
        if out is None:
            STATE["mlp_fallback"] += 1
            return original(self, x)
        STATE["mlp_fp16"] += 1
        return out

    modeling_qwen2.Qwen2MLP.forward = forward
    print("[memory] Qwen2MLP: gate/up/down в fp16", flush=True)


# ---------- C. логиты не переводим в fp32 ----------

def patch_accelerate_logits():
    import accelerate.utils.operations as ops

    if STATE["logits_fp16"]:
        return

    def is_small_half(t):
        return (torch.is_tensor(t) and t.dtype in (torch.float16, torch.bfloat16)
                and not (t.dim() >= 2 and t.shape[-1] >= 50000))

    def convert_to_fp32(tensor):
        return ops.recursively_apply(lambda t: t.float(), tensor, test_type=is_small_half)

    ops.convert_to_fp32 = convert_to_fp32
    STATE["logits_fp16"] = True
    print("[memory] accelerate: логиты остаются fp16 (остальные выходы -> fp32)", flush=True)


def memory_debug():
    return dict(STATE)
