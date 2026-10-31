"""Диагностика (только для своих прогонов, в сдаче выключена): IDA_DIAG=1.

Печатает в train.log:
- типы и формы параметров после всех подготовок (fp32 ли эмбеддинг/lm_head, квантован ли lm_head);
- M (строк в матмуле) и число вызовов наших путей;
- память по фазам: после загрузки, на каждом шаге — allocated/reserved/«как nvidia-smi»;
- шаги 0-1 (компиляция) отдельно;
- снимок памяти на шагах 3-4 -> /kaggle/working/ida_mem.pickle (открыть на pytorch.org/memory_viz,
  вкладка «Active Memory Timeline»: видно, какие тензоры живут в момент пика).

Не трогает reset_peak_memory_stats (harness читает max_memory_allocated).
"""

import os
import time

import torch

ENABLED = os.environ.get("IDA_DIAG") == "1"
SNAPSHOT = os.environ.get("IDA_DIAG_SNAPSHOT", "/kaggle/working/ida_mem.pickle")
T0 = time.perf_counter()


def _mib(x):
    return round(x / 2**20)


def _smi():
    free, total = torch.cuda.mem_get_info()
    return _mib(total - free)


def log(msg):
    if ENABLED:
        print(f"[diag {time.perf_counter() - T0:7.2f}s] {msg}", flush=True)


def triton_variants():
    """Сколько скомпилированных вариантов у каждого нашего ядра (C6: всё ли поймал precompile)."""
    import importlib
    out = {}
    for mod in ("fused_nf4", "fused_rmsnorm", "fused_ce", "fused_entropy", "fused_rope", "swiglu",
                "fused_attn_prologue", "fused_metrics"):
        try:
            m = importlib.import_module(f"{__package__}.{mod}")
        except Exception:  # noqa: BLE001
            continue
        for name, obj in vars(m).items():
            caches = getattr(obj, "device_caches", None)
            if caches is None:
                continue
            n = 0
            for v in caches.values():
                cache = v[0] if isinstance(v, tuple) else v
                try:
                    n += len(cache)
                except TypeError:
                    pass
            if n:
                out[name] = n
    return out


def describe_model(model):
    seen = {}
    for name, p in model.named_parameters():
        key = (type(p).__name__, str(p.dtype), p.requires_grad)
        cnt, num = seen.get(key, (0, 0))
        seen[key] = (cnt + 1, num + p.numel())
    for key, (cnt, num) in sorted(seen.items(), key=lambda kv: -kv[1][1]):
        log(f"params {key}: {cnt} шт, {num / 1e6:.1f} M, {_mib(num * (4 if 'float32' in key[1] else 2))} МиБ*")
    emb, out = model.get_input_embeddings(), model.get_output_embeddings()
    log(f"embedding {type(emb).__name__} {tuple(emb.weight.shape)} {emb.weight.dtype}; "
        f"lm_head {type(out).__name__} {getattr(out.weight, 'dtype', None)} "
        f"tied={out.weight is emb.weight}")


def install():
    if not ENABLED:
        return
    import transformers

    orig_init = transformers.Trainer.__init__

    class DiagCallback(transformers.TrainerCallback):
        def __init__(self):
            self.step = 0
            self.t = None

        def on_train_begin(self, args, state, control, **kw):
            torch.cuda.synchronize()
            log(f"train begin: alloc {_mib(torch.cuda.memory_allocated())} "
                f"reserved {_mib(torch.cuda.memory_reserved())} smi {_smi()} МиБ")

        def on_step_begin(self, args, state, control, **kw):
            if self.step == 3:
                torch.cuda.memory._record_memory_history(max_entries=200000)
            self.t = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            torch.cuda.synchronize()
            dt = time.perf_counter() - self.t
            log(f"step {self.step}: {dt * 1000:.0f} ms, alloc {_mib(torch.cuda.memory_allocated())} "
                f"max_alloc {_mib(torch.cuda.max_memory_allocated())} "
                f"reserved {_mib(torch.cuda.memory_reserved())} smi {_smi()} МиБ")
            if self.step == 4:
                try:
                    torch.cuda.memory._dump_snapshot(SNAPSHOT)
                    log(f"снимок памяти: {SNAPSHOT}")
                finally:
                    torch.cuda.memory._record_memory_history(enabled=None)
            if self.step in (1, 10, 63):
                from . import fused_attn_prologue, fused_ce, fused_nf4, fused_rope, memory_patches
                log(f"nf4 {fused_nf4.nf4_debug()} {fused_nf4.CONFIG} ce {fused_ce.debug_state()} "
                    f"rope {fused_rope.rope_debug()} mem {memory_patches.memory_debug()}")
                log(f"attn {fused_attn_prologue.attn_debug()}")
                from . import fused_metrics
                log(f"metrics/logits {fused_metrics.metrics_debug()}")
                log(f"варианты ядер Triton (больше 1 = перекомпиляция в обучении): "
                    f"{triton_variants()}")
            self.step += 1

    def init(self, *args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        if model is not None:
            torch.cuda.synchronize()
            log(f"Trainer.__init__: alloc {_mib(torch.cuda.memory_allocated())} "
                f"max_alloc {_mib(torch.cuda.max_memory_allocated())} smi {_smi()} МиБ")
            describe_model(model)
        orig_init(self, *args, **kwargs)
        self.add_callback(DiagCallback())

    transformers.Trainer.__init__ = init

    # сколько стоит загрузка модели (из 14 с) и подготовка peft
    try:
        auto = transformers.AutoModelForCausalLM
        orig_fp = auto.from_pretrained

        def from_pretrained(*a, **k):
            t = time.perf_counter()
            model = orig_fp(*a, **k)
            torch.cuda.synchronize()
            log(f"from_pretrained: {time.perf_counter() - t:.2f} с, alloc "
                f"{_mib(torch.cuda.memory_allocated())} МиБ")
            return model

        auto.from_pretrained = from_pretrained
        import peft
        orig_prep = peft.prepare_model_for_kbit_training

        def prep(*a, **k):
            t = time.perf_counter()
            out = orig_prep(*a, **k)
            log(f"prepare_model_for_kbit_training: {time.perf_counter() - t:.2f} с")
            return out

        peft.prepare_model_for_kbit_training = prep
    except Exception as exc:  # noqa: BLE001
        log(f"тайминг загрузки: {exc!r}")

    # M у 4-битных матмулов (сколько строк реально идёт в GEMM)
    try:
        import bitsandbytes.nn as bnn
        orig_fwd = bnn.Linear4bit.forward
        seen = set()

        def fwd(self, x):
            m = x.numel() // x.shape[-1]
            key = (m, tuple(self.weight.quant_state.shape), str(x.dtype))
            if len(seen) < 12 and key not in seen:
                seen.add(key)
                log(f"Linear4bit M={m} W={key[1]} x={x.dtype} compute={self.compute_dtype}")
            return orig_fwd(self, x)

        bnn.Linear4bit.forward = fwd
    except Exception as exc:  # noqa: BLE001
        log(f"Linear4bit hook: {exc!r}")
    log("диагностика включена")
