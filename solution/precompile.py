"""Компиляция наших Triton-ядер в фоновом потоке, пока Soup грузит модель (~14 с).

Почему: каждый замер стартует с пустым кэшем Triton (run_task.py задаёт свой TRITON_CACHE_DIR),
и компиляция ~8 вариантов ядер иначе происходит на 1-2 шаге обучения — последовательно, в общем
времени прогона. Здесь мы честно компилируем их в том же процессе и в том же замере, просто
параллельно с загрузкой весов (она почти вся — чтение файла и CUDA, процессор свободен).
Никаких готовых кэшей: всё компилируется с нуля в каждом запуске.

Каждое ядро вызывается на крошечных тензорах с ТЕМИ ЖЕ типами и признаками специализации
(кратность 16 у целых аргументов, выравнивание указателей), что и в обучении, — тогда в обучении
Triton берёт уже готовый вариант. Если что-то не совпало, ядро просто скомпилируется ещё раз
в обучении, как раньше: на корректность это не влияет.

Выключить: IDA_PRECOMPILE=0.

v5: NF4 и RMSNorm компилируются в том варианте, который выбран fused_nf4.configure /
fused_rmsnorm.configure — поэтому в apply() эти строки стоят ДО precompile.start().
Потери строк rewind (C11) идут тем же fused_ce_forward_kernel, отдельной компиляции не нужно.
"""

import os
import threading
import time

import torch

_THREAD = None
REPORT = {"done": False, "seconds": None, "errors": []}


def _nf4():
    from .fused_nf4 import _launch
    dev = torch.device("cuda")
    n_blocks, bs = 256, 64
    packed = torch.randint(0, 255, (n_blocks * bs // 2,), dtype=torch.uint8, device=dev)
    a8 = torch.randint(0, 255, (n_blocks,), dtype=torch.uint8, device=dev)
    code2 = torch.randn(256, device=dev)
    a2 = torch.rand(n_blocks // 256, device=dev)
    off = torch.zeros(1, device=dev)
    _launch(packed, a8, code2, a2, off, n_blocks * bs, bs, torch.float16)


def _rmsnorm():
    from .fused_rmsnorm import rmsnorm
    x = torch.randn(16, 1536, device="cuda", requires_grad=True)
    w = torch.ones(1536, device="cuda")
    rmsnorm(x, w, 1e-6).backward(torch.randn(16, 1536, device="cuda"))


def _ce_entropy():
    from .fused_ce import fused_cross_entropy, _row_losses
    from .fused_entropy import entropy_from_logits
    v = 151936
    logits = torch.randn(16, v, device="cuda", dtype=torch.float16, requires_grad=True)
    labels = torch.randint(0, v, (16,), device="cuda")
    fused_cross_entropy(logits, labels, -100, "sum").backward()
    with torch.no_grad():
        _row_losses(logits.detach(), labels, -100)
        entropy_from_logits(logits.detach())


def _swiglu():
    from .swiglu import swiglu
    g = torch.randn(16, 8960, device="cuda", dtype=torch.float16, requires_grad=True)
    u = torch.randn(16, 8960, device="cuda", dtype=torch.float16, requires_grad=True)
    swiglu(g, u).backward(torch.randn(16, 8960, device="cuda", dtype=torch.float16))


def _rope():
    from .fused_rope import FusedRoPE
    B, L = 4, 16
    q = torch.randn(B, L, 12, 128, device="cuda").transpose(1, 2).requires_grad_(True)
    k = torch.randn(B, L, 2, 128, device="cuda").transpose(1, 2).requires_grad_(True)
    cos = torch.randn(B, L, 128, device="cuda")
    sin = torch.randn(B, L, 128, device="cuda")
    qo, ko = FusedRoPE.apply(q, k, cos, sin)
    (qo.sum() + ko.sum()).backward()


def _attn():
    from .fused_attn_prologue import AttnPrologue
    B, L = 4, 16
    q = torch.randn(B, L, 12, 128, device="cuda").transpose(1, 2).requires_grad_(True)
    k = torch.randn(B, L, 2, 128, device="cuda").transpose(1, 2).requires_grad_(True)
    v = torch.randn(B, L, 2, 128, device="cuda").transpose(1, 2).requires_grad_(True)
    cos = torch.randn(B, L, 128, device="cuda")
    sin = torch.randn(B, L, 128, device="cuda")
    qo, ko, vo = AttnPrologue.apply(q, k, v, cos, sin)
    (qo.float().sum() + ko.float().sum() + vo.float().sum()).backward()


def _metrics():
    from .fused_ce import fused_cross_entropy
    from .fused_metrics import entropy_argmax
    v = 151936
    base = torch.randn(16, v, device="cuda", dtype=torch.float16, requires_grad=True)
    logits = base * 1.0                                   # не лист: как выход lm_head
    labels = torch.randint(0, v, (16,), device="cuda")
    with torch.no_grad():
        entropy_argmax(logits.view(4, 4, v))
    fused_cross_entropy(logits, labels, -100, "sum", True).backward()


STEPS = [("nf4", _nf4), ("ce+entropy", _ce_entropy), ("rmsnorm", _rmsnorm),
         ("swiglu", _swiglu), ("rope", _rope), ("attn", _attn), ("metrics", _metrics)]


def _run(steps):
    start = time.perf_counter()
    try:
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.enable_grad():
            for name, fn in steps:
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - не компилится тут, скомпилится в обучении
                    REPORT["errors"].append(f"{name}: {exc!r}"[:300])
        stream.synchronize()
    finally:
        REPORT["done"] = True
        REPORT["seconds"] = round(time.perf_counter() - start, 2)
        print(f"[precompile] готово за {REPORT['seconds']} с, ошибки: {REPORT['errors'] or 'нет'}",
              flush=True)


def start(enabled=("nf4", "ce+entropy", "rmsnorm", "swiglu", "rope", "attn", "metrics")):
    global _THREAD
    if os.environ.get("IDA_PRECOMPILE", "1") == "0" or _THREAD is not None:
        return
    steps = [(n, f) for n, f in STEPS if n in enabled]
    _THREAD = threading.Thread(target=_run, args=(steps,), daemon=True, name="ida-precompile")
    _THREAD.start()


def wait(timeout=180):
    if _THREAD is not None:
        _THREAD.join(timeout)
