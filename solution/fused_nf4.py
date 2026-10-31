"""Своё ядро распаковки 4-битных весов NF4 (bitsandbytes 0.50.2) на Triton — версия v4.

Где распаковка происходит в обучении (сверено с исходниками bitsandbytes 0.50.2):

1. ПРЯМОЙ проход. `MatMul4Bit.forward` -> `torch.ops.bitsandbytes.gemm_4bit`. На T4 (sm75) при
   M = 4 x длина > 512 эвристика `_gemm_4bit_use_custom_cuda` выбирает НЕ слитный кернел, а
   `bitsandbytes.backends.cuda.ops._dequant_linear_fallback`: распаковка (+ восстановление
   масштабов двумя отдельными кернелами bnb) -> F.linear. 196 раз за шаг.
   В v3 этот путь шёл мимо нас: мы перехватывали только Python-функцию dequantize_4bit.
2. ОБРАТНЫЙ проход. `MatMul4Bit.backward` -> `bitsandbytes.functional.dequantize_4bit`. 196 раз.

Что изменилось против v3:
- одно ядро вместо двух: масштабы двойного квантования восстанавливаются прямо в ядре;
- нет синхронизаций CPU<->GPU: в v3 на каждом вызове был `float(state.offset)` (offset — тензор
  на GPU) и `torch.tensor(NF4_CODEBOOK, device=...)` (копия с CPU). Это 392 остановки конвейера
  за шаг. Теперь offset читается ядром по указателю, таблица создаётся один раз;
- плитка [GROUPS, 64]: масштаб читается один раз на блок из 64 значений, каждый байт — один раз,
  два нибла раскладываются через tl.join (старший нибл первым — проверено в v3);
- enable_fp_fusion=False: code2*absmax2 и +offset округляются отдельно, как у bnb, поэтому
  результат совпадает с bitsandbytes побитно (тест проверяет torch.equal).
"""

import torch
import triton
import triton.language as tl

NF4_CODEBOOK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]

# Настройка ядра (пункт C5). Меняется одной строкой в apply(): fused_nf4.configure(...).
# mode: "join"  — v4: байт читается один раз, два нибла раскладываются tl.join;
#       "index" — каждое выходное значение читает свой байт (соседние значения делят адрес —
#                 одна транзакция), раскладка выхода простая, запись полностью векторизуется.
# Оба режима дают побитно один и тот же результат (test_c.py). Выбор — по solution/bench_c.py.
CONFIG = {"mode": "join", "groups": 64, "num_warps": 4}
NF4_CALLS = {"backward": 0, "forward": 0, "skipped": 0}
_ORIGINAL_DEQUANT = None
_ORIGINAL_FALLBACK = None
_CODES = {}             # таблица NF4 на устройстве, создаётся один раз
_DUMMY = {}


@triton.jit
def ida_nf4_dequant_kernel(
    packed_ptr,         # uint8 [n/2]: два значения на байт, старший нибл — первое
    absmax_ptr,         # NESTED: uint8 [n_blocks] (сжатые масштабы); иначе fp32 [n_blocks]
    code2_ptr,          # fp32 [256]   — таблица второй ступени (только NESTED)
    absmax2_ptr,        # fp32 [n_blocks/256] — масштабы второй ступени (только NESTED)
    offset_ptr,         # fp32 [1]     — сдвиг масштабов (только NESTED), читаем на GPU
    codes_ptr,          # fp32 [16]    — таблица NF4
    out_ptr,            # fp16/fp32 [n]
    n_blocks,
    NESTED: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    HALF: tl.constexpr = BLOCKSIZE // 2
    pid = tl.program_id(0)
    blk = pid * GROUPS + tl.arange(0, GROUPS)                     # [GROUPS] номера блоков
    bmask = blk < n_blocks

    if NESTED:
        q = tl.load(absmax_ptr + blk, mask=bmask, other=0).to(tl.int32)
        c2 = tl.load(code2_ptr + q, mask=bmask, other=0.0)
        a2 = tl.load(absmax2_ptr + blk // 256, mask=bmask, other=0.0)
        off = tl.load(offset_ptr)
        scale = c2 * a2                                           # как dequantize_blockwise
        scale = scale + off                                       # как `absmax += offset`
    else:
        scale = tl.load(absmax_ptr + blk, mask=bmask, other=0.0).to(tl.float32)

    blk64 = blk.to(tl.int64)
    byte_idx = blk64[:, None] * HALF + tl.arange(0, HALF)[None, :]           # [GROUPS, HALF]
    byte = tl.load(packed_ptr + byte_idx, mask=bmask[:, None], other=0).to(tl.int32)
    hi = tl.load(codes_ptr + (byte >> 4))                         # первое значение пары
    lo = tl.load(codes_ptr + (byte & 0xF))                        # второе
    vals = tl.reshape(tl.join(hi, lo), (GROUPS, BLOCKSIZE))       # hi, lo, hi, lo, ...
    vals = vals * scale[:, None]

    out_idx = blk64[:, None] * BLOCKSIZE + tl.arange(0, BLOCKSIZE)[None, :]
    tl.store(out_ptr + out_idx, vals.to(out_ptr.dtype.element_ty), mask=bmask[:, None])


@triton.jit
def ida_nf4_dequant_idx_kernel(
    packed_ptr, absmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, codes_ptr, out_ptr,
    n_blocks,
    NESTED: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """То же, что ida_nf4_dequant_kernel, но плитка индексируется по выходу."""
    pid = tl.program_id(0)
    blk = pid * GROUPS + tl.arange(0, GROUPS)
    bmask = blk < n_blocks

    if NESTED:
        q = tl.load(absmax_ptr + blk, mask=bmask, other=0).to(tl.int32)
        c2 = tl.load(code2_ptr + q, mask=bmask, other=0.0)
        a2 = tl.load(absmax2_ptr + blk // 256, mask=bmask, other=0.0)
        off = tl.load(offset_ptr)
        scale = c2 * a2
        scale = scale + off
    else:
        scale = tl.load(absmax_ptr + blk, mask=bmask, other=0.0).to(tl.float32)

    j = tl.arange(0, BLOCKSIZE)
    idx = blk.to(tl.int64)[:, None] * BLOCKSIZE + j[None, :]               # [GROUPS, BLOCKSIZE]
    byte = tl.load(packed_ptr + (idx >> 1), mask=bmask[:, None], other=0).to(tl.int32)
    shift = ((1 - (j & 1)) * 4)[None, :]                                   # чётный -> старший
    vals = tl.load(codes_ptr + ((byte >> shift) & 0xF)) * scale[:, None]
    tl.store(out_ptr + idx, vals.to(out_ptr.dtype.element_ty), mask=bmask[:, None])


def configure(mode=None, groups=None, num_warps=None):
    """Выбор варианта ядра (C5). Звать до precompile.start(), чтобы скомпилировался именно он."""
    if mode is not None:
        if mode not in ("join", "index"):
            raise ValueError(f"mode: join | index, а не {mode!r}")
        CONFIG["mode"] = mode
    if groups is not None:
        if groups < 1 or groups & (groups - 1):
            raise ValueError("groups — степень двойки")
        CONFIG["groups"] = int(groups)
    if num_warps is not None:
        CONFIG["num_warps"] = int(num_warps)
    return dict(CONFIG)


def _codes(device):
    key = (device.type, device.index)
    codes = _CODES.get(key)
    if codes is None:
        codes = torch.tensor(NF4_CODEBOOK, dtype=torch.float32, device=device)
        _CODES[key] = codes
    return codes


def _dummy(device):
    key = (device.type, device.index)
    t = _DUMMY.get(key)
    if t is None:
        t = torch.zeros(1, dtype=torch.float32, device=device)
        _DUMMY[key] = t
    return t


def _launch(packed, absmax, code2, absmax2, offset, n, blocksize, out_dtype, out=None):
    """Общий запуск ядра. absmax: uint8 (вложенное квантование) или fp32 (обычное)."""
    device = packed.device
    n_blocks = n // blocksize
    nested = absmax.dtype == torch.uint8
    if out is None:
        out = torch.empty(n, dtype=out_dtype, device=device)
    dummy = _dummy(device)
    if nested:
        code2 = code2.view(-1)
        absmax2 = absmax2.view(-1)
        if offset is None:
            offset = dummy
        elif not torch.is_tensor(offset):
            offset = torch.full((1,), float(offset), dtype=torch.float32, device=device)
        offset = offset.reshape(1)
        if code2.dtype != torch.float32 or absmax2.dtype != torch.float32 \
                or offset.dtype != torch.float32:
            return None
    else:
        code2 = absmax2 = offset = dummy
        if absmax.dtype != torch.float32:
            absmax = absmax.float()
    groups = CONFIG["groups"]
    kernel = ida_nf4_dequant_kernel if CONFIG["mode"] == "join" else ida_nf4_dequant_idx_kernel
    kernel[(triton.cdiv(n_blocks, groups),)](
        packed.view(-1), absmax.view(-1), code2, absmax2, offset, _codes(device), out.view(-1),
        n_blocks, NESTED=nested, BLOCKSIZE=blocksize, GROUPS=groups,
        num_warps=CONFIG["num_warps"], enable_fp_fusion=False,
    )
    return out


def _fits(packed, shape, blocksize, quant_type):
    if not (torch.is_tensor(packed) and packed.is_cuda and packed.dtype == torch.uint8):
        return False
    if quant_type != "nf4" or len(shape) != 2 or blocksize % 2:
        return False
    n = shape[0] * shape[1]
    return n % blocksize == 0 and packed.numel() * 2 == n and packed.is_contiguous()


def nf4_dequantize(packed, state, out_dtype=None):
    """Аналог bitsandbytes.functional.dequantize_4bit(packed, state). None — не берёмся."""
    shape = tuple(state.shape)
    blocksize = int(state.blocksize)
    if not _fits(packed, shape, blocksize, getattr(state, "quant_type", "nf4")):
        return None
    if out_dtype is None:
        out_dtype = state.dtype if state.dtype in (torch.float16, torch.float32) else torch.float16
    n = shape[0] * shape[1]
    if getattr(state, "nested", False):
        s2 = state.state2
        if int(s2.blocksize) != 256:
            return None
        res = _launch(packed, state.absmax, s2.code, s2.absmax, state.offset, n, blocksize,
                      out_dtype)
    else:
        res = _launch(packed, state.absmax, None, None, None, n, blocksize, out_dtype)
    return None if res is None else res.view(shape)


def _patched_dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                             quant_type="fp4"):
    """Перехват bitsandbytes.functional.dequantize_4bit (обратный проход MatMul4Bit)."""
    result = None
    if quant_state is not None and absmax is None:
        result = nf4_dequantize(A, quant_state)
    if result is None:
        NF4_CALLS["skipped"] += 1
        return _ORIGINAL_DEQUANT(A, quant_state, absmax=absmax, out=out, blocksize=blocksize,
                                 quant_type=quant_type)
    NF4_CALLS["backward"] += 1
    if out is not None:
        out.copy_(result)
        return out
    return result


def _patched_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                      absmax_8bit=None, absmax_code=None, absmax_offset=None):
    """Перехват bnb `_dequant_linear_fallback` (прямой проход): распаковка + F.linear."""
    shape = tuple(shapeB)
    ok = _fits(B, shape, int(blocksize), quant_type) and A.dtype in (torch.float16, torch.float32)
    W = None
    if ok:
        n = shape[0] * shape[1]
        if absmax_8bit is not None:
            W = _launch(B, absmax_8bit, absmax_code, absmax, absmax_offset, n, int(blocksize),
                        A.dtype)
        else:
            W = _launch(B, absmax, None, None, None, n, int(blocksize), A.dtype)
    if W is None:
        NF4_CALLS["skipped"] += 1
        return _ORIGINAL_FALLBACK(A, B, shapeB, absmax, blocksize, quant_type, bias,
                                  absmax_8bit=absmax_8bit, absmax_code=absmax_code,
                                  absmax_offset=absmax_offset)
    NF4_CALLS["forward"] += 1
    return torch.nn.functional.linear(A, W.view(shape), bias)


def patch_nf4():
    """Подменить распаковку NF4 своим ядром в обоих проходах."""
    global _ORIGINAL_DEQUANT, _ORIGINAL_FALLBACK
    import bitsandbytes.functional as functional

    if _ORIGINAL_DEQUANT is None:
        _ORIGINAL_DEQUANT = functional.dequantize_4bit
        functional.dequantize_4bit = _patched_dequantize_4bit
    try:
        import bitsandbytes.backends.cuda.ops as cuda_ops
        if _ORIGINAL_FALLBACK is None and hasattr(cuda_ops, "_dequant_linear_fallback"):
            _ORIGINAL_FALLBACK = cuda_ops._dequant_linear_fallback
            cuda_ops._dequant_linear_fallback = _patched_fallback
    except Exception as exc:  # noqa: BLE001 - без CUDA-бэкенда остаётся только обратный проход
        print(f"[fused_nf4] прямой проход не перехвачен: {exc!r}", flush=True)
    print(f"[fused_nf4] v4: backward={_ORIGINAL_DEQUANT is not None} "
          f"forward={_ORIGINAL_FALLBACK is not None}", flush=True)


def nf4_debug():
    return dict(NF4_CALLS)
