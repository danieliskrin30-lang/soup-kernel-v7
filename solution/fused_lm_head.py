"""Своё ядро выходного умножения на словарь (lm_head) на Triton.

Что это за операция. В конце модели скрытое состояние каждого токена (1536 чисел) умножается
на матрицу весов словаря (151 936 × 1536), чтобы получить оценки по всем словам. Это самый
большой тензор за шаг: 940 токенов × 151 936 слов.

Как делает PyTorch: вызывает cuBLAS. Быстро, но это чужая библиотека — нам она в зачёт не идёт.
Как делаем мы: обычный тайловый GEMM на Triton — блок 64×128, накопление в fp32, обход
матриц блоками. Обратный проход нужен только по входу (dX = dY @ W): веса словаря заморожены,
поэтому dW не считаем — это экономит половину работы.

Идея (тайловый GEMM, group-ordering для кэша) — стандартный приём из туториала Triton по matmul
и из cuBLAS. Код написан самостоятельно; от примера из шаблона (example_swiglu) отличается
полностью — другая операция, другая структура, другой патч.
"""

import torch
import triton
import triton.language as tl

LM_CALLS = {"calls": 0, "fused": 0, "skipped": 0}
_ORIGINAL_LINEAR = None


@triton.jit
def fused_lm_head_forward_kernel(
    X, W, Y, M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # offs_k не сдвигаем: он нужен для масок, а двигаются только указатели
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K),
                    other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_K),
                    other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def fused_lm_head_backward_kernel(
    DY, W, DX, M, N, K,
    stride_dym, stride_dyn,
    stride_wk2, stride_wn2,      # для dX = dY @ W: идём по словарю (N) и скрытым (K)
    stride_dxm, stride_dxk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Здесь N = словарь (суммируем по нему), K = скрытый размер (выход)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_l = tl.arange(0, BLOCK_K)

    dy_ptrs = DY + offs_m[:, None] * stride_dym + offs_l[None, :] * stride_dyn
    w_ptrs = W + offs_l[:, None] * stride_wk2 + offs_k[None, :] * stride_wn2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_K)):
        # offs_l не сдвигаем: он нужен для масок, а двигаются только указатели
        dy = tl.load(dy_ptrs, mask=(offs_m[:, None] < M) & (offs_l[None, :] < N - n * BLOCK_K),
                     other=0.0)
        w = tl.load(w_ptrs, mask=(offs_l[:, None] < N - n * BLOCK_K) & (offs_k[None, :] < K),
                    other=0.0)
        acc += tl.dot(dy, w)
        dy_ptrs += BLOCK_K * stride_dyn
        w_ptrs += BLOCK_K * stride_wk2

    dx_ptrs = DX + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk
    tl.store(dx_ptrs, acc.to(DX.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


class FusedLmHead(torch.autograd.Function):
    """Y = X @ W^T для большого словаря; обратный проход только по X (вес заморожен)."""

    @staticmethod
    def forward(ctx, x, weight):
        shape = x.shape
        k = shape[-1]
        x2 = x.reshape(-1, k).contiguous()
        m = x2.shape[0]
        n = weight.shape[0]

        y = torch.empty(m, n, dtype=x.dtype, device=x.device)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        fused_lm_head_forward_kernel[grid](
            x2, weight, y, m, n, k,
            x2.stride(0), x2.stride(1),
            weight.stride(0), weight.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
            num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(weight)
        ctx.out_shape = shape[:-1] + (n,)
        return y.view(ctx.out_shape)

    @staticmethod
    def backward(ctx, grad_out):
        (weight,) = ctx.saved_tensors
        shape = grad_out.shape
        n = shape[-1]
        k = weight.shape[1]
        grad2d = grad_out.reshape(-1, n).contiguous()
        m = grad2d.shape[0]

        dx = torch.empty(m, k, dtype=grad_out.dtype, device=grad_out.device)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(k, BLOCK_N),)
        fused_lm_head_backward_kernel[grid](
            grad2d, weight, dx, m, n, k,
            grad2d.stride(0), grad2d.stride(1),
            weight.stride(0), weight.stride(1),    # W[n, k]: шаг по словарю и по скрытым
            dx.stride(0), dx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
            num_warps=4, num_stages=2,
        )
        return dx.view(ctx.out_shape[:-1] + (k,)), None


def _patched_linear(input, weight, bias=None):
    """Перехват F.linear: своё ядро только для выхода на большой словарь."""
    LM_CALLS["calls"] += 1
    fits = (torch.is_tensor(input) and input.is_cuda and input.dim() >= 2
            and input.dtype in (torch.float16, torch.bfloat16)
            and bias is None and weight is not None and weight.dim() == 2
            and weight.shape[0] >= 50000 and weight.dtype in (torch.float16, torch.bfloat16)
            and input.shape[-1] == weight.shape[1])
    if not fits:
        LM_CALLS["skipped"] += 1
        return _ORIGINAL_LINEAR(input, weight, bias)

    LM_CALLS["fused"] += 1
    if LM_CALLS["fused"] == 1:
        print(f"[fused_lm_head] включилось: x{tuple(input.shape)} w{tuple(weight.shape)}",
              flush=True)
    return FusedLmHead.apply(input, weight)


def patch_lm_head():
    """Подменить F.linear, чтобы выход на словарь считался нашим ядром."""
    global _ORIGINAL_LINEAR
    import torch.nn.functional as functional

    if _ORIGINAL_LINEAR is None:
        _ORIGINAL_LINEAR = functional.linear
        functional.linear = _patched_linear
        print("[fused_lm_head] перехвачен F.linear", flush=True)


def lm_head_debug():
    return dict(LM_CALLS)
