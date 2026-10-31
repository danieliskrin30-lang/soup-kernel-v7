"""Тест распаковки NF4: сверяем с bitsandbytes.functional.dequantize_4bit.

Запуск: python -m pytest solution -q
"""

import pytest
import torch

pytest.importorskip("triton")
bnb_functional = pytest.importorskip("bitsandbytes.functional")

try:
    from solution.fused_nf4 import nf4_dequantize
except ImportError:
    from fused_nf4 import nf4_dequantize


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
@pytest.mark.parametrize("compress", [False, True])
def test_nf4_dequant_matches_bnb(compress):
    torch.manual_seed(0)
    x = torch.randn(96, 1536, device="cuda", dtype=torch.float16)

    packed, state = bnb_functional.quantize_4bit(
        x, blocksize=64, compress_statistics=compress, quant_type="nf4")

    ref = bnb_functional.dequantize_4bit(packed, state)
    ours = nf4_dequantize(packed, state)

    assert ours is not None, "наше ядро отказалось (тип квантования?)"
    assert ours.shape == ref.shape
    err = (ours.float() - ref.float()).abs().max().item()
    assert err < 5e-3, f"максимальная ошибка {err} (compressed={compress})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
def test_nf4_dequant_relative_error():
    """Относительная ошибка должна быть на уровне fp16-округления."""
    torch.manual_seed(1)
    x = torch.randn(64, 8960, device="cuda", dtype=torch.float16)
    packed, state = bnb_functional.quantize_4bit(
        x, blocksize=64, compress_statistics=True, quant_type="nf4")

    ref = bnb_functional.dequantize_4bit(packed, state).float()
    ours = nf4_dequantize(packed, state).float()

    denom = ref.abs().mean().item() + 1e-6
    rel = (ours - ref).abs().mean().item() / denom
    assert rel < 1e-2, f"средняя относительная ошибка {rel:.5f}"
