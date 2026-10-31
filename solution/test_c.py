"""Тесты пункта C (v5): варианты ядер NF4 и RMSNorm, потери строк rewind одним блоком.

Запуск на T4: python -m pytest solution -q
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
pytest.importorskip("triton")


@pytest.fixture
def nf4_config():
    from solution import fused_nf4
    saved = fused_nf4.configure()
    yield fused_nf4
    fused_nf4.configure(**saved)


@pytest.fixture
def rms_config():
    from solution import fused_rmsnorm
    saved = fused_rmsnorm.configure()
    yield fused_rmsnorm
    fused_rmsnorm.configure(**saved)


# ---------- C5: NF4, каждый вариант побитно = bitsandbytes ----------

@pytest.mark.parametrize("mode", ["join", "index"])
@pytest.mark.parametrize("groups,warps", [(16, 4), (64, 4), (128, 8)])
@pytest.mark.parametrize("compress", [True, False])
def test_nf4_variants_bitwise_equal_bnb(nf4_config, mode, groups, warps, compress):
    F = pytest.importorskip("bitsandbytes.functional")
    nf4_config.configure(mode=mode, groups=groups, num_warps=warps)
    torch.manual_seed(7)
    for shape in ((256, 1536), (1536, 8960 // 4)):
        w = torch.randn(*shape, device="cuda", dtype=torch.float16)
        packed, st = F.quantize_4bit(w, blocksize=64, compress_statistics=compress,
                                     quant_type="nf4")
        ref = F.dequantize_4bit(packed, st)
        ours = nf4_config.nf4_dequantize(packed, st)
        assert torch.equal(ours, ref), (mode, groups, warps, shape,
                                        (ours.float() - ref.float()).abs().max().item())


def test_nf4_index_mode_tail_block(nf4_config):
    """Число блоков не кратно GROUPS: последняя программа обрабатывает неполную плитку."""
    F = pytest.importorskip("bitsandbytes.functional")
    nf4_config.configure(mode="index", groups=128, num_warps=4)
    w = torch.randn(3, 64 * 7, device="cuda", dtype=torch.float16)      # 21 блок
    packed, st = F.quantize_4bit(w, blocksize=64, compress_statistics=False, quant_type="nf4")
    assert torch.equal(nf4_config.nf4_dequantize(packed, st), F.dequantize_4bit(packed, st))


# ---------- C10: RMSNorm, плитка из нескольких строк против ядер v4 ----------

@pytest.mark.parametrize("rows_cfg,warps", [(2, 4), (4, 4), (4, 8), (8, 8)])
@pytest.mark.parametrize("m", [940, 1337])
def test_rmsnorm_rows_matches_v4(rms_config, rows_cfg, warps, m):
    torch.manual_seed(m)
    x = torch.randn(m, 1536, device="cuda", requires_grad=True)
    w = torch.randn(1536, device="cuda")
    dy = torch.randn(m, 1536, device="cuda")

    rms_config.configure(rows=1, num_warps=4)
    y0 = rms_config.rmsnorm(x, w, 1e-6)
    (dx0,) = torch.autograd.grad(y0, x, dy)

    rms_config.configure(rows=rows_cfg, num_warps=warps)
    y1 = rms_config.rmsnorm(x, w, 1e-6)
    (dx1,) = torch.autograd.grad(y1, x, dy)

    torch.testing.assert_close(y1, y0, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(dx1, dx0, atol=1e-5, rtol=1e-6)


def test_rmsnorm_rows_weight_grad(rms_config):
    """NEED_DW-ветка (в обучении вес заморожен, но путь должен быть верным)."""
    torch.manual_seed(3)
    x = torch.randn(130, 1536, device="cuda", requires_grad=True)
    w = torch.randn(1536, device="cuda", requires_grad=True)
    dy = torch.randn(130, 1536, device="cuda")
    rms_config.configure(rows=1, num_warps=4)
    gx0, gw0 = torch.autograd.grad(rms_config.rmsnorm(x, w, 1e-6), (x, w), dy)
    rms_config.configure(rows=4, num_warps=4)
    gx1, gw1 = torch.autograd.grad(rms_config.rmsnorm(x, w, 1e-6), (x, w), dy)
    torch.testing.assert_close(gx1, gx0, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(gw1, gw0, atol=1e-3, rtol=1e-5)


# ---------- C11: rewind одним блоком против оригинала Soup ----------

def test_batched_row_losses_matches_soup(monkeypatch):
    rewind_hf = pytest.importorskip("soup_cli.trainer.rewind_hf")
    import solution.fused_ce as fce

    original = fce._ORIGINAL_ROW_LOSSES or rewind_hf.row_losses
    fce._ORIGINAL_ROW_LOSSES = original
    # эталон — Soup в том виде, как он работает в обучении v4: его F.cross_entropy(reduction=
    # "none") уже идёт в наше ядро (fp32). Без этого Soup считал бы CE в fp16.
    if fce._ORIGINAL_CE is None:
        fce._ORIGINAL_CE = torch.nn.functional.cross_entropy
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", fce._patched_cross_entropy)
    torch.manual_seed(11)
    B, T, V = 4, 41, 151936
    logits = (torch.randn(B, T, V, device="cuda") * 2).half()
    labels = torch.randint(0, V, (B, T), device="cuda")
    labels[:, :9] = -100                     # промпт
    labels[2, 30:] = -100                    # паддинг справа
    labels[3, :] = -100                      # строка без ответа -> (0.0, 0)

    ref_loss, ref_tok = original(logits, labels)
    our_loss, our_tok = fce.batched_row_losses(logits, labels)
    assert our_tok == ref_tok
    assert our_loss[3] == 0.0 and our_tok[3] == 0
    for a, b in zip(our_loss, ref_loss):
        assert abs(a - b) <= 1e-5 * max(1.0, abs(b)), (our_loss, ref_loss)
