"""Тесты v5 7+8: метрики TRL одним проходом и градиент логитов на месте.

Запуск на T4: python -m pytest solution -q
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна видеокарта CUDA")
pytest.importorskip("triton")

V = 151936


def _logits(B=3, T=50, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(B, T, V, device="cuda") * 3).half()
    x[0, 5, 100] = x[0, 5, 140000] = 30.0      # равные максимумы в разных кусках -> 100
    x[0, 6, 10] = x[0, 6, 20] = 30.0           # равные максимумы в одном куске -> 10
    x[1, 7, V - 1] = 31.0                      # максимум в последнем неполном куске
    return x


def test_entropy_argmax_kernel():
    from solution.fused_entropy import entropy_from_logits as v4_entropy
    from solution.fused_metrics import entropy_argmax

    x = _logits()
    ent, idx = entropy_argmax(x)
    assert torch.equal(ent, v4_entropy(x)), "энтропия должна совпасть с ядром v4 побитно"
    assert torch.equal(idx, x.argmax(dim=-1))
    assert idx[0, 5].item() == 100 and idx[0, 6].item() == 10 and idx[1, 7].item() == V - 1
    ref = -(torch.log_softmax(x.float(), -1).exp() * torch.log_softmax(x.float(), -1)).sum(-1)
    torch.testing.assert_close(ent, ref, atol=1e-3, rtol=1e-4)


def _fake_trainer(trl_sft):
    t = object.__new__(trl_sft.SFTTrainer)
    t.args = SimpleNamespace(use_liger_kernel=False, loss_type="nll")
    t.model = torch.nn.Linear(1, 1).train()
    t.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x)
    t._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    t.num_virtual_tokens = 0
    t.aux_loss_enabled = False
    t._total_train_tokens = 0
    return t


def test_compute_loss_metrics_match_trl(monkeypatch):
    trl_sft = pytest.importorskip("trl.trainer.sft_trainer")
    import transformers
    import solution.fused_metrics as fm

    original = fm._ORIGINAL_COMPUTE_LOSS or trl_sft.SFTTrainer.compute_loss
    fm._ORIGINAL_COMPUTE_LOSS = original
    x16 = _logits(B=4, T=37, seed=1)
    labels = torch.randint(0, V, (4, 37), device="cuda")
    labels[:, :9] = -100
    am = torch.ones(4, 37, dtype=torch.long, device="cuda")
    am[2, 25:] = 0
    labels[2, 25:] = -100                            # паддинг справа
    loss = torch.tensor(1.234, device="cuda")

    def run(fn, logits):
        monkeypatch.setattr(transformers.Trainer, "compute_loss",
                            lambda self, model, inputs, return_outputs=False,
                            num_items_in_batch=None: (loss, SimpleNamespace(logits=logits)))
        t = _fake_trainer(trl_sft)
        inputs = {"input_ids": labels.clamp(min=0), "labels": labels.clone(),
                  "attention_mask": am.clone()}
        out = fn(t, t.model, inputs, return_outputs=True, num_items_in_batch=None)
        return t, out

    # база: логиты fp32 (как в базе после accelerate), оригинальный TRL
    t_ref, out_ref = run(original, x16.float())
    # v5: логиты fp16, наша ветка
    t_new, out_new = run(fm._compute_loss, x16)

    assert out_new[0] is loss and out_new[1].logits is x16
    m_ref, m_new = t_ref._metrics["train"], t_new._metrics["train"]
    assert sorted(m_ref) == sorted(m_new), (sorted(m_ref), sorted(m_new))
    assert m_new["mean_token_accuracy"] == m_ref["mean_token_accuracy"]
    assert m_new["num_tokens"] == m_ref["num_tokens"]
    assert abs(m_new["entropy"][0] - m_ref["entropy"][0]) <= 1e-4 * abs(m_ref["entropy"][0])


def test_compute_loss_falls_back_for_unusual_trainers(monkeypatch):
    trl_sft = pytest.importorskip("trl.trainer.sft_trainer")
    import solution.fused_metrics as fm

    calls = []
    monkeypatch.setattr(fm, "_ORIGINAL_COMPUTE_LOSS",
                        lambda self, model, inputs, return_outputs=False,
                        num_items_in_batch=None: calls.append(1) or "orig")
    t = _fake_trainer(trl_sft)
    t.args.use_liger_kernel = True
    assert fm._compute_loss(t, t.model, {"labels": 0, "attention_mask": 0}) == "orig"
    t = _fake_trainer(trl_sft)
    assert fm._compute_loss(t, t.model, {"labels": 0, "attention_mask": 0,
                                         "shift_labels": 0}) == "orig"
    assert len(calls) == 2


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_inplace_logits_grad_equals_regular(reduction):
    from solution.fused_ce import INPLACE_GRAD, fused_cross_entropy

    torch.manual_seed(2)
    rows, vocab = 64, 4096 * 3 + 17
    labels = torch.randint(0, vocab, (rows,), device="cuda")
    labels[3] = -100
    grads = []
    for inplace in (False, True):
        torch.manual_seed(5)
        w = (torch.randn(rows, vocab, device="cuda") * 2).half().requires_grad_(True)
        logits = w * 1.0                              # не лист, как выход lm_head
        used = INPLACE_GRAD["used"]
        (fused_cross_entropy(logits, labels, -100, reduction, inplace) * 256).backward()
        assert INPLACE_GRAD["used"] == used + (1 if inplace else 0)
        grads.append(w.grad)
    assert torch.equal(grads[0], grads[1])


def test_causal_lm_loss_inplace_matches_transformers():
    pytest.importorskip("transformers")
    from transformers.loss.loss_utils import ForCausalLMLoss
    import solution.fused_ce as fce

    fce._ORIGINAL_LOSS = fce._ORIGINAL_LOSS or ForCausalLMLoss
    torch.manual_seed(3)
    B, T = 2, 37
    labels = torch.randint(0, V, (B, T), device="cuda")
    labels[:, :5] = -100
    n = (labels[:, 1:] != -100).sum()
    base = (torch.randn(B, T, V, device="cuda") * 3).half()

    w1 = base.clone().requires_grad_(True)
    ref = ForCausalLMLoss(w1 * 1.0, labels, V, num_items_in_batch=n)
    (ref * 1024).backward()

    saved = fce.INPLACE_GRAD["on"]
    fce.INPLACE_GRAD["on"] = True
    try:
        w2 = base.clone().requires_grad_(True)
        ours = fce._our_for_causal_lm_loss(w2 * 1.0, labels, V, num_items_in_batch=n)
        (ours * 1024).backward()
    finally:
        fce.INPLACE_GRAD["on"] = saved
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(w2.grad.float(), w1.grad.float(), atol=2e-3, rtol=1e-2)


def test_plain_cross_entropy_never_inplace():
    """Перехват F.cross_entropy (не потери модели) не трогает буфер входа даже при on=True."""
    import solution.fused_ce as fce

    if fce._ORIGINAL_CE is None:
        fce._ORIGINAL_CE = torch.nn.functional.cross_entropy
    saved = fce.INPLACE_GRAD["on"]
    fce.INPLACE_GRAD["on"] = True
    try:
        w = (torch.randn(8, V, device="cuda")).half().requires_grad_(True)
        logits = w * 1.0
        before = logits.detach().clone()
        labels = torch.randint(0, V, (8,), device="cuda")
        fce._patched_cross_entropy(logits, labels, reduction="mean").backward()
        assert torch.equal(logits.detach(), before)
    finally:
        fce.INPLACE_GRAD["on"] = saved
