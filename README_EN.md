# Soup Kernel Challenge: own Triton kernels for QLoRA on a free T4

**Team:** ID Automation (`IDA`), 1 person · **Track:** AI 1 "AI Engineering by Soup" · **Hackathon:** Decentrathon 6.0
**Repository:** `github.com/BAITC-Hacks/hack-8539eaba-id-automation` · *Russian version (primary): `README.md`*

Final run with the jury protocol: `harness/run_task.py --repeats 2` (base, solution, solution,
base in one Kaggle T4 session) — the saved version of the public notebook
https://www.kaggle.com/code/danieliskrin30/notebookd124ebce8b, medians from `runs/result.json`:

| Metric | Base Soup | Our solution | Ratio |
|---|---|---|---|
| `soup train` time, s (median of 2 runs) | 86.27 | **66.10** | **speedup 1.305×** |
| Peak GPU memory (`nvidia-smi`), MiB | 14 345 | **5 879** | **base/ours 2.440×** |
| Eval loss (public rows) | 2.04962 | 2.04962 | +0.000% |
| Share of our kernels in training time (steps 3+) | — | **9.1%** (lower of two) | threshold ≥5% |
| `harness/check_rules.py` | — | **10/10 PASS** | — |
| Soup tests (`tests/`) / our tests (`pytest solution`) | 79 | **79 passed / 70 passed** | — |

The task, data, batch contents and order, number of steps, LoRA, optimizer and lr schedule are
unchanged. `harness/`, `tests/`, `task/` are unchanged.

---

## 1. What was done and why

### 1.1. Our kernels that run in training

Time is the harness `own_kernel_ms` (CUDA events, steps 3–64), run `solution_2` of the public
notebook — the one check 4 takes as the lower share.

| Kernel (`@triton.jit`) | File | Replaces | Launches per run | Time, ms | Share |
|---|---|---|---|---|---|
| `ida_nf4_dequant_kernel` | `fused_nf4.py` | NF4 unpacking in forward (`_dequant_linear_fallback`) and backward (`dequantize_4bit`) | 24 897 | 2 061.5 | 5.09% |
| `fused_rmsnorm_backward_kernel` | `fused_rmsnorm.py` | `Qwen2RMSNorm`, backward | 3 585 | 386.7 | 0.96% |
| `fused_rmsnorm_forward_kernel` | `fused_rmsnorm.py` | `Qwen2RMSNorm.forward` | 3 649 | 292.8 | 0.72% |
| `ida_attn_prologue_fwd_kernel` | `fused_attn_prologue.py` | RoPE q/k + fp16 cast + `repeat_kv` for SDPA | 1 793 | 233.2 | 0.58% |
| `ida_attn_prologue_bwd_kernel` | `fused_attn_prologue.py` | the same in backward, GQA group sum in fp32 | 1 793 | 220.1 | 0.54% |
| `ida_entropy_argmax_kernel` | `fused_metrics.py` | TRL entropy and argmax for `mean_token_accuracy` | 65 | 203.1 | 0.50% |
| `fused_ce_backward_kernel` | `fused_ce.py` | cross-entropy gradient (written into the logits buffer) | 66 | 177.0 | 0.44% |
| `fused_ce_forward_kernel` | `fused_ce.py` | model cross-entropy and per-row losses for `rewind.jsonl` | 323 | 113.2 | 0.28% |
| **Total** | | | | **3 687.7** | **9.11%** |

The other run (`solution_1`): 9.24%. Launch counts do not change between runs (same code,
same batches).

Only in `precompile.py` (1–2 launches; in training a fallback when the main path does not apply):
`ida_rope_kernel`, `fused_entropy_forward_kernel`. Not launched: `fused_attention_*`,
`fused_lm_head_*` (see 3.2), `ida_nf4_dequant_idx_kernel`, `ida_rmsnorm_*_rows_kernel`
(variants for `configure()`; the benchmark kept the v4 settings). The template example
`example_swiglu_*` runs and does not count.

### 1.2. How the kernels work

**NF4 (`fused_nf4.py`).** In bitsandbytes 0.50.2 on a T4 with M > 512 rows (here M = 4 × length)
the forward pass does not use the fused `gemm_4bit` but `_dequant_linear_fallback`: unpack →
`F.linear`. The backward pass unpacks again in `MatMul4Bit.backward`. That is 392 unpackings
per step (7 matrices × 28 layers × 2 passes). A program handles a [64 blocks × 64 values] tile:
the double-quant scale (`code2[absmax8] * absmax2 + offset`) is computed inside the kernel once
per block, each byte is read once, nibbles are interleaved with `tl.join` (high first), `offset`
is read by pointer — no CPU↔GPU syncs. `enable_fp_fusion=False`: bit-identical to
`dequantize_4bit`.

**RMSNorm (`fused_rmsnorm.py`).** One row per program, `rstd` in fp32 registers, only `rstd`
is kept for backward; `dx = s·g − (s³/n)·x·(g·x)`, `g = dy·w`.

**Attention prologue (`fused_attn_prologue.py`).** Replaces `Qwen2Attention.forward`. The
forward kernel, one program per position (b, l), in one launch: RoPE for 12 q heads and 2 k
heads, fp16 cast and k, v expansion to 12 heads (GQA 12/2) — directly in the SDPA layout. Then
SDPA with `is_causal=True` and no padding mask, backend pinned to `EFFICIENT_ATTENTION`
(cutlass `fmha_cutlassF/B_*_sm75`): if cutlass cannot take it, an exception and the stock path,
never a silent fallback to math. The backward kernel: fp16 → fp32, sum over the 6 heads of a
group in fp32, RoPE backward. Why the mask is not needed — section 4.

**Cross-entropy (`fused_ce.py`).** Online softmax over 4 096-word chunks, only `lse [rows]` kept
for backward. The `LOSS_MAPPING["ForCausalLM"]` entry is replaced (transformers 5.16.1 resolves
the loss function through it), so `logits.float()` never runs: the kernel reads fp16 logits and
computes in fp32 registers. The gradient is written **into the logits buffer itself** (each
element is read and overwritten by the same kernel thread). The same forward kernel computes
per-row losses for `rewind.jsonl` (`reduction="none"`, one sync per step instead of eight).

**Entropy + argmax (`fused_metrics.py`).** One pass over the logits: entropy (same operations in
the same order as the former entropy kernel — bit-identical) and argmax (first maximum, like
`torch.argmax`). This removes the `outputs.logits[..., :-1, :].contiguous()` copy that TRL builds
only for argmax.

### 1.3. Changes without new kernels

| What | Before | After | Effect |
|---|---|---|---|
| Embedding / lm_head 151 936 × 1 536 | fp32 (after `prepare_model_for_kbit_training`); under autocast an fp16 copy every step, kept until backward | stored in fp16; embedding output upcast to fp32 by a hook | −467 MB weight, −467 MB copy (estimate from tensor sizes) |
| MLP gate/up/down | `Linear4bit` returns fp32, SwiGLU saves fp32 | fp16, SwiGLU saves fp16 | ~−1 GB activations (estimate) |
| `outputs.logits` | accelerate converts to fp32 | logits stay fp16 | −1 logits buffer (estimate) |
| TRL accuracy copy of the logits + a separate logits-gradient buffer | two logits-sized buffers at the peak | none (1.2) | −600 MiB peak* |
| Triton compilation | at steps 0–1 | in a thread of the same process while Soup loads the model | −7 s* |

\* Separate A/B runs on the same T4, not from the final `result.json`: without `precompile`
steps 0–1 = 7.65 s and wall 66.8 s, with it 1.44 s and 59.8 s; peak 6 479 → 5 879 MiB before/after
1.2 (metrics + in-place gradient); attention prologue + `is_causal` ≈ −1.2 s (attention kernels
847 → 569 ms over 16 profiled steps).

### 1.4. Sources of ideas (code written by us)

| Idea | Source | Where |
|---|---|---|
| Online softmax, block processing | FlashAttention, [arXiv 2205.14135](https://arxiv.org/abs/2205.14135); Triton tutorial [`06-fused-attention`](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | `fused_ce.py`, `fused_entropy.py`, `fused_metrics.py`, `fused_attention.py` (off) |
| Attention backward without atomics (plan, 3.2) | FlashAttention-2, [arXiv 2307.08691](https://arxiv.org/abs/2307.08691) | — |
| Chunked CE over the vocabulary, keep only `logsumexp` | Unsloth [`unsloth/kernels/cross_entropy_loss.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/cross_entropy_loss.py); Liger-Kernel `FusedLinearCrossEntropy`, [arXiv 2410.10989](https://arxiv.org/abs/2410.10989) | `fused_ce.py` |
| NF4 table, 64-value blocks, double quantization | QLoRA, [arXiv 2305.14314](https://arxiv.org/abs/2305.14314); bitsandbytes 0.50.2 sources | `fused_nf4.py` |
| Tiled grouped GEMM | Triton tutorial [`03-matrix-multiplication`](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | `fused_lm_head.py` (off) |

No banned library (unsloth, liger-kernel, xformers, flash-attn, torchao, …) is imported or
loaded — check 2.

---

## 2. How to run

Public Kaggle notebook (Run All without edits, results saved): **https://www.kaggle.com/code/danieliskrin30/notebookd124ebce8b**
The notebook clones the public mirror of the same commit, `github.com/danieliskrin30-lang/soup-kernel-v7`, so it runs without our
tokens; the platform repository is `github.com/BAITC-Hacks/hack-8539eaba-id-automation`.
Final commit (public mirror of the same code): **https://github.com/danieliskrin30-lang/soup-kernel-v7/commit/655a25344658badfdee1b8107cee6a9066ab68ef**

On a clean Kaggle notebook (Accelerator: GPU T4, Internet: on):

```bash
python -m pip install -q "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" \
    "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
git clone https://github.com/danieliskrin30-lang/soup-kernel-v7.git repo
cd repo && git checkout 655a25344658badfdee1b8107cee6a9066ab68ef
python task/prepare_data.py                               # data/train.jsonl, data/eval_public.jsonl
python -m pytest solution -q                              # our kernels vs references: 70 passed
python harness/run_task.py --repeats 2                    # jury protocol, ~30 min
python harness/check_rules.py --result runs/result.json   # 10 checks
python harness/score.py runs/result.json                  # s and m ratios
```

Faster, solution only (checks that need the base do not pass in this mode):
`python harness/run_task.py --only solution --out runs_try`.

Environment variables (**not set** in the submission): `IDA_DIAG=1` — memory, dtype and path
counters in `train.log`; `IDA_PRECOMPILE=0` — disable threaded compilation. Kernel settings
search: `python solution/bench_c.py` (not a test).

---

## 3. Results and limitations

### 3.1. Results (Kaggle T4, `runs/result.json` and `runs/checks.json`, `--repeats 2`)

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| `soup train` time, s, median | 86.27 | 66.10 | speedup **1.305×** (−20.17 s) |
| Training loop `train_loop_s`, s, median | 58.57 | 42.30 | 1.385× |
| Peak GPU memory, MiB (max over runs) | 14 345 | 5 879 | base/ours **2.440×** (−8 466 MiB) |
| Eval loss (public rows) | 2.04962 | 2.04962 | +0.000% |
| Loss curve vs base (check 6) | — | PASS | tolerance 1% / 3% |
| Share of our kernels (check 4) | — | 9.1% (runs: 9.24% and 9.11%) | threshold 5% |
| Time / memory vs base (check 8) | — | ×0.766 / ×0.410 | tolerance ×1.10 |
| Tests | 79 Soup | 79 passed; `pytest solution`: 70 passed | — |

Spread over repeats:

| Run | Time, s | Peak, MiB |
|---|---|---|
| base_1 / base_2 | 88.78 / 83.75 (spread 5.03 s, 5.8%) | 14 345 / 14 345 |
| solution_1 / solution_2 | 65.80 / 66.40 (spread 0.60 s, 0.9%) | 5 879 / 5 879 |

The base time varies more on a T4 than the solution time; the memory peak repeats to the MiB.

Not measured: the contribution of each change separately (ablations only for `precompile`, the
attention prologue and item 1.2, see the footnote in 1.3), the phase of the memory peak in the
final run.

### 3.2. Limitations

| What we tried | Result | Why it is off |
|---|---|---|
| A full own attention kernel (`fused_attention.py`: block online softmax, GQA 12/2, causal + padding mask) | matches SDPA on synthetic tests; on real data `max_diff` 0.24 at typical values 0.18, eval loss 2.32 instead of 2.05 | mismatch in rows masked by padding and in the last partial block; ~30× slower than built-in because of atomic `dK/dV` adds. Our own prologue + cutlass runs instead (1.2) |
| Own vocabulary projection (`fused_lm_head.py`, 940 × 1536 × 151 936, `dX` only) | tests vs `F.linear` pass | not wired in: lm_head is not quantized (tied to the embedding); lm_head stays on cuBLAS, our GEMM vs cuBLAS speed not measured |
| Not computing padding rows (whole model or lm_head only) | estimate −8…−14 s and −1.5 GB | not done: risk of falling under the `packing` / `padding_free` ban |
| Own 4-bit GEMM, cuBLASLt, fused q/k/v, other NF4/RMSNorm settings (`bench_c.py`) | 0…1% by estimate or benchmark | below the measurement spread |
| Speed | 1.305× vs base | the 1.05 threshold is passed; other teams' results are unknown to us |

**Next:** attention without atomic adds (two FlashAttention-2-style kernels), keeping k/v at 2
heads inside SDPA and recomputing the LoRA input in backward (≈ −0.4 GB by estimate), our own LoRA
forward with a single input cast for q/k/v.

---

## 4. Disclosures (rules 5.4.4 and 5.4.10)

**Attention with `is_causal` instead of the padding mask.** `DataCollatorForLanguageModeling` in
TRL 0.29.1 hard-codes right padding (`padding_side="right"`). A real token at position t only
sees positions ≤ t, all of them real, so the padding mask closes nothing beyond the causal one
for real tokens. Padding rows are computed differently, but their gradient is exactly 0 (labels
−100) and TRL multiplies them by `attention_mask = 0` in the metrics. A guard checks every step
that the mask looks like `1…1 0…0` and that there are no `position_ids`, cache or `use_cache` —
otherwise the stock transformers path runs. Batches, tokens and computations on real tokens are
the same; this is not packing. `test_attn.py` compares the whole layer with stock masked SDPA on
real rows.

**`SFTTrainer.compute_loss`.** `fused_metrics.py` holds a copy of the default branch of the
method from TRL 0.29.1 (Apache-2.0, `trl/trainer/sft_trainer.py`); the only change is that
entropy and argmax come from our kernel instead of `entropy_from_logits` and the logits copy.
Metric names, masks, sums, `gather_for_metrics`, `.item()` are the same lines. Non-default
branches (liger, prompt tuning, `shift_labels`, aux_loss) call the original TRL method.

**In-place logits gradient.** After the CE backward nobody needs the logits: TRL metrics and
Soup per-row losses are computed in forward, lm_head keeps its input and weight for backward,
not its output. Enabled only for the model loss (`LOSS_MAPPING`), never for arbitrary
`F.cross_entropy` calls (`test_metrics.py`).

**Soup patch.** `soup_cli.trainer.rewind_hf.row_losses` is replaced by a version with the same
numbers and one sync instead of eight (`test_c.py` checks against the original).

**Embedding / lm_head in fp16.** Soup loads the model in fp16, peft upcasts non-4-bit parameters
to fp32 without changing values (fp16 → fp32 → fp16 is lossless). The embedding output is upcast
to fp32; the residual stream is fp32 as in the base. The matrix is frozen and not in the adapter.

**MLP in fp16.** In the base, `Linear4bit` casts its input to fp16 itself and widens the output to
fp32 (exactly). We pass fp16 explicitly and do not widen; SwiGLU computes in fp32 registers and
rounds once, as `down_proj` did in the base; gate/up input gradients are summed in fp32. No
rounding moved (`test_v4.py::test_mlp_fp16_bitwise_equal`, `torch.equal`).

**Where results are not bit-identical:** SDPA with `is_causal` vs masked SDPA — fp16 rounding
level on real rows; prologue dk/dv — summation order over 6 heads; cross-entropy, entropy and
RMSNorm — fp32 summation order. All within the tolerances of checks 5–7 (result: eval loss
+0.000% vs base).

**`precompile.py`.** Our kernels compile in a separate thread **of the same process** while Soup
loads the model (done in ~9 s, before `Trainer.__init__`). The Triton cache is empty in every run
(the harness sets its own `TRITON_CACHE_DIR`), no prebuilt caches, all compile time stays inside
the measured wall time.

**AI tools.** DeepSeek — drafts and debugging of kernels v1–v3. Claude (Anthropic) — reading the
bitsandbytes / transformers / peft / accelerate / TRL / Soup sources, finding redundant copies
and syncs, code v4–v7: `fused_nf4.py`, `fused_ce.py`, `fused_rope.py`, `fused_attn_prologue.py`,
`fused_metrics.py`, `memory_patches.py`, `precompile.py`, `diag.py`, `bench_c.py`, the variants in
`fused_rmsnorm.py`, tests `test_v4.py`, `test_c.py`, `test_attn.py`, `test_metrics.py`. All
numbers come from harness runs; every kernel is covered by a test against a reference.

**Third-party components.** Python 3.13.15, `torch` 2.11.0+cu128, `triton` 3.6.0, `transformers`
5.16.1, `trl` 0.29.1, `peft` 0.20.0, `bitsandbytes` 0.50.2, `accelerate` 1.14.0, `datasets` 4.8.5,
`soup-cli` 0.77.0; model `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0); dataset
`AmanMussa/kazakh-instruction-v2` (MIT per its card; a translation of Stanford Alpaca,
CC BY-NC 4.0). Starter kit — Apache-2.0, see `NOTICE`.

---

## 5. Measurement discipline

- `harness/`, `tests/`, `task/` and the measurement settings (allocator, visible GPUs) are
  unchanged; `packages_changed` in `result.json` is empty.
- Diagnostics are off in the submission: `IDA_DIAG` is not set, `diag.py` does nothing.
- `precompile.py` is on by default and runs inside the same process and the same measurement;
  no background processes and no second CUDA streams in training.
- All numbers in sections 1.1 and 3.1 come from `runs/result.json` and `runs/checks.json` of
  the final `--repeats 2` run (the saved version of the public notebook). Numbers from separate A/B runs are marked in 1.3. Anything in
  neither is marked "not measured".

---

## 6. Layout

```
solution/
  __init__.py            apply(): patch order, each patch is one line
  fused_nf4.py           ida_nf4_dequant_kernel — NF4, forward and backward (+ configure)
  fused_rmsnorm.py       RMSNorm, forward + backward (+ configure)
  fused_attn_prologue.py ida_attn_prologue_* — RoPE + fp16 + GQA, SDPA is_causal on cutlass
  fused_ce.py            cross-entropy (LOSS_MAPPING, in-place gradient, reduction none, rewind)
  fused_metrics.py       ida_entropy_argmax_kernel + TRL compute_loss without a logits copy
  fused_entropy.py       entropy (fallback)
  fused_rope.py          ida_rope_kernel — RoPE (fallback)
  memory_patches.py      fp16 embedding, fp16 MLP, fp16 logits in accelerate
  precompile.py          kernel compilation in parallel with model loading
  diag.py                diagnostics, only with IDA_DIAG=1
  bench_c.py             NF4/RMSNorm settings search (not a test)
  swiglu.py              template example (runs, does not count)
  fused_attention.py, fused_lm_head.py   off, see 3.2
  test_*.py              70 kernel tests vs references
harness/, tests/, task/, templates/      starter kit, unchanged
```
