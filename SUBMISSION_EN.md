# Submission: ID Automation (`IDA`)

Track **AI 1 — AI Engineering by Soup**, task **Soup Kernel Challenge**, Decentrathon 6.0.
*(English version. The primary version is `SUBMISSION.md`.)*

## Links

- Repository, final commit: `https://github.com/BAITC-Hacks/hack-8539eaba-id-automation/commit/<COMMIT_SHA>`
- Public Kaggle notebook, saved version (`COMMIT` in it = the sha above; the same link is in
  `README.md`): `<NOTEBOOK_URL>`
- Demo video (up to 3 minutes): `<VIDEO_URL>`
- Presentation: `<SLIDES_URL>`

## Team (1 person)

| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Daniel Iskrin | captain, all work (kernels, integration, measurements, docs) | danieliskrin30-lang | danieliskrin | danieliskrin30@gmail.com |

## Our kernels

Time and share are `own_kernel_ms` of run `solution_2` (loop 37.34 s), the one check 4 takes as
the lower of the two shares.

| Kernel (`@triton.jit` name) | File | What it replaces | Time, ms | Share of the training time |
|---|---|---|---|---|
| `ida_nf4_dequant_kernel` | `solution/fused_nf4.py` | NF4 unpacking in forward and backward, 392 calls/step | 2 047.0 | 5.48% |
| `fused_rmsnorm_backward_kernel` | `solution/fused_rmsnorm.py` | `Qwen2RMSNorm`, backward | 386.0 | 1.03% |
| `fused_rmsnorm_forward_kernel` | `solution/fused_rmsnorm.py` | `Qwen2RMSNorm.forward` | 288.4 | 0.77% |
| `ida_attn_prologue_fwd_kernel` | `solution/fused_attn_prologue.py` | RoPE q/k + fp16 cast + `repeat_kv` before SDPA | 229.6 | 0.61% |
| `ida_attn_prologue_bwd_kernel` | `solution/fused_attn_prologue.py` | the same in backward, GQA group sum in fp32 | 219.3 | 0.59% |
| `ida_entropy_argmax_kernel` | `solution/fused_metrics.py` | TRL entropy and argmax for accuracy (no logits copy) | 181.8 | 0.49% |
| `fused_ce_backward_kernel` | `solution/fused_ce.py` | cross-entropy gradient, in place in the logits buffer | 176.4 | 0.47% |
| `fused_ce_forward_kernel` | `solution/fused_ce.py` | `LOSS_MAPPING["ForCausalLM"]` / `F.cross_entropy`, per-row losses for `rewind.jsonl` | 101.9 | 0.27% |
| **Total, `solution_2`** | | | **3 630.4** | **9.72%** |
| **Total, `solution_1`** (loop 35.82 s) | | | 3 609.9 | 10.08% |
| **Check 4 of `check_rules.py`** (lower of two) | | | | **9.7%** (threshold 5%) |

Only in `precompile.py`, a fallback in training: `ida_rope_kernel`,
`fused_entropy_forward_kernel`. Not launched: `fused_attention_*`, `fused_lm_head_*`,
`ida_nf4_dequant_idx_kernel`, `ida_rmsnorm_*_rows_kernel`. The example `example_swiglu_*` runs
and does not count.

Without new kernels (`memory_patches.py`, `fused_metrics.py`, `precompile.py`): fp16
embedding/lm_head, fp16 MLP, logits are not converted to fp32, no TRL logits copy, Triton
compiles in a thread of the same process while the model loads. Equivalence arguments:
`README_EN.md`, section 4.

Ideas taken from other kernels, with links (we wrote the code ourselves):

- FlashAttention, [arXiv 2205.14135](https://arxiv.org/abs/2205.14135); Triton tutorial
  `06-fused-attention` — block online softmax; FlashAttention-2,
  [arXiv 2307.08691](https://arxiv.org/abs/2307.08691) — plan for attention backward without
  atomics.
- Unsloth `unsloth/kernels/cross_entropy_loss.py`; Liger-Kernel `FusedLinearCrossEntropy`,
  [arXiv 2410.10989](https://arxiv.org/abs/2410.10989) — chunked CE over the vocabulary.
- QLoRA, [arXiv 2305.14314](https://arxiv.org/abs/2305.14314), and bitsandbytes 0.50.2 sources —
  NF4 table, 64-value blocks, double quantization.
- Triton tutorial `03-matrix-multiplication` — tiled GEMM (in the disabled `fused_lm_head.py`).

Borrowed code (not kernels): `fused_metrics._compute_loss` is a copy of the default branch of
`SFTTrainer.compute_loss` from TRL 0.29.1 (Apache-2.0), with entropy and argmax replaced by our
kernel.

AI tools (rule 5.4.10): DeepSeek — kernels v1–v3; Claude (Anthropic) — reading library sources
and code v4–v7. Details: `README_EN.md`, section 4.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above, `--repeats 2`)

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s (median) | 78.34 | 61.15 | speedup ×1.281 |
| Peak GPU memory, MiB | 14 345 | 5 879 | base/ours ×2.440 |
| Eval loss (public rows) | 2.04966 | 2.04964 | −0.001% vs base |

Spread over repeats: base 75.43 / 81.25 s (7.4%), solution 60.4 / 61.9 s (2.5%); the memory peak
is identical in both repeats (14 345 and 5 879 MiB).

- [x] All 10 checks of `harness/check_rules.py` passed — ALL CHECKS PASSED (kernel share 9.7%,
      loss curve 0.01%, time ×0.781 and memory ×0.410 of the base); 79 Soup tests and 70 of our
      tests pass
- [x] README rewritten for our solution: what we did, how to run it, results, limits
- [ ] `<COMMIT_SHA>`, `<NOTEBOOK_URL>`, `<VIDEO_URL>`, `<SLIDES_URL>` filled in

## Presentation covers

Which operations were replaced and why (NF4 unpacking in both passes, RMSNorm, attention
prologue with `is_causal` on cutlass, cross-entropy with an in-place gradient, entropy + argmax;
redundant fp32 copies of the embedding, MLP and logits), how the kernels work, results vs the base
(×1.281 time, ×2.440 memory, −0.001% quality, our kernels 9.7%, T4 spread), limitations and next
steps.

## Limitations

| What we tried | Result | Why it is off |
|---|---|---|
| A full own attention kernel (block online softmax, GQA, masks) | matches on synthetic data; on real data `max_diff` 0.24 at typical 0.18, eval loss 2.32 instead of 2.05 | mismatch in rows masked by padding and in the last partial block; ~30× slower than built-in because of atomic adds. Our prologue + cutlass runs instead |
| Own vocabulary projection (940 × 1536 × 151 936) | tests vs `F.linear` pass | not wired in: lm_head is not quantized (tied to the embedding), stays on cuBLAS; our GEMM speed not measured |
| Not computing padding rows | estimate −8…−14 s, −1.5 GB | not done: risk of falling under the `packing` / `padding_free` ban |
| Speed | ×1.281 vs base | the 1.05 threshold is passed; other teams' results are unknown to us |

**Next:** attention without atomic adds (two FlashAttention-2-style kernels), k/v at 2 heads
inside SDPA and recomputing the LoRA input in backward, our own LoRA forward.
