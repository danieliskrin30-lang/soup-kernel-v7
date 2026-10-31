# Submission: ID Automation (`IDA`)

Track **AI 1 — AI Engineering by Soup**, task **Soup Kernel Challenge**, Decentrathon 6.0.
*(English version. The primary version is `SUBMISSION.md`.)*

## Links

- Repository, final commit (public mirror of the same code): `https://github.com/danieliskrin30-lang/soup-kernel-v7/commit/655a25344658badfdee1b8107cee6a9066ab68ef`
- Public Kaggle notebook, saved version (`COMMIT` in it = the sha above; the same link is in
  `README.md`): https://www.kaggle.com/code/danieliskrin30/notebookd124ebce8b
- Demo video (up to 3 minutes): `<VIDEO_URL>`
- Presentation: `<SLIDES_URL>`

Repository and mirror. The platform repository is `github.com/BAITC-Hacks/hack-8539eaba-id-automation` (jury access requested from
the organizers). A public mirror of the same commit is `github.com/danieliskrin30-lang/soup-kernel-v7`; the public notebook
clones it, so Run All works without our tokens.

## Team (1 person)

| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Daniel Iskrin | captain, all work (kernels, integration, measurements, docs) | danieliskrin30-lang | danieliskrin | danieliskrin30@gmail.com |

## Our kernels

Time and share are `own_kernel_ms` of run `solution_2` of the public notebook, the one check 4 takes as
the lower of the two shares.

| Kernel (`@triton.jit` name) | File | What it replaces | Time, ms | Share of the training time |
|---|---|---|---|---|
| `ida_nf4_dequant_kernel` | `solution/fused_nf4.py` | NF4 unpacking in forward and backward, 392 calls/step | 2 061.5 | 5.09% |
| `fused_rmsnorm_backward_kernel` | `solution/fused_rmsnorm.py` | `Qwen2RMSNorm`, backward | 386.7 | 0.96% |
| `fused_rmsnorm_forward_kernel` | `solution/fused_rmsnorm.py` | `Qwen2RMSNorm.forward` | 292.8 | 0.72% |
| `ida_attn_prologue_fwd_kernel` | `solution/fused_attn_prologue.py` | RoPE q/k + fp16 cast + `repeat_kv` before SDPA | 233.2 | 0.58% |
| `ida_attn_prologue_bwd_kernel` | `solution/fused_attn_prologue.py` | the same in backward, GQA group sum in fp32 | 220.1 | 0.54% |
| `ida_entropy_argmax_kernel` | `solution/fused_metrics.py` | TRL entropy and argmax for accuracy (no logits copy) | 203.1 | 0.50% |
| `fused_ce_backward_kernel` | `solution/fused_ce.py` | cross-entropy gradient, in place in the logits buffer | 177.0 | 0.44% |
| `fused_ce_forward_kernel` | `solution/fused_ce.py` | `LOSS_MAPPING["ForCausalLM"]` / `F.cross_entropy`, per-row losses for `rewind.jsonl` | 113.2 | 0.28% |
| **Total, `solution_2`** | | | **3 687.7** | **9.11%** |
| **Total, `solution_1`** | | | | 9.24% |
| **Check 4 of `check_rules.py`** (lower of two) | | | | **9.1%** (threshold 5%) |

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
| Time of `soup train`, s (median) | 86.27 | 66.10 | speedup ×1.305 |
| Peak GPU memory, MiB | 14 345 | 5 879 | base/ours ×2.440 |
| Eval loss (public rows) | 2.04962 | 2.04962 | +0.000% vs base |

Spread over repeats: base 88.78 / 83.75 s (5.8%), solution 65.80 / 66.40 s (0.9%); the memory peak
is identical in both repeats (14 345 and 5 879 MiB).

- [x] All 10 checks of `harness/check_rules.py` passed — ALL CHECKS PASSED (kernel share 9.1%,
      time ×0.766 and memory ×0.410 of the base); 79 Soup tests and 70 of our
      tests pass
- [x] README rewritten for our solution: what we did, how to run it, results, limits
- [x] Public notebook saved with results (link above)
- [ ] `<COMMIT_SHA>`, `<VIDEO_URL>`, `<SLIDES_URL>` filled in

## Presentation covers

Which operations were replaced and why (NF4 unpacking in both passes, RMSNorm, attention
prologue with `is_causal` on cutlass, cross-entropy with an in-place gradient, entropy + argmax;
redundant fp32 copies of the embedding, MLP and logits), how the kernels work, results vs the base
(×1.305 time, ×2.440 memory, +0.000% quality, our kernels 9.1%, T4 spread), limitations and next
steps.

## Limitations

| What we tried | Result | Why it is off |
|---|---|---|
| A full own attention kernel (block online softmax, GQA, masks) | matches on synthetic data; on real data `max_diff` 0.24 at typical 0.18, eval loss 2.32 instead of 2.05 | mismatch in rows masked by padding and in the last partial block; ~30× slower than built-in because of atomic adds. Our prologue + cutlass runs instead |
| Own vocabulary projection (940 × 1536 × 151 936) | tests vs `F.linear` pass | not wired in: lm_head is not quantized (tied to the embedding), stays on cuBLAS; our GEMM speed not measured |
| Not computing padding rows | estimate −8…−14 s, −1.5 GB | not done: risk of falling under the `packing` / `padding_free` ban |
| Speed | ×1.305 vs base | the 1.05 threshold is passed; other teams' results are unknown to us |

**Next:** attention without atomic adds (two FlashAttention-2-style kernels), k/v at 2 heads
inside SDPA and recomputing the LoRA input in backward, our own LoRA forward.
