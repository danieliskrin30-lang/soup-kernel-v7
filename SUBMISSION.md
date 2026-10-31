# Сдача: ID Automation (`IDA`)

Трек **AI 1 — AI Engineering by Soup**, задача **Soup Kernel Challenge**, Decentrathon 6.0.

## Ссылки

- Репозиторий, финальный коммит: `https://github.com/BAITC-Hacks/hack-8539eaba-id-automation/commit/<COMMIT_SHA>`
- Публичный Kaggle-ноутбук, сохранённая версия (`COMMIT` внутри = хеш выше; та же ссылка в
  `README.md`): `<NOTEBOOK_URL>`
- Демо-видео (до 3 минут): `<VIDEO_URL>`
- Презентация: `<SLIDES_URL>`

## Команда (1 человек)

| Имя | Роль | GitHub | Kaggle | Контакт (капитан) |
|---|---|---|---|---|
| Даниэль Искрин | капитан, вся работа (ядра, интеграция, замеры, документация) | danieliskrin30-lang | danieliskrin | danieliskrin30@gmail.com |

## Наши ядра

Время и доля — `own_kernel_ms` прогона `solution_2` (цикл 37,34 с), по которому проверка №4
берёт меньшую из двух долей.

| Ядро (`@triton.jit`) | Файл | Что заменяет | Время, мс | Доля времени обучения |
|---|---|---|---|---|
| `ida_nf4_dequant_kernel` | `solution/fused_nf4.py` | распаковку NF4 в прямом и обратном проходе, 392 вызова на шаг | 2 047,0 | 5,48% |
| `fused_rmsnorm_backward_kernel` | `solution/fused_rmsnorm.py` | `Qwen2RMSNorm`, обратный проход | 386,0 | 1,03% |
| `fused_rmsnorm_forward_kernel` | `solution/fused_rmsnorm.py` | `Qwen2RMSNorm.forward` | 288,4 | 0,77% |
| `ida_attn_prologue_fwd_kernel` | `solution/fused_attn_prologue.py` | RoPE q/k + приведение к fp16 + `repeat_kv` перед SDPA | 229,6 | 0,61% |
| `ida_attn_prologue_bwd_kernel` | `solution/fused_attn_prologue.py` | то же в backward, сумма по группам GQA в fp32 | 219,3 | 0,59% |
| `ida_entropy_argmax_kernel` | `solution/fused_metrics.py` | энтропию TRL и argmax для точности (без копии логитов) | 181,8 | 0,49% |
| `fused_ce_backward_kernel` | `solution/fused_ce.py` | градиент кросс-энтропии, на месте в буфере логитов | 176,4 | 0,47% |
| `fused_ce_forward_kernel` | `solution/fused_ce.py` | `LOSS_MAPPING["ForCausalLM"]` / `F.cross_entropy`, потери строк для `rewind.jsonl` | 101,9 | 0,27% |
| **Итого, `solution_2`** | | | **3 630,4** | **9,72%** |
| **Итого, `solution_1`** (цикл 35,82 с) | | | 3 609,9 | 10,08% |
| **Проверка №4 `check_rules.py`** (меньшая из двух) | | | | **9,7%** (порог 5%) |

Только в `precompile.py`, в обучении — запасной путь: `ida_rope_kernel`,
`fused_entropy_forward_kernel`. Не запускаются: `fused_attention_*`, `fused_lm_head_*`,
`ida_nf4_dequant_idx_kernel`, `ida_rmsnorm_*_rows_kernel`. Пример `example_swiglu_*` работает,
в зачёт не идёт.

Без новых ядер (`memory_patches.py`, `fused_metrics.py`, `precompile.py`): эмбеддинг/lm_head в
fp16, MLP в fp16, логиты не переводятся в fp32, копии логитов TRL нет, компиляция Triton в потоке
того же процесса во время загрузки модели. Обоснования эквивалентности — `README.md`, раздел 4.

Идеи из чужих кернелов, со ссылками (код писали сами):

- FlashAttention, [arXiv 2205.14135](https://arxiv.org/abs/2205.14135); туториал Triton
  `06-fused-attention` — online-softmax по блокам; FlashAttention-2,
  [arXiv 2307.08691](https://arxiv.org/abs/2307.08691) — план backward внимания без атомиков.
- Unsloth `unsloth/kernels/cross_entropy_loss.py`; Liger-Kernel `FusedLinearCrossEntropy`,
  [arXiv 2410.10989](https://arxiv.org/abs/2410.10989) — кросс-энтропия кусками словаря.
- QLoRA, [arXiv 2305.14314](https://arxiv.org/abs/2305.14314), и исходники bitsandbytes 0.50.2 —
  таблица NF4, блоки по 64, двойное квантование.
- Туториал Triton `03-matrix-multiplication` — тайловый GEMM (в выключенном `fused_lm_head.py`).

Заимствованный код (не ядра): `fused_metrics._compute_loss` — копия штатной ветки
`SFTTrainer.compute_loss` из TRL 0.29.1 (Apache-2.0) с заменой энтропии и argmax на наше ядро.

ИИ-инструменты (п. 5.4.10): DeepSeek — ядра v1–v3; Claude (Anthropic) — разбор исходников
библиотек и код v4–v7. Подробно — `README.md`, раздел 4.

## Результаты на Kaggle T4 (из `runs/result.json` и `runs/checks.json` ноутбука выше, `--repeats 2`)

| | База Soup | Наше | Отношение |
|---|---|---|---|
| Время `soup train`, с (медиана) | 78,34 | 61,15 | speedup ×1,281 |
| Пик памяти GPU, МиБ | 14 345 | 5 879 | база/наше ×2,440 |
| Eval loss (публичные строки) | 2,04966 | 2,04964 | −0,001% к базе |

Разброс по повторам: база 75,43 / 81,25 с (7,4%), решение 60,4 / 61,9 с (2,5%); пик памяти в
обоих повторах одинаковый (14 345 и 5 879 МиБ).

- [x] Все 10 проверок `harness/check_rules.py` пройдены — ALL CHECKS PASSED (доля ядер 9,7%,
      кривая loss 0,01%, время ×0,781 и память ×0,410 от базы); 79 тестов Soup и 70 наших
      тестов пройдены
- [x] `README.md` переписан под решение: что сделано, как запустить, результаты, ограничения
- [ ] Вставлены `<COMMIT_SHA>`, `<NOTEBOOK_URL>`, `<VIDEO_URL>`, `<SLIDES_URL>`

## Что покрывает презентация

Какие операции заменены и почему (распаковка NF4 в обоих проходах, RMSNorm, пролог внимания с
`is_causal` на cutlass, кросс-энтропия с градиентом на месте, энтропия + argmax; лишние
fp32-копии эмбеддинга, MLP и логитов), как устроены ядра, результаты против базы (×1,281 по
времени, ×2,440 по памяти, −0,001% по качеству, доля наших ядер 9,7%, разброс T4), ограничения
и дальнейшие шаги.

## Ограничения

| Что пробовали | Результат | Почему выключено |
|---|---|---|
| Своё ядро внимания целиком (блочный online-softmax, GQA, маски) | на синтетике сходится; на реальных данных `max_diff` 0,24 при типичных 0,18, eval loss 2,32 вместо 2,05 | расхождение в строках, закрытых маской паддинга, и в последнем неполном блоке; ~30× медленнее встроенного из-за атомарных сложений. Вместо него — свой пролог + cutlass |
| Своё умножение на словарь (940 × 1536 × 151 936) | тесты против `F.linear` проходят | не подключено: lm_head не квантован (связан с эмбеддингом), оставлен cuBLAS; скорость своего GEMM не измерена |
| Не считать строки паддинга | оценка −8…−14 с, −1,5 ГБ | не делали: риск попасть под запрет `packing` / `padding_free` |
| Скорость | ×1,281 к базе | порог 1,05 пройден; результаты других команд нам неизвестны |

**Дальше:** внимание без атомарных сложений (два ядра в стиле FlashAttention-2), k/v на 2 головах
внутри SDPA и пересчёт входа LoRA в backward, свой forward LoRA.
