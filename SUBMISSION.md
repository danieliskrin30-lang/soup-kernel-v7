# Сдача: ID Automation (`IDA`)

Трек **AI 1 — AI Engineering by Soup**, задача **Soup Kernel Challenge**, Decentrathon 6.0.

## Ссылки

- Репозиторий, финальный коммит (публичное зеркало того же кода): `https://github.com/danieliskrin30-lang/soup-kernel-v7/commit/655a25344658badfdee1b8107cee6a9066ab68ef`
- Публичный Kaggle-ноутбук, сохранённая версия (`COMMIT` внутри = хеш выше; та же ссылка в
  `README.md`): https://www.kaggle.com/code/danieliskrin/soup-kernel-challenge-ida
- Демо-видео (до 3 минут): `<VIDEO_URL>`
- Презентация: `<SLIDES_URL>`

Репозиторий и зеркало. Основной репозиторий платформы — `github.com/BAITC-Hacks/hack-8539eaba-id-automation` (доступ для жюри
запрошен у организаторов). Публичное зеркало того же коммита — `github.com/danieliskrin30-lang/soup-kernel-v7`; публичный
ноутбук клонирует именно его, чтобы запускаться «Run All» без наших токенов.

## Команда (1 человек)

| Имя | Роль | GitHub | Kaggle | Контакт (капитан) |
|---|---|---|---|---|
| Даниэль Искрин | капитан, вся работа (ядра, интеграция, замеры, документация) | danieliskrin30-lang | danieliskrin | danieliskrin30@gmail.com |

## Наши ядра

Время и доля — `own_kernel_ms` прогона `solution_2` публичного ноутбука, по которому проверка №4
берёт меньшую из двух долей.

| Ядро (`@triton.jit`) | Файл | Что заменяет | Время, мс | Доля времени обучения |
|---|---|---|---|---|
| `ida_nf4_dequant_kernel` | `solution/fused_nf4.py` | распаковку NF4 в прямом и обратном проходе, 392 вызова на шаг | 2 061,5 | 5,09% |
| `fused_rmsnorm_backward_kernel` | `solution/fused_rmsnorm.py` | `Qwen2RMSNorm`, обратный проход | 386,7 | 0,96% |
| `fused_rmsnorm_forward_kernel` | `solution/fused_rmsnorm.py` | `Qwen2RMSNorm.forward` | 292,8 | 0,72% |
| `ida_attn_prologue_fwd_kernel` | `solution/fused_attn_prologue.py` | RoPE q/k + приведение к fp16 + `repeat_kv` перед SDPA | 233,2 | 0,58% |
| `ida_attn_prologue_bwd_kernel` | `solution/fused_attn_prologue.py` | то же в backward, сумма по группам GQA в fp32 | 220,1 | 0,54% |
| `ida_entropy_argmax_kernel` | `solution/fused_metrics.py` | энтропию TRL и argmax для точности (без копии логитов) | 203,1 | 0,50% |
| `fused_ce_backward_kernel` | `solution/fused_ce.py` | градиент кросс-энтропии, на месте в буфере логитов | 177,0 | 0,44% |
| `fused_ce_forward_kernel` | `solution/fused_ce.py` | `LOSS_MAPPING["ForCausalLM"]` / `F.cross_entropy`, потери строк для `rewind.jsonl` | 113,2 | 0,28% |
| **Итого, `solution_2`** | | | **3 687,7** | **9,11%** |
| **Итого, `solution_1`** | | | | 9,24% |
| **Проверка №4 `check_rules.py`** (меньшая из двух) | | | | **9,1%** (порог 5%) |

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
| Время `soup train`, с (медиана) | 86,27 | 66,10 | speedup ×1,305 |
| Пик памяти GPU, МиБ | 14 345 | 5 879 | база/наше ×2,440 |
| Eval loss (публичные строки) | 2,04962 | 2,04962 | +0,000% к базе |

Разброс по повторам: база 88,78 / 83,75 с (5,8%), решение 65,80 / 66,40 с (0,9%); пик памяти в
обоих повторах одинаковый (14 345 и 5 879 МиБ).

- [x] Все 10 проверок `harness/check_rules.py` пройдены — ALL CHECKS PASSED (доля ядер 9,1%,
      время ×0,766 и память ×0,410 от базы); 79 тестов Soup и 70 наших
      тестов пройдены
- [x] `README.md` переписан под решение: что сделано, как запустить, результаты, ограничения
- [x] Публичный ноутбук сохранён с результатами (ссылка выше)
- [ ] Вставлены `<COMMIT_SHA>`, `<VIDEO_URL>`, `<SLIDES_URL>`

## Что покрывает презентация

Какие операции заменены и почему (распаковка NF4 в обоих проходах, RMSNorm, пролог внимания с
`is_causal` на cutlass, кросс-энтропия с градиентом на месте, энтропия + argmax; лишние
fp32-копии эмбеддинга, MLP и логитов), как устроены ядра, результаты против базы (×1,305 по
времени, ×2,440 по памяти, +0,000% по качеству, доля наших ядер 9,1%, разброс T4), ограничения
и дальнейшие шаги.

## Ограничения

| Что пробовали | Результат | Почему выключено |
|---|---|---|
| Своё ядро внимания целиком (блочный online-softmax, GQA, маски) | на синтетике сходится; на реальных данных `max_diff` 0,24 при типичных 0,18, eval loss 2,32 вместо 2,05 | расхождение в строках, закрытых маской паддинга, и в последнем неполном блоке; ~30× медленнее встроенного из-за атомарных сложений. Вместо него — свой пролог + cutlass |
| Своё умножение на словарь (940 × 1536 × 151 936) | тесты против `F.linear` проходят | не подключено: lm_head не квантован (связан с эмбеддингом), оставлен cuBLAS; скорость своего GEMM не измерена |
| Не считать строки паддинга | оценка −8…−14 с, −1,5 ГБ | не делали: риск попасть под запрет `packing` / `padding_free` |
| Скорость | ×1,305 к базе | порог 1,05 пройден; результаты других команд нам неизвестны |

**Дальше:** внимание без атомарных сложений (два ядра в стиле FlashAttention-2), k/v на 2 головах
внутри SDPA и пересчёт входа LoRA в backward, свой forward LoRA.
