# Soup Kernel Challenge: свои Triton-ядра для QLoRA на бесплатной T4

**Команда:** ID Automation (`IDA`), 1 человек · **Трек:** AI 1 «AI Engineering by Soup» · **Хакатон:** Decentrathon 6.0
**Репозиторий:** `github.com/BAITC-Hacks/hack-8539eaba-id-automation` · *English version: `README_EN.md`*

Финальный прогон по протоколу жюри: `harness/run_task.py --repeats 2` (база, решение, решение,
база в одной сессии Kaggle T4) — сохранённая версия публичного ноутбука
https://www.kaggle.com/code/danieliskrin30/notebookd124ebce8b, медианы из `runs/result.json`:

| Показатель | База Soup | Наше решение | Отношение |
|---|---|---|---|
| Время `soup train`, с (медиана 2 прогонов) | 86,27 | **66,10** | **speedup 1,305×** |
| Пик памяти GPU (`nvidia-smi`), МиБ | 14 345 | **5 879** | **база/наше 2,440×** |
| Eval loss (публичные строки) | 2,04962 | 2,04962 | +0,000% |
| Доля наших ядер во времени обучения (с 3-го шага) | — | **9,1%** (меньшая из двух) | порог ≥5% |
| `harness/check_rules.py` | — | **10/10 PASS** | — |
| Тесты Soup (`tests/`) / наши тесты (`pytest solution`) | 79 | **79 passed / 70 passed** | — |

Задача, данные, состав и порядок батчей, число шагов, LoRA, оптимизатор и расписание lr не
менялись. `harness/`, `tests/`, `task/` не менялись.

---

## 1. Что сделано и почему

### 1.1. Наши ядра, которые работают в обучении

Время — `own_kernel_ms` харнесса (CUDA-события, шаги 3–64), прогон `solution_2` публичного
ноутбука — тот, по которому проверка №4 берёт меньшую долю.

| Ядро (`@triton.jit`) | Файл | Что заменяет | Запусков за прогон | Время, мс | Доля |
|---|---|---|---|---|---|
| `ida_nf4_dequant_kernel` | `fused_nf4.py` | распаковку NF4 в прямом (`_dequant_linear_fallback`) и обратном (`dequantize_4bit`) проходе | 24 897 | 2 061,5 | 5,09% |
| `fused_rmsnorm_backward_kernel` | `fused_rmsnorm.py` | `Qwen2RMSNorm`, backward | 3 585 | 386,7 | 0,96% |
| `fused_rmsnorm_forward_kernel` | `fused_rmsnorm.py` | `Qwen2RMSNorm.forward` | 3 649 | 292,8 | 0,72% |
| `ida_attn_prologue_fwd_kernel` | `fused_attn_prologue.py` | RoPE q/k + приведение к fp16 + `repeat_kv` для SDPA | 1 793 | 233,2 | 0,58% |
| `ida_attn_prologue_bwd_kernel` | `fused_attn_prologue.py` | то же в backward, сумма по группам GQA в fp32 | 1 793 | 220,1 | 0,54% |
| `ida_entropy_argmax_kernel` | `fused_metrics.py` | энтропию TRL и argmax для `mean_token_accuracy` | 65 | 203,1 | 0,50% |
| `fused_ce_backward_kernel` | `fused_ce.py` | градиент кросс-энтропии (пишется в буфер логитов) | 66 | 177,0 | 0,44% |
| `fused_ce_forward_kernel` | `fused_ce.py` | кросс-энтропию модели и потери строк для `rewind.jsonl` | 323 | 113,2 | 0,28% |
| **Итого** | | | | **3 687,7** | **9,11%** |

Второй прогон (`solution_1`): 9,24%. Число запусков от прогона к прогону не меняется
(тот же код и те же батчи).

Только в `precompile.py` (по 1–2 запуска, в обучении — запасной путь, если основной не применим):
`ida_rope_kernel`, `fused_entropy_forward_kernel`. Не запускаются: `fused_attention_*`,
`fused_lm_head_*` (см. 3.2), `ida_nf4_dequant_idx_kernel`, `ida_rmsnorm_*_rows_kernel`
(варианты для `configure()`, бенчмарк выбрал настройки v4). Пример из шаблона
`example_swiglu_*` работает и в зачёт не идёт.

### 1.2. Как устроены ядра

**NF4 (`fused_nf4.py`).** В bitsandbytes 0.50.2 на T4 при M > 512 строк (у нас M = 4 × длина)
прямой проход идёт не через слитный `gemm_4bit`, а через `_dequant_linear_fallback`: распаковка →
`F.linear`. Обратный проход распаковывает ещё раз в `MatMul4Bit.backward`. Итого 392 распаковки
за шаг (7 матриц × 28 слоёв × 2 прохода). Программа обрабатывает плитку [64 блока × 64 значения]:
масштаб двойного квантования (`code2[absmax8] * absmax2 + offset`) считается внутри ядра один
раз на блок, байт читается один раз, ниблы раскладываются `tl.join` (старший первым), `offset`
читается по указателю — без синхронизаций CPU↔GPU. `enable_fp_fusion=False`: результат побитно
равен `dequantize_4bit`.

**RMSNorm (`fused_rmsnorm.py`).** Одна строка — одна программа, `rstd` в fp32 в регистрах, на
backward хранится только `rstd`; `dx = s·g − (s³/n)·x·(g·x)`, `g = dy·w`.

**Пролог внимания (`fused_attn_prologue.py`).** Заменён `Qwen2Attention.forward`. Прямое ядро на
позицию (b, l) одним запуском: RoPE для 12 голов q и 2 голов k, приведение к fp16 и разворот k, v
на 12 голов (GQA 12/2) — сразу в формате SDPA. Дальше SDPA с `is_causal=True` без маски
паддинга, бэкенд жёстко `EFFICIENT_ATTENTION` (cutlass `fmha_cutlassF/B_*_sm75`): если cutlass
не может — исключение и штатный путь, а не тихий переход на math. Обратное ядро: fp16 → fp32,
сумма по 6 головам группы в fp32, RoPE назад. Почему маска не нужна — раздел 4.

**Кросс-энтропия (`fused_ce.py`).** Online-softmax кусками по 4 096 слов, на backward хранится
только `lse [rows]`. Подменена запись `LOSS_MAPPING["ForCausalLM"]` (через неё transformers
5.16.1 находит функцию потерь), поэтому `logits.float()` не выполняется: ядро читает
fp16-логиты, считает в fp32 в регистрах. Градиент пишется **в буфер самих логитов** (каждый
элемент читается и перезаписывается одним потоком ядра). Тот же прямой kernel считает потери
строк для `rewind.jsonl` (`reduction="none"`, одна синхронизация на шаг вместо восьми).

**Энтропия + argmax (`fused_metrics.py`).** За один проход по логитам: энтропия (те же операции
в том же порядке, что в прежнем ядре энтропии, — побитно то же) и argmax (первый максимум, как
`torch.argmax`). Это позволяет не делать копию `outputs.logits[..., :-1, :].contiguous()`,
которую TRL строит только ради argmax.

### 1.3. Изменения без новых ядер

| Что | Было | Стало | Эффект |
|---|---|---|---|
| Эмбеддинг / lm_head 151 936 × 1 536 | fp32 (после `prepare_model_for_kbit_training`), под autocast каждый шаг fp16-копия до backward | хранится в fp16, выход эмбеддинга поднимается в fp32 хуком | −467 МБ веса, −467 МБ копии (оценка по размерам тензоров) |
| gate/up/down в MLP | `Linear4bit` возвращает fp32, SwiGLU хранит fp32 | fp16, SwiGLU хранит fp16 | ~−1 ГБ активаций (оценка) |
| `outputs.logits` | accelerate переводит в fp32 | логиты остаются fp16 | −1 буфер логитов (оценка) |
| Копия логитов TRL для точности + отдельный буфер градиента логитов | два буфера по размеру логитов на пике | ни одного (1.2) | −600 МиБ пика* |
| Компиляция Triton | на шагах 0–1 | в потоке того же процесса, пока Soup грузит модель | −7 с* |

\* Отдельные A/B-прогоны на той же T4, не из финального `result.json`: без `precompile` шаги 0–1
= 7,65 с и wall 66,8 с, с ним — 1,44 с и 59,8 с; пик 6 479 → 5 879 МиБ до/после 1.2 (метрики +
градиент на месте); пролог внимания + `is_causal` — около −1,2 с (ядра внимания 847 → 569 мс
на 16 профилированных шагах).

### 1.4. Источники идей (код написан самостоятельно)

| Идея | Источник | Где у нас |
|---|---|---|
| Online-softmax, блочная обработка | FlashAttention, [arXiv 2205.14135](https://arxiv.org/abs/2205.14135); туториал Triton [`06-fused-attention`](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | `fused_ce.py`, `fused_entropy.py`, `fused_metrics.py`, `fused_attention.py` (выкл.) |
| Схема backward внимания без атомиков (план, 3.2) | FlashAttention-2, [arXiv 2307.08691](https://arxiv.org/abs/2307.08691) | — |
| Кросс-энтропия кусками словаря, хранить только `logsumexp` | Unsloth [`unsloth/kernels/cross_entropy_loss.py`](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/cross_entropy_loss.py); Liger-Kernel `FusedLinearCrossEntropy`, [arXiv 2410.10989](https://arxiv.org/abs/2410.10989) | `fused_ce.py` |
| Таблица NF4, блоки по 64, двойное квантование | QLoRA, [arXiv 2305.14314](https://arxiv.org/abs/2305.14314); исходники bitsandbytes 0.50.2 | `fused_nf4.py` |
| Тайловый GEMM с группировкой | туториал Triton [`03-matrix-multiplication`](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | `fused_lm_head.py` (выкл.) |

Запрещённые библиотеки (unsloth, liger-kernel, xformers, flash-attn, torchao и др.) не
импортируются и не загружаются — проверка №2.

---

## 2. Как запустить

Публичный Kaggle-ноутбук (Run All без правок, с сохранёнными результатами): **https://www.kaggle.com/code/danieliskrin30/notebookd124ebce8b**
Ноутбук клонирует публичное зеркало того же коммита `github.com/danieliskrin30-lang/soup-kernel-v7`, чтобы запускаться без
наших токенов; основной репозиторий платформы — `github.com/BAITC-Hacks/hack-8539eaba-id-automation`.
Финальный коммит (публичное зеркало того же кода): **https://github.com/danieliskrin30-lang/soup-kernel-v7/commit/655a25344658badfdee1b8107cee6a9066ab68ef**

На чистом Kaggle-ноутбуке (Accelerator: GPU T4, Internet: on):

```bash
python -m pip install -q "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" \
    "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
git clone https://github.com/danieliskrin30-lang/soup-kernel-v7.git repo
cd repo && git checkout 655a25344658badfdee1b8107cee6a9066ab68ef
python task/prepare_data.py                               # data/train.jsonl, data/eval_public.jsonl
python -m pytest solution -q                              # наши ядра против эталонов: 70 passed
python harness/run_task.py --repeats 2                    # протокол жюри, ~30 мин
python harness/check_rules.py --result runs/result.json   # 10 проверок
python harness/score.py runs/result.json                  # отношения s и m
```

Быстрее, только решение (проверки, которым нужна база, в этом режиме не проходят):
`python harness/run_task.py --only solution --out runs_try`.

Переменные окружения (в сдаче **не ставятся**): `IDA_DIAG=1` — диагностика памяти, типов и
счётчиков путей в `train.log`; `IDA_PRECOMPILE=0` — выключить компиляцию в потоке.
Подбор настроек ядер: `python solution/bench_c.py` (не тест).

---

## 3. Результаты и ограничения

### 3.1. Результаты (Kaggle T4, `runs/result.json` и `runs/checks.json`, `--repeats 2`)

| | База Soup | Наше | Отношение |
|---|---|---|---|
| Время `soup train`, с, медиана | 86,27 | 66,10 | speedup **1,305×** (−20,17 с) |
| Цикл обучения `train_loop_s`, с, медиана | 58,57 | 42,30 | 1,385× |
| Пик памяти GPU, МиБ (максимум по прогонам) | 14 345 | 5 879 | база/наше **2,440×** (−8 466 МиБ) |
| Eval loss (публичные строки) | 2,04962 | 2,04962 | +0,000% |
| Кривая loss против базы (проверка №6) | — | PASS | допуск 1% / 3% |
| Доля наших ядер (проверка №4) | — | 9,1% (прогоны: 9,24% и 9,11%) | порог 5% |
| Время / память против базы (проверка №8) | — | ×0,766 / ×0,410 | допуск ×1,10 |
| Тесты | 79 Soup | 79 passed; `pytest solution`: 70 passed | — |

Разброс по повторам:

| Прогон | Время, с | Пик, МиБ |
|---|---|---|
| base_1 / base_2 | 88,78 / 83,75 (разброс 5,03 с, 5,8%) | 14 345 / 14 345 |
| solution_1 / solution_2 | 65,80 / 66,40 (разброс 0,60 с, 0,9%) | 5 879 / 5 879 |

Время базы на T4 гуляет сильнее, чем время решения; пик памяти воспроизводится до мегабайта.

Не измерено: вклад каждого изменения по отдельности (абляции — только для `precompile`, пролога
внимания и пункта 1.2, см. сноску в 1.3), фаза пика памяти в финальном прогоне.

### 3.2. Ограничения

| Что пробовали | Результат | Почему выключено |
|---|---|---|
| Своё ядро внимания целиком (`fused_attention.py`: блочный online-softmax, GQA 12/2, причинная маска + маска паддинга) | тесты против SDPA на синтетике сходятся; на реальных данных `max_diff` 0,24 при типичных значениях 0,18, eval loss 2,32 вместо 2,05 | расхождение в строках, закрытых маской паддинга, и в последнем неполном блоке; ~30× медленнее встроенного из-за атомарных сложений `dK/dV`. Вместо него работает свой пролог + cutlass (1.2) |
| Своё умножение на словарь (`fused_lm_head.py`, 940 × 1536 × 151 936, только `dX`) | тесты против `F.linear` проходят | не подключено: lm_head не квантован (связан с эмбеддингом); lm_head оставлен cuBLAS, скорость своего GEMM против cuBLAS не измерена |
| Не считать строки паддинга (во всей модели или только в lm_head) | оценка −8…−14 с и −1,5 ГБ | не делали: риск попасть под запрет `packing` / `padding_free` |
| Свой 4-битный GEMM, cuBLASLt, слияние q/k/v, другие настройки NF4/RMSNorm (`bench_c.py`) | 0…1% по оценке или бенчмарку | ниже разброса замера |
| Скорость | 1,305× к базе | порог 1,05 пройден; результаты других команд нам неизвестны |

**Дальше:** внимание без атомарных сложений (два ядра в стиле FlashAttention-2), хранение k/v
в SDPA на 2 головах и пересчёт входа LoRA в backward (по оценке −0,4 ГБ), свой forward LoRA с
одним приведением входа на q/k/v.

---

## 4. Раскрытия (п. 5.4.4 и 5.4.10)

**Внимание с `is_causal` вместо маски паддинга.** `DataCollatorForLanguageModeling` в TRL 0.29.1
жёстко дополняет батч справа (`padding_side="right"`). Реальный токен на позиции t видит только
позиции ≤ t, а они все реальные, — маска паддинга для реальных токенов ничего не закрывает
сверх причинной. Строки паддинга считаются иначе, но их градиент ровно 0 (метки −100), а в
метриках TRL они умножаются на `attention_mask = 0`. Guard на каждом шаге проверяет, что маска
имеет вид `1…1 0…0`, нет `position_ids`, кэша и `use_cache`, — иначе работает штатный путь
transformers. Батчи, токены и вычисления над реальными токенами те же, это не упаковка.
`test_attn.py` сверяет весь слой со штатным SDPA с маской на реальных строках.

**`SFTTrainer.compute_loss`.** В `fused_metrics.py` — копия штатной ветки метода из TRL 0.29.1
(Apache-2.0, `trl/trainer/sft_trainer.py`); изменено только то, что энтропия и argmax берутся
из нашего ядра вместо `entropy_from_logits` и копии логитов. Имена метрик, маски, суммы,
`gather_for_metrics`, `.item()` — те же строки. Нестандартные ветки (liger, prompt tuning,
`shift_labels`, aux_loss) вызывают оригинальный метод TRL целиком.

**Градиент логитов на месте.** После CE-backward логиты никому не нужны: метрики TRL и потери
строк Soup считаются в forward, lm_head хранит для backward вход и вес, а не выход. Включается
только для потерь модели (`LOSS_MAPPING`), не для произвольных `F.cross_entropy`
(`test_metrics.py`).

**Патч Soup.** `soup_cli.trainer.rewind_hf.row_losses` заменён на версию с теми же числами и
одной синхронизацией вместо восьми (`test_c.py` сверяет с оригиналом).

**Эмбеддинг / lm_head в fp16.** Soup грузит модель в fp16, peft поднимает не-4-битные параметры в
fp32 без изменения значений (fp16 → fp32 → fp16 без потерь). Выход эмбеддинга поднимается в fp32,
остаточный поток — fp32, как в базе. Матрица заморожена, в адаптер не входит.

**MLP в fp16.** `Linear4bit` в базе приводит вход к fp16 сам и расширяет выход до fp32 (точно).
Мы подаём fp16 явно и не расширяем; SwiGLU считает в fp32 в регистрах и округляет один раз, как
`down_proj` в базе; градиенты по входу gate/up складываются в fp32. Ни одно округление не
сдвинулось (`test_v4.py::test_mlp_fp16_bitwise_equal`, `torch.equal`).

**Где совпадение не побитное:** SDPA с `is_causal` против SDPA с маской — на уровне округления
fp16 на реальных строках; dk/dv пролога — порядок суммы по 6 головам; кросс-энтропия, энтропия и
RMSNorm — порядок суммирования в fp32. Всё в пределах допусков проверок №5–№7 (итог: eval loss
+0,000% к базе).

**`precompile.py`.** Наши ядра компилируются в отдельном потоке **того же процесса**, пока Soup
грузит модель (готово за ~9 с, до `Trainer.__init__`). Кэш Triton пустой в каждом прогоне
(harness задаёт свой `TRITON_CACHE_DIR`), готовых кэшей нет, всё время компиляции — внутри
измеряемого wall-времени.

**ИИ-инструменты.** DeepSeek — черновики и отладка ядер v1–v3. Claude (Anthropic) — разбор
исходников bitsandbytes / transformers / peft / accelerate / TRL / Soup, поиск лишних копий и
синхронизаций, код v4–v7: `fused_nf4.py`, `fused_ce.py`, `fused_rope.py`, `fused_attn_prologue.py`,
`fused_metrics.py`, `memory_patches.py`, `precompile.py`, `diag.py`, `bench_c.py`, варианты в
`fused_rmsnorm.py`, тесты `test_v4.py`, `test_c.py`, `test_attn.py`, `test_metrics.py`. Все числа —
из прогонов harness, каждое ядро покрыто тестом против эталона.

**Сторонние компоненты.** Python 3.13.15, `torch` 2.11.0+cu128, `triton` 3.6.0, `transformers`
5.16.1, `trl` 0.29.1, `peft` 0.20.0, `bitsandbytes` 0.50.2, `accelerate` 1.14.0, `datasets` 4.8.5,
`soup-cli` 0.77.0; модель `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0); датасет
`AmanMussa/kazakh-instruction-v2` (MIT по карточке; перевод Stanford Alpaca, CC BY-NC 4.0).
Стартовый набор — Apache-2.0, см. `NOTICE`.

---

## 5. Дисциплина измерений

- `harness/`, `tests/`, `task/` и настройки замера (аллокатор, видимые GPU) не менялись;
  `packages_changed` в `result.json` пуст.
- Диагностика в сдаче выключена: `IDA_DIAG` не ставится, `diag.py` ничего не делает.
- `precompile.py` включён по умолчанию и работает внутри того же процесса и того же замера;
  фоновых процессов и вторых CUDA-потоков в обучении нет.
- Все числа в разделах 1.1 и 3.1 — из `runs/result.json` и `runs/checks.json` финального
  прогона `--repeats 2` (сохранённая версия публичного ноутбука). Числа из отдельных A/B-прогонов помечены в 1.3. Чего нет ни там, ни
  там, помечено «не измерено».

---

## 6. Структура

```
solution/
  __init__.py            apply(): порядок патчей, каждый включается одной строкой
  fused_nf4.py           ida_nf4_dequant_kernel — NF4, прямой и обратный проход (+ configure)
  fused_rmsnorm.py       RMSNorm, forward + backward (+ configure)
  fused_attn_prologue.py ida_attn_prologue_* — RoPE + fp16 + GQA, SDPA is_causal на cutlass
  fused_ce.py            кросс-энтропия (LOSS_MAPPING, градиент на месте, reduction none, rewind)
  fused_metrics.py       ida_entropy_argmax_kernel + compute_loss TRL без копии логитов
  fused_entropy.py       энтропия (запасной путь)
  fused_rope.py          ida_rope_kernel — RoPE (запасной путь)
  memory_patches.py      эмбеддинг fp16, MLP fp16, логиты fp16 в accelerate
  precompile.py          компиляция ядер параллельно с загрузкой модели
  diag.py                диагностика, только при IDA_DIAG=1
  bench_c.py             подбор настроек NF4/RMSNorm (не тест)
  swiglu.py              пример из шаблона (работает, в зачёт не идёт)
  fused_attention.py, fused_lm_head.py   выключены, см. 3.2
  test_*.py              70 тестов ядер против эталонов
harness/, tests/, task/, templates/      стартовый набор без изменений
```
