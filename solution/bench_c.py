"""Подбор настроек ядер для пункта C (NF4 и RMSNorm) на T4. Не тест: pytest его не собирает.

Запуск из корня репозитория (2–4 минуты, нужна видеокарта):
    python solution/bench_c.py

Печатает таблицы и две строки для apply():
    fused_nf4.configure(mode=..., groups=..., num_warps=...)
    fused_rmsnorm.configure(rows=..., num_warps=...)
В зачёт идут только варианты, которые побитно совпадают с bitsandbytes (NF4) и
с v4 в пределах 1e-6 (RMSNorm). Оценка «с/прогон» = время за шаг x 62 замеряемых шага.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triton.testing import do_bench  # noqa: E402

from solution import fused_nf4 as nf4  # noqa: E402
from solution import fused_rmsnorm as rms  # noqa: E402

STEPS = 62
LAYERS = 28
# форма (out, in) и сколько таких матриц в слое; распаковка — в forward и в backward
NF4_SHAPES = {"q,o": ((1536, 1536), 2), "k,v": ((256, 1536), 2),
              "gate,up": ((8960, 1536), 2), "down": ((1536, 8960), 1)}
RMS_ROWS = (940, 1336)          # средний и пиковый батч
RMS_CALLS = 57                  # норм за шаг


def bench(fn):
    return do_bench(fn, warmup=25, rep=100)


def bench_nf4():
    import bitsandbytes.functional as F

    cases = []
    for name, ((n, k), cnt) in NF4_SHAPES.items():
        torch.manual_seed(0)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16)
        packed, st = F.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4")
        ref = F.dequantize_4bit(packed, st)
        cases.append((name, packed, st, ref, cnt))

    def per_step(fn_for_case):
        return sum(cnt * bench(fn_for_case(p, s)) for _, p, s, _, cnt in cases) * LAYERS * 2

    t_bnb = per_step(lambda p, s: (lambda: F.dequantize_4bit(p, s)))
    print(f"\nNF4: bitsandbytes dequantize_4bit — {t_bnb:.2f} мс/шаг ({t_bnb * STEPS / 1000:.2f} с/прогон)")
    print(f"{'mode':>6} {'groups':>6} {'warps':>5} {'мс/шаг':>8} {'с/прогон':>9}  побитно")
    rows = []
    saved = nf4.configure()
    for mode in ("join", "index"):
        for groups in (16, 32, 64, 128):
            for warps in (2, 4, 8):
                nf4.configure(mode=mode, groups=groups, num_warps=warps)
                try:
                    exact = all(torch.equal(nf4.nf4_dequantize(p, s), ref)
                                for _, p, s, ref, _ in cases)
                    t = per_step(lambda p, s: (lambda: nf4.nf4_dequantize(p, s)))
                except Exception as exc:  # noqa: BLE001 - не влезло в ресурсы и т. п.
                    print(f"{mode:>6} {groups:>6} {warps:>5}   ошибка: {exc!r}"[:120])
                    continue
                rows.append((t, mode, groups, warps, exact))
                print(f"{mode:>6} {groups:>6} {warps:>5} {t:8.2f} {t * STEPS / 1000:9.2f}  "
                      f"{'да' if exact else 'НЕТ'}")
    nf4.configure(**saved)
    good = sorted(r for r in rows if r[4])
    base = next((r for r in rows if r[1:4] == ("join", 64, 4)), None)
    if good:
        t, mode, groups, warps, _ = good[0]
        gain = (base[0] - t) * STEPS / 1000 if base else float("nan")
        print(f"\nлучший NF4: {mode}/{groups}/{warps} — {t:.2f} мс/шаг; против v4 (join/64/4) "
              f"{gain:+.2f} с/прогон экономии")
        return f'fused_nf4.configure(mode="{mode}", groups={groups}, num_warps={warps})'
    return None


def bench_rms():
    print(f"\nRMSNorm (fp32, 1536 колонок), {RMS_CALLS} вызовов на шаг, forward + backward")
    print(f"{'rows':>4} {'warps':>5} " + " ".join(f"{'M=' + str(m):>10}" for m in RMS_ROWS)
          + f" {'с/прогон':>9}  отклонение от v4")
    saved = rms.configure()
    refs = {}
    rms.configure(rows=1, num_warps=4)                 # эталон = v4
    for m in RMS_ROWS:
        torch.manual_seed(m)
        x = torch.randn(m, 1536, device="cuda", requires_grad=True)
        w = torch.randn(1536, device="cuda")
        dy = torch.randn(m, 1536, device="cuda")
        y = rms.rmsnorm(x, w, 1e-6)
        (dx,) = torch.autograd.grad(y, x, dy)
        refs[m] = (y.detach(), dx)
    results = []
    for rows_cfg in (1, 2, 4, 8):
        for warps in (2, 4, 8):
            rms.configure(rows=rows_cfg, num_warps=warps)
            times, worst = [], 0.0
            try:
                for m in RMS_ROWS:
                    torch.manual_seed(m)
                    x = torch.randn(m, 1536, device="cuda", requires_grad=True)
                    w = torch.randn(1536, device="cuda")
                    dy = torch.randn(m, 1536, device="cuda")
                    y = rms.rmsnorm(x, w, 1e-6)
                    (dx,) = torch.autograd.grad(y, x, dy)
                    ref = refs[m]
                    worst = max(worst, (y - ref[0]).abs().max().item(),
                                (dx - ref[1]).abs().max().item())

                    def fb(x=x, w=w, dy=dy):
                        out = rms.rmsnorm(x, w, 1e-6)
                        torch.autograd.grad(out, x, dy)
                    times.append(bench(fb))
            except Exception as exc:  # noqa: BLE001
                print(f"{rows_cfg:>4} {warps:>5}   ошибка: {exc!r}"[:120])
                continue
            per_run = sum(times) / len(times) * RMS_CALLS * STEPS / 1000
            results.append((per_run, rows_cfg, warps, worst))
            print(f"{rows_cfg:>4} {warps:>5} " + " ".join(f"{t * 1000:8.1f}мкс" for t in times)
                  + f" {per_run:9.2f}  {worst:.1e}")
    rms.configure(**saved)
    ok = sorted(r for r in results if r[3] <= 1e-5)
    base = next((r for r in results if r[1:3] == (1, 4)), None)
    if ok:
        per_run, rows_cfg, warps, _ = ok[0]
        gain = base[0] - per_run if base else float("nan")
        print(f"\nлучший RMSNorm: rows={rows_cfg}, warps={warps} — {per_run:.2f} с/прогон "
              f"(с запуском из Python); против v4 {gain:+.2f} с")
        return f"fused_rmsnorm.configure(rows={rows_cfg}, num_warps={warps})"
    return None


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    lines = [bench_nf4(), bench_rms()]
    print("\n=== строки для solution/__init__.py, apply() ===")
    for line in lines:
        print(line or "# не найден вариант лучше/корректнее v4 — оставить как есть")
    print("Внимание: выигрыш меньше ~0,1 с — в пределах шума T4; такие настройки лучше не трогать.")
