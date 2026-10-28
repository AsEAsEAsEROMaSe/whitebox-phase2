"""ЧИ РОЗВАЛЮЄТЬСЯ 1024 ВЗАГАЛІ — АБО ЛИШЕ ПРИ БЮДЖЕТІ, ЯКОГО НЕ БУДЕ? (С81)

## 🚨 ЗНАХІДКА, ЩО Б'Є ПО ВЛАСНОМУ ВИСНОВКУ С80
С80 міряла ширину 1024 при БЮДЖЕТІ 2.72e11 — тобто ТОМУ САМОМУ, що й на 256.
Я його не перевіряв, просто лишив. А в старткіті лежить прямий доказ, що
спонсор так НЕ робить (`whestbench/cli.py:451`, дослівно):
    «invariant under the 4×/4× depth-and-budget scaling from the warmup round»
⟹ коли форму МІНЯЛИ (глибина 8 → 32), бюджет підняли ТЕЖ учетверо, щоб
   утилізація лишилась та сама. `rules.md` §5.4.3 каже «same FLOP-budget
   CALIBRATION», а калібрування — це формула, а не константа.

## 🔑 ЩО ЦЕ МІНЯЄ
Вартість одного семпла МК = `2·глибина·ширина²`. 256 → 1024 це **16×**.
Отже «утилізаційно-інваріантний» бюджет на 1024 = `16 × 2.72e11 = 4.35e12`.
Проба на 1 мережі вже показала: `C/B` для sub91 падає **0.4611 → 0.1039**,
тобто при ×16 ми знову ПІД підлогою 0.1 — рівно як на 256 (0.0999).

## ЩО МІРЯЄМО
Ширина 1024, глибина 32, 6 мереж; множник бюджету 1…32. Для кожного:
  ⛔ чистий МК (база) · 🟠 sub91 ЯК Є · 🟢 рецепт С80 (2048 пар, KSIG=1,
  без вибілювання) — той, що дав 1.4843× при множнику 1.

⛔ ПЛОМБА: ширина 256, множник 1 ⟹ sub91 мусить дати ≈**3.24×** (dims.py).
   Якщо ні — стенд бреше, і решту таблиці читати не можна.

📣 ПЕРЕДБАЧЕНО: sub91 при ×16 ВІДНОВИТЬСЯ до ~2…3×, бо (а) утилізація
   повертається під підлогу, (б) пар стає ≥ ширини і вибілюванню вистачає
   рангу. Перелом — між ×2 і ×8. Якщо так, увесь «протилежний рецепт» С80
   потрібен лише у сценарії, який спонсор власним прецедентом виключає.
   Якщо ж sub91 лишається розваленим і при ×32 — розвал СПРАВЖНІЙ, і
   розгалуження за шириною обов'язкове.

Запуск: python3 side66/wbudget.py
"""
import sys
import time

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)
import numpy as np                                       # noqa: E402
import flopscope as flops                                # noqa: E402
import flopscope.numpy as fnp                            # noqa: E402
from whestbench import MLP                               # noqa: E402
from bench import load_estimator                         # noqa: E402
from flopscope.errors import BudgetExhaustedError        # noqa: E402

flops.configure(symmetry_warnings=False)
B0, LAM, D, NETS = 272_000_000_000, 1e11, 32, 6
MULTS = [1, 2, 4, 8, 16, 32]
Z = np.load('/mnt/win_d/llm/White-Box/side66/logs/dims.npz')
W1024 = '/mnt/win_d/llm/White-Box/side66/w1024/estimator_w1024_best.py'
CAND = [('⛔ чистий МК', SK + '/estimator_mc_plain.py'),
        ('🟠 sub91 ЯК Є', SK + '/estimator_sub91.py'),
        ('🟢 рецепт С80 (1024)', W1024)]
print(__doc__.split('Запуск')[0], flush=True)


def run(path, width, mult):
    """Повертає (MSE, ціна C/B, скільки мереж вилетіло за бюджет)."""
    B = B0 * mult
    E = load_estimator(path)()
    ms, cs, ov = [], [], 0
    for n in range(NETS):
        key = f'{width}x{D}x{n}'
        Wn = [fnp.array(Z[f'{key}_w{i}']) for i in range(D)]
        T = Z[key + '_T'].astype(np.float64)
        blown = False
        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            try:
                o = E.predict(MLP(width=width, depth=D, weights=Wn,
                                  seed=n + 1), B)
            except BudgetExhaustedError:
                blown, o = True, np.zeros((D, width))
        C = c.flops_used + LAM * c.summary_dict()['residual_wall_time_s']
        a = np.asarray(o)[-1].astype(np.float64)
        if blown or C > B:                     # 🔑 правило грейдера: нулі, ×1.0
            a = np.zeros_like(a)
            ov += 1
        ms.append(float(((a - T) ** 2).mean()))
        cs.append(1.0 if (blown or C > B) else max(0.1, C / B))
    return (float(np.mean(ms)), float(np.mean(cs)),
            float(np.mean([x * y for x, y in zip(ms, cs)])), ov)


print('⛔ПЛОМБА — ширина 256, множник 1 (мусить бути ≈3.24×)')
_, _, s_mc, _ = run(SK + '/estimator_mc_plain.py', 256, 1)
_, _, s_91, _ = run(SK + '/estimator_sub91.py', 256, 1)
print(f'   sub91 проти МК = {s_mc / s_91:.4f}×   '
      f'{"✅ОК" if 2.6 < s_mc / s_91 < 4.0 else "🚨СТЕНД БРЕШЕ"}\n')

print(f'ШИРИНА 1024 · глибина 32 · {NETS} мереж')
print(f'{"множник":>8s} {"варіант":>22s} {"MSE":>11s} {"ціна":>7s} '
      f'{"БАЛ":>11s} {"проти МК":>9s} {"вилет":>6s}')
print('-' * 82)
for mult in MULTS:
    base = None
    for nm, path in CAND:
        t = time.time()
        mse, price, sc, ov = run(path, 1024, mult)
        if base is None:
            base = sc
        print(f'{"×" + str(mult):>8s} {nm:>22s} {mse:11.4e} {price:7.4f} '
              f'{sc:11.4e} {base / sc:8.4f}× {ov:3d}/{NETS}'
              f'   {time.time() - t:5.1f}s', flush=True)
    print('-' * 82)
print('\n💰 утилізаційно-інваріантний множник для 1024 = 16 '
      '(вартість семпла ∝ ширина²)')
print('📊 БАЛ 1.954e-07 — вимір вирішує, чи потрібне розгалуження за шириною')
