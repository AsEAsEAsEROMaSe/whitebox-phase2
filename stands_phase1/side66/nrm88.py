"""ПЕРЕВІРКА КАЛІБРУВАЛЬНОЇ КОНСТАНТИ `RMS(істина)²`. (С88)

Пам'ять каже **0.88404** (12 мереж × 2²⁰, `nrmcheck87.py`).
`dag88` на 40 мережах × 2¹⁹ дав **1.02637** — розбіжність 16%.
Це та сама константа, на якій С87 уже одного разу обпеклася в 4.61×,
тому міряю розкид ЯВНО: чи це мала вибірка, чи різні об'єкти.

📣 ПЕРЕДБАЧЕННЯ: `RMS(істина)²` МІЖ МЕРЕЖАМИ розкидане сильно (лог-нормально:
добуток 32 пошарових підсилень), тож 12 мереж дають ±20% на середньому.
Шанс 70%. Якщо розкид малий — значить, різні ОБ'ЄКТИ (інші ваги/нормування).
"""
import glob, math
import numpy as np

SP = '/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad'
files = sorted(glob.glob(f'{SP}/mc88cache/n*.npz'))
v = np.array([(np.load(f)['out'] ** 2).mean() for f in files])
print(__doc__.split('📣')[1])
print(f'\nмереж: {len(v)}')
print(f'середнє RMS(істина)² = {v.mean():.5f}   (пам\'ять 0.88404)')
print(f'медіана              = {np.median(v):.5f}')
print(f'std МІЖ мережами     = {v.std():.5f}  ({v.std()/v.mean():.1%} від середнього)')
print(f'станд. похибка серед. = {v.std()/math.sqrt(len(v)):.5f} '
      f'({v.std()/math.sqrt(len(v))/v.mean():.1%})')
print(f'min / max            = {v.min():.5f} / {v.max():.5f}')
for k in (12, 24, 40):
    if k <= len(v):
        rng = np.random.default_rng(0)
        s = np.array([v[rng.choice(len(v), k, replace=False)].mean() for _ in range(4000)])
        print(f'  підвибірка {k:2d} мереж: середнє {s.mean():.5f} · '
              f'std {s.std():.5f} · 5–95% [{np.quantile(s,0.05):.4f}, {np.quantile(s,0.95):.4f}]')
print('\n📌 якщо 0.88404 у 5–95% інтервалі для 12 мереж — це просто мала вибірка')
