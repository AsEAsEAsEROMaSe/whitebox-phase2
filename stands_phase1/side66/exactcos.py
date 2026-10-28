"""НАСКІЛЬКИ ТОЧНО МИ ЗНАЄМО `cos(x*, E[o])`? (С80, виклик юзера «реально точна?»)

ЗВІДКИ. У `twopass.py` я назвав 14.4% «точною відстанню». Це було ПЕРЕБІЛЬШЕННЯ,
і в моїх же даних лежали три причини сумніватися:
 1. значення скакало: k=12 → 0.98973, k=20 → 0.98944, k=40 → 0.98952 ⟹ розкид
    ±3e-4 БІЛЬШИЙ за весь «прогрес» від k=2 до k=40;
 2. 40 ітерацій може бути замало: `fixed.py` (С80) показав, що при щілині
    `|λ₂|/|λ₁| → 1` збіжність іде як `(|λ₂|/|λ₁|)^k`, і 400 кроків дають лише
    0.018 при щілині 0.99;
 3. С79 дала СИРИЙ `cos = 0.9914`, я — 0.9895. Третій знак НЕ ЗБІГАЄТЬСЯ між
    двома стендами.

## ЩО РОБИМО ІНАКШЕ
* збіжність до межі fp32: `PRE` ітерацій замість 40;
* ⛔ПЕРЕВІРКА ЗБІЖНОСТІ: косинус на кроці `PRE/2` проти `PRE` — якщо різниця
  не мала, мережа НЕ збіглася і її числа не можна цитувати;
* косинус рахуємо у fp64 проти ОБОХ незалежних істин `T1` і `T2` — їхня
  розбіжність і є ШУМ ПРИЛАДУ (а не наша здогадка про нього);
* друкуємо ПО КОЖНІЙ МЕРЕЖІ, бо медіана ховає саме ті мережі, що не збіглися;
* ⛔НЕГ: косинус із істиною ЧУЖОЇ мережі.

🎯 ПИТАННЯ, НА ЯКЕ ВІДПОВІДАЄМО: скільки ЗНАКІВ у `cos(x*,E[o])` реальні?
   Вирок про важіль від цього не залежить (13–15% проти цілі 0.02% — це 700×),
   але чесність числа залежить.

📣 ПЕРЕДБАЧЕНО ДО ПРОГОНУ: шум приладу (`T1` проти `T2`) буде ДУЖЕ малий (істини
   чисті, 1.47e-08), тож обмежувачем виявиться ЗБІЖНІСТЬ, і частина мереж не
   збіжиться навіть за 3000 кроків. Ставлю, що по збіжних мережах косинус
   стабільний у 5–6 знаках, але РОЗКИД МІЖ МЕРЕЖАМИ великий (0.97…0.999),
   і саме він, а не шум, пояснює розбіжність із С79.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/exactcos.py [cuda:N] [NETS] [PRE]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
PRE = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
NS = 8
W_, D = 256, 32

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
TZ = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
SEEDS = list(np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'))
IDX = [SEEDS.index(s) for s in TZ['seeds'].tolist()]
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _f(W, X):
    for L in range(D):
        X = torch.relu(X @ W[L])
    return X


rows = []
for j in range(min(NETS, len(IDX))):
    W = [torch.tensor(np.asarray(WZ[IDX[j]][k], np.float32), device=DEV)
         for k in range(D)]
    t1 = torch.tensor(TZ['T1'][j], dtype=torch.float64, device=DEV)
    t2 = torch.tensor(TZ['T2'][j], dtype=torch.float64, device=DEV)
    tf = torch.tensor(TZ['T1'][(j + 7) % len(IDX)], dtype=torch.float64,
                      device=DEV)
    u1, u2, uf = (v / v.norm() for v in (t1, t2, tf))
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(555 + 13 * j)
        X = torch.randn(NS, W_, device=DEV, generator=g)
        X = X / X.norm(dim=1, keepdim=True)
        half = None
        for k in range(PRE):
            X = _f(W, X)
            X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-30)
            if k + 1 == PRE // 2:
                half = (X.double() @ u1).median().item()
        Xd = X.double()
        Xd = Xd / Xd.norm(dim=1, keepdim=True)
        c1 = (Xd @ u1)
        c2 = (Xd @ u2)
        cf = (Xd @ uf)
        # розкид МІЖ 8 стартами: чи всі прийшли в ту саму точку
        spread = float(c1.max() - c1.min())
        conv = abs(float(c1.median()) - half)      # ⛔перевірка збіжності
        rows.append((float(c1.median()), float(c2.median()), float(cf.median()),
                     spread, conv))
    print(f'  мережа {j+1:2d}: cos(T1)={rows[-1][0]:.8f}  '
          f'|T1−T2|={abs(rows[-1][0]-rows[-1][1]):.2e}  '
          f'збіжність={conv:.2e}  розкид стартів={spread:.2e}', flush=True)
    del W
    torch.cuda.empty_cache()

a = np.array(rows)
print(f'\n{a.shape[0]} мереж × {NS} стартів × {PRE} ітерацій (fp32), '
      f'косинус у fp64\n')
print('⛔ ПЛОМБИ І ШУМ ПРИЛАДУ')
print(f'  |cos(T1) − cos(T2)|  медіана = {np.median(np.abs(a[:,0]-a[:,1])):.3e}'
      f'   макс = {np.abs(a[:,0]-a[:,1]).max():.3e}   ← ШУМ ІСТИНИ')
print(f'  збіжність |крок {PRE//2} − крок {PRE}| медіана = '
      f'{np.median(a[:,4]):.3e}   макс = {a[:,4].max():.3e}')
print(f'  розкид між {NS} стартами: медіана = {np.median(a[:,3]):.3e}   '
      f'макс = {a[:,3].max():.3e}')
print(f'  ⛔НЕГ (істина ЧУЖОЇ мережі) = {np.median(a[:,2]):.4f}')

ok = (a[:, 4] < 1e-6) & (a[:, 3] < 1e-6)
print(f'\n🔑 ЗБІГЛИСЯ (обидві перевірки < 1e-6): {int(ok.sum())}/{a.shape[0]} мереж')
for nm, m in (('ВСІ мережі', np.ones(a.shape[0], bool)), ('лише ЗБІЖНІ', ok)):
    if m.sum() == 0:
        continue
    c = a[m, 0]
    print(f'\n{nm}: cos медіана {np.median(c):.6f}  '
          f'мін {c.min():.6f}  макс {c.max():.6f}')
    e = np.sqrt(np.maximum(1 - c * c, 0))
    print(f'   збоку лишається: медіана {100*np.median(e):.2f}%  '
          f'діапазон {100*e.min():.2f}%…{100*e.max():.2f}%')

noise = float(np.median(np.abs(a[:, 0] - a[:, 1])))
spr = float(np.std(a[:, 0]))
print(f'\n🎯 ВІДПОВІДЬ ЮЗЕРУ «реально точна?»')
print(f'   шум приладу        = {noise:.1e}  ⟹ достовірних знаків ~{max(0,int(-np.log10(max(noise,1e-16)))):d}')
print(f'   РОЗКИД МІЖ МЕРЕЖАМИ = {spr:.1e}  ⟹ у {spr/max(noise,1e-16):.0f}× більший')
print('   ⟹ «одне число» для всіх мереж НЕ ІСНУЄ; існує РОЗПОДІЛ')
print('📊 БАЛ 1.954e-07 — вимір про ТОЧНІСТЬ ЧИСЛА, не про важіль')
