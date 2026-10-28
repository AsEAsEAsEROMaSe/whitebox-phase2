"""ОДНА ПРЯМА, НА ЯКІЙ ЛЕЖИТЬ УСЕ (С80, ідея юзера).

Юзер: «якщо є зміщена сіра лінія, то вже на початках синя починає гуляти
навколо якоїсь ВІРТУАЛЬНОЇ ЛІНІЇ. має існувати теоретичний механізм, щоб
зрозуміти відношення. якщо взяти червону МК і наш топовий — візуалізуй».

🔑 ВІДПОВІДЬ НА «ЩО ЗА ВІРТУАЛЬНА ЛІНІЯ» — вона є, і теорія для неї відома:
    похибка(N) = σ/√N          — пряма з нахилом −1/2 у log-log
    гуляння навколо неї         — це НЕ шум приладу, а χ² з `k` ступенями волі,
                                  амплітуда `√(2/k)`; зміряно `k≈6`
                                  ([[finding-error-lives-in-six-directions]])
                                  ⟹ очікуване гуляння `√(2/6) = 58%`.
🎯 ТОБТО «ВІРТУАЛЬНА ЛІНІЯ» = МЕДІАНА, а розмах навколо неї заданий тим, що
   похибка живе лише в ~6 незалежних напрямках із 256.

🔑 І ГОЛОВНЕ — ЄДИНА ВІСЬ, ЩО ЗВОДИТЬ УСЕ ДОКУПИ: `N_еф` (ефективні семпли).
   Будь-який метод = точка на ТІЙ САМІЙ прямій, питання лише де.
       атрактор        → `N_еф ≈ 1`
       один семпл      → 1
       наш топовий     → `N × (виграш машинерії)`
       лідер           → у 24.4× менша MSE ⟹ у 24.4× більше `N_еф`
   Це і є «механізм відношення», якого бракувало.

## ПАНЕЛІ
 1. усе на одній прямій: МК, атрактор, наш топовий, лідер, ціль;
 2. гуляння синьої навколо віртуальної лінії + теоретична смуга `√(2/k)`;
 3. скільки `N_еф` купує наша машинерія (і скільки треба лідеру);
 4. 🎯 ЧОМУ РУХ ПО ПРЯМІЙ НІЧОГО НЕ ДАЄ: `бал = MSE × max(0.1, C/B)` —
    вище підлоги `MSE×бюджет = const`, тобто бал ПЛАСКИЙ.

⛔ ПЛОМБИ: нахил −0.5 · гуляння проти теорії `√(2/k)` при `k=6` ·
   `N_еф` нашого топового мусить збігтися з незалежно відомим виграшем машинерії.

📣 ПЕРЕДБАЧЕНО: наш топовий купує `N_еф/N ≈ 1.5–2×` (вибілювання 1.48× плюс
   канал КЗ), а лідеру треба ≈24× ⟹ саме тому семплінгом туди не дійти.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/oneline.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NDRAW = 96
NS = [1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384]
W_, D = 256, 32
PROD_MSE = 1.95e-06          # наш топовий (sub87), зміряно грейдером
PROD_N = 8192
LEADER_RATIO = 24.4          # у стільки разів менша MSE в лідера

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
TZ = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
SEEDS = list(np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'))
IDX = [SEEDS.index(s) for s in TZ['seeds'].tolist()]
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _f(W, X):
    for L in range(D):
        X = torch.relu(X @ W[L])
    return X


def _attr(W, gen, it=800):
    x = torch.randn(1, W_, device=DEV, generator=gen)
    x = x / x.norm()
    for _ in range(it):
        y = _f(W, x)
        x = y / y.norm().clamp_min(1e-30)
    return x.reshape(-1).double()


ERR = {n: [] for n in NS}
att, nrm2 = [], []
for j in range(min(NETS, len(IDX))):
    W = [torch.tensor(np.asarray(WZ[IDX[j]][k], np.float32), device=DEV)
         for k in range(D)]
    mu = torch.tensor(TZ['T1'][j], dtype=torch.float64, device=DEV)
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(777 + 41 * j)
        xs = _attr(W, g)
        # найкращий масштаб для атрактора (оракульний — на його користь)
        c = float((xs @ mu) / (xs @ xs))
        att.append(float((c * xs - mu).norm() / mu.norm()))
        nrm2.append(float(mu @ mu))
        for n in NS:
            for _ in range(NDRAW):
                z = torch.randn(n, W_, device=DEV, generator=g)
                p = _f(W, z).double().mean(0)
                ERR[n].append(float((p - mu).norm() / mu.norm()))
    if j % 8 == 0:
        print(f'  … мережа {j+1}/{min(NETS,len(IDX))}', flush=True)
    del W
    torch.cuda.empty_cache()

# 💾 зберігаємо СИРІ дані, щоб інтерактивний стенд `linegui.py` відкривався
#    миттєво й не рахував GPU наново
np.savez('/mnt/win_d/llm/White-Box/side66/logs/oneline.npz',
         NS=np.array(NS), ERR=np.array([ERR[n] for n in NS]),
         att=np.array(att), nrm2=np.array(nrm2), ndraw=NDRAW,
         prod_mse=PROD_MSE, prod_n=PROD_N, leader=LEADER_RATIO)

med = np.array([np.median(ERR[n]) for n in NS])
sig = med[0]                              # похибка ОДНОГО семпла = σ
slope = np.polyfit(np.log(NS), np.log(med), 1)[0]
att_med = float(np.median(att))
# наш топовий у ТІЙ САМІЙ метриці
prod_rel = float(np.sqrt(256 * PROD_MSE / np.median(nrm2)))
n_eff_prod = (sig / prod_rel) ** 2
n_eff_att = (sig / att_med) ** 2
n_eff_lead = n_eff_prod * LEADER_RATIO
prod_gain = n_eff_prod / PROD_N
# гуляння: відносний розкид похибки навколо медіани
wob = np.array([np.std(ERR[n]) / np.median(ERR[n]) for n in NS])
k_eff = 2.0 / np.median(wob[3:]) ** 2 / 4.0   # похибка ~√MSE ⟹ розкид удвічі менший

print(f'\n{min(NETS,len(IDX))} мереж × {NDRAW} жеребів\n')
print('⛔ ПЛОМБИ')
print(f'  нахил у log-log = {slope:+.4f}   (теорія −0.5)')
print(f'  σ (похибка ОДНОГО семпла) = {100*sig:.2f}%')
print(f'\n{"величина":34s} {"відн.похибка":>13s} {"N_еф":>12s}')
print('-' * 62)
for nm, e in (('один семпл', sig), ('🩶 АТРАКТОР (оракульний масштаб)', att_med),
              ('🔵 наш топовий (sub87)', prod_rel),
              ('🔴 рівень лідера', prod_rel / np.sqrt(LEADER_RATIO)),
              ('🎯 ціль 0.02%', 2e-4)):
    print(f'{nm:34s} {100*e:12.4f}% {(sig/e)**2:12.0f}')
print(f'\n🔑 НАША МАШИНЕРІЯ КУПУЄ N_еф/N = {prod_gain:.2f}×  '
      f'(при {PROD_N} реальних семплах)')
print(f'   ЛІДЕРУ треба N_еф = {n_eff_lead:,.0f} ⟹ у {LEADER_RATIO:.1f}× більше')
print(f'\n📐 ГУЛЯННЯ навколо віртуальної лінії: зміряно '
      f'{100*np.median(wob[3:]):.1f}%  ⟹ ефективних напрямків k ≈ {k_eff:.1f}')
print(f'   (корпус: k ≈ 6 із двох незалежних вимірів)')

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Одна пряма, на якій лежить усе  ·  '
             f'{min(NETS,len(IDX))} мереж × {NDRAW} жеребів', fontsize=13)

a0 = ax[0, 0]
lo = [np.percentile(ERR[n], 10) for n in NS]
hi = [np.percentile(ERR[n], 90) for n in NS]
a0.fill_between(NS, lo, hi, alpha=.2, color='tab:blue')
a0.loglog(NS, med, 'o-', color='tab:blue', lw=2, label='МК (виміряно)')
a0.axhline(att_med, color='grey', ls='--', lw=2,
           label=f'🩶 атрактор {100*att_med:.1f}%  (N_еф≈{n_eff_att:.0f})')
a0.plot([n_eff_prod], [prod_rel], 'o', color='tab:blue', ms=14, mec='k',
        zorder=6, label=f'🔵 наш топовий (N_еф={n_eff_prod:,.0f})')
a0.plot([n_eff_lead], [prod_rel / np.sqrt(LEADER_RATIO)], 'o', color='crimson',
        ms=14, mec='k', zorder=6, label=f'🔴 лідер (N_еф={n_eff_lead:,.0f})')
a0.axhline(2e-4, color='green', ls='-.', lw=1.5, label='🎯 ціль 0.02%')
a0.set_xlabel('ефективні семпли  N_еф'); a0.set_ylabel('відносна похибка')
a0.set_title('1. Усі методи — точки на ОДНІЙ прямій σ/√N.\n'
             'Питання лише в тому, ДЕ ти на ній сидиш')
a0.legend(fontsize=8); a0.grid(alpha=.3, which='both')

a1 = ax[0, 1]
for i, n in enumerate(NS):
    e = np.array(ERR[n]) / np.median(ERR[n])
    a1.scatter([n] * len(e), e, s=3, alpha=.15, color='tab:blue')
a1.axhline(1, color='crimson', lw=2, label='віртуальна лінія = медіана')
a1.axhline(1 + np.median(wob[3:]), color='crimson', ls=':', lw=1.5)
a1.axhline(1 - np.median(wob[3:]), color='crimson', ls=':', lw=1.5,
           label=f'±{100*np.median(wob[3:]):.0f}% (χ², k≈{k_eff:.0f})')
a1.set_xscale('log'); a1.set_xlabel('N'); a1.set_ylabel('похибка ÷ медіана')
a1.set_title('2. Що таке «гуляння»: χ² з k≈6 ступенями волі.\n'
             'Розмах НЕ спадає з N — це властивість, а не шум')
a1.legend(fontsize=8); a1.grid(alpha=.3)

a2 = ax[1, 0]
names = ['атрактор', 'один\nсемпл', 'наш\nтоповий', 'лідер', 'ціль\n0.02%']
vals = [n_eff_att, 1, n_eff_prod, n_eff_lead, (sig / 2e-4) ** 2]
cols = ['grey', 'black', 'tab:blue', 'crimson', 'green']
a2.bar(names, vals, color=cols, alpha=.8)
a2.set_yscale('log'); a2.set_ylabel('N_еф (ефективні семпли)')
for i, v in enumerate(vals):
    a2.text(i, v * 1.4, f'{v:,.0f}', ha='center', fontsize=9)
a2.set_title('3. Скільки ефективних семплів коштує кожен рівень\n'
             f'наша машинерія купує ×{prod_gain:.2f} понад сирий МК')
a2.grid(alpha=.3, axis='y')

a3 = ax[1, 1]
Ng = np.logspace(0, 6, 200)
mse = (sig * np.sqrt(PROD_N / Ng) * prod_rel / prod_rel) ** 2
mse = (sig / np.sqrt(Ng)) ** 2
budget = np.maximum(0.1, 0.0997 * Ng / PROD_N)
score = mse * budget
a3.loglog(Ng, score / score[np.argmin(np.abs(Ng - PROD_N))], lw=2)
a3.axvline(PROD_N, color='tab:blue', ls='--', label=f'ми: N={PROD_N}')
a3.axvline(0.1 / 0.0997 * PROD_N, color='k', ls=':',
           label='підлога множника 10%')
a3.set_xlabel('N (семплів)'); a3.set_ylabel('бал, віднормований до нашого')
a3.set_title('4. ЧОМУ РУХ ПО ПРЯМІЙ НЕ ДАЄ НІЧОГО:\n'
             'вище підлоги MSE×бюджет = const ⟹ бал ПЛАСКИЙ')
a3.legend(fontsize=8); a3.grid(alpha=.3, which='both')

plt.tight_layout()
out = '/mnt/win_d/llm/White-Box/side66/shots/oneline.png'
plt.savefig(out, dpi=110)
print(f'\n🖼  збережено: {out}')
print(f'📊 БАЛ 1.954e-07 — рухатись по прямій безглуздо (панель 4); '
      f'треба ЗМІНИТИ ПРЯМУ')
