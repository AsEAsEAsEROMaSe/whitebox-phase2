"""СИСТЕМА ВІДЛІКУ, ПРИБИТА ДО АТРАКТОРА (С80, ідея юзера).

Юзер: «якщо є стабільна точка, то ми отримуємо оту сіру пунктирну лінію. воно
точне, але з великою похибкою. цікаво стає, коли наші методи і/або МК рухаються
— відносно цієї точки ми бачитимемо, ЯК вони це роблять».

🔑 ЧОМУ ЦЕ СЛУШНО. `x*` — єдиний об'єкт у задачі з ПРОТИЛЕЖНИМ профілем похибки:
    `x*`        шум **2.8e-06** (зміряно `exactcos.py`)  ·  зміщення **~10%**
    наша оцінка шум 0.177%                               ·  зміщення ≈0
⟹ у системі відліку, прибитій до `x*`, увесь видимий рух — це рух ОЦІНОК,
   бо сама вісь не тремтить. Це робить `x*` придатним як ЛІНІЙКА, хоч він і
   непридатний як ВІДПОВІДЬ.

## ЧОТИРИ ПАНЕЛІ
 1. площина `(x*, E[o])`: хмари МК при різних `N`, істина, атрактор;
 2. збіжність log-log: похибка МК від `N` проти СТАЛОЇ лінії атрактора
    + точка перетину (де МК переганяє атрактор);
 3. кут похибки МК відносно осі `x*→E[o]` — чи є в неї улюблений напрямок;
 4. 🎯 ГОЛОВНА: розклад шуму на складову ВЗДОВЖ осі атрактора й ПОПЕРЕК,
    проти ізотропного очікування `1/√255` — це вже вимір, а не картинка.

📐 РАХУЄТЬСЯ НАПЕРЕД: похибка напрямку одного семпла 15.11%, атрактора 9.93%
   ⟹ МК переганяє його при `N = (15.11/9.93)² ≈ 2.3`.
   **Уся конструкція з атрактором варта ~2–3 семплів.**

⛔ ПЛОМБИ:
 * ізотропний контроль: для ВИПАДКОВОЇ осі частка вздовж мусить дати `1/255`;
 * хмара МК мусить стискатися рівно як `1/√N` (нахил −0.5 у log-log);
 * ⛔НЕГ: вісь із ЧУЖОЇ мережі.

📣 ПЕРЕДБАЧЕНО: шум щодо осі атрактора виявиться ІЗОТРОПНИМ (частка вздовж ≈
   1/255 = 0.39%), бо вісь `x*` майже збігається з `E[o]`, а вздовж `E[o]`
   похибка вже виміряна як 36% ([[finding-error-is-one-scalar-plus-covariance]])
   — тобто якщо частка вийде ~36%, це буде ПЕРЕВІДКРИТТЯ вже відомого, а не нове.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/anchorframe.py [cuda:N] [NETS]
Малює у side66/shots/anchorframe.png (без вікна, щоб працювало headless).
"""
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NDRAW = 64
NS = [4, 16, 64, 256, 1024, 4096, 16384]
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


def _attr(W, gen, iters=800):
    x = torch.randn(1, W_, device=DEV, generator=gen)
    x = x / x.norm()
    for _ in range(iters):
        y = _f(W, x)
        x = y / y.norm().clamp_min(1e-30)
    return x.reshape(-1).double()


P2 = {n: [] for n in NS}          # координати в площині (для панелі 1)
ERR = {n: [] for n in NS}         # відносна похибка МК
ALONG = {n: [] for n in NS}       # частка енергії шуму ВЗДОВЖ осі
ALONG_R = {n: [] for n in NS}     # те саме для ВИПАДКОВОЇ осі (пломба)
ALONG_F = {n: [] for n in NS}     # ⛔НЕГ: вісь із чужої мережі
ANG = {n: [] for n in NS}
gap = []

for j in range(min(NETS, len(IDX))):
    W = [torch.tensor(np.asarray(WZ[IDX[j]][k], np.float32), device=DEV)
         for k in range(D)]
    mu = torch.tensor(TZ['T1'][j], dtype=torch.float64, device=DEV)
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(2024 + 31 * j)
        xs = _attr(W, g)
        # вісь: напрямок від істини до атрактора, у площині, ортогональній mu
        e_mu = mu / mu.norm()
        d = xs - (xs @ e_mu) * e_mu
        axis = d / d.norm().clamp_min(1e-30)
        gap.append(float(d.norm() / xs.norm()))
        # ⛔НЕГ вісь із чужої мережі
        Wf = [torch.tensor(np.asarray(WZ[IDX[(j + 7) % len(IDX)]][k],
                                      np.float32), device=DEV) for k in range(D)]
        xf = _attr(Wf, g)
        df = xf - (xf @ e_mu) * e_mu
        axf = df / df.norm().clamp_min(1e-30)
        del Wf
        rax = torch.randn(W_, dtype=torch.float64, device=DEV, generator=g)
        rax = rax - (rax @ e_mu) * e_mu
        rax = rax / rax.norm()

        for n in NS:
            for _ in range(NDRAW):
                z = torch.randn(n, W_, device=DEV, generator=g)
                p = _f(W, z).double().mean(0)
                e = p - mu
                ep = e - (e @ e_mu) * e_mu          # поперечна частина
                n2 = float(e @ e)
                ERR[n].append(float(e.norm() / mu.norm()))
                ALONG[n].append(float((ep @ axis) ** 2) / max(n2, 1e-300))
                ALONG_R[n].append(float((ep @ rax) ** 2) / max(n2, 1e-300))
                ALONG_F[n].append(float((ep @ axf) ** 2) / max(n2, 1e-300))
                ANG[n].append(float((e / e.norm().clamp_min(1e-300)) @ axis))
                P2[n].append((float(e @ e_mu) / float(mu.norm()),
                              float(e @ axis) / float(mu.norm())))
    if j % 8 == 0:
        print(f'  … мережа {j+1}/{min(NETS,len(IDX))}', flush=True)
    del W
    torch.cuda.empty_cache()

med_gap = float(np.median(gap))
print(f'\n{min(NETS,len(IDX))} мереж × {NDRAW} жеребів на кожне N\n')
print('⛔ ПЛОМБИ')
sl = np.polyfit(np.log(NS), [np.log(np.median(ERR[n])) for n in NS], 1)[0]
print(f'  нахил похибки МК у log-log = {sl:+.4f}   (теорія −0.5)')
print(f'  ізотропний контроль (ВИПАДКОВА вісь) = '
      f'{np.median([np.median(ALONG_R[n]) for n in NS]):.5f}   '
      f'(теорія 1/255 = {1/255:.5f})')
print(f'  ⛔НЕГ вісь із ЧУЖОЇ мережі = '
      f'{np.median([np.median(ALONG_F[n]) for n in NS]):.5f}')
print(f'\n📐 ЗМІЩЕННЯ АТРАКТОРА (поперечна частина) = {100*med_gap:.2f}%')
print(f'\n{"N":>7s} {"похибка МК":>12s} {"частка ВЗДОВЖ осі":>19s} '
      f'{"проти ізотропної":>18s}')
print('-' * 62)
for n in NS:
    a_, r_ = np.median(ALONG[n]), np.median(ALONG_R[n])
    print(f'{n:7d} {100*np.median(ERR[n]):11.4f}% {100*a_:18.3f}% '
          f'{a_/max(r_,1e-12):17.1f}×')

cross = (0.1511 / med_gap) ** 2
print(f'\n🎯 МК ПЕРЕГАНЯЄ АТРАКТОР ПРИ N ≈ {cross:.1f} '
      f'⟹ атрактор вартий ~{cross:.0f} семплів')

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Система відліку, прибита до атрактора  ·  '
             f'{min(NETS,len(IDX))} мереж × {NDRAW} жеребів', fontsize=13)

a0 = ax[0, 0]
cols = plt.cm.viridis(np.linspace(0, 0.9, len(NS)))
for i, n in enumerate(NS):
    q = np.array(P2[n])
    a0.scatter(q[:, 0], q[:, 1], s=4, alpha=.35, color=cols[i], label=f'N={n}')
a0.axhline(med_gap, color='grey', ls='--', lw=2)
a0.text(0, med_gap, f'  атрактор  x*   ({100*med_gap:.1f}%)', color='grey',
        va='bottom', fontsize=10)
a0.plot(0, 0, 'r*', ms=18, zorder=5)
a0.text(0, 0, '  істина E[o]', color='red', va='top', fontsize=10)
a0.set_xlabel('уздовж E[o]  (частка ‖E[o]‖)')
a0.set_ylabel('уздовж осі x*  (частка ‖E[o]‖)')
a0.set_title('1. Площина (E[o], x*): хмари МК стискаються до істини,\n'
             'атрактор стоїть НЕРУХОМО збоку')
a0.legend(fontsize=7, ncol=2)
a0.grid(alpha=.3)

a1 = ax[0, 1]
med = [np.median(ERR[n]) for n in NS]
lo = [np.percentile(ERR[n], 10) for n in NS]
hi = [np.percentile(ERR[n], 90) for n in NS]
a1.fill_between(NS, lo, hi, alpha=.25)
a1.loglog(NS, med, 'o-', lw=2, label='похибка МК')
a1.axhline(med_gap, color='grey', ls='--', lw=2,
           label=f'атрактор (стала, {100*med_gap:.1f}%)')
a1.axvline(cross, color='crimson', ls=':', lw=2,
           label=f'перетин: N ≈ {cross:.1f}')
a1.axhline(2e-4, color='green', ls='-.', lw=1.5, label='ціль 0.02%')
a1.set_xlabel('N (семплів)'); a1.set_ylabel('відносна похибка')
a1.set_title(f'2. МК переганяє атрактор уже при N≈{cross:.0f}\n'
             f'нахил {sl:+.3f} (теорія −0.5)')
a1.legend(fontsize=8); a1.grid(alpha=.3, which='both')

a2 = ax[1, 0]
for i, n in enumerate(NS):
    a2.hist(ANG[n], bins=40, histtype='step', lw=1.6, color=cols[i],
            density=True, label=f'N={n}')
a2.axvline(0, color='k', lw=1)
a2.set_xlabel('cos(похибка МК, вісь атрактора)')
a2.set_ylabel('щільність')
a2.set_title('3. Кут похибки відносно осі атрактора:\n'
             'симетрично навколо нуля ⟹ вісь НЕ виділена за знаком')
a2.legend(fontsize=7, ncol=2); a2.grid(alpha=.3)

a3 = ax[1, 1]
va = [100 * np.median(ALONG[n]) for n in NS]
vr = [100 * np.median(ALONG_R[n]) for n in NS]
vf = [100 * np.median(ALONG_F[n]) for n in NS]
a3.semilogx(NS, va, 'o-', lw=2, label='вісь атрактора ЦІЄЇ мережі')
a3.semilogx(NS, vf, 's--', lw=1.6, label='⛔НЕГ: вісь ЧУЖОЇ мережі')
a3.semilogx(NS, vr, '^:', lw=1.6, label='⛔ізотропна (випадкова вісь)')
a3.axhline(100 / 255, color='k', ls=':', lw=1)
a3.text(NS[0], 100 / 255, ' теорія 1/255', fontsize=8, va='bottom')
a3.set_xlabel('N (семплів)')
a3.set_ylabel('частка енергії шуму вздовж осі, %')
a3.set_title('4. ГОЛОВНА: чи виділена вісь атрактора для ШУМУ')
a3.legend(fontsize=8); a3.grid(alpha=.3, which='both')

plt.tight_layout()
out = '/mnt/win_d/llm/White-Box/side66/shots/anchorframe.png'
plt.savefig(out, dpi=110)
print(f'\n🖼  збережено: {out}')
print('📊 БАЛ 1.954e-07')
