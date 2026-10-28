"""КУДИ ВКАЗУЄ `Δ = p₈₇ − p_v2`: картинка до вироку `twoest.py`.

«Вказувати на середнє» означає дивитись рівно ПРОТИ власної похибки `e₈₇`:
щоб поправка зменшила помилку, `Δ` мусить мати компоненту вздовж `−e₈₇`.
Тому міряємо КУТ між `Δ` і `e₈₇`:
    180° — `Δ` дивиться ТОЧНО на істину (мрія);
     90° — нікуди (ортогональна, нуль інформації);
      0° — точно ВІД істини (шкідлива).

🚨 У 256 вимірах два НЕЗАЛЕЖНІ вектори лежать під 90° ± ~3.6° (розкид 1/√n).
Тому одного піка на 90° мало — треба порівняти з ЕТАЛОНОМ із випадкових
векторів. Якщо гістограми збігаються, `Δ` не відрізнити від шуму.

Панелі:
  ⬅⬆ полярна: кожна точка — жереб; кут = напрямок `Δ`, радіус = |Δ|/|e₈₇|;
  ⬅⬇ гістограма кута проти ЕТАЛОНА випадкових векторів;
  ➡⬆ скатер по координатах: `Δ` проти `e₈₇` — так виглядає НУЛЬ (`R²`);
  ➡⬇ те саме для `e_v2` проти `e₈₇` — так виглядає СПРАВЖНІЙ зв'язок (`ρ=0.91`).
"""
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
import jitanim                                                    # noqa: E402
import curvegui                                                   # noqa: E402
from base_real import cut_and_budget, closure_parts, W_           # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
OUT = '/mnt/win_d/llm/White-Box/side66/deltafig.png'
BG, FG, MUT = '#0d1117', '#e6edf3', '#8b949e'
RED, CYAN, GOLD, GRN = '#ff2d55', '#00e5ff', '#f0c674', '#3fb950'

TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
IDX = {int(v): i for i, v in enumerate(TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
HAVE = {int(v): j for j, v in enumerate(SD)}

CACHE = '/mnt/win_d/llm/White-Box/side66/deltafig_data.npz'
ang, rad, angref = [], [], []
sc_d, sc_e, sc_v = [], [], []
import os
if os.path.exists(CACHE) and '--fresh' not in sys.argv:
    _c = np.load(CACHE)
    ang, rad, angref = list(_c['ang']), list(_c['rad']), list(_c['angref'])
    sc_d, sc_e, sc_v = [_c['D']], [_c['E']], [_c['V']]
    print('кеш підвантажено; --fresh щоб перерахувати')
for n, s0 in enumerate([] if ang else list(IDX)[:NETS]):
    W = [torch.tensor(np.asarray(WZ[HAVE[s0]][k], np.float32), device=DEV)
         for k in range(32)]
    tm = (TT['T1'][IDX[s0]] + TT['T2'][IDX[s0]]) / 2.0
    with torch.no_grad():
        wc, npair = cut_and_budget(W, DEV)
        cl = closure_parts(wc, 8, DEV, extras=True)
        for r in range(DRAWS):
            g = torch.Generator(device=DEV).manual_seed(51_000 + 977 * r + n)
            z = torch.randn(npair, W_, device=DEV, generator=g)
            p87 = jitanim._run_sub87(wc, npair, cl, DEV, 0, z_ext=z).cpu().numpy()
            pv2 = curvegui._v2(W, npair, DEV, z).double().cpu().numpy()
            e87, ev2 = p87 - tm, pv2 - tm
            d = p87 - pv2
            c = float(d @ e87 / (np.linalg.norm(d) * np.linalg.norm(e87)))
            ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
            rad.append(float(np.linalg.norm(d) / np.linalg.norm(e87)))
            # ⛔ЕТАЛОН: випадковий вектор тієї самої довжини замість `Δ`
            q = np.random.default_rng(7000 + 31 * n + r).standard_normal(e87.size)
            angref.append(np.degrees(np.arccos(np.clip(
                float(q @ e87 / (np.linalg.norm(q) * np.linalg.norm(e87))), -1, 1))))
            if r < 8:
                s = np.linalg.norm(e87)
                sc_d.append(d / s); sc_e.append(e87 / s); sc_v.append(ev2 / s)
    del W
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS} {s0}', flush=True)

ang = np.array(ang); rad = np.array(rad); angref = np.array(angref)
D = np.concatenate(sc_d); E = np.concatenate(sc_e); V = np.concatenate(sc_v)
np.savez(CACHE, ang=ang, rad=rad, angref=angref, D=D, E=E, V=V)
r2d = float(np.corrcoef(D, E)[0, 1] ** 2)
rv = float(np.corrcoef(V, E)[0, 1])
# частка |Δ|, спрямована НА істину (проєкція на −ê₈₇)
share = float(np.mean(-np.cos(np.radians(ang))))

fig = plt.figure(figsize=(15.5, 8.6), facecolor=BG)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=.32, wspace=.2)


def sty(ax, t, xl='', yl=''):
    ax.set_facecolor(BG)
    ax.set_title(t, color=FG, fontsize=11.5, pad=10)
    ax.set_xlabel(xl, color=MUT, fontsize=9)
    ax.set_ylabel(yl, color=MUT, fontsize=9)
    ax.tick_params(colors=MUT, labelsize=8)
    for s in ax.spines.values():
        s.set_color('#30363d')
    ax.grid(alpha=.12, color=FG)


a1 = fig.add_subplot(gs[0, 0], projection='polar')
a1.set_facecolor(BG)
a1.scatter(np.radians(ang), rad, s=9, color=GOLD, alpha=.45, edgecolors='none')
a1.set_theta_zero_location('E')
a1.set_thetagrids([0, 45, 90, 135, 180],
                  ['0°\nВІД істини', '45°', '90°\nНІКУДИ', '135°', '180°\nНА істину'],
                  color=FG, fontsize=8.5)
a1.set_thetamin(0); a1.set_thetamax(180)
a1.tick_params(colors=MUT, labelsize=7.5)
a1.grid(alpha=.16, color=FG)
a1.set_title('КУДИ ВКАЗУЄ Δ  ·  кут до власної похибки, радіус = |Δ|/|e₈₇|',
             color=FG, fontsize=11.5, pad=18)
a1.plot([np.pi, np.pi], [0, rad.max()], color=GRN, lw=2.4, alpha=.8)
a1.plot([np.pi / 2, np.pi / 2], [0, rad.max()], color=MUT, lw=1.4, ls=':')
a1.set_rlabel_position(112)
a1.set_yticks([0.5, 1.0, 1.5])
a1.set_yticklabels(['0.5', '1.0', '1.5'], color=MUT, fontsize=7)

a2 = fig.add_subplot(gs[1, 0])
bins = np.linspace(40, 140, 81)
a2.hist(angref, bins=bins, color=MUT, alpha=.55, label='ЕТАЛОН: випадковий вектор (нуль-гіпотеза)')
a2.hist(ang, bins=bins, color=GOLD, alpha=.75, label='Δ = p₈₇ − p_v2')
a2.axvline(90, color=MUT, ls=':', lw=1.4)
a2.text(90.6, a2.get_ylim()[1]*.92, '90° = нікуди', color=MUT, fontsize=8.5)
sty(a2, f'ГІСТОГРАМА КУТА  ·  Δ ширша за еталон у {ang.std()/angref.std():.1f}× — структура Є, ЗНАКУ нема',
    'кут між Δ і e₈₇, градуси', 'жеребів')
a2.legend(facecolor=BG, edgecolor='#30363d', labelcolor=FG, fontsize=9)

a3 = fig.add_subplot(gs[0, 1])
a3.scatter(E, D, s=3, color=GOLD, alpha=.18, edgecolors='none')
sty(a3, f'Δ ПРОТИ e₈₇ по координатах  ·  R² = {r2d:.4f}  ← ТАК ВИГЛЯДАЄ НУЛЬ',
    'наша похибка e₈₇ (норм.)', 'Δ (норм.)')

a4 = fig.add_subplot(gs[1, 1])
a4.scatter(E, V, s=3, color=CYAN, alpha=.18, edgecolors='none')
sty(a4, f'для контрасту: e_v2 ПРОТИ e₈₇  ·  ρ = {rv:.3f}  ← ТАК ВИГЛЯДАЄ ЗВʼЯЗОК',
    'наша похибка e₈₇ (норм.)', 'похибка v2 (норм.)')

fig.text(.5, .975, f'Δ дивиться під {ang.mean():.1f}° (еталон випадкового: '
         f'{angref.mean():.1f}°) · на істину спрямовано {100*share:+.2f}% довжини Δ · '
         f'{NETS} мереж × {DRAWS} жеребів, крос-ціль',
         ha='center', color=GOLD, fontsize=13, weight='bold')
fig.savefig(OUT, dpi=110, facecolor=BG, bbox_inches='tight')
print(f'\nкут Δ: {ang.mean():.2f}° ± {ang.std():.2f}  ·  '
      f'еталон: {angref.mean():.2f}° ± {angref.std():.2f}')
print(f'на істину спрямовано {100*share:+.3f}% довжини Δ  ·  R²(Δ,e₈₇)={r2d:.4f}  '
      f'·  ρ(e_v2,e₈₇)={rv:.3f}')
print(f'📊 → {OUT}')
