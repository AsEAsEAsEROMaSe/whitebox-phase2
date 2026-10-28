"""Статичний PNG форми по шарах із `arena62/logs/shapeviz.npz`. (С86)"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

BG, FG, MUT = '#0d1117', '#e6edf3', '#8b949e'
RED, CYAN, GOLD, GRN = '#ff2d55', '#00e5ff', '#f0c674', '#3fb950'
D = np.load('/mnt/win_d/llm/White-Box/arena62/logs/shapeviz.npz')
P, G3, G4, CTR, DT = D['P'], D['G3'], D['G4'], D['ctr'], float(D['dt'])
L_ = P.shape[0]
PHI = np.exp(-CTR ** 2 / 2) / np.sqrt(2 * np.pi)
E_G = float(np.sqrt(2 / np.pi))
EM = np.array([(np.abs(CTR) * P[i]).sum() * DT for i in range(L_)])
EM = (EM / E_G - 1.0) * 100.0
ED = (-G4.mean(1) / 24.0) * 100.0

fig, ax = plt.subplots(2, 2, figsize=(16, 8.6), facecolor=BG)
fig.subplots_adjust(hspace=0.34, wspace=0.2, left=0.055, right=0.985,
                    top=0.895, bottom=0.075)
a1, a2, a3, a4 = ax.ravel()
for a in (a1, a2, a3, a4):
    a.set_facecolor('#010409')
    for s in a.spines.values():
        s.set_color('#30363d')
    a.tick_params(colors=MUT, labelsize=9)
    a.grid(alpha=0.13, color=MUT, lw=0.5)

# ① щільність, лог
a1.semilogy(CTR, np.maximum(PHI, 1e-9), color=FG, lw=2.2, ls=':',
            label='гаусова φ(t) — те, що ми ПРИПУСКАЄМО')
for ly, c in ((1, '#3fb95088'), (8, CYAN), (32, RED)):
    a1.semilogy(CTR, np.maximum(P[ly - 1], 1e-9), color=c, lw=1.5,
                label=f'шар {ly}')
a1.set_ylim(1e-7, 0.8); a1.set_xlim(-6, 6)
a1.set_xlabel('t = (h − μ)/σ', color=MUT, fontsize=10)
a1.set_title('① щільність передактивацій проти гаусової (лог)',
             color=FG, fontsize=12, loc='left')
a1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=FG, fontsize=9)

# ② форма = p − φ
a2.axhline(0, color=MUT, lw=1.0)
for ly, c, w in ((1, '#3fb95088', 1.2), (2, '#58a6ff', 1.2),
                 (4, '#a371f7', 1.3), (8, CYAN, 1.5),
                 (16, GOLD, 1.6), (32, RED, 2.2)):
    a2.plot(CTR, P[ly - 1] - PHI, color=c, lw=w, label=f'шар {ly}')
a2.fill_between(CTR, 0, P[31] - PHI, color=RED, alpha=0.13)
a2.set_xlim(-4.6, 4.6)
a2.set_xlabel('t', color=MUT, fontsize=10)
a2.set_title('② ФОРМА: p(t) − φ(t) — ось уся наша похибка',
             color=FG, fontsize=12, loc='left')
a2.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=FG,
          fontsize=8.5, ncol=2)

# ③ негаусовість по шарах
x = np.arange(1, L_ + 1)
a3.plot(x, G4.mean(1), color=GOLD, lw=2.2, marker='o', ms=3.5,
        label='γ₄ ексцес — ВИННИЙ, росте ЛІНІЙНО')
a3.plot(x, np.abs(G3).mean(1), color=CYAN, lw=1.5, ls='--',
        label='⟨|γ₃|⟩ модуль — теж росте')
a3.plot(x, G3.mean(1), color=RED, lw=2.0,
        label='⟨γ₃⟩ ЗІ ЗНАКОМ ≈ 0  ⟹ Еджворт-3 приречений')
a3.axhline(0, color=MUT, lw=1.0)
a3.set_xlabel('шар', color=MUT, fontsize=10)
a3.set_title('③ негаусовість наростає з глибиною', color=FG,
             fontsize=12, loc='left')
a3.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=FG, fontsize=9,
          loc='upper left')

# ④ похибка E|t|
a4.plot(x, EM, color=RED, lw=2.4, marker='o', ms=3.5,
        label='ЗМІРЯНА похибка E|t| (форма)')
a4.plot(x, ED, color=GOLD, lw=1.8, ls='--',
        label='Еджворт: −γ₄/24')
a4.axhline(0, color=MUT, lw=1.0)
a4.set_xlabel('шар', color=MUT, fontsize=10)
a4.set_ylabel('%', color=MUT, fontsize=10)
a4.set_title('④ у скільки % форма зсуває E|h| — і чи ловить це Еджворт',
             color=FG, fontsize=12, loc='left')
a4.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=FG, fontsize=9,
          loc='lower left')
a4.text(0.985, 0.96,
        f'шар 32:  зміряно {EM[31]:+.3f}%   Еджворт {ED[31]:+.3f}%\n'
        f'⛔пломба шар 1: {EM[0]:+.4f}%  (мусить бути 0)',
        transform=a4.transAxes, ha='right', va='top', color=FG, fontsize=10)

fig.suptitle('ФОРМА РОЗПОДІЛУ h НА КОЖНОМУ ШАРІ · 8 мереж × 2²⁰ · '
             'кожен нейрон стандартизовано ОКРЕМО', color=FG, fontsize=13.5)
out = '/mnt/win_d/llm/White-Box/arena62/logs/shape.png'
fig.savefig(out, dpi=110, facecolor=BG)
print('збережено', out)
print(f'⛔ПЛОМБА шар 1: max|p−φ| = {np.abs(P[0]-PHI).max():.3e} · '
      f'γ₄ = {G4[0].mean():+.5f} · похибка E|t| = {EM[0]:+.4f}%')
print(f'\n{"шар":>4s} {"γ₄":>8s} {"⟨γ₃⟩":>9s} {"⟨|γ₃|⟩":>8s} '
      f'{"E|t| зміряно%":>13s} {"Еджворт%":>9s} {"відношення":>11s}')
for i in list(range(6)) + [7, 11, 15, 19, 23, 27, 31]:
    r = EM[i] / ED[i] if abs(ED[i]) > 1e-9 else float('nan')
    print(f'{i+1:4d} {G4[i].mean():+8.4f} {G3[i].mean():+9.5f} '
          f'{np.abs(G3[i]).mean():8.4f} {EM[i]:+13.4f} {ED[i]:+9.4f} '
          f'{r:11.3f}')
