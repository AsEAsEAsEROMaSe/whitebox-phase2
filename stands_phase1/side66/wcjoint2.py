"""БЛЕНД, МАСШТАБ І ПІВПОДІЛ РАЗОМ — ЧИ ЦЕ ОДНА РУЧКА (С83)

## ЗВІДКИ
`outscale.py` (48×16, банк): оракульний масштаб виходу дає **×1.0911** над
базою `sub91`, а предиктор `‖ours‖/‖m_deep‖` бере **×1.0097** (корел +0.355).
🚨 Але це та сама вісь, на якій уже грає бленд `w`, і база була `w=0.0651`,
тоді як подавати збираємось `w=0.0883`. Отже невідомо, чи масштаб дає щось
ПОВЕРХ правильного бленду, чи це він і є, переодягнений.

## 🔑 ЯК РОЗДІЛИТИ РАЗ І НАЗАВЖДИ
Прогноз двопараметричний:  `pr(w,c) = c·(ours + w·Δ)`,  `Δ = (1+α)m_deep − ours`.
Усе, що треба для MSE при БУДЬ-ЯКИХ `(w,c)`, — шість скалярів на жереб:
    OO=‖ours‖²  OD=⟨ours,Δ⟩  DD=‖Δ‖²
    OT=⟨ours,T1+T2⟩  DT=⟨Δ,T1+T2⟩  TT=⟨T1,T2⟩
Тоді точно (крос-ціль):
    MSE(w,c) = c²(OO + 2w·OD + w²·DD) − c(OT + w·DT) + TT
⟹ вся площина (w,c) обраховується офлайн задарма, і видно, чи `c` дає щось
поверх оптимального `w`, чи вони колінеарні.

## ВАРІАНТИ
  A ⛔ (w=0.0651, c=1)   — sub91
  B ⛔ (w=0.0883, c=1)   — sub92, те, що подаємо
  C 🎯 (w*, c=1)          — найкраща КОНСТАНТНА пара, LOO
  D 🎯 (w*, c*) спільно   — обидві корпусні константи, LOO
  E 🎯 c(мережа) з ознаки, w=w*  — персоналізований масштаб, LOO
  F ⛔ (w*, c*(мережа) ОРАКУЛ)  — стеля масштабу ПОВЕРХ оптимального бленду
  G ⛔ НЕГ · H ⛔ ШУМ

📣 ПЕРЕДБАЧЕНО: якщо `c` — переодягнений бленд, то F над C буде ≈1.00, бо
   оптимальний `w` уже вибрав цю вісь. Якщо ж F над C лишиться ≳1.05 — це
   ОКРЕМА ручка, і тоді предиктор масштабу вартий роботи.
   🔑 Ключове число — не сам виграш, а F/C: скільки лишається ПОВЕРХ бленду.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/wcjoint.py [cuda:N] [NETS] [NDRAW]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, closure91, predict91,      # noqa: E402
                        ALPHA)
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 16
L_ = 32
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
NETS = len(FS)
K = ['OO', 'OD', 'DD', 'OT', 'DT', 'TT', 'SP', 'MM']
S = {k: np.zeros((NETS, NDRAW)) for k in K}
FEAT = np.zeros((NETS, 5))
FN = ['‖ours‖/‖m_deep‖', 'vout/‖Δ‖²', 'n_pairs', 'частка живих', '⟨ours,Δ⟩/‖ours‖²']

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        alive = sum(float(w.shape[1]) for w in wc[:-1]) / (256.0*(len(wc)-1))
        cl = closure91(wc, 8, DEV, kdeep=L_)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        TS = T1 + T2
        tt = float((T1 * T2).sum())
        acc = np.zeros(3)
        for d in range(NDRAW):
            _, _, _, (ours, m_deep, vout) = predict91(
                wc, npair, DEV, seed=77_000_000 + 733*n + d, cl=cl,
                diag=True, raw=True)
            dl = (1.0 + ALPHA) * m_deep - ours
            S['OO'][n, d] = float((ours*ours).sum())
            S['OD'][n, d] = float((ours*dl).sum())
            S['DD'][n, d] = float((dl*dl).sum())
            S['OT'][n, d] = float((ours*TS).sum())
            S['DT'][n, d] = float((dl*TS).sum())
            S['TT'][n, d] = tt
            # 🔑 ПІВПОДІЛ: два прогони по npair/2 = ті самі флопи, що один по
            #    npair. Дає РЕАЛІЗОВАНУ оцінку шуму ⟹ шанс очистити ознаку від
            #    пожеребної компоненти, яка має корел −1.000 з оптимумом.
            _, _, _, (oA, _, _) = predict91(wc, npair//2, DEV,
                                            seed=(77_000_000+733*n+d)*2+1,
                                            cl=cl, diag=True, raw=True)
            _, _, _, (oB, _, _) = predict91(wc, npair//2, DEV,
                                            seed=(77_000_000+733*n+d)*2+2,
                                            cl=cl, diag=True, raw=True)
            S['SP'][n, d] = float(((oA-oB)**2).sum())/4.0
            S['MM'][n, d] = float((m_deep*m_deep).sum())
            acc += np.array([float(ours.norm()/m_deep.norm()),
                             vout/max(float((dl*dl).sum()), 1e-300),
                             float((ours*dl).sum())/max(float((ours*ours).sum()), 1e-300)])
        acc /= NDRAW
        FEAT[n] = [acc[0], acc[1], npair, alive, acc[2]]
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/wcjoint2.npz', FEAT=FEAT,
         **{k: S[k] for k in K})
OO, OD, DD, OT, DT, TT = (S[k] for k in K)


def mse(w, c):
    w = np.asarray(w, float); c = np.asarray(c, float)
    if w.ndim == 1: w = w[:, None]
    if c.ndim == 1: c = c[:, None]
    A = OO + 2*w*OD + w*w*DD
    return float(np.sum(c*c*A - c*(OT + w*DT) + TT))


def best_w(idx, c=1.0):
    "мінімум по w при фіксованому c (скаляр або вектор по мережах)"
    cc = c if np.ndim(c) else np.full(len(idx), c)
    num = (cc[:, None]*DT[idx]).sum() - 2*(cc[:, None]**2*OD[idx]).sum()
    den = 2*(cc[:, None]**2*DD[idx]).sum()
    return float(num/den)


def best_c(idx, w):
    ww = w if np.ndim(w) else np.full(len(idx), w)
    A = OO[idx] + 2*ww[:, None]*OD[idx] + ww[:, None]**2*DD[idx]
    return float((OT[idx] + ww[:, None]*DT[idx]).sum() / (2*A.sum()))


ALL = np.arange(NETS)
b91 = mse(0.0651, 1.0); b92 = mse(0.0883, 1.0)
print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ, банкова істина.')
print(f'⛔ПЛОМБА sub91: {b91:.6e}  (wvout дав 4.992362e-01)')
w_st = best_w(ALL); c_st = best_c(ALL, w_st)
for _ in range(6):
    w_st = best_w(ALL, c_st); c_st = best_c(ALL, w_st)
print(f'🔑 спільний оптимум КОНСТАНТ: w = **{w_st:.4f}**, c = **{c_st:.6f}**')

# оракульний c(мережа) при w=w_st
A_st = OO + 2*w_st*OD + w_st*w_st*DD
c_or = (OT + w_st*DT).sum(1) / (2*A_st.sum(1))
print(f'🔑 c_орак(мережа): сер {c_or.mean():.6f} розкид {c_or.std():.6f}')

rows = [('A ⛔ sub91 (w=0.0651, c=1)', b91),
        ('B ⛔ sub92 (w=0.0883, c=1)', b92)]
# C, D — LOO по мережах
mC = mD = 0.0
for n in range(NETS):
    i = np.array([m for m in range(NETS) if m != n])
    wl = best_w(i); mC += mse(np.full(1, wl), 1.0) * 0 + \
        float(np.sum((OO[n]+2*wl*OD[n]+wl*wl*DD[n]) - (OT[n]+wl*DT[n]) + TT[n]))
    wl2 = best_w(i); cl2 = best_c(i, wl2)
    for _ in range(6):
        wl2 = best_w(i, cl2); cl2 = best_c(i, wl2)
    A = OO[n]+2*wl2*OD[n]+wl2*wl2*DD[n]
    mD += float(np.sum(cl2*cl2*A - cl2*(OT[n]+wl2*DT[n]) + TT[n]))
rows += [('C 🎯 w* константа, c=1 (LOO)', mC),
         ('D 🎯 w* і c* константи (LOO)', mD)]
# F — оракульний c(мережа) при константному w*
rows += [('F ⛔ w*, c*(мережа) ОРАКУЛ', mse(w_st, c_or)),
         ('G ⛔ НЕГ: c* зсунутої мережі', mse(w_st, np.roll(c_or, 1)))]
rng = np.random.default_rng(3)
rows += [('H ⛔ ШУМ у c', mse(w_st, 1.0 + c_or.std()*rng.standard_normal(NETS)))]

print(f'\n{"варіант":34s} {"MSE":>13s} {"× над sub92":>12s}')
print('─' * 62)
for nm, v in rows:
    print(f'{nm:34s} {v:13.6e} {b92/v:12.4f}' + ('  🟢' if b92/v > 1.002 else ''))
print(f'\n🎯 ГОЛОВНЕ ЧИСЛО — F над C (скільки масштаб дає ПОВЕРХ бленду): '
      f'**×{mC/mse(w_st, c_or):.4f}**')

print(f'\n{"E 🎯 предиктор c(мережа), LOO":34s} {"корел":>9s} {"× над sub92":>12s}')
print('─' * 62)
for j, nm in enumerate(FN):
    if FEAT[:, j].std() < 1e-300:
        continue
    cc = np.zeros(NETS)
    for n in range(NETS):
        i = np.array([m for m in range(NETS) if m != n])
        u = FEAT[i, j] - FEAT[i, j].mean()
        Ai = (OO[i] + 2*w_st*OD[i] + w_st*w_st*DD[i]).sum(1)
        Bi = (OT[i] + w_st*DT[i]).sum(1)
        M = np.array([[Ai.sum(), (u*Ai).sum()], [(u*Ai).sum(), (u*u*Ai).sum()]])
        bb = 0.5*np.array([Bi.sum(), (u*Bi).sum()])
        try:
            a, k = np.linalg.solve(M, bb)
        except np.linalg.LinAlgError:
            a, k = c_st, 0.0
        cc[n] = a + k*(FEAT[n, j] - FEAT[i, j].mean())
    print(f'   {nm:31s} {np.corrcoef(FEAT[:, j], c_or)[0, 1]:+9.3f} '
          f'{b92/mse(w_st, cc):12.4f}' + ('  🟢' if b92/mse(w_st, cc) > 1.002 else ''))
print(f'\n📊 БАЛ: sub92 1.946e-07 · стеля масштабу поверх бленду '
      f'{1.946e-07/(b92/mse(w_st, c_or)):.4e}')
