"""ГЛОБАЛЬНИЙ МАСШТАБ ВИХОДУ — НА БАНКУ (вказівка ЮЗЕРА «шукай глобально на банку», С83)

## ЗВІДКИ
1. С80: **40.9% енергії промаху = МАСШТАБ**, ефективних напрямків 2.9/256
   ([[finding-mc-miss-lives-in-three-directions-not-256]]). Один скаляр.
2. Сьогодні: 81.8% MSE — шум вибірки, 7.6% — замикання. Клас МК має стелю
   ~×7 (ми на ×3.4), лідерові треба ×24.4 ⟹ мікровідсотки не ведуть.
3. 🚨 ГЛОБАЛЬНИЙ МНОЖНИК НА ВИХІД НЕ ПЕРЕВІРЯВСЯ. `_ALPHA=−0.00751` діє лише
   на `m_deep` з вагою `w=0.088` ⟹ зсуває масштаб прогнозу на **0.07%**.
   Якщо в масштабі справді 40.9% енергії, там може лежати до **×1.69**.

## ЩО МІРЯЄМО (48 мереж × 16 жеребів, банкова істина, крос-ціль)
`pr' = c·pr`. Крос-ціль дає точний розклад:
    MSE(c) = c²‖pr‖² − c·⟨pr, T1+T2⟩ + ⟨T1,T2⟩
    ⟹ c*(мережа) = ⟨pr, T1+T2⟩ / (2‖pr‖²)
Варіанти:
  A ⛔ c=1 (нинішній сабміт)
  B 🎯 ОДНА корпусна константа `c` (LOO по мережах) — ціна НУЛЬ флопів
  C 🎯 c(мережа) з ознак, доступних зсередини (LOO)
  D ⛔ c*(мережа) оракульний — СТЕЛЯ гілки
  E ⛔ НЕГ: c* зі зсунутої мережі · F ⛔ ЧИСТИЙ ШУМ

🔑 ЧОМУ ЦЕ МОЖЕ БУТИ НЕ НУЛЬ: `ours = last − v`, де `last` — середнє ReLU по
вибірці, а `v` — накопичена поправка. Обидва мають власні систематичні
масштабні зсуви (обрізання пілотом ⟹ втрата хвоста; `gate` з 1024 рядків
⟹ зміщена частка). Жоден із них не калібрувався ГЛОБАЛЬНО на істині.

⛔ ПЛОМБИ: (а) при c=1 MSE мусить збігтися з `wvout.npz` до 1e-12;
   (б) оракульний c мусить давати НЕ ГІРШЕ за c=1 на тих самих даних;
   (в) НЕГ і ШУМ мусять бути ≤1.000.

📣 ПЕРЕДБАЧЕНО: якщо є систематичний зсув корпусу — варіант B дасть >1.00 за
   нуль ціни. Оракул D покаже, скільки взагалі лежить у масштабі: чекаю
   ×1.3…1.7, якщо число 40.9% чинне для НАШОГО прогнозу (а не для сирого МК,
   на якому воно міряне). Якщо D ≈ 1.0 — число 40.9% було властивістю сирого
   МК, і сабміт цю компоненту вже вибрав; гілка закривається одним прогоном.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/outscale.py [cuda:N] [NETS] [NDRAW]
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
# покоефіцієнтний розклад MSE(c) = c²·P − c·Q + R,  усе пожеребно
P = np.zeros((NETS, NDRAW))   # ‖pr‖²
Q = np.zeros((NETS, NDRAW))   # ⟨pr, T1+T2⟩
R = np.zeros((NETS, NDRAW))   # ⟨T1,T2⟩
# глобальні ознаки мережі, доступні зсередини
FT = np.zeros((NETS, 6))
NAMES_F = ['‖ours‖/‖m_deep‖', 'частка живих', 'n_pairs',
           'Σgate/n', '‖pr‖²', 'медіана s_cl вих.']

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        alive = sum(float(w.shape[1]) for w in wc[:-1]) / (256.0 * (len(wc)-1))
        cl = closure91(wc, 8, DEV, kdeep=L_)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        rr = float((T1 * T2).sum())
        acc = np.zeros(3)
        for d in range(NDRAW):
            pr, _, _, (ours, m_deep, vout) = predict91(
                wc, npair, DEV, seed=77_000_000 + 733 * n + d, cl=cl,
                diag=True, raw=True)
            P[n, d] = float((pr * pr).sum())
            Q[n, d] = float((pr * (T1 + T2)).sum())
            R[n, d] = rr
            acc += np.array([
                float(ours.norm() / m_deep.norm()),
                float((pr * pr).sum()),
                float(cl[0][7].median())])
        acc /= NDRAW
        FT[n] = [acc[0], alive, npair, 0.0, acc[1], acc[2]]
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/outscale.npz',
         P=P, Q=Q, R=R, FT=FT)


def mse(c):
    c = np.asarray(c, float)
    if c.ndim == 1:
        c = c[:, None]
    return float(np.sum(c * c * P - c * Q + R))


# ⛔ПЛОМБА (а): c=1 проти wvout.npz
base = mse(np.ones(NETS))
try:
    V = np.load('/mnt/win_d/llm/White-Box/arena62/logs/wvout.npz')
    w = 0.0883
    ref = float(np.sum(V['ZZ'] + w * V['NU'] + w * w * V['DE']))
    print(f'\n⛔ПЛОМБА c=1 vs wvout.npz: {base:.6e} vs {ref:.6e} '
          f'відн. {abs(base-ref)/abs(ref):.2e} '
          f'{"✅" if abs(base-ref)/abs(ref) < 1e-9 else "🚨"}')
except FileNotFoundError:
    pass

c_or = Q.sum(1) / (2.0 * P.sum(1))
print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ, банкова істина.')
print(f'🔑 c_орак: сер **{c_or.mean():.6f}** розкид {c_or.std():.6f} '
      f'[{c_or.min():.6f}…{c_or.max():.6f}]')
print(f'   ⟹ систематичний зсув масштабу корпусу: '
      f'**{(c_or.mean()-1)*100:+.3f}%**')
c_one = float(Q.sum() / (2.0 * P.sum()))
print(f'🔑 ОДНА корпусна константа (усі дані): c = **{c_one:.6f}**')

rng = np.random.default_rng(3)
CAND = [('A ⛔ c=1 (нинішній сабміт)', np.ones(NETS)),
        ('D ⛔ c*(мережа) ОРАКУЛ — СТЕЛЯ', c_or),
        ('E ⛔ НЕГ: c* зсунутої мережі', np.roll(c_or, 1)),
        ('F ⛔ ЧИСТИЙ ШУМ у масштабі c', 1.0 + c_or.std()*rng.standard_normal(NETS))]
print(f'\n{"варіант":36s} {"MSE":>13s} {"× над c=1":>10s}')
print('─' * 62)
for nm, c in CAND:
    print(f'{nm:36s} {mse(c):13.6e} {base/mse(c):10.4f}'
          + ('  🟢' if base / mse(c) > 1.002 else ''))

# B: одна константа, LOO по мережах
cb = np.zeros(NETS)
for n in range(NETS):
    i = [m for m in range(NETS) if m != n]
    cb[n] = float(Q[i].sum() / (2.0 * P[i].sum()))
print(f'{"B 🎯 ОДНА константа, LOO":36s} {mse(cb):13.6e} '
      f'{base/mse(cb):10.4f}' + ('  🟢' if base / mse(cb) > 1.002 else ''))

# C: c(мережа) з ознак, LOO
print(f'\n{"C 🎯 предиктор c(мережа), LOO":36s} {"корел":>9s} {"× над c=1":>10s}')
print('─' * 62)
for j, nm in enumerate(NAMES_F):
    if FT[:, j].std() < 1e-300:
        continue
    cc = np.zeros(NETS)
    for n in range(NETS):
        i = np.array([m for m in range(NETS) if m != n])
        u = FT[i, j] - FT[i, j].mean()
        M = np.array([[P[i].sum(), (u * P[i].sum(1)).sum()],
                      [(u * P[i].sum(1)).sum(), (u * u * P[i].sum(1)).sum()]])
        b = 0.5 * np.array([Q[i].sum(), (u * Q[i].sum(1)).sum()])
        try:
            a, k = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            a, k = c_one, 0.0
        cc[n] = a + k * (FT[n, j] - FT[i, j].mean())
    print(f'   {nm:33s} {np.corrcoef(FT[:, j], c_or)[0, 1]:+9.3f} '
          f'{base/mse(cc):10.4f}' + ('  🟢' if base / mse(cc) > 1.002 else ''))

print(f'\n📊 БАЛ: sub92 1.946e-07 → одна константа '
      f'{1.946e-07/(base/mse(cb)):.4e} · стеля масштабу '
      f'{1.946e-07/(base/mse(c_or)):.4e}')
