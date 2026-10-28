"""ФОРМА РОЗПОДІЛУ `h` НА КОЖНОМУ ШАРІ — ЗБІР ДАНИХ І ГРАФІК. (С86)

## ЗВІДКИ — ЗАПИТ ЮЗЕРА
«Візуалізуй форму для мене на кожному слої.» Уся наша похибка сидить рівно в
одному місці: `E[|h|] = σ√(2/π)·(1 − γ₄/24 + …)`, де перший множник точний, а
дужка — невідома ФОРМА. Тут ми її дивимось напряму.

## ЩО ЗБИРАЄТЬСЯ
Кожен нейрон стандартизується ОКРЕМО: `t = (h_i − μ_i)/σ_i` — інакше суміш
256 різних нормувань дасть не форму `h`, а артефакт змішування.
1. гістограма `p(t)` по всіх нейронах і зразках, 400 бінів на [−6, 6];
2. `γ₃` (скошеність) і `γ₄` (ексцес) по нейронах: середнє ЗІ ЗНАКОМ і модуль;
3. внесок у `E|t|`: `∫|t|(p−φ)dt` накопичувально ⟹ ВИДНО, ЯКА ЧАСТИНА
   розподілу робить похибку — центр чи хвости.

⛔ПЛОМБА: шар 1 (`h₁ = zᵀW₀`) СТРОГО гаусів ⟹ `p−φ ≈ 0`, `γ₃=γ₄=0`.
   Усе, що там видно, — шум приладу, і це масштабна лінійка для решти.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/shapeviz.py [cuda:N] [NETS] [LOG2N]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
BATCH = 1 << 14
L_, W_ = 32, 256
NB, LO, HI = 400, -6.0, 6.0
EDGE = torch.linspace(LO, HI, NB + 1)
CTR = ((EDGE[1:] + EDGE[:-1]) * 0.5).numpy()
DT = float(EDGE[1] - EDGE[0])

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж, вибірка 2^{LOG2N} = {NS}, {DEV}', flush=True)


def run(W, dev, ns, batch, seed):
    ed = EDGE.to(dev)
    M1 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    M2 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    done = 0
    while done < ns:                                   # прохід 1: μ, σ
        b = min(batch, ns - done)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            h = a @ W[i]
            hd = h.double()
            M1[i] += hd.sum(0); M2[i] += (hd * hd).sum(0)
            a = torch.relu(h)
        done += b
    MU = [m / ns for m in M1]
    SG = [(m2 / ns - m * m).clamp_min(1e-30).sqrt()
          for m, m2 in zip(MU, M2)]
    HIS = [torch.zeros(NB, dtype=torch.float64, device=dev) for _ in range(L_)]
    C3 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    C4 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    done = 0
    while done < ns:                                   # прохід 2: форма
        b = min(batch, ns - done)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            h = a @ W[i]
            t = (h.double() - MU[i]) / SG[i]           # стандартизація ПОНЕЙРОННО
            t2 = t * t
            C3[i] += (t2 * t).sum(0); C4[i] += (t2 * t2).sum(0)
            idx = torch.bucketize(t.reshape(-1), ed) - 1
            ok = (idx >= 0) & (idx < NB)
            HIS[i] += torch.bincount(idx[ok], minlength=NB).double()
            a = torch.relu(h)
        done += b
    tot = float(ns) * W_
    P = [(hh / tot / DT).cpu().numpy() for hh in HIS]   # щільність
    G3 = [(c / ns).cpu().numpy() for c in C3]
    G4 = [((c / ns) - 3.0).cpu().numpy() for c in C4]
    return P, G3, G4


PA = np.zeros((L_, NB)); G3A = np.zeros((L_, W_ * len(FS)))
G4A = np.zeros((L_, W_ * len(FS)))
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        P, G3, G4 = run(W, DEV, NS, BATCH, 4200 + n)
    for i in range(L_):
        PA[i] += P[i] / len(FS)
        G3A[i, n*W_:(n+1)*W_] = G3[i]
        G4A[i, n*W_:(n+1)*W_] = G4[i]
    print(f'  … {n+1}/{len(FS)} seed {sd0} · ⛔ПЛОМБА шар1: '
          f'γ₄={G4[0].mean():+.4f} (мусить бути ~0)', flush=True)
    del W
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/shapeviz.npz',
         P=PA, G3=G3A, G4=G4A, ctr=CTR, dt=DT)

PHI = np.exp(-CTR**2 / 2) / np.sqrt(2 * np.pi)
print(f'\n⛔ПЛОМБА шар 1 (h строго гаусів): max|p−φ| = '
      f'{np.abs(PA[0]-PHI).max():.3e} · γ₄ = {G4A[0].mean():+.5f}')
print(f'\n{"шар":>4s} {"⟨γ₃⟩ знак":>10s} {"⟨|γ₃|⟩":>9s} {"⟨γ₄⟩":>8s} '
      f'{"max|p−φ|":>10s} {"E|t| емп":>9s} {"похибка %":>10s}')
print('─' * 64)
ROWS = []
for i in range(L_):
    e_emp = float((np.abs(CTR) * PA[i]).sum() * DT)
    e_g = np.sqrt(2 / np.pi)
    row = (i + 1, G3A[i].mean(), np.abs(G3A[i]).mean(), G4A[i].mean(),
           np.abs(PA[i] - PHI).max(), e_emp, (e_emp / e_g - 1) * 100)
    ROWS.append(row)
    if i < 6 or (i + 1) % 4 == 0 or i == L_ - 1:
        print(f'{row[0]:4d} {row[1]:+10.5f} {row[2]:9.4f} {row[3]:+8.4f} '
              f'{row[4]:10.2e} {row[5]:9.5f} {row[6]:+10.3f}')
print('─' * 64)
np.save('/mnt/win_d/llm/White-Box/arena62/logs/shapeviz_rows.npy',
        np.array(ROWS))
print('дані: arena62/logs/shapeviz.npz')
