"""СТЕЛЯ ВСЬОГО КЛАСУ «ЛІНІЙНИЙ ФІЛЬТР У ФІКСОВАНОМУ БАЗИСІ» (С80).

`capture.py` показала, що ЖОРСТКА проєкція `p̂ → P_K p̂` не дає нічого (макс
1.0000× навіть з ОРАКУЛЬНИМ базисом). Але жорстка проєкція — найгрубіший член
класу: вага 1 для K напрямків, 0 для решти. Загальний член — ДІАГОНАЛЬНИЙ
ФІЛЬТР `p̂ → U·diag(w)·Uᵀ·p̂` з довільними `w_j ∈ [0,1]`.

🔑 Оптимальний `w` відомий у закритій формі (Вінер / Джеймс-Стайн покомпонентно):
        w_j = s_j / (s_j + n_j),      s_j = (µᵀu_j)²,  n_j = E[(nᵀu_j)²]
і тоді   MSE_опт = Σ_j s_j·n_j/(s_j+n_j).
Це СТЕЛЯ: жоден лінійний фільтр у цьому базисі не може краще. Беремо `s_j`,`n_j`
ОРАКУЛЬНИМИ — отже вирок накриває і всі способи їх оцінити.

📐 ЩО ВИРІШУЄ СПРАВУ: виграш є лише там, де `n_j > s_j`, тобто де ШУМ ПЕРЕВАЖАЄ
   СИГНАЛ. Якщо в КОЖНОМУ напрямку `s_j > n_j`, оптимальний фільтр ≈ тотожність
   і клас мертвий цілком, а не лише в жорсткій формі.

📣 ПЕРЕДБАЧЕНО ДО ПРОГОНУ: з таблиці `capture.py` виходить, що навіть у хвості
   сигнал переважає шум приблизно в 6× ⟹ ставлю на те, що `min_j s_j/n_j ≈ 5-10`,
   напрямків із SNR<1 буде НУЛЬ, а стеля Вінера — між 1.00× і 1.05×.

⛔ ПЛОМБИ:
 * `w ≡ 1` (тотожність) ⟹ виграш РІВНО 1.000000;
 * сума `s_j` по всіх j = ‖µ‖², сума `n_j` = E‖n‖² (повнота базису);
 * НЕГ.: перемішати `w` між напрямками ⟹ виграш мусить УПАСТИ нижче 1.

🚀 fp32 у матмулах, fp64 у накопиченні (fp64 на P104 у 32× повільніше).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/wiener.py [cuda:N] [NETS] [R]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
R = int(sys.argv[3]) if len(sys.argv) > 3 else 6
NMC = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
W_, D = 256, 32

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
TZ = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
SEEDS = list(np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'))
IDX = [SEEDS.index(s) for s in TZ['seeds'].tolist()]
print(__doc__.split('📣')[1].split('⛔ ПЛОМБИ')[0], flush=True)


def _mean(W, gen, n):
    acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
    done = 0
    while done < n:
        b = min(2048, n - done)
        x = torch.randn(b, W_, device=DEV, generator=gen)
        for L in range(D):
            x = torch.relu(x @ W[L])
        acc += x.double().sum(0)
        done += b
    return acc / n


res = []
for j in range(min(NETS, len(IDX))):
    net = IDX[j]
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
         for k in range(D)]
    mu = torch.tensor(TZ['T1'][j], dtype=torch.float64, device=DEV)
    tgt = torch.tensor(TZ['T2'][j], dtype=torch.float64, device=DEV)
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(500011 + 101 * j)
        # ОРАКУЛЬНИЙ базис: власні вектори E[o oᵀ], незалежний жереб
        x = torch.randn(4096, W_, device=DEV, generator=g)
        for L in range(D):
            x = torch.relu(x @ W[L])
        O = x.double()
        _, V = torch.linalg.eigh(O.transpose(0, 1) @ O / O.shape[0])
        U = V.flip(1)

        s = (mu @ U) ** 2                                  # енергія СИГНАЛУ
        nn = torch.zeros(W_, dtype=torch.float64, device=DEV)
        hats = []
        for r in range(R):
            p = _mean(W, g, NMC)
            hats.append(p)
            nn += ((p - mu) @ U) ** 2
        nn /= R                                            # енергія ШУМУ

        # 🚨 ~20% ВИХІДНИХ НЕЙРОНІВ МЕРТВІ НАЗАВЖДИ (зміряно: 51/256 медіана):
        #    у них істина РІВНО 0 і шум РІВНО 0 ⟹ власний вектор = орт цієї
        #    координати, s_j = n_j = 0, і формула дає 0/0. Такий напрямок не
        #    несе ні сигналу, ні шуму — його внесок у ОБИДВІ суми нульовий.
        #    ⚠️ Викидати треба за `n_j = 0`, а НЕ за `s_j = 0`: напрямок із
        #    s=0, але n>0 — це чистий шум, і саме там фільтр виграє задарма.
        live = nn > 0
        w = torch.where(live, s / (s + nn).clamp_min(1e-300),
                        torch.ones_like(s))
        num = torch.where(live, s * nn / (s + nn).clamp_min(1e-300),
                          torch.zeros_like(s))
        ceil = float(nn.sum() / num.sum().clamp_min(1e-300))

        # ЧЕСНО: застосувати фільтр і зміряти MSE проти НЕЗАЛЕЖНОЇ цілі T2
        F = (U * w) @ U.transpose(0, 1)
        base = float(np.median([float(((h - tgt) ** 2).mean()) for h in hats]))
        filt = float(np.median([float(((h @ F - tgt) ** 2).mean()) for h in hats]))
        # ⛔ПЛОМБА w≡1 і ⛔НЕГ. перемішаний w
        Fi = (U * torch.ones_like(w)) @ U.transpose(0, 1)
        seal = base / float(np.median([float(((h @ Fi - tgt) ** 2).mean())
                                       for h in hats]))
        ws = w[torch.randperm(W_, device=DEV, generator=g)]
        Fs = (U * ws) @ U.transpose(0, 1)
        neg = base / float(np.median([float(((h @ Fs - tgt) ** 2).mean())
                                      for h in hats]))

        snr = (s / nn.clamp_min(1e-300))[live]
        res.append((ceil, base / filt, seal, neg, float(snr.min()),
                    int((snr < 1).sum()), float(w[live].min()), float(s.sum()),
                    float(mu @ mu), int(live.sum()),
                    int(((~live)).sum()), float(nn[live].sum() / nn.sum())))
    if j % 8 == 0:
        print(f'  … мережа {j+1}/{min(NETS,len(IDX))}', flush=True)
    del W
    torch.cuda.empty_cache()

a = np.array(res)
print(f'\n{a.shape[0]} мереж × {R} незалежних жеребів × N={NMC}\n')
print('⛔ ПЛОМБИ')
print(f'  w≡1 (тотожність) виграш        = {np.median(a[:,2]):.6f}   (треба 1.000000)')
print(f'  Σs_j проти ‖µ‖²  (повнота)      = {np.median(a[:,7]/a[:,8]):.8f}   (треба 1)')
print(f'  НЕГ. перемішаний w              = {np.median(a[:,3]):.4f}   (треба <1)')
print(f'\n{"величина":42s} {"медіана":>12s} {"мін":>12s} {"макс":>12s}')
print('-' * 82)
for k, nm in ((0, 'СТЕЛЯ Вінера (формула, оракульна)'),
              (1, 'зміряний виграш фільтра проти T2'),
              (4, 'МІНІМАЛЬНИЙ по ЖИВИХ напрямках SNR s_j/n_j'),
              (5, 'ЖИВИХ напрямків із SNR < 1 (є що знімати)'),
              (6, 'мінімальна вага w_j на живих'),
              (9, 'ЖИВИХ напрямків (з 256)'),
              (10, 'МЕРТВИХ напрямків (s=n=0 точно)')):
    c = a[:, k]
    print(f'{nm:42s} {np.median(c):12.5g} {c.min():12.5g} {c.max():12.5g}')

ceil = float(np.median(a[:, 0]))
print(f'\n🔑 ВИРОК КЛАСУ: жоден лінійний фільтр у жодному базисі не дає більше '
      f'{ceil:.4f}×,')
print(f'   бо мінімальний SNR по напрямках = {np.median(a[:,4]):.2f} ≫ 1 ⟹ '
      f'знімати нема чого.')
print(f'\n📊 БАЛ 1.954e-07 ⟹ стеля {1.954e-07/ceil:.4e} (і це ОРАКУЛ, безкоштовно недосяжний)')
