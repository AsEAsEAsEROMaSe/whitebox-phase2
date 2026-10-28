"""ФІЛЬТР ВІНЕРА: ЧЕСНО, БЕЗ ТАВТОЛОГІЇ, І В ДОСЯЖНІЙ ФОРМІ (С80).

`wiener.py` дала оракульну стелю 1.2449× і зміряний виграш 1.3472× — уперше за
сесію понад одиницю. 🚨 АЛЕ там був ДЕФЕКТ: енергія шуму `n_j` оцінювалась із
ТИХ САМИХ 6 жеребів, які потім фільтрувались і мірялись. Фільтр підганявся під
реалізацію шуму, яку сам же й знімав ⟹ виграш ЗАВИЩЕНИЙ (і це видно з того, що
зміряне 1.347 перевищило власну формульну стелю 1.245 — так не буває).

Цей стенд розділяє три речі, які там злиплись:
 1. ⛔ПЛОМБА тотожності `w≡1`;
 2. 🎯 ОРАКУЛЬНА стеля: `s_j` з істини `T1`, `n_j` з M−1 ЧУЖИХ жеребів,
    застосовано до ВІДКЛАДЕНОГО жеребу `r`. Оракул — бо `s_j` вимагає істини;
 3. 🟢 ДОСЯЖНА форма (емпіричний Баєс): усе з ОДНОГО жеребу, як у продакшні.
    `n̂_j` — з розкиду 8 фолдів усередині жеребу (це чесна оцінка дисперсії),
    `ŝ_j = max(0, (p̂ᵀu_j)² − n̂_j)` — незміщена оцінка енергії сигналу.
    Саме ця форма або дає важіль, або ні.

📐 ЧОМУ ВЗАГАЛІ Є ЩО ЗНІМАТИ (пояснення числа 1.24× з `capture.py`):
   сигнал ГРАНИЧНО зосереджений — 99.998% енергії `µ` в ОДНОМУ напрямку, тоді як
   шум розмазаний (у топ-1 лише 23%). Тому в глибокому хвості спектра шум
   ПЕРЕВАЖАЄ сигнал, і там фільтр законно ставить `w≈0`.
   Зміряно: ЖИВИХ напрямків із `s_j < n_j` — **128 із 255** (медіана).

📣 ПЕРЕДБАЧЕНО ДО ПРОГОНУ: оракульна стеля на відкладеному жеребі впаде з 1.35×
   до ~1.20× (зникне підгонка під реалізацію). Емпіричний Баєс дасть ІСТОТНО
   менше за оракул, бо `ŝ_j` у хвості — це різниця двох майже рівних величин,
   тобто чистий шум; ставлю на 1.00-1.05×. Якщо ЕБ дасть >1.1× — це важіль.

⛔ ПЛОМБИ: `w≡1` → 1.000000 · перемішаний `w` → <1 · `Σŝ_j ≈ ‖µ‖²` ·
   ⛔НЕГ. базис із ЧУЖОЇ мережі → виграш мусить зникнути.

🚀 fp32 у матмулах, fp64 у накопиченні (fp64 на P104 у 32× повільніше).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/wiener2.py [cuda:N] [NETS] [M]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
M = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NMC = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
NF = 8                                    # фолдів усередині жеребу
W_, D = 256, 32

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
TZ = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
SEEDS = list(np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'))
IDX = [SEEDS.index(s) for s in TZ['seeds'].tolist()]
print(__doc__.split('📣')[1].split('⛔ ПЛОМБИ')[0], flush=True)


def _fwd(W, x):
    for L in range(D):
        x = torch.relu(x @ W[L])
    return x


def _folds(W, gen, n, nf):
    """Повертає nf середніх по рівних фолдах (їх середнє = повна оцінка)."""
    per = n // nf
    out = torch.zeros(nf, W_, dtype=torch.float64, device=DEV)
    for f in range(nf):
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        done = 0
        while done < per:
            b = min(2048, per - done)
            acc += _fwd(W, torch.randn(b, W_, device=DEV, generator=gen)
                        ).double().sum(0)
            done += b
        out[f] = acc / per
    return out


def _apply(p, U, w):
    return ((p @ U) * w) @ U.transpose(0, 1)


res = []
for j in range(min(NETS, len(IDX))):
    net = IDX[j]
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
         for k in range(D)]
    mu = torch.tensor(TZ['T1'][j], dtype=torch.float64, device=DEV)
    tgt = torch.tensor(TZ['T2'][j], dtype=torch.float64, device=DEV)
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(500011 + 101 * j)
        O = _fwd(W, torch.randn(4096, W_, device=DEV, generator=g)).double()
        _, V = torch.linalg.eigh(O.transpose(0, 1) @ O / O.shape[0])
        U = V.flip(1)
        # ⛔НЕГ.БАЗИС: та сама конструкція, але з ЧУЖОЇ мережі
        Wf = [torch.tensor(np.asarray(WZ[IDX[(j + 7) % len(IDX)]][k], np.float32),
                           device=DEV) for k in range(D)]
        Of = _fwd(Wf, torch.randn(4096, W_, device=DEV, generator=g)).double()
        _, Vf = torch.linalg.eigh(Of.transpose(0, 1) @ Of / Of.shape[0])
        Un = Vf.flip(1)
        del Wf, Of

        FL = torch.stack([_folds(W, g, NMC, NF) for _ in range(M)])   # M×NF×256
        P = FL.mean(1)                                                # M×256
        E = (P - mu) @ U                                              # M×256
        s = (mu @ U) ** 2

        b_, o_, e_, n_, sl_, ng_, ss_ = [], [], [], [], [], [], []
        on_, os_ = [], []
        for r in range(M):
            # ── ОРАКУЛ: n_j з ЧУЖИХ жеребів (r виключено) ─────────────
            oth = torch.cat([E[:r], E[r + 1:]], 0)
            nor = (oth ** 2).mean(0)
            wo = s / (s + nor).clamp_min(1e-300)
            # ── ЕМПІРИЧНИЙ БАЄС: усе з ОДНОГО жеребу r ────────────────
            fm = FL[r] @ U                                   # NF×256
            neb = fm.var(0, unbiased=True) / NF              # дисп. середнього
            pu = P[r] @ U
            seb = (pu ** 2 - neb).clamp_min(0.0)
            we = seb / (seb + neb).clamp_min(1e-300)
            # ── РОЗКЛАД РОЗРИВУ оракул→ЕБ: що саме винне, `ŝ` чи `n̂`? ──
            #    змішуємо по одному оракульному члену за раз
            s_on = (pu ** 2 - nor).clamp_min(0.0)     # оракульний n, свій s
            w_on = s_on / (s_on + nor).clamp_min(1e-300)
            w_os = s / (s + neb).clamp_min(1e-300)    # оракульний s, свій n
            # ── ЕБ у ЧУЖОМУ базисі (НЕГ.контроль) ─────────────────────
            fmn = FL[r] @ Un
            nebn = fmn.var(0, unbiased=True) / NF
            pun = P[r] @ Un
            sebn = (pun ** 2 - nebn).clamp_min(0.0)
            wen = sebn / (sebn + nebn).clamp_min(1e-300)

            base = float(((P[r] - tgt) ** 2).mean())
            b_.append(base)
            o_.append(base / float(((_apply(P[r], U, wo) - tgt) ** 2).mean()))
            e_.append(base / float(((_apply(P[r], U, we) - tgt) ** 2).mean()))
            n_.append(base / float(((_apply(P[r], Un, wen) - tgt) ** 2).mean()))
            sl_.append(base / float(((_apply(P[r], U, torch.ones_like(s))
                                      - tgt) ** 2).mean()))
            perm = torch.randperm(W_, device=DEV, generator=g)
            ng_.append(base / float(((_apply(P[r], U, wo[perm])
                                      - tgt) ** 2).mean()))
            ss_.append(float(seb.sum() / (mu @ mu)))
            on_.append(base / float(((_apply(P[r], U, w_on) - tgt) ** 2).mean()))
            os_.append(base / float(((_apply(P[r], U, w_os) - tgt) ** 2).mean()))
        res.append((np.median(o_), np.median(e_), np.median(sl_),
                    np.median(ng_), np.median(n_), np.median(ss_),
                    np.median(b_), float((s < (E ** 2).mean(0)).sum()),
                    np.median(on_), np.median(os_)))
    if j % 8 == 0:
        print(f'  … мережа {j+1}/{min(NETS,len(IDX))}', flush=True)
    del W
    torch.cuda.empty_cache()

a = np.array(res)
print(f'\n{a.shape[0]} мереж × M={M} незалежних жеребів × N={NMC}, фолдів {NF}')
print(f'ціль T2 (крос), базова MSE = {np.median(a[:,6]):.4e}\n')
print('⛔ ПЛОМБИ')
print(f'  w≡1 (тотожність)                = {np.median(a[:,2]):.6f}   (треба 1.000000)')
print(f'  НЕГ. перемішаний w              = {np.median(a[:,3]):.4f}   (треба <1)')
print(f'  Σŝ_j / ‖µ‖²                     = {np.median(a[:,5]):.6f}   (треба ≈1)')
print(f'  ⛔НЕГ. ЕБ у ЧУЖОМУ базисі        = {np.median(a[:,4]):.4f}   (треба ≈1 або <1)')
print(f'\n{"фільтр":44s} {"медіана":>11s} {"мін":>11s} {"макс":>11s} {"частка >1":>10s}')
print('-' * 92)
for k, nm in ((0, '🎯 ОРАКУЛ (s з істини, n з чужих жеребів)'),
              (9, '   ↳ оракульний s, свій n̂ (8 фолдів)'),
              (8, '   ↳ свій ŝ, оракульний n'),
              (1, '🟢 ЕМПІР.БАЄС — усе з ОДНОГО жеребу')):
    c = a[:, k]
    print(f'{nm:44s} {np.median(c):11.4f} {c.min():11.4f} {c.max():11.4f}'
          f' {100*np.mean(c>1):9.0f}%')
print(f'\nнапрямків із s_j < n_j (є що знімати): медіана {np.median(a[:,7]):.0f} із 256')
eb = float(np.median(a[:, 1])); orc = float(np.median(a[:, 0]))
print(f'\n🔑 ВИРОК: оракул {orc:.4f}× · ДОСЯЖНЕ {eb:.4f}×')
print(f'📊 БАЛ 1.954e-07 ⟹ досяжне {1.954e-07/eb:.4e} · оракул {1.954e-07/orc:.4e}')
