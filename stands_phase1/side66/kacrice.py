"""ДРУГИЙ ОЦІНЮВАЧ ЧЕРЕЗ ОДНОРІДНІСТЬ + СТАЙНА: `E[f] = E[Δf]`. (С73)

════════ ВИВЕДЕННЯ (перевірене вручну) ════════
Мережа БЕЗ ЗСУВІВ ⟹ `f` однорідна степеня 1 ⟹ теорема Ейлера: `z·∇f = f`.
Лема Стайна для гауса: `E[z_i g] = E[∂_i g]`; беремо `g = ∂_i f` і сумуємо:
    `E[z·∇f] = E[Δf]`   ⟹   **`E[f] = E[Δf]`**
🚨 `f` кусково-ЛІНІЙНА ⟹ всередині конуса `Δf ≡ 0`, і вся маса `Δf` сидить
ДЕЛЬТАМИ на границях конусів (7936 гіперповерхонь `h=0`). Порядок шарів у
праву частину НЕ входить — це принципово НЕПОШАРОВИЙ оцінювач, єдиний, що
лишився живим після закриття RMT/реплік ([[MEMORY]] §теорема-огорожа).

════════ ЯК МІРЯЄМО ════════
Лапласіан — стохастично (Хатчінсон) на випадковому напрямку `v ~ N(0,I)`:
    `D(z,v) = [f(z+εv) − 2f(z) + f(z−εv)] / ε²`,  `E_v[D] → Δf` при `ε→0`.
Ціна: **3 проходи** замість 1.

📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (з пам'яті): δ-оцінювачі мають дисперсію ~`1/ε²`
і зміщення ~`ε²` ⟹ ставлю, що `Var(D) ≫ Var(f)` на 2+ порядки, і пара дасть
виграш <1.05×. ЖИВИЙ, якщо комбінація б'є 1.2× ПРИ ЦІНІ 3× (тобто треба >3×,
щоб окупити проходи).

🚨 ПЛОМБИ
  ✅ ГОЛОВНА: `E[D]` мусить збігтися з `E[f]` — це перевірка САМОЇ ТОТОЖНОСТІ;
  ✅ однорідність: `f(2z) = 2f(z)` точно;
  ⛔ КОНТРОЛЬ: `D` з ЧУЖОЮ мережею в збуренні ⟹ тотожність мусить зламатись.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_                      # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NS = 4096
EPS = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def fwd(wc, x):
    a = x
    for w in wc:
        a = torch.relu(a @ w)
    return a.double().sum(1)


def main(nets=3, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    R = {e: {'bias': [], 'vr': [], 'corr': [], 'comb': []} for e in EPS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            g = torch.Generator(device=dev).manual_seed(5150 + n)
            z = torch.randn(NS, W_, device=dev, generator=g)
            fz = fwd(wc, z)
            if n == 0:
                h = float((fwd(wc, 2 * z) / (2 * fz)).mean())
                print(f'  ✅ПЛОМБА однорідності: f(2z)/(2f(z)) = {h:.8f}')
            for e in EPS:
                v = torch.randn(NS, W_, device=dev, generator=g)
                D = (fwd(wc, z + e * v) - 2 * fz + fwd(wc, z - e * v)) / (e * e)
                R[e]['bias'].append(float(D.mean() / fz.mean()))
                R[e]['vr'].append(float(D.var() / fz.var()))
                c = float(torch.corrcoef(torch.stack([fz, D]))[0, 1])
                R[e]['corr'].append(c)
                # оптимальна комбінація двох незміщених оцінювачів
                vf, vd = float(fz.var()), float(D.var())
                cov = c * (vf * vd) ** 0.5
                w = (vd - cov) / max(vf + vd - 2 * cov, 1e-30)
                R[e]['comb'].append(vf / max(w * w * vf + (1 - w) ** 2 * vd
                                             + 2 * w * (1 - w) * cov, 1e-30))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    m = np.median
    print(f'\n{nets} мереж × {NS} семплів; ціна оцінювача `D` — 3 проходи\n')
    print(f'{"ε":>8s} {"E[D]/E[f] (=1?)":>17s} {"Var(D)/Var(f)":>15s} '
          f'{"corr(f,D)":>11s} {"пара, ×":>9s}')
    print('-' * 66)
    for e in EPS:
        print(f'{e:8.3g} {m(R[e]["bias"]):17.4f} {m(R[e]["vr"]):15.3e} '
              f'{m(R[e]["corr"]):11.4f} {m(R[e]["comb"]):9.4f}')
    print('-' * 66)
    print('✅ГОЛОВНА ПЛОМБА — колонка `E[D]/E[f]`: тотожність `E[f]=E[Δf]` вимагає '
          '1.0000.\n   Якщо не 1 при малих ε — виведення або реалізація хибні.')
    print('🎯 ГЕЙТ: пара мусить дати >3×, щоб окупити 3 проходи. '
          '<1.05× ⟹ клас закрито.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
