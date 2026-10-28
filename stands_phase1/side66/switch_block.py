"""ЧИ ЖИВЕ НЕГАУСОВІСТЬ У БЛОЦІ ПЕРЕМИКНИХ НЕЙРОНІВ. (мета 1e-8)

ЗВІДКИ. `tropical_count.py`: на глибині перемикається лише 58 нейронів із 256.
Замерзлий relu ЛІНІЙНИЙ ⟹ негаусовість на кроці народжується тільки там, де
затвор справді перемикається. Якщо так, то спільний тензор κ₃, який С64
закрила ціною («тензор = 114% бюджету»), потрібен на блоці 58³, а не 256³ —
на глибині це **86× дешевше**, і гілка [[finding-joint-kappa3...]] (R²=0.971,
5.87×) стає доступною. Сьогоднішня стеля каже: 5.87× ⟹ r 4.92→0.84 ⟹ ~2.4×.

ЩО МІРЯЄМО. Одношагова нев'язка з ІСТИННИМИ входами (це САМЕ те, що С64
пояснювала спільними κ₃):
    R = Cov_істина[relu(p)] − relu_cov_gauss(m_істина, C_істина)
і питаємо, як ‖R‖² розподілена по блоках, де S = перемикні (0.05<g<0.95):
    S×S,  S×F,  F×F     проти очікуваних часток при рівномірному розподілі.

🚨 ГОЛОВНА ПАСТКА — ШУМ ІСТИНИ. `R` — це різниця двох близьких чисел, і шум
вибірки істини впаде в неї цілком. Тому істина рахується ДВОМА НЕЗАЛЕЖНИМИ
половинами, а всі частки беруться з ПЕРЕХРЕСНОГО добутку ⟨R₁,R₂⟩: шум половин
незалежний, тож у перехресному добутку він у СЕРЕДНЬОМУ нуль.

КОНТРОЛІ
  * друкуємо ‖R₁−R₂‖²/‖R₁‖² — прямий рівень шуму; якщо він ~1, вимір не має сенсу;
  * поруч ⟨R₁,R₂⟩/‖R₁‖² — яка частка `R` взагалі СПРАВЖНЯ;
  * очікувані частки блоків при рівномірному ‖R‖² — базова лінія;
  * ⛔ ВИПАДКОВИЙ набір тієї ж потужності |S| замість перемикних.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import CHUNK, cut_net                        # noqa: E402
from closure_precision import NQ, relu_cov_gauss               # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
HALF_N = 2_097_152
LS = (1, 4, 8, 12, 16, 20, 24, 28, 31)


def half_moments(wc, dev, seed, n, layers):
    """Істинні (m, C) досигмоїдних `p` і Cov[relu(p)] для заданих шарів."""
    sp = {i: torch.zeros(wc[i].shape[1], dtype=torch.float64, device=dev)
          for i in layers}
    sr = {i: torch.zeros(wc[i].shape[1], dtype=torch.float64, device=dev)
          for i in layers}
    Cp = {i: torch.zeros(wc[i].shape[1], wc[i].shape[1],
                         dtype=torch.float64, device=dev) for i in layers}
    Cr = {i: torch.zeros(wc[i].shape[1], wc[i].shape[1],
                         dtype=torch.float64, device=dev) for i in layers}
    gt = {i: torch.zeros(wc[i].shape[1], dtype=torch.float64, device=dev)
          for i in layers}
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < n:
        mch = min(CHUNK, n - done)
        z = torch.tensor(rng.standard_normal((mch // 2, W_)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        x = torch.relu(x @ wc[0])
        for i in range(1, L_):
            p = x @ wc[i]
            r = torch.relu(p)
            if i in layers:
                sp[i] += p.sum(0).double()
                sr[i] += r.sum(0).double()
                Cp[i] += (p.T @ p).double()
                Cr[i] += (r.T @ r).double()
                gt[i] += (p > 0).sum(0).double()
            x = r
        done += mch
    out = {}
    for i in layers:
        mp, mr = sp[i] / done, sr[i] / done
        out[i] = (mp, Cp[i] / done - torch.outer(mp, mp),
                  Cr[i] / done - torch.outer(mr, mr), gt[i] / done)
    return out


def main(nets=4, dev=DEV):
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    ROWS = ('|S| із 256', 'шум ‖R₁−R₂‖²/‖R₁‖²', 'справжня частка ⟨R₁,R₂⟩/‖R₁‖²',
            'S×S частка', 'S×S очікувано', 'S×F частка', 'F×F частка',
            '⛔ випадк.×випадк. частка')
    acc = {k: {i: [] for i in LS} for k in ROWS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            A = half_moments(wc, dev, 70001 + n, HALF_N, LS)
            B = half_moments(wc, dev, 80001 + n, HALF_N, LS)
            for i in LS:
                R = []
                for H in (A, B):
                    mp, Cpp, Crr, _ = H[i]
                    Cg, _ = relu_cov_gauss(mp, Cpp, nodes, wts)
                    R.append(Crr - Cg)
                R1, R2 = R
                g = 0.5 * (A[i][3] + B[i][3])
                S = ((g > 0.05) & (g < 0.95))
                nw = int(S.shape[0])
                k = int(S.sum())
                rnd = torch.zeros_like(S)
                rnd[torch.randperm(nw, device=dev)[:max(k, 1)]] = True
                n1 = float((R1 * R1).sum())
                cross = float((R1 * R2).sum())
                acc['|S| із 256'][i].append(k)
                acc['шум ‖R₁−R₂‖²/‖R₁‖²'][i].append(
                    float(((R1 - R2) ** 2).sum()) / n1)
                acc['справжня частка ⟨R₁,R₂⟩/‖R₁‖²'][i].append(cross / n1)
                mSS = S.unsqueeze(1) & S.unsqueeze(0)
                mSF = S.unsqueeze(1) ^ S.unsqueeze(0)
                mFF = (~S).unsqueeze(1) & (~S).unsqueeze(0)
                mRR = rnd.unsqueeze(1) & rnd.unsqueeze(0)
                for key, msk in (('S×S частка', mSS), ('S×F частка', mSF),
                                 ('F×F частка', mFF),
                                 ('⛔ випадк.×випадк. частка', mRR)):
                    acc[key][i].append(float((R1 * R2)[msk].sum()) / cross)
                acc['S×S очікувано'][i].append((k / nw) ** 2)
        print(f'  мережа {n+1}/{len(fs)}  шар31: |S|={acc["|S| із 256"][31][-1]} '
              f'шум={acc["шум ‖R₁−R₂‖²/‖R₁‖²"][31][-1]:.3f} '
              f'S×S={acc["S×S частка"][31][-1]:.3f}', flush=True)

    print(f'\n{len(fs)} мереж, істина 2×{HALF_N} (дві незалежні половини)')
    print(f'{"шар":>4s}' + ''.join(f'{k:>18s}' for k in ROWS[1:]) + f'{"|S|":>7s}')
    for i in LS:
        print(f'{i:4d}' + ''.join(f'{np.mean(acc[k][i]):18.3f}' for k in ROWS[1:])
              + f'{np.mean(acc["|S| із 256"][i]):7.0f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
