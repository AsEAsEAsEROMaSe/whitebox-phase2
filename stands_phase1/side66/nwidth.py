"""ПОПЕРЕЧНИКИ КОЛМОГОРОВА НЕВ'ЯЗКИ НЕГАУСОВОСТІ. (ідея юзера)

ПЕРЕФОРМУЛЮВАННЯ ПИТАННЯ. Усю дорогу ми питали «який базис вибрати» і
перебирали відповіді: Такер r=32 (10%), зрізи (0), низькорангова тріангуляція,
навчена модель над групами, Джеймс-Стайн… Правильне ПОПЕРЕДНЄ питання —
який у цього класу спектр поперечників. Якщо він спадає повільно, то всі ті
невдачі були не десятьма фактами, а ОДНИМ, і жоден базис не врятує — це
теорема, а не невдача методу.

ЩО МІРЯЄМО. Поперечник Колмогорова в L² — це рівно спектр сингулярних чисел,
а найкращий підпростір розмірності n — головні компоненти. Об'єкт:
    R = Cov_істина[relu(p)] − relu_cov_gauss(m_істина, C_істина)
тобто РІВНО та частина, якої гаусовому замиканню бракує (`switch_block.py`
показав, що вона реальна: шум 0.000, справжня частка 1.000).

ДВА ПИТАННЯ
  A. поперечники ОДНОГО об'єкта: чи наближається `R` матрицею рангу n;
  B. поперечники КЛАСУ: чи є СПІЛЬНИЙ підпростір, у якому лежать `R` різних
     шарів і мереж — це те, що мовчки припускала кожна навчена/перенесена
     поправка. Координати беруться у власному базисі `C_p` (упорядкованому за
     власним числом), інакше нейрон j різних мереж — різні нейрони.

🚨 ШУМ. `R` — різниця близьких чисел. Істина рахується ДВОМА незалежними
половинами; підпростір будується на `R₁`, а частка захопленого міряється
ПЕРЕХРЕСНО через ⟨·, R₂⟩ ⟹ шум у середньому нуль і НЕ може роздути захоплення.

КОНТРОЛІ
  * ⛔ випадковий підпростір рангу n — базова лінія;
  * ✅ n = повний ранг мусить дати 1.000;
  * друкуємо показник спаду α у `частка(n) ≈ 1 − c·n^(−α)`: α<1 = повільно.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                               # noqa: E402
from closure_precision import NQ, relu_cov_gauss               # noqa: E402
from switch_block import half_moments, HALF_N, LS              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NS = (1, 2, 4, 8, 16, 32, 64, 128)
KB = 48


def capture(R1, R2, n):
    """Частка ⟨R₁,R₂⟩, яку бере ранг-n наближення `R₁`. Перехресно ⟹ без шуму."""
    U, S, Vh = torch.linalg.svd(R1)
    Rn = (U[:, :n] * S[:n]) @ Vh[:n]
    return float((Rn * R2).sum()) / float((R1 * R2).sum())


def main(nets=6, dev=DEV):
    torch.manual_seed(4242)
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    capA = {i: {n: [] for n in NS} for i in LS}
    capR = {i: {n: [] for n in NS} for i in LS}
    fam, deliv, delivR = ({i: [] for i in LS} for _ in range(3))
    for k, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            A = half_moments(wc, dev, 70001 + k, HALF_N, LS)
            B = half_moments(wc, dev, 80001 + k, HALF_N, LS)
            for i in LS:
                R = []
                for H in (A, B):
                    mp, Cpp, Crr, _ = H[i]
                    Cg, _ = relu_cov_gauss(mp, Cpp, nodes, wts)
                    R.append(Crr - Cg)
                R1, R2 = R
                nw = R1.shape[0]
                for n in NS:
                    if n > nw:
                        continue
                    capA[i][n].append(capture(R1, R2, n))
                    Q, _ = torch.linalg.qr(torch.randn(nw, n, dtype=torch.float64,
                                                       device=dev))
                    Rr = Q @ (Q.T @ R1)
                    capR[i][n].append(float((Rr * R2).sum())
                                      / float((R1 * R2).sum()))
                # ── B: координати у власному базисі C_p (топ-KB), спільні для
                # мереж різної ширини після обрізання
                _, U = torch.linalg.eigh(A[i][1])
                U = U.flip(1)[:, :KB]
                nrm = float((R1 * R2).sum()) ** 0.5
                fam[i].append(((U.T @ R1 @ U) / max(nrm, 1e-300)).reshape(-1))

                # ── C: чи рахуються верхні напрямки `R` З ВАГ (без істини)
                mp, Cpp = A[i][0], A[i][1]
                sp = Cpp.diagonal().clamp_min(1e-30).sqrt()
                t = mp / sp
                Ph = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
                ph = torch.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
                cand = torch.stack([mp * Ph + sp * ph, sp, ph, Ph,
                                    torch.ones_like(sp), sp * ph], 1)
                Qc, _ = torch.linalg.qr(cand)
                Ur = torch.linalg.svd(R1)[0][:, :2]
                deliv[i].append(float((Qc.T @ Ur).pow(2).sum()) / 2.0)
                Qr, _ = torch.linalg.qr(torch.randn(nw, cand.shape[1],
                                                    dtype=torch.float64, device=dev))
                delivR[i].append(float((Qr.T @ Ur).pow(2).sum()) / 2.0)
        print(f'  мережа {k+1}/{len(fs)}  шар31 ранг-8 бере '
              f'{capA[31][8][-1]:.3f} (⛔ випадк. {capR[31][8][-1]:.3f})',
              flush=True)

    print(f'\n{len(fs)} мереж, істина 2×{HALF_N} двома незалежними половинами')
    print('\nA. ПОПЕРЕЧНИКИ ОДНОГО ОБ\'ЄКТА — частка нев\'язки, яку бере ранг n')
    print(f'{"шар":>4s}' + ''.join(f'{"n="+str(n):>9s}' for n in NS)
          + f'{"⛔вип.n=32":>11s}{"α":>7s}')
    for i in LS:
        row = [np.mean(capA[i][n]) for n in NS]
        lg = np.log(np.maximum(1e-9, 1.0 - np.array(row[:-1])))
        al = -np.polyfit(np.log(NS[:-1]), lg, 1)[0]
        print(f'{i:4d}' + ''.join(f'{v:9.3f}' for v in row)
              + f'{np.mean(capR[i][32]):11.3f}{al:7.2f}')

    print(f'\nB. ПОПЕРЕЧНИКИ КЛАСУ — спільний підпростір для РІЗНИХ МЕРЕЖ '
          f'(топ-{KB} власних координат; {len(fs)} мереж ⟹ максимум {len(fs)} мод)')
    print(f'{"шар":>4s}   накопичена частка по модах класу')
    for i in LS:
        M = torch.stack(fam[i])
        s = torch.linalg.svdvals(M.double())
        cum = torch.cumsum((s * s) / (s * s).sum(), 0)
        print(f'{i:4d}   ' + '  '.join(f'{float(cum[j]):.3f}'
                                       for j in range(min(5, cum.shape[0]))))

    print('\nC. ЧИ РАХУЮТЬСЯ ВЕРХНІ 2 НАПРЯМКИ `R` З ВАГ '
          '(перекриття з 6 кандидатами із замикання)')
    print(f'{"шар":>4s} {"з ваг":>9s} {"⛔випадк.":>11s}')
    for i in LS:
        print(f'{i:4d} {np.mean(deliv[i]):9.3f} {np.mean(delivR[i]):11.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
