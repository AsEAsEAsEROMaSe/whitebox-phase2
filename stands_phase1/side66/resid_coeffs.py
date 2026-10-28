"""ЧИ УНІВЕРСАЛЬНІ КОЕФІЦІЄНТИ НЕВ'ЯЗКИ В БАЗИСІ ІЗ ЗАМИКАННЯ. (мета 1e-8)

ЗВІДКИ (`nwidth.py`, 6 мереж, істина двома половинами, шум 0.000):
  A. нев'язка `R = Cov_іст[relu] − relu_cov_gauss` майже РАНГ-2:
     ранг-2 бере 0.917, ранг-8 — 0.975 (⛔випадковий ранг-32 лише 0.12);
  B. але СПІЛЬНОГО підпростору між мережами НЕМА: моди класу дають по ~25%
     кожна, тобто нев'язки різних мереж майже ортогональні;
  C. 🔑 верхні 2 напрямки `R` лежать у span 6 дешевих векторів із замикання —
     перекриття **0.940** на шарі 31 (⛔випадково 0.021).
⟹ шукати СПІЛЬНИЙ БАЗИС було хибним питанням: базис у кожної мережі свій, але
він рахується з її ж ваг. Лишається останнє питання — КОЕФІЦІЄНТИ.

ЩО ПЕРЕВІРЯЄМО. У канонічному базисі `Q` (6 векторів із ЗАМИКАННЯ, не з
істини — інакше це не доставка) нев'язка описується матрицею 6×6:
    G = Qᵀ R Q      (21 число на шар)
Питання: чи однакова `G` між мережами. Якщо так — її можна зашити константами
і замикання стане точнішим ЗАДУРНО.

ВАРІАНТИ
    ідеальна G цієї мережі      ← стеля (оракул)
    G, усереднена по ІНШИХ мережах (LOO)   ★ доставка
    ⛔ G з ІНШОГО шару           ← контроль «аби якась G»
    ⛔ G випадкова з тим же слідом

КОНТРОЛІ
  * усе перехресно: `Q` і `G` будуються на половині 1, захоплення міряється
    через ⟨·, R₂⟩ ⟹ шум не може роздути результат;
  * базис канонізовано за знаком (інакше LOO зіпсують випадкові фліпи QR);
  * ⛔ випадкова `G` і `G` з чужого шару — обидві мусять дати ~0.
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
from lyapunov_dirs import closure_chain                        # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
UPTO = 31
SQRT2 = math.sqrt(2.0)


def basis_from_closure(m, s, dev):
    """6 дешевих векторів із замикання; канонізований ортонормований базис."""
    t = m / s
    Ph = 0.5 * (1.0 + torch.erf(t / SQRT2))
    ph = torch.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    cand = torch.stack([m * Ph + s * ph, s, ph, Ph,
                        torch.ones_like(s), s * ph], 1)
    Q, _ = torch.linalg.qr(cand)
    sg = torch.sign((Q * cand).sum(0))
    sg = torch.where(sg == 0, torch.ones_like(sg), sg)
    return Q * sg.unsqueeze(0)


def main(nets=8, dev=DEV):
    torch.manual_seed(20260729)
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    st = {}          # (net, layer) -> (Q, R1, R2, scale, G)
    for k, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            clo = closure_chain(wc, nodes, wts, UPTO)
            A = half_moments(wc, dev, 70001 + k, HALF_N, LS)
            B = half_moments(wc, dev, 80001 + k, HALF_N, LS)
            for i in LS:
                R = []
                for H in (A, B):
                    mp, Cpp, Crr, _ = H[i]
                    Cg, _ = relu_cov_gauss(mp, Cpp, nodes, wts)
                    R.append(Crr - Cg)
                R1, R2 = R
                Q = basis_from_closure(clo[i][0], clo[i][1], dev)
                sc = float(Cg.diagonal().mean())
                # шум виміряно раніше як 0.000 ⟹ беремо середнє половин
                Rm = 0.5 * (R1 + R2)
                st[(k, i)] = (Q.cpu(), Rm.cpu(), sc,
                              ((Q.T @ Rm @ Q) / sc).cpu(),
                              float(((R1 - R2) ** 2).sum() / (R1 * R1).sum()))
        print(f'  мережа {k+1}/{len(fs)} готова', flush=True)

    def cap(k, i, G):
        """ЧЕСНА частка знятої нев'язки: 1 − ‖R − QĜQᵀ‖²/‖R‖²."""
        Q, Rm, sc, _, _ = st[(k, i)]
        E = Q @ (G * sc) @ Q.T
        return 1.0 - float(((Rm - E) ** 2).sum() / (Rm * Rm).sum())

    rows = ('ідеальна G (оракул)', '★LOO по мережах', '⛔G з іншого шару',
            '⛔випадкова G')
    acc = {r: {i: [] for i in LS} for r in rows}
    for i in LS:
        for k in range(len(fs)):
            G = st[(k, i)][3]
            acc['ідеальна G (оракул)'][i].append(cap(k, i, G))
            Gl = torch.stack([st[(j, i)][3] for j in range(len(fs))
                              if j != k]).mean(0)
            acc['★LOO по мережах'][i].append(cap(k, i, Gl))
            oth = LS[(LS.index(i) + 4) % len(LS)]
            acc['⛔G з іншого шару'][i].append(cap(k, i, st[(k, oth)][3]))
            Z = torch.randn_like(G)
            Z = 0.5 * (Z + Z.T)
            Z = Z * (G.norm() / Z.norm())
            acc['⛔випадкова G'][i].append(cap(k, i, Z))

    print(f'\n{len(fs)} мереж, істина 2×{HALF_N}, базис 6 векторів ІЗ ЗАМИКАННЯ')
    print(f'{"шар":>4s}' + ''.join(f'{r:>22s}' for r in rows))
    for i in LS:
        print(f'{i:4d}' + ''.join(f'{np.mean(acc[r][i]):22.3f}' for r in rows))

    print('\nрозкид G між мережами (‖G_k − Ḡ‖/‖Ḡ‖) і ✅шум ‖R₁−R₂‖²/‖R₁‖²:')
    for i in LS:
        Gs = torch.stack([st[(k, i)][3] for k in range(len(fs))])
        Gm = Gs.mean(0)
        nz = np.mean([st[(k, i)][4] for k in range(len(fs))])
        print(f'  шар {i:2d}: розкид {float((Gs-Gm).norm(dim=(1,2)).mean()/Gm.norm()):.3f}'
              f'   ✅шум {nz:.4f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
