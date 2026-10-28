"""ЧИ Є КОЛАПС У 6 НАПРЯМКІВ ЛЯПУНОВСЬКИМ. (ідея юзера: Фюрстенберг/Осєлєдець)

ТВЕРДЖЕННЯ, ЯКЕ ПЕРЕВІРЯЄМО. 32 шари — це добуток 32 матриць. Теорема
Осєлєдця: у добутку багатьох матриць майже будь-який вектор притягується до
одного виділеного напрямку, а швидкість притягання задає розрив показників
Ляпунова. Якщо колапс 256 → ~6 (С65, `caustic_dirs`) саме такий, то:
  (1) напрямки задані ВАГАМИ, без жодного семплу;
  (2) вони ОДНІ Й ТІ САМІ для похибки, народженої на РІЗНІЙ глибині —
      бо всі часткові добутки `M_k→31` мають спільну верхню сингулярну
      підпросторову межу (пряма підмножина фільтрації Осєлєдця);
  (3) розрив між σ₁ і σ₂ мусить бути помітним, інакше пояснення хибне.

ЩО РАХУЄМО (жодної істини, жодного прогону мережі)
    g_i = Φ(m_i/s_i)  ← затвор із ЗАМИКАННЯ, тобто з самих ваг
    M_k = W_{k+1} G_{k+1} … G_30 W_31 · diag(Φ(t₃₁))     — перенос з шару k у фінал
і дивимось верхні ПРАВІ сингулярні вектори `M_k` — це напрямки, у яких похибка
ПРИБУВАЄ на вихід.

ТРИ ВИМІРИ
  A. спектр Ляпунова `M_1`: PR спектра σ², розрив σ₁/σ₂ — чи є колапс узагалі;
  B. збіг верхніх підпросторів `M_k` для різних k — чи спільний напрямок;
  C. збіг із ФАКТИЧНИМ підпростором шуму фіналу (коваріація по 32 жеребах,
     істина НЕ потрібна) — чи це той самий підпростір, у якому живе похибка.

КОНТРОЛІ
  * ⛔ випадковий підпростір тієї ж вимірності — базова лінія перекриття k/256;
  * ⛔ ті самі ваги з ПЕРЕМІШАНИМИ затворами (g зсунуто по шарах) — якщо збіг
    не впаде, він не про Ляпунова, а про самі `W`;
  * поруч друкуємо перекриття з базисом замикання (`caustic_dirs`, PR 5.0) —
    щоб бачити, чи Ляпунов дає ТЕ САМЕ дешевше, чи щось інше.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import (INV_SQRT_2PI, MIN_PAIRS,              # noqa: E402
                         PILOT_PAIRS, TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                               # noqa: E402
from born_source import _Phi                                   # noqa: E402
from closure_precision import NQ, relu_cov0, relu_cov_gauss    # noqa: E402
from caustic_dirs import run_one                               # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
UPTO = 31
NSEED = 32
SEED0 = 400_000
KS = (1, 5, 10, 20, 25, 30)


def closure_chain(wc, nodes, wts, upto):
    """(m, s) доактиваційних `p` шарів 1..upto із самих ваг."""
    w0 = wc[0].double()
    m_est = (w0.pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
    C_est = relu_cov0(wc[0])
    out = {}
    for i in range(1, upto + 1):
        Wd = wc[i].double()
        C_p = Wd.T @ (C_est @ Wd)
        m_p = m_est @ Wd
        out[i] = (m_p, C_p.diagonal().clamp_min(1e-30).sqrt())
        if i < upto:
            C_est, m_est = relu_cov_gauss(m_p, C_p, nodes, wts)
    return out


def transport(wc, g, k, phi_last):
    """M_k = W_{k+1} G_{k+1} … G_30 W_31 · diag(Φ(t₃₁)) — перенос з шару k у фінал."""
    M = wc[k + 1].double()
    for i in range(k + 1, L_ - 1):
        M = (M * g[i].unsqueeze(0)) @ wc[i + 1].double()
    return M * phi_last.unsqueeze(0)


def top_right(M, k):
    """Верхні k правих сингулярних векторів (256, k), ортонормовані."""
    _, _, Vh = torch.linalg.svd(M)
    return Vh[:k].T.contiguous()


def overlap(P, Q):
    """Перекриття підпросторів: ‖PᵀQ‖²_F / k ∈ [0,1]; випадкове ≈ k/256."""
    return float((P.T @ Q).pow(2).sum() / P.shape[1])


def main(nets=12, dev=DEV, nseed=NSEED):
    torch.manual_seed(999)
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    KTOP = 6
    acc = {'PR': [], 'gap': [], 'ln1': [], 'ширина': [],
           'M_k↔M_1': {k: [] for k in KS[1:]},
           'факт↔Ляпунов': [], 'факт↔замикання': [], 'факт↔випадк.': [],
           'факт↔перемішані затвори': [], 'Ляпунов↔замикання': [],
           'частка MSE у топ-6 Ляпунов': [], 'частка MSE у топ-6 замикання': [],
           'частка MSE у топ-6 випадк.': []}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            clo = closure_chain(wc, nodes, wts, UPTO)
            g = {i: _Phi(clo[i][0] / clo[i][1]) for i in range(1, L_ - 1)}
            phi_last = _Phi(clo[UPTO][0] / clo[UPTO][1])

            M1 = transport(wc, g, 1, phi_last)
            sv = torch.linalg.svdvals(M1).double()
            e = sv * sv
            acc['PR'].append(float(e.sum() ** 2 / (e * e).sum()))
            acc['gap'].append(float(sv[0] / sv[1]))
            acc['ln1'].append(float(torch.log(sv[0] / sv[-1])))
            P1 = top_right(M1, KTOP)
            for k in KS[1:]:
                acc['M_k↔M_1'][k].append(overlap(P1, top_right(
                    transport(wc, g, k, phi_last), KTOP)))

            # ⛔ контроль: ті самі W і ті самі значення затворів, але
            # перемішані ВСЕРЕДИНІ шару — руйнує узгодження затвора з вагами
            gs = {i: g[i][torch.randperm(g[i].shape[0], device=dev)]
                  for i in range(1, L_ - 1)}
            Ps = top_right(transport(wc, gs, 1, phi_last), KTOP)

            # фактичний підпростір шуму фіналу — по жеребах, БЕЗ істини
            Y = torch.stack([run_one(wc, SEED0 + 1009 * j, dev)[0]
                             for j in range(nseed)])
            Yc = Y - Y.mean(0, keepdim=True)
            _, Ue = torch.linalg.eigh(Yc.T @ Yc)
            Pe = Ue.flip(1)[:, :KTOP].contiguous()

            # базис замикання (як у caustic_dirs)
            C_est = relu_cov0(wc[0])
            m_est = (wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
            for i in range(1, UPTO + 1):
                Wi = wc[i].double()
                C_p = Wi.T @ (C_est @ Wi)
                m_p = m_est @ Wi
                if i < UPTO:
                    C_est, m_est = relu_cov_gauss(m_p, C_p, nodes, wts)
            Cn = (phi_last.unsqueeze(1) * C_p) * phi_last.unsqueeze(0)
            _, Uc = torch.linalg.eigh(Cn)
            Pc = Uc.flip(1)[:, :KTOP].contiguous()

            nw = phi_last.shape[0]
            Q, _ = torch.linalg.qr(torch.randn(nw, KTOP, dtype=torch.float64, device=dev))
            tot = float(Yc.pow(2).sum())
            for nm, P in (('Ляпунов', P1), ('замикання', Pc), ('випадк.', Q)):
                acc['частка MSE у топ-6 ' + nm].append(
                    float((Yc @ P).pow(2).sum()) / tot)
            acc['ширина'].append(nw)
            acc['факт↔Ляпунов'].append(overlap(Pe, P1))
            acc['факт↔замикання'].append(overlap(Pe, Pc))
            acc['факт↔випадк.'].append(overlap(Pe, Q))
            acc['факт↔перемішані затвори'].append(overlap(Pe, Ps))
            acc['Ляпунов↔замикання'].append(overlap(P1, Pc))
        print(f'  мережа {n+1}/{len(fs)}  PR(M₁)={acc["PR"][-1]:.1f} '
              f'σ₁/σ₂={acc["gap"][-1]:.2f} '
              f'факт↔Ляпунов={acc["факт↔Ляпунов"][-1]:.3f}', flush=True)

    print(f'\n{len(fs)} мереж, KTOP={KTOP}, ширина фіналу '
          f'{np.mean(acc["ширина"]):.0f}, випадкова база '
          f'{KTOP/np.mean(acc["ширина"]):.3f}')
    print(f'\nA. спектр переносу M₁: PR = {np.mean(acc["PR"]):.1f} ± '
          f'{np.std(acc["PR"]):.1f}, розрив σ₁/σ₂ = {np.mean(acc["gap"]):.2f} '
          f'± {np.std(acc["gap"]):.2f}, ln(σ₁/σ₂₅₆) = {np.mean(acc["ln1"]):.1f}')
    print('\nB. чи спільний напрямок у похибки з різної глибини:')
    for k in KS[1:]:
        v = acc['M_k↔M_1'][k]
        print(f'   M_{k:<2d} ↔ M_1 : {np.mean(v):.3f} ± {np.std(v):.3f}')
    print('\nC. збіг із ФАКТИЧНИМ підпростором шуму фіналу:')
    for k in ('факт↔Ляпунов', 'факт↔замикання', 'факт↔перемішані затвори',
              'факт↔випадк.', 'Ляпунов↔замикання'):
        v = acc[k]
        print(f'   {k:26s}: {np.mean(v):.3f} ± {np.std(v):.3f}')
    print('\nD. яку частку РЕАЛЬНОЇ похибки фіналу накриває підпростір:')
    for k in ('частка MSE у топ-6 Ляпунов', 'частка MSE у топ-6 замикання',
              'частка MSE у топ-6 випадк.'):
        v = acc[k]
        print(f'   {k:28s}: {np.mean(v):.3f} ± {np.std(v):.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 12)
