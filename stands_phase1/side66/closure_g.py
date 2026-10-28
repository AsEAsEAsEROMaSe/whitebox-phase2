"""ЗАМИКАННЯ З УНІВЕРСАЛЬНОЮ ПОПРАВКОЮ G. (мета 1e-8)

ЗВІДКИ (`nwidth.py` + `resid_coeffs.py`, істина двома половинами, шум 1e-4):
  * нев'язка `R = Cov_іст[relu] − relu_cov_gauss` майже РАНГ-2 (0.917);
  * її верхні напрямки лежать у span 6 ДЕШЕВИХ векторів із замикання (0.940);
  * матриця `G = QᵀRQ / масштаб` (21 число на шар) майже УНІВЕРСАЛЬНА:
    розкид між мережами 0.10–0.28, а LOO-усереднена `G` знімає **72–78%**
    нев'язки на шарах 8–28 (⛔випадкова G дає ВІД'ЄМНЕ, ⛔чужий шар гірше).

ЩО ЦЕЙ СТЕНД РОБИТЬ. Ставить цю поправку В САМУ РЕКУРСІЮ на ВСІ шари:
    C_relu ← relu_cov_gauss(m_p, C_p) + Q(m_p,s_p) · Ĝ_шар · Q(m_p,s_p)ᵀ · масштаб
де `Ĝ_шар` узято З ІНШИХ МЕРЕЖ (LOO) ⟹ жодної істини цієї мережі.
Питання одне: наскільки впаде `r` = похибка_замикання / шум_вибірки на шарі 31.

КОНТРОЛІ
  * ✅ `Ĝ=0` мусить відтворити гаусове замикання ДО ОСТАННЬОЇ ЦИФРИ;
  * ⛔ випадкова `Ĝ` того ж масштабу — має бути ГІРШЕ за гаусове;
  * оракульна `Ĝ` цієї ж мережі — стеля;
  * після поправки `C` проєктується на PSD (друкуємо, скільки маси зрізано);
  * `r` міряється проти РЕАЛЬНОГО шуму вибірки на тих самих мережах.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import INV_SQRT_2PI                          # noqa: E402
from transport_r2 import CHUNK, cut_net                       # noqa: E402
from closure_precision import NQ, relu_cov0, relu_cov_gauss   # noqa: E402
from caustic_dirs import run_one                              # noqa: E402
from resid_coeffs import basis_from_closure                   # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
UPTO = 31
HALF_N = 1_048_576
NSEED = 8
SEED0 = 400_000
ALL = tuple(range(1, UPTO))


def truth_all(wc, dev, seed, n):
    """(m, C) досигмоїдних `p` і Cov[relu(p)] на ВСІХ шарах 1..UPTO-1."""
    sp = {i: torch.zeros(wc[i].shape[1], dtype=torch.float64, device=dev)
          for i in ALL}
    sr, Cp, Cr = ({i: torch.zeros_like(sp[i]) for i in ALL},
                  {i: torch.zeros(sp[i].shape[0], sp[i].shape[0],
                                  dtype=torch.float64, device=dev) for i in ALL},
                  None)
    Cr = {i: torch.zeros_like(Cp[i]) for i in ALL}
    mlast = torch.zeros(wc[UPTO].shape[1], dtype=torch.float64, device=dev)
    slast = torch.zeros_like(mlast)
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
            if i == UPTO:
                pd = p.double()
                mlast += pd.sum(0)
                slast += (pd * pd).sum(0)
            elif i in ALL:
                r = torch.relu(p)
                sp[i] += p.sum(0).double()
                sr[i] += r.sum(0).double()
                Cp[i] += (p.T @ p).double()
                Cr[i] += (r.T @ r).double()
            x = torch.relu(p)
        done += mch
    out = {}
    for i in ALL:
        mp, mr = sp[i] / done, sr[i] / done
        out[i] = (mp, Cp[i] / done - torch.outer(mp, mp),
                  Cr[i] / done - torch.outer(mr, mr))
    ml = mlast / done
    return out, ml, (slast / done - ml * ml).clamp_min(1e-30).sqrt()


def closure_g(wc, nodes, wts, G, dev, trace=None):
    """Замикання з поправкою `G` (dict шар → 6×6) або без неї, якщо G is None.

    trace: якщо dict — складає туди (m_p, s_p) КОЖНОГО шару, щоб бачити, який
    саме ланцюг поправка лагодить, а який ламає.
    """
    w0 = wc[0].double()
    m_est = (w0.pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
    C_est = relu_cov0(wc[0])
    cut = 0.0
    for i in range(1, UPTO + 1):
        Wd = wc[i].double()
        C_p = Wd.T @ (C_est @ Wd)
        m_p = m_est @ Wd
        s_p = C_p.diagonal().clamp_min(1e-30).sqrt()
        if trace is not None:
            trace[i] = (m_p, s_p)
        if i == UPTO:
            return m_p, s_p, cut
        C_g, m_g = relu_cov_gauss(m_p, C_p, nodes, wts)
        if G is not None and i in G:
            Q = basis_from_closure(m_p, s_p, dev)
            C_g = C_g + Q @ (G[i] * float(C_g.diagonal().mean())) @ Q.T
            ev, U = torch.linalg.eigh(0.5 * (C_g + C_g.T))
            neg = float(ev.clamp(max=0.0).abs().sum() / ev.abs().sum())
            cut = max(cut, neg)
            C_g = (U * ev.clamp_min(0.0)) @ U.T
        C_est, m_est = C_g, m_g
    raise AssertionError


def main(nets=6, dev=DEV):
    torch.manual_seed(5150)
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    keep = []
    for k, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            T, mlast, slast = truth_all(wc, dev, 70001 + k, HALF_N)
            G = {}
            for i in ALL:
                mp, Cpp, Crr = T[i]
                Cg, _ = relu_cov_gauss(mp, Cpp, nodes, wts)
                Q = basis_from_closure(mp, Cpp.diagonal().clamp_min(1e-30).sqrt(),
                                       dev)
                G[i] = (Q.T @ (Crr - Cg) @ Q) / float(Cg.diagonal().mean())
            nz = float(np.mean([
                float(((run_one(wc, SEED0 + 1009 * j, dev)[1] - mlast).abs()
                       / slast).mean()) for j in range(NSEED)]))
            tm = {i: (T[i][0], T[i][1].diagonal().clamp_min(1e-30).sqrt())
                  for i in ALL}
            tm[UPTO] = (mlast, slast)
            keep.append((wc, G, mlast, slast, nz, tm))
        print(f'  істина мережі {k+1}/{len(fs)}, шум {nz:.3e}', flush=True)

    rows = ('гаусове (нинішнє)', '✅контроль Ĝ=0', '★LOO по мережах',
            'оракул Ĝ цієї мережі', '⛔випадкова Ĝ')
    res = {r: [] for r in rows}
    cuts = []
    PL = (1, 2, 4, 8, 12, 16, 20, 24, 28, 31)
    prof = {r: {i: [[], []] for i in PL} for r in (rows[0], rows[2], rows[3])}
    for k, (wc, G, mlast, slast, _, tm) in enumerate(keep):
        Gl = {i: torch.stack([o[1][i] for j, o in enumerate(keep) if j != k]
                             ).mean(0) for i in ALL}
        Z = {}
        for i in ALL:
            z = torch.randn_like(G[i])
            z = 0.5 * (z + z.T)
            Z[i] = z * (Gl[i].norm() / z.norm())
        with torch.no_grad():
            for r, g in ((rows[0], None), (rows[1], {i: G[i] * 0 for i in ALL}),
                         (rows[2], Gl), (rows[3], G), (rows[4], Z)):
                tr = {} if r in prof else None
                mc, _, cut = closure_g(wc, nodes, wts, g, dev, tr)
                res[r].append(float(((mc - mlast).abs() / slast).mean()))
                if tr is not None:
                    for i in PL:
                        mm, ss = tr[i]
                        prof[r][i][0].append(
                            float(((mm - tm[i][0]).abs() / tm[i][1]).mean()))
                        prof[r][i][1].append(
                            float(((ss - tm[i][1]).abs() / tm[i][1]).mean()))
                if r == rows[2]:
                    cuts.append(cut)
        print(f'  замикання {k+1}/{len(keep)}  гаус {res[rows[0]][-1]:.3e} → '
              f'LOO {res[rows[2]][-1]:.3e}', flush=True)

    print('\nПОШАРОВО — який ланцюг поправка лагодить, а який ламає '
          '(похибка у частках істинного s)')
    print(f'{"шар":>4s} {"m: гаус":>10s} {"m: LOO":>10s} {"m: оракул":>10s}'
          f'{"s: гаус":>12s} {"s: LOO":>10s} {"s: оракул":>10s}')
    for i in PL:
        v = [np.mean(prof[r][i][j]) for j in (0, 1)
             for r in (rows[0], rows[2], rows[3])]
        print(f'{i:4d} {v[0]:10.3e} {v[1]:10.3e} {v[2]:10.3e}'
              f'{v[3]:12.3e} {v[4]:10.3e} {v[5]:10.3e}')

    nz = float(np.mean([o[4] for o in keep]))
    d = abs(np.mean(res[rows[1]]) / np.mean(res[rows[0]]) - 1.0)
    print(f'\n{len(keep)} мереж, істина {HALF_N}, шум вибірки {nz:.3e}')
    print(f'✅ контроль Ĝ=0 проти гаусового: {d:.2e} (має бути ~0)')
    print(f'PSD-зріз (макс. частка від\'ємної маси): {np.mean(cuts):.2e}')
    print(f'\n{"варіант":24s} {"похибка":>11s} {"r":>7s} {"β":>7s} {"виграш":>8s}')
    print('-' * 62)
    for r in rows:
        e = float(np.mean(res[r]))
        rr = e / nz
        print(f'{r:24s} {e:11.3e} {rr:7.2f} {1/(1+rr*rr):7.3f} '
              f'{1.0 + 1.0/(rr*rr):8.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
