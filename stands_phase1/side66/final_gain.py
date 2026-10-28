"""ФІНАЛЬНИЙ СТЕНД: КАЛІБРОВАНЕ ЗАМИКАННЯ + β-ВАГИ, УСЕ В ОДНОМУ ПРОГОНІ.

СКЛАДАЄ ДВА СЬОГОДНІШНІ ЗДОБУТКИ, ЯКІ МНОЖАТЬСЯ:
  1. β-ваги (`beta_gain.py`): β = 1/(1+r²) ⟹ поправку можна вмикати на ВСІХ
     шарах, зашкодити вона не може за побудовою. Дало 1.264× (медіана 1.212).
  2. калібрування (`feat_calib.py`): регресія похибки замикання на (t, t²),
     навчена на банку, знімає R²=0.82 і зменшує r удвічі-втричі на глибині
     (шар 8: 3.12→1.85, шар 16: 6.26→3.05, шар 31: 11.8→5.0).
Менший r автоматично дає більший β — множення відбувається САМО, без підбору.

ФАЗА 1 (мережі 0..TRAIN-1): на кожному шарі навчаємо rel ≈ a + b·t + c·t²
   окремо для s і для m; лишок дає r_s(i), r_m(i) ⟹ β_s(i), β_m(i).
ФАЗА 2 (мережі SKIP..SKIP+NETS): застосовуємо, міряємо бал.
Мережі фаз НЕ ПЕРЕТИНАЮТЬСЯ.

КОНТРОЛІ
  * «без калібрування, β» мусить відтворити ~1.25× з `beta_gain.py`;
  * «калібрування, БЕЗ β (плоско)» мусить програвати — як 1..8 без ваг (0.750);
  * ОРАКУЛ σ — стеля гілки (~2.6-2.9×);
  * SHUFFLE калібрувальних коефіцієнтів по шарах ⟹ мусить зіпсувати назад;
  * медіана поруч із середнім (на грейдер іде ближче до медіани).
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import (GATE_N, INV_SQRT_2PI, MIN_PAIRS,     # noqa: E402
                         PILOT_PAIRS, TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                              # noqa: E402
from born_source import truth_moments, _Phi, _phi             # noqa: E402
from closure_precision import relu_cov0, relu_cov_gauss, NQ   # noqa: E402
from mu_gain import closure_ms                                # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331, 5150, 60613, 727272, 8181)
TRAIN, VAL, SKIP, UPTO = 5, 3, 8, 31   # 0..4 навчання COEF, 5..7 валідація β


def npairs(wc):
    per_cut = sum(2.0 * w.shape[0] * w.shape[1] for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * L_ * W_ * W_)
    return max(MIN_PAIRS, int((TARGET_UTIL * B_BUDGET - used) / (2.0 * per_cut)))


def sample_pass(wc, seed, dev):
    """Прогін sub66; повертає raw, v і вибіркові (m, s) кожного шару."""
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    z = torch.tensor(rng.standard_normal((npairs(wc), W_)).astype(np.float32),
                     device=dev)
    x = torch.cat([z, -z], 0)
    n2 = float(x.shape[0])
    cov = (x.T @ x) / n2
    ev, evec = torch.linalg.eigh(cov.double())
    wh = (evec * (1.0 / ev.clamp_min(1e-8).sqrt()) @ evec.T).float()
    x = torch.relu(x @ (wh @ wc[0]))
    sg0 = wc[0].pow(2).sum(0).clamp_min(1e-30).sqrt()
    v0 = (x.double().sum(0) / n2 - (sg0 * INV_SQRT_2PI).double())
    ms, gs, ps = [], [], []
    v = v0.clone()
    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        pd = p.double()
        m = pd.sum(0) / n2
        s = ((pd * pd).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt()
        ms.append((m, s)); gs.append(g); ps.append(None)
        v = (v @ wc[i].double()) * g
        x = torch.relu(p)
    return x.double().sum(0) / n2, v, v0, ms, gs, n2


def main(nets=8, dev=DEV):
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    allf = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have]

    def load(f):
        s = int(np.load(f)['seed'])
        return [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
                for q in range(L_)]

    # ---------- ФАЗА 1: навчання калібрування ----------
    acc = {i: {'t': [], 's': [], 'm': []} for i in range(1, UPTO + 1)}
    for k, f in enumerate(allf[:TRAIN]):
        with torch.no_grad():
            wc, _, _ = cut_net(load(f), dev)
            _, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            clo = closure_ms(wc, nodes, wts, UPTO)
            for i in range(1, UPTO + 1):
                m_c, s_c = clo[i]
                st, mt = sg_t[i - 1].double(), mu_t[i - 1].double()
                acc[i]['t'].append((m_c / s_c).cpu().numpy())
                acc[i]['s'].append(((s_c - st) / st).cpu().numpy())
                acc[i]['m'].append(((m_c - mt) / st).cpu().numpy())
        print(f'  навчання {k+1}/{TRAIN}', flush=True)

    COEF = {}
    for i in range(1, UPTO + 1):
        t = np.concatenate(acc[i]['t'])
        A = np.stack([np.ones_like(t), t, t * t], 1)
        for key in ('s', 'm'):
            y = np.concatenate(acc[i][key])
            COEF[(i, key)], *_ = np.linalg.lstsq(A, y, rcond=None)

    # ---------- ФАЗА 1b: β на ВІДКЛАДЕНИХ мережах, з ВИМІРЯНИМ шумом вибірки ----------
    # 🚨 r НЕ МОЖНА рахувати там, де навчали COEF (занижує r ⟹ завищує β),
    #    і шум треба брати виміряний, а не аналітичний: завищений удвічі β
    #    дає РІВНО НУЛЬ виграшу, трохи більший — уже шкоду.
    val = {i: {'raw': [], 'cal': [], 'nz': []} for i in range(1, UPTO + 1)}
    for k, f in enumerate(allf[TRAIN:TRAIN + VAL]):
        with torch.no_grad():
            wc, _, _ = cut_net(load(f), dev)
            _, mu_t, sg_t = truth_moments(wc, dev, 80001 + k, TRUTH_N)
            clo = closure_ms(wc, nodes, wts, UPTO)
            _, _, _, ms, _, _ = sample_pass(wc, 12345, dev)
            for i in range(1, UPTO + 1):
                m_c, s_c = clo[i]
                st = sg_t[i - 1].double()
                t_ = (m_c / s_c).cpu().numpy()
                A = np.stack([np.ones_like(t_), t_, t_ * t_], 1)
                y = ((s_c - st) / st).cpu().numpy()
                rk = (1.0 + y) / (1.0 + A @ COEF[(i, 's')]) - 1.0
                val[i]['raw'].append(np.abs(y).mean())
                val[i]['cal'].append(np.abs(rk).mean())
                val[i]['nz'].append(float(((ms[i - 1][1] - st).abs() / st).mean()))
        print(f'  валідація {k+1}/{VAL}', flush=True)

    BETA = {}
    for i in range(1, UPTO + 1):
        nz = float(np.mean(val[i]['nz']))
        BETA[(i, 's')] = 1.0 / (1.0 + (float(np.mean(val[i]['cal'])) / nz) ** 2)
        BETA[(i, 's0')] = 1.0 / (1.0 + (float(np.mean(val[i]['raw'])) / nz) ** 2)
        BETA[(i, 'm')] = 0.0
    nz_s = nz_m = float(np.mean(val[1]['nz']))

    print(f'\nшум вибірки (ВИМІРЯНИЙ, шар 1): {nz_s:.3e}')
    print('β_s після калібр.: ' + ' '.join(
        f'{i}:{BETA[(i,"s")]:.3f}' for i in (1, 2, 4, 8, 16, 31)))
    print('β_s без калібр.:   ' + ' '.join(
        f'{i}:{BETA[(i,"s0")]:.3f}' for i in (1, 2, 4, 8, 16, 31)))

    # ---------- ФАЗА 2: бал на інших мережах ----------
    variants = ['oracle', 'beta_raw', 'calib_flat', 'calib_beta', 'calib_beta_ms']
    E_b, VAR = [], {q: [] for q in variants}
    for k, f in enumerate(allf[SKIP:SKIP + nets]):
        with torch.no_grad():
            wc, _, _ = cut_net(load(f), dev)
            tr, mu_t, sg_t = truth_moments(wc, dev, 70001 + k, TRUTH_N)
            clo = closure_ms(wc, nodes, wts, UPTO)
            t_last = tr[-1]
            eb = []
            ev = {q: [] for q in variants}
            for sd in SEEDS:
                raw, v, v0, ms, gs, n2 = sample_pass(wc, sd, dev)
                V = {q: v0.clone() for q in variants}
                for i in range(1, L_):
                    m, s = ms[i - 1]
                    g = gs[i - 1]
                    Wd = wc[i].double()
                    t = m / s
                    Fs, Fm = _phi(t), _Phi(t)
                    add = {}
                    if i <= UPTO:
                        m_c, s_c = clo[i]
                        tc = (m_c / s_c)
                        A = torch.stack([torch.ones_like(tc), tc, tc * tc], 1)
                        cs = torch.tensor(COEF[(i, 's')], dtype=torch.float64,
                                          device=dev)
                        cm = torch.tensor(COEF[(i, 'm')], dtype=torch.float64,
                                          device=dev)
                        # 🚨 регресія з t² може дати pred≈−1 на окремих нейронах
                        #    ⟹ знаменник ≈0 і s_k вибухає; одна така компонента
                        #    отруює весь шар. Клампуємо ДІЛЬНИК, не результат.
                        den = (1.0 + A @ cs).clamp(0.6, 1.6)
                        s_k = s_c / den
                        m_k = m_c - (A @ cm).clamp(-0.4, 0.4) * s_k
                        add['calib_flat'] = Fs * (s - s_k)
                        add['calib_beta'] = Fs * (s - s_k) * BETA[(i, 's')]
                        add['calib_beta_ms'] = (Fs * (s - s_k) * BETA[(i, 's')]
                                                + Fm * (m - m_k) * BETA[(i, 'm')])
                        add['beta_raw'] = Fs * (s - s_c) * BETA[(i, 's0')]
                    add['oracle'] = Fs * (s - sg_t[i - 1].double())
                    for q in variants:
                        V[q] = (V[q] @ Wd) * g + add.get(q, 0.0)
                eb.append((raw - v - t_last).cpu().numpy())
                for q in variants:
                    ev[q].append((raw - V[q] - t_last).cpu().numpy())
            E_b.append(eb)
            for q in variants:
                VAR[q].append(ev[q])
        print(f'  тест {k+1}/{nets}', flush=True)

    E_b = np.array(E_b)
    base = float((E_b ** 2).mean())

    def med(E):
        return float(np.median((E_b ** 2).mean(axis=(1, 2))
                               / (E ** 2).mean(axis=(1, 2))))

    names = {'oracle': 'ОРАКУЛ σ усі шари (стеля)',
             'beta_raw': 'β без калібрування (опора 1.25)',
             'calib_flat': 'калібр. БЕЗ β (контроль)',
             'calib_beta': 'калібр. + β  ★',
             'calib_beta_ms': 'калібр. + β, з μ ★'}
    print(f'\n{nets} мереж (тест з {SKIP}-ї, навчання 0..{TRAIN-1}) '
          f'× {len(SEEDS)} жеребів')
    print(f'база MSE {base:.4e}')
    print(f'\n{"варіант":34s} {"MSE":>11s} {"× серед.":>9s} {"× медіана":>10s}')
    print('-' * 68)
    for q in variants:
        E = np.array(VAR[q])
        m = float((E ** 2).mean())
        print(f'{names[q]:34s} {m:11.4e} {base/m:9.3f} {med(E):10.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
