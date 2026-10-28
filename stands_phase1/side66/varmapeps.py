"""КАРТА (K, ε): ЯКОЇ ТОЧНОСТІ ОПОРИ ТРЕБА НА ГЛИБИНІ K.

`varmap.py` зміряв ІДЕАЛЬНИЙ оракул: точна правда на шарі K (K=28 → 15.35×,
K=30 → 44.80×). Але точної правди в нас не буде НІКОЛИ: наше замикання на шарі
29 має **0.9%**, у лідера за нашою оцінкою ~**0.02%**. Драбина по одній осі
мовчить про те, що МІЖ цими точками — а це рівно те питання, яким ми живемо.

ЩО МІРЯЄМО: те саме заміщення `v_K ← x̄_K − m̃_K`, але опора ЗІПСОВАНА на ε:
  `scale` : m̃ = m·(1+ε)        — рівно мода нашого замикання (κ=+0.753%,
                                   усі 24 мережі ОДНОГО знаку);
  `iso`   : m̃ = m + ε‖m‖ξ/√n   — щоб відділити «помиляємось УЗГОДЖЕНО»
                                   від «помиляємось абияк».
🚨 ξ — ОДИН вектор на (мережу, роботу), а НЕ свіжий на кожен жереб: похибка
замикання це ТОЙ САМИЙ вектор для всіх жеребів мережі (⛔НЕГ «сусідній жереб»).
Свіжий шум на жереб усереднився б і збрехав у наш бік.

📣 ПЕРЕДБАЧЕНО (записати ДО прогону):
  1. на глибині виграш обвалюється ШВИДКО за ε: при K=30 ε=0.9% лишить
     ≲3×, а не 44.80× — бо на глибині 1% від `m` більший за всю нашу похибку;
  2. `scale` буде ПОБЛАЖЛИВІШИЙ за `iso` на тому ж ε (узгоджена похибка
     частково знімається вільним членом — див. «вільний член з'їдає зміщення»);
  3. лідерські 24.4× вимагатимуть ε ≈ 0.02–0.05% на K≈29.
🟢 ЖИВЕ, якщо при K=29-30 і ε=0.009 виграш ≥1.30×: тоді різати виклики в
   `sub91` варто. ⛔ЯКЩО ≤1.05× — гілка `sub91` мертва ДО інженерії.
🚨 ПЛОМБИ: ε=0 мусить відтворити драбину `varmap` · тавтологічний K=d−1
   калібрує шум істини · ⛔контроль «істина з ЧУЖОЇ мережі».
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts, W_, GATE_N     # noqa: E402
from base72 import run_side1 as run72                               # noqa: E402
from tilt_wide import truth_last                                    # noqa: E402
from varsub import S72, A72, Q72, G72                               # noqa: E402

L_ = 32
NETS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:1'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
TR_PAIRS, CHUNK = 524_288, 32768

KS = (8, 16, 20, 24, 28, 30)
EPS = (0.0, 0.0002, 0.001, 0.003, 0.009, 0.03)
JOBS = ([(K, 0.0, 'scale') for K in KS]
        + [(K, e, m) for K in KS for e in EPS[1:] for m in ('scale', 'iso')])


def truth_means(wc, dev, seed):
    depth = len(wc)
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(depth):
            a = torch.relu(a @ wc[i])
            acc[i] += a.double().sum(0)
        done += k
    return [t / float(2 * TR_PAIRS) for t in acc]


def spoil(mt, jobs, dev, seed):
    """m̃ для кожної роботи. ξ фіксований на (мережу, роботу)."""
    g = torch.Generator(device=dev).manual_seed(int(seed))
    out = []
    for K, e, mode in jobs:
        m = mt[K]
        if e == 0.0:
            out.append(m)
        elif mode == 'scale':
            out.append(m * (1.0 + e))
        else:
            xi = torch.randn(m.shape[0], generator=g, device=dev, dtype=torch.float64)
            out.append(m + e * m.norm() * xi / float(m.shape[0]) ** 0.5)
    return out


def run_map(wc, npair, dev, seed, mpert, jobs):
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cov = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xa = torch.relu(x @ (wh @ wc[0]))
    VM = torch.zeros(len(jobs), wc[0].shape[1], dtype=torch.float64, device=dev)
    for j, (K, _e, _m) in enumerate(jobs):
        if K == 0:
            VM[j] = xa.double().mean(0) - mpert[j]
    for i in range(1, depth):
        p = xa @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        VM = (VM @ wc[i].double()) * gate.unsqueeze(0)
        xa = torch.relu(p)
        for j, (K, _e, _m) in enumerate(jobs):
            if K == i:
                VM[j] = xa.double().mean(0) - mpert[j]
    xl = xa.double().mean(0)
    return xl, xl.unsqueeze(0) - VM


def main():
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:NETS]
    print(__doc__.split('📣')[1].split('🚨 ПЛОМБИ')[0], flush=True)
    print(f'   мереж {NETS} × {NDRAW} жеребів, КРОС-ЦІЛЬ, {len(JOBS)} робіт\n')

    mse_pure, mse_sub, mse_ctl = [], [], []
    mse_j = {j: [] for j in range(len(JOBS))}
    mse_tau = []
    prevm = None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, DEV)
            d = len(wc)
            jobs = [(min(K, d - 1), e, m) for K, e, m in JOBS] + [(d - 1, 0.0, 'scale')]
            mt = truth_means(wc, DEV, 310_000 + 877 * n)
            T1 = truth_last(wc, DEV, 510_000 + 7919 * n)
            T2 = truth_last(wc, DEV, 610_000 + 7919 * n)
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, 8, DEV, extras=True)
            mp = spoil(mt, jobs, DEV, 424_242 + 31 * n)
            ctl = None
            if prevm is not None:
                t = torch.zeros(wc[d - 1].shape[1], dtype=torch.float64, device=DEV)
                kk = min(prevm.shape[0], t.shape[0])
                t[:kk] = prevm[:kk]
                ctl = [t]

            ap = asb = ac = 0.0
            aj = np.zeros(len(jobs))
            for dr in range(NDRAW):
                sd = 77_000_000 + 733 * n + dr
                xl, preds = run_map(wc, npair, DEV, sd, mp, jobs)
                ap += float(((xl - T1) * (xl - T2)).sum())
                e12 = ((preds - T1.unsqueeze(0)) * (preds - T2.unsqueeze(0))).sum(1)
                aj += e12.cpu().numpy()
                pb, _ = run72(wc, npair, s_cl, m_cl, DEV, sd, lam_s=S72, lam_a=A72,
                              lam_q=Q72, lam_g=G72, q_ext=q_cl, p_ext=f_cl)
                asb += float(((pb - T1) * (pb - T2)).sum())
                if ctl is not None:
                    _, pc = run_map(wc, npair, DEV, sd, ctl, [(d - 1, 0.0, 'scale')])
                    ac += float(((pc[0] - T1) * (pc[0] - T2)).sum())
            mse_pure.append(ap / NDRAW)
            mse_sub.append(asb / NDRAW)
            for j in range(len(JOBS)):
                mse_j[j].append(aj[j] / NDRAW)
            mse_tau.append(aj[-1] / NDRAW)
            if ctl is not None:
                mse_ctl.append(ac / NDRAW)
            prevm = mt[d - 1].clone()
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

    P = float(np.sum(mse_pure))
    noise = float(np.sum(mse_tau))
    print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ.')
    print(f'✅ПЛОМБА: сабміт проти чистого `x̄` = {P/np.sum(mse_sub):.3f}× '
          f'(у varmap24b було 1.205×)')
    print(f'✅КАЛІБР: шум істини (тавтологічний K) = {noise/NETS:.4e}\n')

    def clean(j):
        v = float(np.sum(mse_j[j]))
        return (P - noise) / max(v - noise, 1e-300)

    idx = {(K, e, m): j for j, (K, e, m) in enumerate(JOBS)}
    for mode in ('scale', 'iso'):
        print(f'=== ОЧИЩЕНИЙ ВИГРАШ ×, модель похибки опори: {mode} ===')
        print('   K  ' + '  '.join(f'ε={e*100:6.2f}%' for e in EPS))
        print('-' * (6 + 11 * len(EPS)))
        for K in KS:
            row = []
            for e in EPS:
                key = (K, e, 'scale' if e == 0.0 else mode)
                row.append(f'{clean(idx[key]):9.2f}×' if key in idx else '        —')
            print(f'{K:4d}  ' + '  '.join(row))
        print()
    if mse_ctl:
        c = float(np.sum(mse_ctl))
        print(f'⛔КОНТРОЛЬ (істина з ЧУЖОЇ мережі): {P*(len(mse_ctl)/NETS)/c:.3f}× '
              f'{"✅гірше за чистий x̄" if c/len(mse_ctl) > P/NETS else "🛑 НЕ гірше"}')
    print('\n🔑 НАША ТОЧКА: замикання на 29 має ε≈0.9% ⟹ дивитись стовпець ε=0.90%')
    print('🔑 ЛІДЕР 24.4×: знайти, за якого ε цей рівень досяжний на K≈29')


main()
