"""ПОРТ ДІЮЧОГО САБМІТА `estimator_sub91.py` У TORCH — ПРАВИЛЬНА БАЗА (С82)

## НАВІЩО ЦЕ З'ЯВИЛОСЬ
`side66/base_real.py` — порт `estimator_side1.py`, у якому лише σ-поправка на
8 шарах. Діючий сабміт має **ЧОТИРИ** родини (`_LAM_S`, `_LAM_A`, `_LAM_Q`,
`_LAM_G`) плюс бленд із глибоким замиканням. Міряючи «нові» канали проти
`base_real`, я знайшов «сильний новий канал 2-го моменту» — а це рівно
`_LAM_Q`, що стоїть у сабміті з С66.
🎓🎓 Це ТРЕТІЙ раз, коли спрацьовує та сама пастка
([[finding-free-second-moment-channel]]): **ПЛОМБИ ПЕРЕВІРЯЮТЬ СТЕНД, А НЕ
ВИБІР БАЗИ.** Тому база тепер не пишеться «за мотивами», а звіряється з
самим сабмітом.

## ⛔ ПЛОМБА, ЩО РОБИТЬ ЦЕЙ ФАЙЛ ЛЕГІТИМНИМ
`estimator_sub91.py` запускається НАЖИВО (numpy+flopscope), його вихід
кладеться поруч, і порт мусить збігтися. Той самий потік `numpy` RNG:
пілот 128 пар, потім основна вибірка — тож `z` беруться ЗВІДТИ, а не
генеруються заново.

## ЩО ВІДРІЗНЯЄТЬСЯ СВІДОМО
* замикання у fp64 (у сабміті fp32) — різниця окремо перевірена як несуттєва;
* `_Phi` через `erf` замість Абрамовіца-Стігана (max похибка А&С 1.5e-7);
⟹ збіг очікується на рівні **1e-5…1e-6 відносно**, не побітовий.

## ЩО ДАЄ ЗВЕРХ САБМІТА
`collect=True` повертає матрицю δ-ознак УСІХ чотирьох родин на ДОВІЛЬНИХ
шарах, уже перенесених до виходу ланцюгом воріт. Саме те, чого бракувало:
сабміт має канали лише до шару 8, а питання — що дають шари 16/24/31.

Запуск (самоперевірка): /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/sub91_port.py
"""
import numpy as np
import torch

W_ = 256
INV_SQRT_2PI = 0.3989422804014327
TARGET_UTIL = 0.0817
B_BUDGET = 272_000_000_000
MIN_PAIRS, SEED, PILOT_PAIRS, GATE_N = 128, 12345, 128, 1024
KSIG = 8
ALPHA, W_BLEND = -0.00751, 0.0651
LAM_S = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
         0.0088285, 0.0007922, 0.0015044, 0.0001062)
LAM_A = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
         -0.0025822, -0.0030778, 0.0006822, 0.0068061)
LAM_Q = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
         0.0189783, 0.0250559, 0.0340802, 0.0415433)
LAM_G = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
         -0.0146129, -0.0100612, 0.0167223, 0.0168109)


def _phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQRT_2PI


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def closure91(wc, ksig, dev, kdeep=None):
    """Порт `_closure_parts`: s, m, q=E[a²], P=Φ(t) для шарів 1..ksig + m_deep."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    s_o, m_o, q_o, p_o = [], [], [], []
    nit = ksig + 1 if kdeep is None else kdeep
    for i in range(nit):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        if 1 <= i <= ksig:
            s_o.append(s)
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        if 1 <= i <= ksig:
            p_o.append(Ph)
        m = mp * Ph + s * ph
        if i == nit - 1 and i > ksig:
            break
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s,
                                                                    ph / s)
        d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(d - C.diagonal())
        if 1 <= i <= ksig:
            m_o.append(m)
            q_o.append(C.diagonal() + m * m)
    return s_o, m_o, q_o, p_o, m


def cut_budget91(W, dev):
    """Пілот+обрізання+n_pairs — з `_TARGET_UTIL=0.0817` САБМІТА (не 0.0942!)."""
    depth = len(W)
    rng = np.random.default_rng(SEED)
    zp = torch.tensor(rng.standard_normal((PILOT_PAIRS, W_)).astype(np.float32),
                      device=dev)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(depth):
        h = a @ W[i]
        if i < depth - 1:
            alive.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    wc = []
    for i in range(depth):
        wi = W[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < depth - 1:
            wi = wi[:, alive[i]]
        wc.append(wi.contiguous())
    per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * depth * W_ * W_)
    left = TARGET_UTIL * float(B_BUDGET) - used
    return wc, max(MIN_PAIRS, int(left / (2.0 * max(per_cut, 1.0)))), rng


def predict91(wc, npair, dev, rng=None, seed=None, cl=None, collect=None,
              lam_off=False, diag=False, raw=False):
    """Один прогін сабміту. `collect` — список шарів (0-індексовані) для δ-ознак.

    Повертає (прогноз останнього шару, R), де рядок `4·k + f` матриці `R` —
    δ-ознака родини `f` (0:σ, 1:μ, 2:E[a²], 3:маска) шару `collect[k]+1`,
    уже перенесена до виходу.
    """
    depth = len(wc)
    if rng is None:
        rng = np.random.default_rng(int(seed))
        rng.standard_normal((PILOT_PAIRS, W_))          # ⟵ з'їсти пілот
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                     device=dev)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    w0 = wh @ wc[0]
    y = z @ w0
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    denom = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / denom - sg0 * INV_SQRT_2PI
    s_cl, m_cl, q_cl, P_cl, m_deep = (cl if cl is not None else
                                      closure91(wc, KSIG, dev, kdeep=depth))
    # 🔑 diag: ПОШАРОВІ скаляри для формули усадки, усі — з ВЖЕ обчисленого:
    #    D2 = Σ_j δ_j²  (δ = x̄ − m_замик — те, що канал БАЧИТЬ)
    #    E2 = Σ_j Var(a_j)/(2·npair)  (шум вибіркового середнього — АРИФМЕТИКА)
    #    ⟹ B² = D2 − E2 і λ* = E2/D2, БЕЗ жодної істини. Ціна: дві редукції.
    DG = [] if diag else None
    NC = 0 if collect is None else len(collect)
    R = (torch.zeros(4 * NC, wc[0].shape[1], dtype=torch.float64, device=dev)
         if NC else None)
    last = None
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        if R is not None:
            R = (R @ wd) * gate.unsqueeze(0)
        pd = p.double()
        mh = pd.sum(0) / denom
        sh = (pd.pow(2).sum(0) / denom - mh * mh).clamp_min(1e-30).sqrt()
        x = torch.relu(p)
        xm = x.double().sum(0) / denom
        xq = x.double().pow(2).sum(0) / denom
        if i - 1 < len(s_cl):
            d = (_phi(mh / sh) * (sh - s_cl[i - 1]), xm - m_cl[i - 1],
                 xq - q_cl[i - 1], gate - P_cl[i - 1])
            if i <= KSIG and not lam_off:
                v = (v + LAM_S[i - 1] * d[0] + LAM_A[i - 1] * d[1]
                     + LAM_Q[i - 1] * d[2] + LAM_G[i - 1] * d[3])
            if DG is not None:
                va = (xq - xm * xm).clamp_min(0.0)
                DG.append((float((d[1] * d[1]).sum()),
                           float(va.sum()) / denom, i))
            if R is not None and (i - 1) in collect:
                k = collect.index(i - 1)
                for f in range(4):
                    R[4 * k + f] = R[4 * k + f] + d[f]
        if i == depth - 1:
            last = x.double().sum(0) / denom
            # 🔑 raw: глобальні числа ВИХОДУ для динамічного бленду.
            #    VO = Σ_j Var(a_j)/(2·npair) — дисперсія НАШОГО прогнозу,
            #    чиста арифметика по вже обчисленому останньому шару.
            vvec = ((x.double().pow(2).sum(0) / denom
                     - last * last).clamp_min(0.0)) / denom
            vout = float(vvec.sum())
    ours = last - v
    pr = ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)
    if raw:
        # 🆕 С91: `vvec` — ПОНЕЙРОННА дисперсія НАШОГО прогнозу (Var(x̄_j)),
        #    додано в кінець кортежу ⟹ старі виклики не ламаються.
        return pr, R, DG, (ours, m_deep, vout, vvec)
    return (pr, R, DG) if diag else (pr, R)


if __name__ == '__main__':
    import glob
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
    ref = np.load('/tmp/claude-1000/-mnt-win-d-llm-White-Box/'
                  '5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/'
                  'sub91_ref.npy')
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    FS = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:1]
    sd0 = int(np.load(FS[0])['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
         for q in range(32)]
    with torch.no_grad():
        wc, npair, rng = cut_budget91(W, dev)
        pr, _ = predict91(wc, npair, dev, rng=rng)
    pr = pr.cpu().numpy()
    rel = float(np.linalg.norm(pr - ref) / np.linalg.norm(ref))
    print(f'\nn_pairs = {npair}')
    print(f'⛔ ПЛОМБА ПРОТИ ЖИВОГО `estimator_sub91.py`: '
          f'відносна різниця {rel:.3e} '
          f'{"✅" if rel < 3e-5 else "🚨 ПОРТ РОЗІЙШОВСЯ З САБМІТОМ"}')
    print(f'   норми: порт {np.linalg.norm(pr):.6f} · сабміт '
          f'{np.linalg.norm(ref):.6f}')
