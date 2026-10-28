"""КОРЕЛЯЦІЇ МІЖ СЕМПЛАМИ НА ШАРІ 29: чи несуть вони ЗНАК похибки виходу.

════════ ЗВІДКИ ПИТАННЯ (С77, юзер) ════════
«А якщо в тупу шукати не середнє, а міряти кореляцію між всіма семплами на
всіх шарах?» → сфокусовано: ШАР 29 (там сидить приз — лідер = точна опора
на 29). Міряємо, чи кореляційна структура хмари семплів на глибокому шарі
знає щось про те, КУДИ гойднувся центр хмари на виході в ЦЬОМУ жеребі.

════════ ОЗНАКИ ЖЕРЕБУ (на шарах 8 / 24 / 29) ════════
маскові (патерни перемикання, «хто з ким вмикається»):
    jit  = mean_a f_a(1−f_a)      — дрижання гейтів
    sim  = mean_a f_a²            — середня попарна схожість масок семплів
    rare = частка a: 0<f_a<0.02   — рідко перемикні (носії похибки)
формові (кореляція АКТИВАЦІЙ між семплами = форма хмари):
    top  = λ₁/tr вибіркової коваріації
    pr   = (Σλ)²/Σλ²              — PR-ранг
крос-шарові по-семплові:
    link = corr по семплах(проєкція шару K на її середнє, те саме на виході)
    q2   = дисперсія по-семплової проєкції на середнє шару K

ЦІЛЬ: t_d = ⟨x̄₃₁(жереб) − сер_жеребів, û⟩, û = напрям середнього (масштабна
складова = 95%+ шуму, cos(m,u₁)=0.978, M49).

════════ ДВА ВИРОКИ ════════
A) СТЕЛЯ З МЕРЕЖЕВОЮ КАЛІБРОВКОЮ: corr(F, t) по жеребах усередині мережі
   (недоступно на грейдері — там 1 жереб; це верхня межа «чи є інформація»).
B) ГРЕЙДЕРНА ФОРМА: LOO ПО МЕРЕЖАХ, ridge t̂=β·F БЕЗ пер-мережевого
   центрування; виграш = mean(t−t̂)²/mean t² на випущеній мережі (geo-mean).

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (два закони С75) ════════
📣 формові (top, pr): |corr| ≤ 0.1 — для дзвоноподібної хмари ФОРМА ⊥ ЦЕНТР
   (та сама парність, що вбила фолдові ознаки й канал дисперсії `u`);
📣 маскові: |corr| ≤ 0.2, чисті на глибині (родина G), але порожні;
📣 link на 29: по-жеребна corr помітна (рідня q31, R²=0.347), АЛЕ
📣 LOO (грейдерна) ВСІХ ознак разом: ≤ підлоги 1.03× ⟹ вирок НУЛЬ.
🟢 ГЕЙТ «ЖИВА»: LOO ≥ 1.05× — тоді шукати конструкцію в сабміт.

🚨 ПЛОМБИ
  ✅ NEG-жереб: corr(F, roll(t по жеребах)) → мусить бути ≈0;
  ✅ NEG-мережа: у LOO ознаки СУСІДНЬОЇ мережі → виграш мусить бути ≤1.00×;
  ✅ пломба половин: знак топ-кореляції збігається на двох половинах жеребів;
  ✅ формові на шарі 8 ≈ 0 (гаусоподібність після вибілювання — парність).
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import cut_and_budget, W_                                # noqa: E402

L_ = 32
KS = (8, 24, 29)
FEAT = ('jit', 'sim', 'rare', 'top', 'pr', 'link', 'q2')
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
OUT = '/mnt/win_d/llm/White-Box/side66/corr29_raw.npz'


def one_draw(wc, npair, dev, seed):
    """Прогін одного жеребу: ознаки на шарах KS + середнє виходу."""
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cov = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xa = torch.relu(x @ (wh @ wc[0]))
    depth = len(wc)
    keep = {}
    for i in range(1, depth):
        p = xa @ wc[i]
        xa = torch.relu(p)
        if i in KS and i < depth - 1:
            keep[i] = xa.double()
    xL = xa.double()
    mL = xL.mean(0)
    uL = mL / mL.norm().clamp_min(1e-30)
    vL = (xL - mL) @ uL
    fv = {}
    for K, a in keep.items():
        f = (a > 0).double().mean(0)
        jit = float((f * (1.0 - f)).mean())
        sim = float((f * f).mean())
        rare = float(((f > 0) & (f < 0.02)).double().mean())
        am = a.mean(0)
        ac = a - am
        C = (ac.T @ ac) / float(a.shape[0])
        evK = torch.linalg.eigvalsh(C)
        tr = float(evK.sum())
        top = float(evK[-1]) / max(tr, 1e-30)
        pr = tr * tr / max(float(evK.pow(2).sum()), 1e-300)
        u = ac @ (am / am.norm().clamp_min(1e-30))
        link = float((u * vL).mean() /
                     (u.std() * vL.std()).clamp_min(1e-30))
        q2 = float((u * u).mean())
        fv[K] = [jit, sim, rare, top, pr, link, q2]
    return fv, mL.cpu().numpy()


def collect(nets, dev, ndraw):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]
    F = np.zeros((nets, ndraw, len(KS), len(FEAT)))
    ML = np.zeros((nets, ndraw, W_))
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                          device=dev) for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            for dr in range(ndraw):
                fv, mL = one_draw(wc, npair, dev,
                                  91_000_000 + 733 * n + dr)
                for ki, K in enumerate(KS):
                    if K in fv:
                        F[n, dr, ki] = fv[K]
                ML[n, dr, :len(mL)] = mL
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets} (npair={npair})', flush=True)
    np.savez(OUT, F=F, ML=ML, KS=np.array(KS), FEAT=np.array(FEAT))
    return F, ML


def analyse(F, ML):
    nets, ndraw = F.shape[:2]
    # ціль: масштабна похибка виходу жеребу
    M = ML.mean(1)                                   # (nets, W)
    U = M / np.linalg.norm(M, axis=1, keepdims=True).clip(1e-30)
    t = np.einsum('ndw,nw->nd', ML - M[:, None], U)  # (nets, ndraw)

    print('\n════ A) СТЕЛЯ З МЕРЕЖЕВОЮ КАЛІБРОВКОЮ: corr(F, t) по жеребах ════')
    print(f'{"шар":>4s} ' + ' '.join(f'{q:>7s}' for q in FEAT) + '     NEG')
    half_ok = None
    for ki, K in enumerate(KS):
        cells, negs, halves = [], [], []
        for qi in range(len(FEAT)):
            cs, ng, h1, h2 = [], [], [], []
            for n in range(nets):
                x, y = F[n, :, ki, qi], t[n]
                if np.std(x) < 1e-30 or np.std(x[:ndraw // 2]) < 1e-30 \
                        or np.std(x[ndraw // 2:]) < 1e-30:
                    continue
                cs.append(np.corrcoef(x, y)[0, 1])
                ng.append(np.corrcoef(x, np.roll(y, 1))[0, 1])
                h = ndraw // 2
                h1.append(np.corrcoef(x[:h], y[:h])[0, 1])
                h2.append(np.corrcoef(x[h:], y[h:])[0, 1])
            cells.append(np.median(cs) if cs else np.nan)
            negs.append(np.median(ng) if ng else np.nan)
            halves.append((np.median(h1), np.median(h2)) if h1 else (0.0, 0.0))
        if K == 29:
            qi_top = int(np.nanargmax([abs(c) for c in cells]))
            half_ok = halves[qi_top] + (FEAT[qi_top],)
        print(f'{K:4d} ' + ' '.join(f'{c:7.3f}' for c in cells)
              + f'   {np.median(negs):7.3f}')

    print('\n════ B) ГРЕЙДЕРНА ФОРМА: LOO ПО МЕРЕЖАХ, ridge, без центрування ════')
    X = F.reshape(nets, ndraw, -1)
    nf = X.shape[-1]
    ratios, ratios_neg = [], []
    for lo in range(nets):
        tr_idx = [n for n in range(nets) if n != lo]
        Xtr = X[tr_idx].reshape(-1, nf)
        ytr = t[tr_idx].reshape(-1)
        mu, sd = Xtr.mean(0), Xtr.std(0).clip(1e-30)
        Xs = (Xtr - mu) / sd
        G = Xs.T @ Xs + 1e-3 * len(ytr) * np.eye(nf)
        beta = np.linalg.solve(G, Xs.T @ ytr)
        b0 = ytr.mean()
        Xte = (X[lo] - mu) / sd
        pred = Xte @ beta + b0
        ratios.append(np.mean((t[lo] - pred) ** 2) / np.mean(t[lo] ** 2))
        Xne = (X[(lo + 1) % nets] - mu) / sd
        pne = Xne @ beta + b0
        ratios_neg.append(np.mean((t[lo] - pne) ** 2) / np.mean(t[lo] ** 2))
    g = float(np.exp(np.mean(np.log(ratios))))
    gn = float(np.exp(np.mean(np.log(ratios_neg))))
    print(f'виграш LOO = {1.0 / g:.4f}×   (частка MSE, geo-mean; >1 = знімає)')
    print(f'NEG-мережа  = {1.0 / gn:.4f}×  (мусить бути ≤1.00)')

    # ═ оракульна колонка: пер-мережеве центрування ознак (НЕдоступне грейдеру)
    Xc = X - X.mean(1, keepdims=True)
    tc = t - t.mean(1, keepdims=True)
    rat_o = []
    for lo in range(nets):
        tr_idx = [n for n in range(nets) if n != lo]
        Xtr = Xc[tr_idx].reshape(-1, nf)
        ytr = tc[tr_idx].reshape(-1)
        sd = Xtr.std(0).clip(1e-30)
        Xs = Xtr / sd
        G = Xs.T @ Xs + 1e-3 * len(ytr) * np.eye(nf)
        beta = np.linalg.solve(G, Xs.T @ ytr)
        pred = (Xc[lo] / sd) @ beta
        rat_o.append(np.mean((tc[lo] - pred) ** 2) / np.mean(tc[lo] ** 2))
    go = float(np.exp(np.mean(np.log(rat_o))))
    print(f'ОРАКУЛ (центрування по мережі) = {1.0 / go:.4f}× '
          f'— стеля, грейдеру недоступна')

    if half_ok is not None:
        s1, s2, qn = half_ok
        print(f'\n✅ПЛОМБА ПОЛОВИН (топ-ознака шару 29 = {qn}): '
              f'{s1:+.3f} / {s2:+.3f} — знак '
              f'{"ЗБІГСЯ" if s1 * s2 > 0 else "НЕ ЗБІГСЯ"}')
    print('\n🔑 ВИРОК: гейт «жива» = LOO ≥ 1.05× '
          f'(підлога шуму 1.03×); зміряно {1.0 / g:.4f}×')


def main():
    nets = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    print(f'📣 ДО ПРОГОНУ: формові ≈0 (парність) · маскові ≤0.2 і порожні · '
          f'link(29) помітний по жеребах · LOO ≤1.03× ⟹ НУЛЬ; гейт 1.05×')
    print(f'   {nets} мереж × {NDRAW} жеребів, шари {KS}, ціль — масштабна '
          f'похибка виходу\n')
    F, ML = collect(nets, DEV, NDRAW)
    analyse(F, ML)


if __name__ == '__main__':
    main()
