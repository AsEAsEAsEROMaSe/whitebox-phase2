"""ПІДЛОГА ЗСУВУ v2 — чи є в нашого оцінювача систематична похибка, незалежна від N?

ПРИВІД: у arccos_cv.py колонка зсув² дала 2.3e-7 (нет0) при MSE 2.8e-6 — 8%, і однаково
для всіх конфігурацій CV, включно з ВИПАДКОВИМИ напрямками ⟹ зсув не від CV, а від САМОГО v2.
Власна похибка «істини» (1e6 семплів) пояснює лише ~1.8e-8 із них.

ЧОМУ ЦЕ КРИТИЧНО: якщо підлога зсуву ≈3e-7, то жодне зменшення дисперсії не опустить MSE
нижче неї. Лідери сидять на MSE 2.67e-7 — РІВНО там, де в нас підлога.

ГІПОТЕЗА: винне ZCA. Вибілювання силоміць робить емпіричну коваріацію вибірки = I, тобто
це НЕ вибірка з N(0,I), а детермінований її перекрут ⟹ зсув O(1/N).

ПЕРЕВІРКА:
  • зсув(N) для ZCA-антитетики при N = 1600 … 100000. Якщо ∝1/N — підпис ZCA.
  • те саме для ЧИСТОЇ антитетики (без ZCA) — там зсув має бути нуль.
  • істина рахується ЧИСТОЮ антитетикою на 4e6 семплів (незміщена за побудовою),
    її власна похибка віднімається аналітично.
⚠️ Зсув міряємо як |серед. по R повторах (est − TH)|², і ОКРЕМО друкуємо шумову підлогу
   цієї оцінки (MSE/R), щоб не переплутати зсув із недобором повторів.
"""
import torch, math, numpy as np, time

dev = 'cuda:4'; D = 256; L = 32; T0 = time.time()
def log(m): print('[{:6.1f}s] {}'.format(time.time() - T0, m), flush=True)
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d' % (n, i)].astype(np.float32), device=dev) for i in range(L)]
def fwd(W, x):
    a = x
    for i in range(L): a = torch.relu(a @ W[i])
    return a
def zca(x):
    C = (x.t() @ x) / x.shape[0]
    e, V = torch.linalg.eigh(C.double())
    return (V @ torch.diag(e.clamp(min=1e-8).rsqrt()) @ V.t()).float()

for net in (0, 1):
    W = load(net); log('=' * 86); log('МЕРЕЖА %d' % net)
    # ---------- ІСТИНА: чиста антитетика (незміщена), 4e6 семплів ----------
    S = torch.zeros(D, dtype=torch.float64, device=dev); n = 0
    for c in range(40):
        g = torch.Generator(device=dev).manual_seed(910000 + c)
        z = torch.randn(50000, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
        S += fwd(W, xx).double().sum(0); n += xx.shape[0]
    TH = S / n
    # дисперсія одного семпла — для шумових підлог
    g = torch.Generator(device=dev).manual_seed(1)
    z = torch.randn(50000, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
    yy = fwd(W, xx).double(); vper = float(yy.var(0).mean())
    th_err = vper / n
    del yy, xx, z
    log('  Var одного семпла = {:.4e};  власна похибка істини = {:.3e} (віднімається)'
        .format(vper, th_err))
    log('  {:>8s} {:>6s} {:>5s} {:>12s} {:>12s} {:>12s} {:>10s}'
        .format('схема', 'N', 'R', 'MSE', 'зсув² сир.', 'зсув² чист.', 'шум оцінки'))
    for scheme in ('ZCA', 'чиста'):
        for N, R in ((1600, 400), (6484, 240), (26000, 120), (100000, 60)):
            devs = []; mses = []
            for r in range(R):
                g = torch.Generator(device=dev).manual_seed(600000 + 7919 * r + 13 * N + net)
                z = torch.randn(N // 2, D, device=dev, generator=g)
                x = torch.cat([z, -z], 0)
                if scheme == 'ZCA': x = x @ zca(x)
                est = fwd(W, x).double().mean(0)
                d = (est - TH)
                devs.append(d.cpu().numpy()); mses.append(float((d * d).mean()))
            mse = float(np.mean(mses))
            raw = float((np.mean(np.stack(devs), axis=0) ** 2).mean())
            floor = mse / R                       # шумова підлога оцінки зсуву
            clean = max(raw - floor - th_err, 0.0)
            log('  {:>8s} {:6d} {:5d} {:12.4e} {:12.4e} {:12.4e} {:10.3e}'
                .format(scheme, N, R, mse, raw, clean, floor))
    log('  ⟹ якщо «зсув² чист.» падає ∝1/N² — це звичайний O(1/N) зсув оцінювача;')
    log('    якщо виходить на ПОЛИЦЮ — це справжня підлога, і вона обмежує все.')
log('done.')
