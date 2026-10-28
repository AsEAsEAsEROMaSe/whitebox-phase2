"""СВІП ПІЛОТА: чи падає зсув синтетичного старту як 1/P? (аудит С48)

ЗАГАДКА З lowrank_ablate2.py: база майже не поліпшилась із пілотом
(P=1000 -> 3.15e-5 ; P=4000 -> 2.67e-5), хоча Var(m_hat) ∝ 1/P мала б дати 4×.
Якщо похибка СИСТЕМАТИЧНА — збільшувати пілот безглуздо, і треба шукати джерело зсуву.

ПІДОЗРА НА СТАРИЙ СТЕНД: там `manual_seed(70000+done)`, done=0 для всіх P ⟹ вибірка P=4000
МІСТИТЬ вибірку P=1000 (torch.randn заповнює по рядках) ⟹ два виміри сильно корельовані,
і це одна реалізація, а не оцінка середнього. Тут: НЕЗАЛЕЖНИЙ seed на кожен (P, повтор),
R повторів, звітуємо СЕРЕДНЄ.

ДІАГНОСТИКА, якої не було: поряд зі зсувом друкуємо ‖m_hat − m_true‖²/D.
  - якщо ‖Δm‖² падає як 1/P, а зсув НЕ падає -> зсув не від m, шукати деінде
  - якщо обидва падають як 1/P -> усе гаразд, стара цифра була однією реалізацією
  - якщо ‖Δm‖² НЕ падає як 1/P -> баг у пілоті

КОНТРОЛЬ ПІДЛОГИ ПРИЛАДУ: зсув «істинні m,K» на РІЗНИХ синтетичних сідах — щоб знати,
скільки з нього це просто MC-шум синтетичної вибірки, а скільки справжнє замикання.
"""
import torch, math, numpy as np, time
LOG = '/mnt/win_d/llm/White-Box/pilot_sweep.log'; T0 = time.time()
open(LOG, 'w').close()
def log(m):
    s = '[{:6.1f}s] {}'.format(time.time() - T0, m); print(s, flush=True)
    open(LOG, 'a').write(s + '\n')

dev = 'cuda:1'; D = 256; L = 32; MSYN = 200000; CH = 20000
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d' % (n, i)].astype(np.float32), device=dev) for i in range(L)]
def pre_at(W, x, l):
    a = x
    for i in range(l): a = torch.relu(a @ W[i])
    return a @ W[l]
def from_pre(W, h, l):
    a = torch.relu(h)
    for i in range(l + 1, L): a = torch.relu(a @ W[i])
    return a

def synth(W, l, mm, KK, r, TH, sseed=31000):
    ev, V = torch.linalg.eigh(KK); ev = ev.flip(0).clamp(min=0); V = V.flip(1)
    lam = ev[:r]; U = V[:, :r]
    resid = max(float(ev.sum()) - float(lam.sum()), 0.) / D
    S = torch.zeros(D, dtype=torch.float64, device=dev); k = 0
    for c in range(MSYN // CH):
        g = torch.Generator(device=dev).manual_seed(sseed + c)
        u = torch.randn(CH // 2, r, device=dev, generator=g, dtype=torch.float64)
        uu = torch.cat([u, -u], 0)
        h = mm.unsqueeze(0) + (uu * lam.sqrt().unsqueeze(0)) @ U.t()
        e = torch.randn(CH // 2, D, device=dev, generator=g, dtype=torch.float64)
        h = h + torch.cat([e, -e], 0) * math.sqrt(resid)
        S += from_pre(W, h.float(), l).double().sum(0); k += uu.shape[0]
    return float(((S / k - TH) ** 2).mean())

PS = [250, 1000, 4000, 16000, 64000]
R = 8
for net in (0, 1):
    W = load(net)
    S = torch.zeros(D, dtype=torch.float64, device=dev); n = 0
    for c in range(20):
        g = torch.Generator(device=dev).manual_seed(990000 + c)
        z = torch.randn(50000, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
        S += from_pre(W, pre_at(W, xx, 0), 0).double().sum(0); n += xx.shape[0]
    TH = S / n

    for l in (24, 31):
        log('=' * 92); log('МЕРЕЖА %d  ШАР %d' % (net, l))
        Sm = torch.zeros(D, dtype=torch.float64, device=dev)
        SC = torch.zeros(D, D, dtype=torch.float64, device=dev); m0 = 0
        for c in range(20):
            g = torch.Generator(device=dev).manual_seed(660000 + c)
            z = torch.randn(50000, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
            h = pre_at(W, xx, l).double(); Sm += h.sum(0); SC += h.t() @ h; m0 += xx.shape[0]
        mT = Sm / m0; KT = SC / m0 - torch.outer(mT, mT)

        # підлога приладу: та сама конфігурація, різні синтетичні сіди
        fl = [synth(W, l, mT, KT, 32, TH, sseed=31000 + 777 * j) for j in range(4)]
        log('  ПІДЛОГА (істинні m,K; 4 синтетичні сіди): {}  сер {:.3e}'
            .format(' '.join('{:.2e}'.format(v) for v in fl), sum(fl) / len(fl)))
        log('  {:>7s} {:>12s} {:>12s} {:>12s} {:>10s}'
            .format('P', 'зсув(база)', 'зсув(іст.m)', '‖Δm‖²/D', 'P·‖Δm‖²/D'))
        for P in PS:
            bb = []; bm = []; dm = []
            for rep in range(R):
                hs = []; done = 0
                gseed = 5_000_000 + 9973 * rep + 13 * P      # НЕЗАЛЕЖНИЙ на (P, повтор)
                while done < P:
                    b = min(CH, P - done)
                    g = torch.Generator(device=dev).manual_seed(gseed + done)
                    z = torch.randn(b // 2, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
                    hs.append(pre_at(W, xx, l).double()); done += b
                H = torch.cat(hs); mh = H.mean(0); Hc = H - mh
                Kh = (Hc.t() @ Hc) / H.shape[0]
                dm.append(float(((mh - mT) ** 2).mean()))
                bb.append(synth(W, l, mh, Kh, 32, TH))
                bm.append(synth(W, l, mT, Kh, 32, TH))
            mb, mm_, md = sum(bb) / R, sum(bm) / R, sum(dm) / R
            log('  {:7d} {:12.3e} {:12.3e} {:12.3e} {:10.4f}'.format(P, mb, mm_, md, P * md))
log('done.')
