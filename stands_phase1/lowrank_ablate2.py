"""АБЛЯЦІЯ m ПРОТИ K НА ГЛИБИНІ 24/28 (аудит С48).

ЧОМУ ПЕРЕРОБКА. lowrank_ablate.py стояв на l=31, а там from_pre(W,h,31) = relu(h)
(цикл range(32,32) порожній). E[relu(h_j)] залежить ЛИШЕ від маргіналі h_j, тобто від
m_j і K_jj ⟹ ПОЗАДІАГОНАЛЬНА K структурно не могла вплинути ні на що. Стенд не міг
відповісти інакше. На l=24/28 лишається 7/3 шарів змішування ⟹ питання стає осмисленим.

НОВИЙ РОЗРІЗ (його не було): розділяємо K на ДІАГОНАЛЬ і ПОЗАДІАГОНАЛЬ.
  - істинна діаг + оцінена позадіаг
  - оцінена діаг + ІСТИННА позадіаг      <- ось це і є справжнє питання
Якщо позадіагональ нічого не дає і на l=24 — тоді ранг-редукція K справді марна,
і це вже висновок про мережу, а не про стенд.
"""
import torch, math, numpy as np, time
LOG = '/mnt/win_d/llm/White-Box/lowrank_ablate2.log'; T0 = time.time()
open(LOG, 'w').close()
def log(m):
    s = '[{:5.1f}s] {}'.format(time.time() - T0, m); print(s, flush=True)
    open(LOG, 'a').write(s + '\n')

dev = 'cuda:1'; D = 256; L = 32; M = 200000; CH = 20000
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

for net in (0, 1):
    W = load(net)
    # істинна ціль E[a^L] (спільна для всіх шарів)
    S = torch.zeros(D, dtype=torch.float64, device=dev); n = 0
    for c in range(20):
        g = torch.Generator(device=dev).manual_seed(990000 + c)
        z = torch.randn(50000, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
        S += from_pre(W, pre_at(W, xx, 0), 0).double().sum(0); n += xx.shape[0]
    TH = S / n

    for l in (24, 28, 31):
        log('=' * 88); log('МЕРЕЖА %d  ШАР %d   (лишається %d шарів змішування після relu)' % (net, l, L - l - 1))
        Sm = torch.zeros(D, dtype=torch.float64, device=dev)
        SC = torch.zeros(D, D, dtype=torch.float64, device=dev); m0 = 0
        for c in range(20):
            g = torch.Generator(device=dev).manual_seed(660000 + c)
            z = torch.randn(50000, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
            h = pre_at(W, xx, l).double(); Sm += h.sum(0); SC += h.t() @ h; m0 += xx.shape[0]
        mT = Sm / m0; KT = SC / m0 - torch.outer(mT, mT)

        def run(mm, KK, r, tag):
            ev, V = torch.linalg.eigh(KK); ev = ev.flip(0).clamp(min=0); V = V.flip(1)
            lam = ev[:r]; U = V[:, :r]
            resid = max(float(ev.sum()) - float(lam.sum()), 0.) / D
            Ssyn = torch.zeros(D, dtype=torch.float64, device=dev); k = 0
            for c in range(M // CH):
                g = torch.Generator(device=dev).manual_seed(31000 + c)
                u = torch.randn(CH // 2, r, device=dev, generator=g, dtype=torch.float64)
                uu = torch.cat([u, -u], 0)
                h = mm.unsqueeze(0) + (uu * lam.sqrt().unsqueeze(0)) @ U.t()
                e = torch.randn(CH // 2, D, device=dev, generator=g, dtype=torch.float64)
                h = h + torch.cat([e, -e], 0) * math.sqrt(resid)
                Ssyn += from_pre(W, h.float(), l).double().sum(0); k += uu.shape[0]
            b = float(((Ssyn / k - TH) ** 2).mean())
            log('    {:44s} зсув {:.3e}'.format(tag, b)); return b

        for P in (1000, 4000):
            hs = []; done = 0
            while done < P:
                b = min(CH, P - done)
                g = torch.Generator(device=dev).manual_seed(70000 + done)
                z = torch.randn(b // 2, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
                hs.append(pre_at(W, xx, l).double()); done += b
            H = torch.cat(hs); mh = H.mean(0); Hc = H - mh; Kh = (Hc.t() @ Hc) / H.shape[0]
            # розділення діагоналі та позадіагоналі
            K_trueDiag = Kh - torch.diag(torch.diagonal(Kh)) + torch.diag(torch.diagonal(KT))
            K_trueOff = KT - torch.diag(torch.diagonal(KT)) + torch.diag(torch.diagonal(Kh))
            log('  пілот P={:d}:'.format(P))
            run(mh, Kh, 32, 'оцінені m і K  (база)')
            run(mT, Kh, 32, 'ІСТИННЕ m + оцінена K')
            run(mh, KT, 32, 'оцінена m + ІСТИННА K')
            run(mh, K_trueDiag, 32, 'оцінена m + істинна ДІАГОНАЛЬ K')
            run(mh, K_trueOff, 32, 'оцінена m + ІСТИННА ПОЗАДІАГОНАЛЬ K')
        log('  контроль стелі:')
        run(mT, KT, 32, 'ІСТИННІ m і K, ранг 32')
        run(mT, KT, 256, 'ІСТИННІ m і K, ПОВНИЙ ранг')
log('done.')
