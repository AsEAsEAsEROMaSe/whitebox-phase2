"""ОРТОГОНАЛІЗОВАНА γ З ВИБІРКИ — ШЛЯХ ДО САБМІТА. (С87)

## ЗВІДКИ
`gcut87` (оракульна γ): пряма γ у ланцюгу = ×0.521 після α, а ОРТОГОНАЛІЗОВАНА
до базового прогнозу = **×1.618**. Механізм виведено: `γ` несе МАСШТАБ+ФОРМУ;
масштаб дублює `_ALPHA` і програє їй (бо діє неоднорідно по мережах), форма
губиться в цьому конфлікті. Проєкція в ортогональне доповнення лишає чисту
форму — те, чого один скаляр `α` дати НЕ МОЖЕ в принципі.

## 🎯 ЩО ТУТ ПЕРЕВІРЯЄТЬСЯ — ЧИ ПЕРЕЖИВЕ ЦЕ РЕАЛЬНУ γ
Оракульна γ недосяжна. Але ВИБІРКОВА доступна безкоштовно: сабміт уже жене
6000 антитетичних семплів, з них `γ̂₃,γ̂₄` беруться однією редукцією на шар.
```
точність γ̂₃ при N=6000:  √(6/N) ≈ 3.2%
вимога з ТЗ (ladder87):   3%      ⟹ ВПРИТУЛ
```
🚨 Гібрид із вибірковою `γ̂` уже провалювався (×0.26,
`finding-sample-moments-into-the-closure-recursion-beats-sub97`) — АЛЕ там γ̂
йшла БЕЗ ортогоналізації, тобто в режимі конкуренції з α. Це інший об'єкт.

## ⛔ПЛОМБИ
 P1 оракульна γ з ортогоналізацією мусить відтворити ×1.6 із `gcut87`.
 P2 ⛔ЧУЖА γ̂ (з іншої мережі) — мусить бути помітно гіршою.
 P3 ⛔ЗНАК: `−γ̂` мусить погіршити.
 P4 α-калібровка LOO по мережах скрізь.
 P5 бленд із прямою МК — вага LOO по мережах (як `_W_BLEND`).

📣 ПЕРЕДБАЧЕННЯ:
 (а) вибіркова γ̂ дасть менше за оракульну, але ≥×1.15. Шанс 45%.
 (б) після бленду з МК виграш проти повної бази ≥×1.05. Шанс 30%.
 (в) ⛔чужа γ̂ буде помітно гіршою. Шанс 80%.
"""
import glob, math, sys
import numpy as np, torch
DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NSAMP = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
NTRUE = 1 << (int(sys.argv[4]) if len(sys.argv) > 4 else 20)
BATCH = 1 << 13; L_, W_ = 32, 256; CUT = 2.0
assert DEV not in ('cuda:0', 'cuda:4')
print(__doc__.split('📣')[1], flush=True)
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
SUB97 = 1.873e-06


def chain(W, G=None, cut=CUT, ortho=True):
    C = W[0].T @ W[0]
    m = torch.zeros(W_, dtype=torch.float64, device=W[0].device)
    for i in range(1, L_ + 1):
        s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        t = m / s; P, f = Phi(t), phi(t)
        if G is not None:
            g3, g4 = G[i - 1]
            msk = torch.abs(t) < cut
            g3 = torch.where(msk, g3, torch.zeros_like(g3))
            g4 = torch.where(msk, g4, torch.zeros_like(g4))
        else:
            g3 = g4 = torch.zeros_like(s)
        G3_, G4_ = -t * f, (t * t - 1.0) * f
        G6_ = (t ** 4 - 6.0 * t * t + 3.0) * f
        m_b = s * (t * P + f)
        corr = s * ((g3 / 6.0) * G3_ + (g4 / 24.0) * G4_ + (g3 * g3 / 72.0) * G6_)
        if ortho:
            den = float((m_b * m_b).sum())
            if den > 1e-30:
                corr = corr - m_b * (float((corr * m_b).sum()) / den)
        m_a = m_b + corr
        if i == L_:
            return m_a
        s2 = s * s
        q2 = ((m * m + s2) * P + m * s * f + (g3 / 3.0) * s2 * f
              - (g4 / 12.0) * s2 * t * f
              + (g3 * g3 / 36.0) * s2 * (3.0 * t - t ** 3) * f)
        Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30) - torch.diagonal(Ca))
        m = m_a @ W[i]; C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · N={NSAMP} · cut={CUT} · {DEV}\n', flush=True)
TAGS = ('база', 'ОРАКУЛ ⊥γ', 'ВИБІРКА ⊥γ̂', '⛔ЧУЖА ⊥γ̂', '⛔ЗНАК −γ̂', 'пряма МК')
P = {t: np.zeros((len(FS), W_)) for t in TAGS}
TRUE_ = np.zeros((len(FS), W_)); GS = []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV) for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(4100 + n); d = 0
        while d < NTRUE:
            b = min(BATCH, NTRUE - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]; hd = h.double(); p = hd
                for kk in range(4):
                    MOM[q, kk] += p.sum(0); p = p * hd
                a = torch.relu(h)
            acc += a.sum(0, dtype=torch.float64); d += b
        TRUE_[n] = (acc / NTRUE).cpu().numpy(); MOM /= NTRUE
        GO = []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0*m1*m2 + 2.0*m1**3
            c4 = m4 - 4.0*m1*m3 + 6.0*m1*m1*m2 - 3.0*m1**4 - 3.0*c2*c2
            s_ = torch.sqrt(c2); GO.append((c3/s_**3, c4/s_**4))
        # ── ВИБІРКОВА γ̂ на N=6000 з антитетикою (як у сабміті) ──
        gw = torch.Generator(device=DEV); gw.manual_seed(5100 + n)
        z = torch.randn(NSAMP // 2, W_, device=DEV, generator=gw)
        a = torch.cat([z, -z], 0); MS_ = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        for q in range(L_):
            h = a @ W32[q]; hd = h.double(); p = hd
            for kk in range(4):
                MS_[q, kk] += p.sum(0); p = p * hd
            a = torch.relu(h)
        P['пряма МК'][n] = a.double().mean(0).cpu().numpy()
        MS_ /= NSAMP
        GH = []
        for q in range(L_):
            m1, m2, m3, m4 = MS_[q]
            c2 = torch.clamp(m2 - m1*m1, min=1e-30)
            c3 = m3 - 3.0*m1*m2 + 2.0*m1**3
            c4 = m4 - 4.0*m1*m3 + 6.0*m1*m1*m2 - 3.0*m1**4 - 3.0*c2*c2
            s_ = torch.sqrt(c2); GH.append((c3/s_**3, c4/s_**4))
        GS.append([(x.cpu(), y.cpu()) for x, y in GH])
        P['база'][n] = chain(W, None).cpu().numpy()
        P['ОРАКУЛ ⊥γ'][n] = chain(W, GO).cpu().numpy()
        P['ВИБІРКА ⊥γ̂'][n] = chain(W, GH).cpu().numpy()
        P['⛔ЗНАК −γ̂'][n] = chain(W, [(-x, -y) for x, y in GH]).cpu().numpy()
    print(f'  net{n}', flush=True)
for n in range(len(FS)):
    W32, W = None, None
    sd0 = int(np.load(FS[n])['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float64), device=DEV) for q in range(L_)]
    GX = [(x.to(DEV), y.to(DEV)) for x, y in GS[(n + 1) % len(FS)]]
    with torch.no_grad():
        P['⛔ЧУЖА ⊥γ̂'][n] = chain(W, GX).cpu().numpy()


def am(t):
    d = P[t] - TRUE_; pt = (P[t]*TRUE_).sum(1); pp = (P[t]*P[t]).sum(1)
    tt = (TRUE_*TRUE_).sum(1); a_n = pt/np.maximum(pp,1e-30); out=np.zeros(len(FS))
    for i in range(len(FS)):
        msk = np.arange(len(FS)) != i
        a = a_n[msk].mean() if len(FS)>1 else a_n[i]
        out[i] = (a*a*pp[i] - 2*a*pt[i] + tt[i])/W_
    return out.mean()


def blend(A, B):
    dA, dB = A-TRUE_, B-TRUE_
    w_n = (dB*(dB-dA)).sum(1)/np.maximum(((dB-dA)**2).sum(1),1e-30)
    out=np.zeros(len(FS)); ws=np.zeros(len(FS))
    for i in range(len(FS)):
        msk = np.arange(len(FS))!=i
        w = float(np.clip(w_n[msk].mean(),-0.5,1.5)) if len(FS)>1 else float(w_n[i])
        ws[i]=w; r = w*dA[i]+(1-w)*dB[i]; out[i]=float((r*r).mean())
    return out.mean(), ws.mean()


b = am('база')
print(f'\n{"="*72}\nЛАНЦЮГ ПІСЛЯ α (без бленду)')
print(f'{"варіант":16s} {"ПІСЛЯ α":>11s} {"vs база":>9s}')
for t in TAGS[:-1]:
    print(f'{t:16s} {am(t):11.3e} {"x%.3f" % (b/am(t)):>9s}')
print(f'\n{"="*72}\nПІСЛЯ БЛЕНДУ З ПРЯМОЮ МК (N={NSAMP}) — ЦЕ ЧИСЛО ЙДЕ В БАЛ')
mk = P['пряма МК']
base_b, wb = blend(P['база'], mk)
print(f'{"варіант":16s} {"MSE бленду":>12s} {"w":>7s} {"vs повна база":>14s} {"бал":>10s}')
print(f'{"ПОВНА БАЗА":16s} {base_b:12.4e} {wb:7.4f} {"x1.000":>14s} {0.1*base_b:10.3e}')
for t in ('ОРАКУЛ ⊥γ', 'ВИБІРКА ⊥γ̂', '⛔ЧУЖА ⊥γ̂', '⛔ЗНАК −γ̂'):
    mb, w = blend(P[t], mk)
    print(f'{t:16s} {mb:12.4e} {w:7.4f} {"x%.3f" % (base_b/mb):>14s} {0.1*mb:10.3e}')
mv, _ = blend(P['ВИБІРКА ⊥γ̂'], mk); mx, _ = blend(P['⛔ЧУЖА ⊥γ̂'], mk)
print(f'\nВИРОК: вибіркова ⊥γ̂ проти повної бази = x{base_b/mv:.4f}')
print(f'   ⛔чужа x{base_b/mx:.4f} · sub97 = 1.873e-07')
