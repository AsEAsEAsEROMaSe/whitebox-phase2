"""ЗБІР ВЕКТОРІВ ДЛЯ ПЕРЕФОРМУЛЮВАННЯ КАЛІБРОВКИ. (С88)

Один стенд знімає ВСІ кінці закономірності «⊥ працює для m_a, не працює для C»
на ОДНИХ І ТИХ САМИХ мережах, семплах і істині. Далі — офлайн-алгебра калібровок
(`calib88a.py`) на збережених векторах.

## ЩО ЗБИРАЄТЬСЯ (на мережу)
    y        істина E[a_last] (МК 2^LOG2N, антитетика)
    base     чисте замикання Мелера
    gamF     γ у всіх шарах (має відтворити ×0.52 після α)
    gam2     γ з маскою |t|<2
    gamX     ⛔γ ЧУЖОЇ мережі (контроль)
    gamN     ⛔−γ (контроль знаку)
    sigALL   оракульна σ підставлена в УСЮ рекурсію
    sigMA    оракульна σ як поправка до m_a на КОЖНОМУ шарі (=sorth87 S1)
    sigOUT   оракульна σ ЛИШЕ в останньому шарі (точна підстановка)
    sigOU1   те саме, перший порядок δm = φ(t)·δσ
    sigOUX   ⛔σ ЧУЖОЇ мережі лише у виході (контроль)

🚨 ЖОДНОЇ калібровки тут немає — вектори сирі. Уся α-алгебра в `calib88a.py`,
щоб схеми калібровки порівнювались на БІТ-ІДЕНТИЧНИХ прогнозах.

## ⛔ПЛОМБИ (перевіряються в calib88a)
 P1 `gamF` після α мусить дати ×0.52±0.05 (gcut87/gamalpha87).
 P2 `⊥gamF` після α мусить дати ×1.02±0.03 (gcut87).
 P3 `sigMA` мусить дати ×0.96±0.03 (layerk87 L6).
 P4 `sigOUT` мусить дати ×1.21±0.05 (табличка команди).
 Якщо P1..P4 не відтворюються — числа команди знімались на РІЗНИХ об'єктах,
 і закономірність «⊥ для m_a / не для C» будувати НЕ МОЖНА.
"""
import glob, math, sys, time
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
OUT = sys.argv[4] if len(sys.argv) > 4 else '/mnt/win_d/llm/White-Box/side66/calib88.npz'
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
assert DEV not in ('cuda:0', 'cuda:4')
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)


def chain(W, G=None, cut=None, SO=None, smode='none', ortho=False, ret_ts=False):
    """Замикання Мелера з опційними поправками.

    G/cut  — γ-поправка до `m_a` (і до q2), маска |t|<cut.
    SO     — цільова (оракульна) σ по шарах.
    smode  — 'all'  σ підставляється замість s УСЮДИ (t, m_a, Ca)
             'ma'   σ входить лише як поправка δm=φ(t)(σ̂−s) до m_a, кожен шар
             'out'  точна підстановка σ̂ ТІЛЬКИ в останньому шарі
             'out1' перший порядок δm=φ(t)(σ̂−s) ТІЛЬКИ в останньому шарі
    """
    C = W[0].T @ W[0]
    m = torch.zeros(W_, dtype=torch.float64, device=W[0].device)
    for i in range(1, L_ + 1):
        s_own = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        last = (i == L_)
        s = s_own
        if SO is not None and (smode == 'all' or (last and smode == 'out')):
            s = SO[i - 1]
        t = m / s
        P, f = Phi(t), phi(t)
        m_b = m * P + s * f
        corr = torch.zeros_like(m_b)
        if G is not None:
            g3, g4 = G[i - 1]
            if cut is not None:
                msk = torch.abs(t) < cut
                g3 = torch.where(msk, g3, torch.zeros_like(g3))
                g4 = torch.where(msk, g4, torch.zeros_like(g4))
            G3_, G4_ = -t * f, (t * t - 1.0) * f
            G6_ = (t ** 4 - 6.0 * t * t + 3.0) * f
            corr = corr + s * ((g3 / 6.0) * G3_ + (g4 / 24.0) * G4_
                               + (g3 * g3 / 72.0) * G6_)
        else:
            g3 = g4 = torch.zeros_like(s)
        if SO is not None and (smode == 'ma' or (last and smode == 'out1')):
            corr = corr + f * (SO[i - 1] - s_own)
        if ortho:      # ⊥ ПОШАРОВО — рівно як у gcut87 (пломба відтворення)
            den = float((m_b * m_b).sum())
            if den > 1e-30:
                corr = corr - m_b * (float((corr * m_b).sum()) / den)
        m_a = m_b + corr
        if last:
            if not ret_ts:
                return m_a
            # 🔑 Var(a) з САМОГО замикання — доступна в сабміті БЕЗКОШТОВНО
            q2l = (m * m + s * s) * P + m * s * f
            return m_a, t, s, torch.clamp(q2l - m_a * m_a, min=1e-30)
        s2 = s * s
        q2 = ((m * m + s2) * P + m * s * f + (g3 / 3.0) * s2 * f
              - (g4 / 12.0) * s2 * t * f
              + (g3 * g3 / 36.0) * s2 * (3.0 * t - t ** 3) * f)
        Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30)
                             - torch.diagonal(Ca))
        m = m_a @ W[i]
        C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
# 🚨 `pool3/net_NNNNN.npz` має seed = 6000000+NNNNN (перевірено), а `SD` уже
#    відсортований ⟹ набір і ПОРЯДОК мереж тотожні gcut87/sorth87/learnc87,
#    але без сканування 2454 файлів на дуже повільному томі (≈1 c/файл).
FS = [int(v) for v in sorted(SD)][:NETS]
N = len(FS)
print(f'{N} мереж · 2^{LOG2N} · {DEV}', flush=True)

NMC = 6000      # ≈ обсяг вибірки сабміта при util 0.0875
NREP = 8        # незалежних жеребів МК-проксі на мережу
TAGS = ('base', 'gamF', 'gam2', 'gamX', 'gamN', 'gamOL2', 'gamOL99',
        'sigALL', 'sigMA', 'sigMAO', 'sigOUT', 'sigOU1', 'sigOUX')
P = {t: np.zeros((N, W_)) for t in TAGS}
Y = np.zeros((N, W_))
MCP = np.zeros((N, NREP, W_))   # МК-проксі `ours` (чиста МК N=NMC, антитетика)
VA = np.zeros((N, W_))          # Var(a_last) з великої МК — предиктор шуму МК
TL = np.zeros((N, W_))          # t останнього шару із замикання — предиктор зміщення
SL = np.zeros((N, W_))          # s останнього шару із замикання
VC = np.zeros((N, W_))          # Var(a_last), ПЕРЕДБАЧЕНА замиканням (безкоштовна)
GBANK, SBANK, WSEED = [], [], []
t0 = time.time()
for n, sd0 in enumerate(FS):
    WSEED.append(sd0)
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    with torch.no_grad():
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        acc2 = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(8800 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d) // 2
            z = torch.randn(b, W_, device=DEV, generator=g)
            a = torch.cat([z, -z], 0)          # антитетика
            for q in range(L_):
                h = a @ W32[q]; hd = h.double(); p = hd
                for kk in range(4):
                    MOM[q, kk] += p.sum(0); p = p * hd
                a = torch.relu(h)
            ad = a.double()
            acc += ad.sum(0); acc2 += (ad * ad).sum(0); d += 2 * b
        Y[n] = (acc / d).cpu().numpy(); MOM /= d
        VA[n] = (acc2 / d - (acc / d) ** 2).cpu().numpy()
        # 🎲 МК-ПРОКСІ `ours`: NREP незалежних чистих МК по NMC семплів
        gm = torch.Generator(device=DEV); gm.manual_seed(4400 + n)
        for r in range(NREP):
            z = torch.randn(NMC // 2, W_, device=DEV, generator=gm)
            a = torch.cat([z, -z], 0)
            for q in range(L_):
                a = torch.relu(a @ W32[q])
            MCP[n, r] = (a.double().mean(0)).cpu().numpy()
        G, SO = [], []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q, 0], MOM[q, 1], MOM[q, 2], MOM[q, 3]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = m4 - 4.0*m1*m3 + 6.0*m1*m1*m2 - 3.0*m1**4 - 3.0*c2*c2
            s_ = torch.sqrt(c2)
            G.append((c3 / s_ ** 3, c4 / s_ ** 4)); SO.append(s_)
        GBANK.append([(a.cpu(), b.cpu()) for a, b in G])
        SBANK.append([x.cpu() for x in SO])
        W = [w.double() for w in W32]
        b0, t0_, s0_, va0_ = chain(W, ret_ts=True)
        P['base'][n] = b0.cpu().numpy()
        TL[n] = t0_.cpu().numpy(); SL[n] = s0_.cpu().numpy()
        VC[n] = va0_.cpu().numpy()
        P['gamF'][n] = chain(W, G=G, cut=None).cpu().numpy()
        P['gam2'][n] = chain(W, G=G, cut=2.0).cpu().numpy()
        P['gamN'][n] = chain(W, G=[(-a, -b) for a, b in G]).cpu().numpy()
        P['gamOL2'][n] = chain(W, G=G, cut=2.0, ortho=True).cpu().numpy()
        P['gamOL99'][n] = chain(W, G=G, cut=None, ortho=True).cpu().numpy()
        P['sigALL'][n] = chain(W, SO=SO, smode='all').cpu().numpy()
        P['sigMA'][n] = chain(W, SO=SO, smode='ma').cpu().numpy()
        P['sigMAO'][n] = chain(W, SO=SO, smode='ma', ortho=True).cpu().numpy()
        P['sigOUT'][n] = chain(W, SO=SO, smode='out').cpu().numpy()
        P['sigOU1'][n] = chain(W, SO=SO, smode='out1').cpu().numpy()
    e = math.sqrt(((P['base'][n] - Y[n]) ** 2).mean() / (Y[n] ** 2).mean())
    print(f'  net{n} база {e:.4%} · {time.time()-t0:.0f}s', flush=True)

# ⛔КОНТРОЛІ З ЧУЖОЮ ІНФОРМАЦІЄЮ (окремий прохід — потрібен сусід)
for n in range(N):
    W = [torch.tensor(np.asarray(WZ[have[WSEED[n]]][q], np.float64), device=DEV)
         for q in range(L_)]
    j = (n + 1) % N
    GX = [(a.to(DEV), b.to(DEV)) for a, b in GBANK[j]]
    SX = [x.to(DEV) for x in SBANK[j]]
    with torch.no_grad():
        P['gamX'][n] = chain(W, G=GX, cut=None).cpu().numpy()
        P['sigOUX'][n] = chain(W, SO=SX, smode='out').cpu().numpy()
print('контролі готові', flush=True)

np.savez_compressed(OUT, Y=Y, seeds=np.array(WSEED), MCP=MCP, VA=VA, TL=TL,
                    SL=SL, VC=VC, NMC=np.array(NMC), **{t: P[t] for t in TAGS})
print(f'збережено {OUT} · {time.time()-t0:.0f}s', flush=True)
for t in TAGS:
    e = math.sqrt(((P[t] - Y) ** 2).mean() / (Y ** 2).mean())
    print(f'  {t:8s} сира віднос. похибка {e:.4%}')
