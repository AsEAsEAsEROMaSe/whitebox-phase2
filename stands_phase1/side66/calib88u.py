"""ЗБІР ВЕКТОРІВ ДЛЯ ПЕРЕФОРМУЛЮВАННЯ КАЛІБРОВКИ — numpy/CPU. (С88)

🚨 ЧОМУ CPU: `libtorch_cuda.so` = 1.07 ГБ на томі ntfs-3g; під час сесії машина
в свопі (18/31 ГБ зайнято чужою задачею), і імпорт torch віддавав 25 КБ/с —
за 13 хв не дочитав. numpy на 2 потоках дає 169 GFLOPS fp32, чого досить:
2^18 семплів × 32 шари ≈ 22 c/мережу.

Один стенд знімає ВСІ кінці закономірності «⊥ працює для m_a, не працює для C»
на ОДНИХ І ТИХ САМИХ мережах, семплах і істині.

## 🚨 КРОС-ЦІЛЬ
Істина ділиться на ДВІ НЕЗАЛЕЖНІ половини `yA`,`yB`. Оцінка
`MSE = ⟨p−yA, p−yB⟩/W` НЕЗМІЩЕНА (шум МК має нульове середнє і незалежний
між половинами) ⟹ скінченна вибірка НЕ роздуває MSE і не стискає відношення.
Пломба: `⟨yA,yB⟩` мусить дати `RMS(істина)² = 0.88404`.

## ЩО ЗБИРАЄТЬСЯ (на мережу)
    base     чисте замикання Мелера          gamF/gam2   γ всюди / з маскою |t|<2
    gamOL2   γ|t|<2 з ПОШАРОВОЮ ⊥            gamOL99     γ всюди з ПОШАРОВОЮ ⊥
    gamX     ⛔γ ЧУЖОЇ мережі                 gamN        ⛔−γ
    sigALL   оракульна σ в УСЮ рекурсію      sigMA       σ як поправка до m_a щошару
    sigMAO   те саме з пошаровою ⊥           sigOUT/OU1  σ ЛИШЕ у виході (точна/1-й пор.)
    sigOUX   ⛔σ ЧУЖОЇ мережі у виході
    MCP      NREP чистих МК по NMC семплів — проксі `ours` для бленду
    VA/VC    Var(a_last) з МК / ПЕРЕДБАЧЕНА замиканням (безкоштовна в сабміті)
    TL/SL    t,s останнього шару із замикання — предиктори зміщення

## ⛔ПЛОМБИ
 P0 ⟨yA,yB⟩/W = 0.88404 (прилад).
 P1 `gamF` після α ≈ ×0.52 (gamalpha87)          P2 `gamOL2` після α ≈ ×1.02 (gcut87)
 P3 `sigMA` ≈ ×0.96 (layerk87 L6)                P4 `sigOUT` ≈ ×1.21
 Якщо P1..P4 не відтворюються — числа знімались на РІЗНИХ об'єктах.
"""
import math, sys, time
import numpy as np

NETS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
LOG2N = int(sys.argv[2]) if len(sys.argv) > 2 else 18
OUT = sys.argv[3] if len(sys.argv) > 3 else '/mnt/win_d/llm/White-Box/side66/calib88.npz'
NS = 1 << LOG2N          # СУМАРНО на дві половини
BATCH = 8192
L_, W_ = 32, 256
NMC = 6000
NREP = 8
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: np.exp(-0.5 * x * x) * I2
from math import erfc as _me
_verfc = np.frompyfunc(_me, 1, 1)   # stdlib, БЕЗ scipy: імпорт scipy з ntfs-3g
                                    # під час сесії віддавав 53 КБ/с і вішав стенд


def Phi(x):
    return 0.5 * _verfc(-x * 0.7071067811865476).astype(np.float64)


def chain(W, G=None, cut=None, SO=None, smode='none', ortho=False, ret_ts=False):
    """Замикання Мелера з опційними поправками (див. докстрінг модуля)."""
    C = W[0].T @ W[0]
    m = np.zeros(W_)
    for i in range(1, L_ + 1):
        s_own = np.sqrt(np.maximum(np.diagonal(C), 1e-30))
        last = (i == L_)
        s = s_own
        if SO is not None and (smode == 'all' or (last and smode == 'out')):
            s = SO[i - 1]
        elif SO is not None and smode == 'shape':
            # 🆕 ЛИШЕ ФОРМА `σ`: відношення σ̂/s ділиться на своє геометричне
            #    середнє ⟹ РІВЕНЬ (те, чим володіє α) знято ТОЧНО.
            r = SO[i - 1] / s_own
            s = s_own * (r / math.exp(float(np.log(r).mean())))
        t = m / s
        Ph, f = Phi(t), phi(t)
        m_b = m * Ph + s * f
        corr = np.zeros(W_)
        if G is not None:
            g3, g4 = G[i - 1]
            if cut is not None:
                msk = np.abs(t) < cut
                g3 = np.where(msk, g3, 0.0); g4 = np.where(msk, g4, 0.0)
            G3_, G4_ = -t * f, (t * t - 1.0) * f
            G6_ = (t ** 4 - 6.0 * t * t + 3.0) * f
            corr = corr + s * ((g3 / 6.0) * G3_ + (g4 / 24.0) * G4_
                               + (g3 * g3 / 72.0) * G6_)
        else:
            g3 = g4 = np.zeros(W_)
        if SO is not None and (smode == 'ma' or (last and smode == 'out1')):
            corr = corr + f * (SO[i - 1] - s_own)
        if smode == 'tshift':
            # 🆕 ТА САМА γ-поправка, але як зсув БЕЗРОЗМІРНОГО `t` при s=const.
            #    ∂m_a/∂t|_s = s·Φ(t) ⟹ δt = δm_a/(s·Φ). Далі крок рахується
            #    ПОВНІСТЮ заново зі стану (s, t+δt) — тобто поправка живе в
            #    координаті, якої `α` не бачить ЗА ПОБУДОВОЮ (1-однорідність).
            dt = np.clip(corr / (s * np.maximum(Ph, 0.05)), -0.5, 0.5)
            t = t + dt; m = s * t
            Ph, f = Phi(t), phi(t)
            m_b = m * Ph + s * f
            corr = np.zeros(W_)
        if ortho:                     # ⊥ ПОШАРОВО — рівно як у gcut87
            den = float(m_b @ m_b)
            if den > 1e-30:
                corr = corr - m_b * (float(corr @ m_b) / den)
        m_a = m_b + corr
        if last:
            if not ret_ts:
                return m_a
            q2l = (m * m + s * s) * Ph + m * s * f
            return m_a, t, s, np.maximum(q2l - m_a * m_a, 1e-30)
        s2 = s * s
        q2 = ((m * m + s2) * Ph + m * s * f + (g3 / 3.0) * s2 * f
              - (g4 / 12.0) * s2 * t * f
              + (g3 * g3 / 36.0) * s2 * (3.0 * t - t ** 3) * f)
        Ca = C * np.outer(Ph, Ph) + (0.5 * (C * C)) * np.outer(f / s, f / s)
        np.fill_diagonal(Ca, np.maximum(q2 - m_a * m_a, 1e-30))
        m = m_a @ W[i]
        C = W[i].T @ (Ca @ W[i])
    raise RuntimeError


def mc_half(W32, seed, n_half):
    """Половина істини + 4 моменти передактивацій + Var(a_last). Антитетика."""
    rng = np.random.default_rng(seed)
    MOM = np.zeros((L_, 4, W_)); acc = np.zeros(W_); acc2 = np.zeros(W_); d = 0
    while d < n_half:
        b = min(BATCH, n_half - d) // 2
        z = rng.standard_normal((b, W_), dtype=np.float32)
        a = np.concatenate([z, -z], 0)
        for q in range(L_):
            h = a @ W32[q]
            hd = h.astype(np.float64); p = hd
            for kk in range(4):
                MOM[q, kk] += p.sum(0); p = p * hd
            a = np.maximum(h, 0.0)
        ad = a.astype(np.float64)
        acc += ad.sum(0); acc2 += (ad * ad).sum(0); d += 2 * b
    return acc / d, acc2 / d, MOM / d, d


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [int(v) for v in sorted(SD)][:NETS]
N = len(FS)
print(f'{N} мереж · 2^{LOG2N} (дві половини) · numpy CPU', flush=True)

TAGS = ('base', 'gamF', 'gam2', 'gamX', 'gamN', 'gamOL2', 'gamOL99',
        'sigALL', 'sigMA', 'sigMAO', 'sigOUT', 'sigOU1', 'sigOUX', 'gamT', 'sigSH')
P = {t: np.zeros((N, W_)) for t in TAGS}
YA = np.zeros((N, W_)); YB = np.zeros((N, W_))
MCP = np.zeros((N, NREP, W_)); VA = np.zeros((N, W_))
TL = np.zeros((N, W_)); SL = np.zeros((N, W_)); VC = np.zeros((N, W_))
GBANK, SBANK = [], []
t0 = time.time()
for n, sd0 in enumerate(FS):
    W32 = [np.ascontiguousarray(WZ[have[sd0]][q], np.float32) for q in range(L_)]
    W = [w.astype(np.float64) for w in W32]
    # 🚨 ОРАКУЛ — З НЕЗАЛЕЖНОГО БЛОКУ. Інакше шум γ,σ корелює з шумом істини
    #    і крос-ціль перестає бути незміщеною (самофіт, [[feedback ... його ж виходами]]).
    _, _, MOM, _ = mc_half(W32, 5500 + n, NS // 2)
    a1, q1, M1, d1 = mc_half(W32, 8800 + 2 * n, NS // 4)
    a2, q2_, M2, d2 = mc_half(W32, 8801 + 2 * n, NS // 4)
    YA[n], YB[n] = a1, a2
    VA[n] = 0.5 * (q1 + q2_) - (0.5 * (a1 + a2)) ** 2
    rng = np.random.default_rng(4400 + n)
    for r in range(NREP):
        z = rng.standard_normal((NMC // 2, W_), dtype=np.float32)
        a = np.concatenate([z, -z], 0)
        for q in range(L_):
            a = np.maximum(a @ W32[q], 0.0)
        MCP[n, r] = a.astype(np.float64).mean(0)
    G, SO = [], []
    for q in range(L_):
        m1, m2, m3, m4 = MOM[q]
        c2 = np.maximum(m2 - m1 * m1, 1e-30)
        c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        c4 = m4 - 4.0*m1*m3 + 6.0*m1*m1*m2 - 3.0*m1**4 - 3.0*c2*c2
        s_ = np.sqrt(c2)
        G.append((c3 / s_ ** 3, c4 / s_ ** 4)); SO.append(s_)
    GBANK.append(G); SBANK.append(SO)
    b0, t0_, s0_, va0 = chain(W, ret_ts=True)
    P['base'][n] = b0; TL[n] = t0_; SL[n] = s0_; VC[n] = va0
    P['gamF'][n] = chain(W, G=G)
    P['gam2'][n] = chain(W, G=G, cut=2.0)
    P['gamN'][n] = chain(W, G=[(-a, -b) for a, b in G])
    P['gamOL2'][n] = chain(W, G=G, cut=2.0, ortho=True)
    P['gamOL99'][n] = chain(W, G=G, ortho=True)
    P['sigALL'][n] = chain(W, SO=SO, smode='all')
    P['sigMA'][n] = chain(W, SO=SO, smode='ma')
    P['sigMAO'][n] = chain(W, SO=SO, smode='ma', ortho=True)
    P['sigOUT'][n] = chain(W, SO=SO, smode='out')
    P['sigOU1'][n] = chain(W, SO=SO, smode='out1')
    P['gamT'][n] = chain(W, G=G, smode='tshift')
    P['sigSH'][n] = chain(W, SO=SO, smode='shape')
    yb = 0.5 * (a1 + a2)
    e = math.sqrt(float(((b0 - a1) * (b0 - a2)).mean() / (a1 @ a2 / W_)))
    print(f'  net{n} база {e:.4%} · {time.time()-t0:.0f}s', flush=True)

for n, sd0 in enumerate(FS):                       # ⛔контролі з чужою інформацією
    W = [np.asarray(WZ[have[sd0]][q], np.float64) for q in range(L_)]
    j = (n + 1) % N
    P['gamX'][n] = chain(W, G=GBANK[j])
    P['sigOUX'][n] = chain(W, SO=SBANK[j], smode='out')
print('контролі готові', flush=True)

np.savez_compressed(OUT, YA=YA, YB=YB, seeds=np.array(FS), MCP=MCP, VA=VA,
                    TL=TL, SL=SL, VC=VC, NMC=np.array(NMC),
                    **{t: P[t] for t in TAGS})
print(f'збережено {OUT} · {time.time()-t0:.0f}s', flush=True)
tt = float((YA * YB).sum(1).mean() / W_)
print(f'⛔ПЛОМБА P0: ⟨yA,yB⟩/W = {tt:.5f}  (мусить бути 0.88404)')
for t in TAGS:
    mse = float(((P[t] - YA) * (P[t] - YB)).mean())
    print(f'  {t:8s} крос-MSE {mse:.4e} · віднос. {math.sqrt(max(mse,0)/tt):.4%}')
