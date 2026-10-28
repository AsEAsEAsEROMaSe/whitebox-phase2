"""ПОПРАВКА В БЕЗРОЗМІРНИХ КООРДИНАТАХ — α-ОРТОГОНАЛЬНА ЗА ПОБУДОВОЮ. (С88)

## 📐 ТЕОРЕМА (пломба пройдена: похибка 0…3e-13)
Замикання `m_a = s(tΦ(t)+φ(t))`, `Ca = CΦΦᵀ + ½C²φφᵀ/(ss)` СТРОГО
позитивно-1-однорідне по стану `(m, C^{1/2})`. Зміряно: множення ваг шару 0 на
`c` дає вихід РІВНО `×c`; множення ВСІХ 32 шарів на 1.03 дає рівно `×1.03³²`.

⟹ 🔑 **УСІ масштабні збурення в УСІХ шарах згортаються в ОДИН множник на
виході — рівно ту одну ступінь свободи, якою володіє `_ALPHA`.** Це не
«майже», це тотожність. Тому 32 шарові скаляри дали ×1.05 не випадково.

⟹ 🔑 **РОЗКЛАД СТАНУ НА МАСШТАБ І ФОРМУ ТОЧНИЙ:** `(m,C) ↔ (s, t, R)`, де
`s_j=√C_jj` (масштаб), `t_j=m_j/s_j` і `R=C/(ss)` — БЕЗРОЗМІРНІ. `α` бачить
ЛИШЕ `s`. Поправка, сформульована в `(t,R)`, НЕ МОЖЕ конкурувати з `α`
ЗА ПОБУДОВОЮ — без жодної проєкції.

## 🎯 ЩО ПЕРЕВІРЯЄТЬСЯ
Та сама оракульна `γ`, подана ТРЬОМА способами:
```
gamF   δm_a = s·Σγ·G       розмірна поправка (як у всіх 4 мертвих гілках)
gamT   δt = δm_a/(s·Φ(t))  🆕 ТА САМА поправка, переведена в δt при s=const,
                              і крок ПЕРЕРАХОВАНИЙ повністю (m'=s·t', Ca(t'))
gamOL2 пошарова ⊥-проєкція  спосіб gcut87
```
і оракульна `σ` двома:
```
sigALL  σ̂ підставлена як є        (масштаб + форма разом)
sigSH   🆕 лише ФОРМА: σ̂/s поділене на геометричне середнє ⟹ рівень знято
```
⛔КОНТРОЛІ: чужа γ через δt · знак −γ через δt · чужа σ-форма.

## ⛔ПЛОМБИ
 P0 ⟨yA,yB⟩ крос-ціль. P1 `gamF` ≈ ×0.52 після α. P2 δt=0 ⟹ РІВНО база.
 P3 ⛔чужа/знак мусять провалитись. P4 частка клампу δt має бути мала.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (щоб не тлумачити заднім числом):
 (а) `gamT` побʼє `gamF` (×0.52) — шанс 80%, це майже тавтологія механізму.
 (б) `gamT` дасть ≥×1.10 проти бази після α — шанс 35%.
 (в) `gamT` побʼє пошарову ⊥ (`gamOL2` ≈ ×1.02) — шанс 45%.
 (г) `sigSH` дасть ≥×1.10 — шанс 30% (σ-гілка глухла чотири рази).
 ⚠️РИЗИК: `Φ(t)` мале при `t≪0` ⟹ δt роздувається; кламп може вбити ефект.
"""
import math, sys, time
import numpy as np

NETS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
LOG2N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT = sys.argv[3] if len(sys.argv) > 3 else '/tmp/calib88t.npz'
NS = 1 << LOG2N
BATCH = 8192
L_, W_ = 32, 256
DTMAX = 0.5
PHIMIN = 0.05
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: np.exp(-0.5 * x * x) * I2
from math import erfc as _me
_verfc = np.frompyfunc(_me, 1, 1)


def Phi(x):
    return 0.5 * _verfc(-x * 0.7071067811865476).astype(np.float64)


CLAMP = [0, 0]


def chain(W, G=None, SO=None, mode='base'):
    """mode: base · gamF · gamT · sigALL · sigSH · (з G/SO відповідно)."""
    C = W[0].T @ W[0]
    m = np.zeros(W_)
    for i in range(1, L_ + 1):
        s = np.sqrt(np.maximum(np.diagonal(C), 1e-30))
        if mode == 'sigALL':
            s = SO[i - 1]
        elif mode == 'sigSH':
            r = SO[i - 1] / s
            s = s * (r / math.exp(float(np.log(r).mean())))    # рівень знято
        t = m / s
        Ph, f = Phi(t), phi(t)
        if mode in ('gamF', 'gamT'):
            g3, g4 = G[i - 1]
            G3_, G4_ = -t * f, (t * t - 1.0) * f
            G6_ = (t ** 4 - 6.0 * t * t + 3.0) * f
            dm = s * ((g3 / 6.0) * G3_ + (g4 / 24.0) * G4_
                      + (g3 * g3 / 72.0) * G6_)
        else:
            g3 = g4 = np.zeros(W_); dm = np.zeros(W_)
        if mode == 'gamT':
            # 🔑 та сама поправка, але як зсув БЕЗРОЗМІРНОГО t при s=const
            dt = dm / (s * np.maximum(Ph, PHIMIN))
            CLAMP[0] += int((np.abs(dt) > DTMAX).sum()); CLAMP[1] += W_
            t = t + np.clip(dt, -DTMAX, DTMAX)
            m = s * t
            Ph, f = Phi(t), phi(t)
            dm = np.zeros(W_)
        m_a = m * Ph + s * f + dm
        if i == L_:
            return m_a
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
    rng = np.random.default_rng(seed)
    MOM = np.zeros((L_, 4, W_)); acc = np.zeros(W_); d = 0
    while d < n_half:
        b = min(BATCH, n_half - d) // 2
        z = rng.standard_normal((b, W_), dtype=np.float32)
        a = np.concatenate([z, -z], 0)
        for q in range(L_):
            h = a @ W32[q]; hd = h.astype(np.float64); p = hd
            for kk in range(4):
                MOM[q, kk] += p.sum(0); p = p * hd
            a = np.maximum(h, 0.0)
        acc += a.astype(np.float64).sum(0); d += 2 * b
    return acc / d, MOM / d


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [int(v) for v in sorted(SD)][:NETS]
N = len(FS)
print(__doc__.split('📣')[1], flush=True)
print(f'{N} мереж · 2^{LOG2N} · numpy CPU', flush=True)
TAGS = ('base', 'gamF', 'gamT', 'sigALL', 'sigSH',
        '⛔gamT чужа', '⛔gamT знак', '⛔sigSH чужа')
P = {t: np.zeros((N, W_)) for t in TAGS}
YA = np.zeros((N, W_)); YB = np.zeros((N, W_))
GB, SB = [], []
t0 = time.time()
for n, sd0 in enumerate(FS):
    W32 = [np.ascontiguousarray(WZ[have[sd0]][q], np.float32) for q in range(L_)]
    W = [w.astype(np.float64) for w in W32]
    a1, M1 = mc_half(W32, 8800 + 2 * n, NS // 2)
    a2, M2 = mc_half(W32, 8801 + 2 * n, NS // 2)
    YA[n], YB[n] = a1, a2
    MOM = 0.5 * (M1 + M2)
    G, SO = [], []
    for q in range(L_):
        m1, m2, m3, m4 = MOM[q]
        c2 = np.maximum(m2 - m1 * m1, 1e-30)
        c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        c4 = m4 - 4.0*m1*m3 + 6.0*m1*m1*m2 - 3.0*m1**4 - 3.0*c2*c2
        s_ = np.sqrt(c2)
        G.append((c3 / s_ ** 3, c4 / s_ ** 4)); SO.append(s_)
    GB.append(G); SB.append(SO)
    P['base'][n] = chain(W)
    P['gamF'][n] = chain(W, G=G, mode='gamF')
    P['gamT'][n] = chain(W, G=G, mode='gamT')
    P['sigALL'][n] = chain(W, SO=SO, mode='sigALL')
    P['sigSH'][n] = chain(W, SO=SO, mode='sigSH')
    P['⛔gamT знак'][n] = chain(W, G=[(-a, -b) for a, b in G], mode='gamT')
    print(f'  net{n} · {time.time()-t0:.0f}s', flush=True)
for n, sd0 in enumerate(FS):
    W = [np.asarray(WZ[have[sd0]][q], np.float64) for q in range(L_)]
    j = (n + 1) % N
    P['⛔gamT чужа'][n] = chain(W, G=GB[j], mode='gamT')
    P['⛔sigSH чужа'][n] = chain(W, SO=SB[j], mode='sigSH')

YM = 0.5 * (YA + YB); TTc = (YA * YB).sum(1); NRM = TTc.mean() / W_
B = P['base']; bb = (B * B).sum(1)


def loo(cols):
    k = len(cols)
    G_ = np.zeros((N, k, k)); H = np.zeros((N, k))
    for a in range(k):
        H[:, a] = (cols[a] * YM).sum(1)
        for b_ in range(k):
            G_[:, a, b_] = (cols[a] * cols[b_]).sum(1)
    out = np.zeros(N)
    for i in range(N):
        m = np.arange(N) != i
        A = G_[m].sum(0)
        th = np.linalg.solve(A + 1e-12 * np.trace(A) * np.eye(k), H[m].sum(0))
        out[i] = (th @ G_[i] @ th - 2 * th @ H[i] + TTc[i]) / W_
    return out.mean()


b0 = loo([B])
print(f'\n⛔ПЛОМБА клампу δt: {CLAMP[0]}/{CLAMP[1]} = {CLAMP[0]/max(CLAMP[1],1):.2%}')
print(f'⟨yA,yB⟩/W = {NRM:.5f} · база після α: віднос. {math.sqrt(b0/NRM):.4%}')
print(f'\n{"варіант":14s} {"сира%":>8s} {"A1 вага1":>9s} {"A3 α,β⊥":>9s}')
for t in TAGS:
    if t == 'base':
        continue
    Cc = P[t] - B
    kap = (Cc * B).sum(1) / bb
    CO = Cc - kap[:, None] * B
    r1 = loo([B + Cc]); r3 = loo([B, CO])
    rr = ((P[t] - YA) * (P[t] - YB)).mean()
    print(f'{t:14s} {math.sqrt(max(rr,0)/NRM):8.4%} x{b0/r1:8.3f} x{b0/r3:8.3f}')
np.savez_compressed(OUT, YA=YA, YB=YB, **{t: P[t] for t in TAGS})
print(f'\nзбережено {OUT} · {time.time()-t0:.0f}s')
