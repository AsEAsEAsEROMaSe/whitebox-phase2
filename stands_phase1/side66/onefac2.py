"""ЛАНЦЮГ 32 ШАРИ: ЗАМИКАННЯ З ФОРМОЮ ІЗ ФАКТОРНОЇ КВАДРАТУРИ. (С87)

## ЗВІДКИ
`side66/onefac.py` зміряв ОДИН крок і дав механізм:
```
FAC сам по собі  ×0.05–0.21  ⛔ бо похибка МАСШТАБУ `s_p` = 4–10% (проти 0.5–2% у замикання)
SC = форма FAC + масштаб замикання  ×1.03–1.33  ✅
SO = форма FAC + ОРАКУЛЬНИЙ масштаб  б'є навіть оракульний гаус
```
🔑 ВИРОК ОДНОГО КРОКУ: цінна лише ФОРМА (`γ₃`,`γ₄`), масштаб мусить лишитись
від Мелера. Це рівно [[finding-remove-beats-predict-across-30-verdicts]]
«40.9% = МАСШТАБ» і закон «ВАЖІЛЬ НА ДВІ ТОЧНОСТІ = ДОБУТОК».

## 📐 ЩО РОБИТЬ ЦЕЙ СТЕНД
Один крок ≠ 32 кроки: у ланцюгу похибка КОМПАУНДИТЬСЯ, і поправка на кожному
кроці може дати більше (або менше) за одношаровий вимір. Ведемо стан
`(m_g, C_g)` ПЕРЕДАКТИВАЦІЙ і на кожному кроці:
```
БАЗА:  m_a = m_g·Φ(t) + s·φ(t)                      ← гаусове замикання сабміта
FAC :  m_a = s·[G₀ + (γ₃/6)G₃ + (γ₄/24)G₄ + (γ₃²/72)G₆]
       де `γ₃,γ₄` — ПОВНІ кумулянти `p` із факторної суміші попереднього кроку
```
`C_a` в обох випадках — Мелер K=2 + точна діагональ (масштаб не чіпаємо).

## ⛔ПЛОМБИ
 C1 ШАР 1: `p₁ = W₀ᵀz` гаусів ТОЧНО ⟹ `κ₃=κ₄≈0` ⟹ обидва ланцюги мусять
    збігтися побітово на першому кроці. Це ловить помилку індексації.
 C2 ШУМ ІСТИНИ: половина МК проти повного.
 C3 ⛔КОНТРОЛЬ: `γ` від ЧУЖОЇ мережі ⟹ ланцюг мусить стати ГІРШИМ за базу.
 C4 РЕЖИМ ОБ'ЄКТА: `k` від 1 до 3 — якщо виграш не росте з `k`, форма впіймана
    вже одним фактором (і це здешевлює гілку вдесятеро).

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) БАЗА дасть 0.9–1.0% — саме це знає банк (`side66/clgeom.py`: 0.9645%).
     Якщо вийде інше — баг у приладі, а не результат. Шанс 80%.
 (б) FAC-ланцюг ≤0.7%: шанс **45%**. Один крок дав лише ×1.03–1.09 на глибині,
     але компаундинг може підсилити.
 (в) ⚠️РИЗИК: поправка форми входить у `m_a`, а `C_a` лишається Мелером ⟹
     ланцюг стає НЕУЗГОДЖЕНИМ (середнє точніше за коваріацію) і може
     РОЗІЙТИСЬ. Шанс, що FAC зробить ГІРШЕ: 30%.
 (г) Навіть при успіху ЦІНА: k=1,Q=24 ⟹ ~4 матмули (24,256)×(256,256) на шар
     = 4e8 флопів (0.15% бюджету) + ~35 викликів `fnp`/шар = 1120 викликів
     (0.54%). Разом ~0.7% бюджету ⟹ зрізати `util` на 8% ⟹ MSE +7%.
     🔑 ГЕЙТ: щоб окупитись через БЛЕНД (зараз `w`=0.0883, виграш ×1.097),
     похибка замикання мусить упасти щонайменше в **1.6×** (тоді `w`→0.20,
     виграш ×1.25, приріст ×1.14 > ціна ×1.07).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/onefac2.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NS = 1 << LOG2N
BATCH = 1 << 12
L_, W_ = 32, 256
KS = (1, 2, 3)
NQ = {1: 24, 2: 16, 3: 10}
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

SQ2PI = math.sqrt(2.0 * math.pi)
INV_SQ2PI = 1.0 / SQ2PI


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def relu_moments(mu, sg):
    t = mu / sg
    P, f = Phi(t), phi(t)
    s2 = sg * sg
    m1 = mu * P + sg * f
    m2 = (mu * mu + s2) * P + mu * sg * f
    m3 = mu * (mu * mu + 3.0 * s2) * P + sg * (mu * mu + 2.0 * s2) * f
    m4 = ((mu ** 4 + 6.0 * mu * mu * s2 + 3.0 * s2 * s2) * P
          + sg * mu * (mu * mu + 5.0 * s2) * f)
    return m1, m2, m3, m4


def edge_relu(m, s, g3, g4):
    t = m / s
    P, f = Phi(t), phi(t)
    G0 = t * P + f
    G3 = -t * f
    G4 = (t * t - 1.0) * f
    G6 = (t ** 4 - 6.0 * t * t + 3.0) * f
    return s * (G0 + (g3 / 6.0) * G3 + (g4 / 24.0) * G4 + (g3 * g3 / 72.0) * G6)


def herm_nodes(nq, k, dev):
    x, w = np.polynomial.hermite_e.hermegauss(nq)
    w = w / w.sum()
    if k == 1:
        return (torch.tensor(x[:, None], dtype=torch.float64, device=dev),
                torch.tensor(w, dtype=torch.float64, device=dev))
    grids = np.meshgrid(*([x] * k), indexing='ij')
    wg = w
    for _ in range(k - 1):
        wg = np.multiply.outer(wg, w)
    return (torch.tensor(np.stack([q.ravel() for q in grids], axis=1),
                         dtype=torch.float64, device=dev),
            torch.tensor(wg.ravel(), dtype=torch.float64, device=dev))


def fac_shape(m_g, C_g, W, pts, wq, k):
    """`γ₃`,`γ₄` НАСТУПНОЇ передактивації `p' = Wᵀ ReLU(p)` із факторної суміші.

    Повертає ЛИШЕ форму — масштаб свідомо викидається (вимір `onefac.py`)."""
    ev, U = torch.linalg.eigh(C_g)
    idx = torch.argsort(ev, descending=True)
    ev, U = torch.clamp(ev[idx], min=0.0), U[:, idx]
    V = U[:, :k] * torch.sqrt(ev[:k])
    D = torch.clamp(torch.diagonal(C_g) - (V * V).sum(1), min=1e-12)
    mu = m_g[None, :] + pts @ V.T
    m1, m2, m3, m4 = relu_moments(mu, torch.sqrt(D)[None, :])
    c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
    c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    c4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4 - 3.0 * c2 * c2
    W2 = W * W
    mq = m1 @ W
    vq = torch.clamp(c2 @ W2, min=1e-30)
    k3q = c3 @ (W2 * W)
    k4q = c4 @ (W2 * W2)
    w = wq[:, None]
    M1 = (w * mq).sum(0)
    M2 = (w * (mq * mq + vq)).sum(0)
    M3 = (w * (mq ** 3 + 3.0 * mq * vq + k3q)).sum(0)
    M4 = (w * (mq ** 4 + 6.0 * mq * mq * vq + 4.0 * mq * k3q
               + k4q + 3.0 * vq * vq)).sum(0)
    K2 = torch.clamp(M2 - M1 * M1, min=1e-30)
    K3 = M3 - 3.0 * M1 * M2 + 2.0 * M1 ** 3
    K4 = M4 - 4.0 * M1 * M3 + 6.0 * M1 * M1 * M2 - 3.0 * M1 ** 4 - 3.0 * K2 * K2
    S = torch.sqrt(K2)
    return K3 / (S ** 3), K4 / (S ** 4)


def chain(W, k, pts, wq, alt=None):
    """Ланцюг замикання. `k=0` ⟹ БАЗА (чисто гаусове). Повертає `E[a_last]`."""
    n = W[0].shape[1]
    C_g = W[0].T @ W[0]                       # p₁ = W₀ᵀz, z~N(0,I) — ТОЧНО гаусів
    m_g = torch.zeros(n, dtype=torch.float64, device=W[0].device)
    g3 = torch.zeros(n, dtype=torch.float64, device=W[0].device)
    g4 = torch.zeros_like(g3)
    shapes = []
    for i in range(1, len(W) + 1):
        s = torch.sqrt(torch.clamp(torch.diagonal(C_g), min=1e-30))
        t = m_g / s
        P, f = Phi(t), phi(t)
        m_a = edge_relu(m_g, s, g3, g4) if k else m_g * P + s * f
        if i == len(W):
            return m_a, shapes
        C_a = C_g * torch.outer(P, P) + (0.5 * (C_g * C_g)) * torch.outer(f / s, f / s)
        d = (m_g * m_g + s * s) * P + (m_g * s) * f - m_a * m_a
        C_a = C_a + torch.diag(torch.clamp(d, min=1e-30) - torch.diagonal(C_a))
        if k:
            g3, g4 = fac_shape(m_g, C_g, W[i], pts, wq, k)
            if alt is not None:
                g3, g4 = alt[i - 1]
            shapes.append((g3.clone(), g4.clone()))
        m_g = m_a @ W[i]
        C_g = W[i].T @ (C_a @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · k={KS} · NQ={NQ} · {DEV}\n', flush=True)

E_B = np.zeros(len(FS))
E_K = {k: np.zeros(len(FS)) for k in KS}
E_ALT = np.zeros(len(FS))
NOISE = np.zeros(len(FS))
SEAL_C1 = np.zeros(len(FS))
RB = np.zeros(len(FS))
RK = {k: np.zeros(len(FS)) for k in KS}
SHAPE_REF = None

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        acch = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(3300 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                a = torch.relu(a @ W32[q])
            r = a.sum(0, dtype=torch.float64)
            acc += r
            if d < NS // 2:
                acch += r
            d += b
        tru = acc / NS
        nrm = float(torch.sqrt((tru ** 2).mean()))
        NOISE[n] = float(torch.sqrt((((acch / (NS // 2)) - tru) ** 2).mean())) / nrm

        base, _ = chain(W, 0, None, None)
        E_B[n] = float(torch.sqrt(((base - tru) ** 2).mean())) / nrm
        RB[n] = float((base * tru).sum() / (tru * tru).sum())      # масштаб
        for k in KS:
            pts, wq = herm_nodes(NQ[k], k, DEV)
            pr, shp = chain(W, k, pts, wq)
            E_K[k][n] = float(torch.sqrt(((pr - tru) ** 2).mean())) / nrm
            RK[k][n] = float((pr * tru).sum() / (tru * tru).sum())
            if k == 1:
                if SHAPE_REF is None:
                    SHAPE_REF = shp
                else:                                   # ⛔C3: чужа форма
                    pa, _ = chain(W, 1, pts, wq, alt=SHAPE_REF)
                    E_ALT[n] = float(torch.sqrt(((pa - tru) ** 2).mean())) / nrm
        # ⛔C1: перший крок мусить збігтися (γ=0 на гаусовому p₁)
        pts, wq = herm_nodes(NQ[1], 1, DEV)
        C0 = W[0].T @ W[0]
        m0 = torch.zeros(W_, dtype=torch.float64, device=DEV)
        s0 = torch.sqrt(torch.diagonal(C0))
        a_b = m0 * Phi(m0 / s0) + s0 * phi(m0 / s0)
        a_f = edge_relu(m0, s0, torch.zeros_like(s0), torch.zeros_like(s0))
        SEAL_C1[n] = float(torch.max(torch.abs(a_b - a_f)) / torch.max(torch.abs(a_b)))
    print(f'  net{n} (seed {sd0}) БАЗА {E_B[n]:.4%} · ' +
          ' · '.join(f'k{k} {E_K[k][n]:.4%}' for k in KS), flush=True)


def loo_resid(err, rat):
    """Похибка після зняття СПІЛЬНОГО скалярного масштабу, LOO по мережах.

    `pred = tru·r + ε` ⟹ `e² = (r−1)² + ‖ε‖²`; після ділення на LOO-середнє `r̄`
    лишається `(r − r̄)² + ‖ε‖²`. Саме це робить `_ALPHA` у сабміті."""
    out = np.zeros_like(err)
    for i in range(len(err)):
        msk = np.arange(len(err)) != i
        out[i] = math.sqrt(max(err[i] ** 2 - (rat[i] - 1.0) ** 2
                               + (rat[i] - rat[msk].mean()) ** 2, 1e-30))
    return out


print(f'\n⛔C1 перший крок база vs FAC: {SEAL_C1.max():.3e} (мусить бути 0)')
print(f'⛔C2 шум істини (півМК): {NOISE.mean():.4%}\n')
print('══ ПОХИБКА `E[a_last]` ЛАНЦЮГА, віднос. RMS по мережах ══')
print(f'БАЗА (гаусове замикання) : {E_B.mean():.4%}  ± {E_B.std():.4%}  '
      f'· масштаб {RB.mean():.5f}')
for k in KS:
    r = (E_B / np.maximum(E_K[k], 1e-30)) ** 2
    print(f'FAC-{k} (форма з квадратури): {E_K[k].mean():.4%}  ± {E_K[k].std():.4%}  '
      f'· масштаб {RK[k].mean():.5f}  ⟹ ×{r.mean():.3f} по MSE')
if len(FS) > 1:
    print(f'⛔C3 ЧУЖА форма            : {E_ALT[1:].mean():.4%}  '
          f'(мусить бути ГІРШЕ за базу {E_B[1:].mean():.4%})')

print('\n══ ПІСЛЯ СКАЛЯРНОЇ α-КАЛІБРОВКИ (як у сабміті, LOO по мережах) ══')
RES = {}
for tag, arr, rr in [('БАЗА', E_B, RB)] + [(f'FAC-{k}', E_K[k], RK[k]) for k in KS]:
    RES[tag] = loo_resid(arr, rr)
    ex = (f'  ⟹ ×{(RES["БАЗА"] / np.maximum(RES[tag], 1e-30)).mean() ** 2:.3f}'
          if tag != 'БАЗА' else '')
    print(f'{tag:7s}: {RES[tag].mean():.4%}{ex}')

print('\n🎯 ГЕЙТ БЛЕНДУ: `w = σ²/(σ²+b²)`; зараз w=0.0883 ⟹ b² = 10.32σ².')
b0 = E_B.mean()
for k in KS:
    ratio = (b0 / max(E_K[k].mean(), 1e-30)) ** 2
    b2 = 10.32 / ratio
    w = 1.0 / (1.0 + b2)
    print(f'FAC-{k}: b² падає в {ratio:5.3f}× ⟹ w={w:.4f} ⟹ виграш бленду '
          f'×{(1.0 / (1.0 - w)) / (1.0 / (1.0 - 0.0883)):.4f} (ціна ×1.07)')
