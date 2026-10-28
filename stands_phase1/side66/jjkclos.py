"""ПЛОМБА 2: чи виживає `S_jk = E[ã_j²ã_k]` БЕЗ ВИБІРКИ, із ЗАМИКАННЯ. (С86)

## ЗВІДКИ
`jjk.py` довів: член (j,j,k) = **93.6%** норми `κ₃(h)` (корел +0.9282,
масштаб 0.9359 ± 2.3%), тоді як діагональ (j,j,j) — лише 7.8%. Але `S` там
бралася З ВИБІРКИ. Для замикання вона потрібна З ВАГ. Це і перевіряємо.

## 📐 ВИВЕДЕНА ФОРМУЛА (нова, ніде раніше не стояла)
`A_jk = E[ReLU(h_j)²·ReLU(h_k)]` для гаусової пари, `t=μ/σ`, `ρ=Cov/(σσ)`:
    `A_jk = σ_j²·σ_k · Σ_n (ρⁿ/n!)·F_n(t_j)·A_n(t_k)`
де `A_n` — ЗВИЧАЙНІ коефіцієнти Мелера (вже в сабміті):
    `A₀=tΦ+φ · A₁=Φ · A₂=φ · A₃=−tφ · A₄=(t²−1)φ`
а `F_n = E[ReLU(Z+t)²He_n(Z)]` виводиться з `E[f·He_{n+1}] = E[f'·He_n]`
(інтегрування частинами) і `(ReLU²)' = 2ReLU`:
    🔑 **`F_{n+1} = 2·A_n`**,  `F_0 = (t²+1)Φ + tφ`
⟹ НОВИХ СПЕЦФУНКЦІЙ НЕ ПОТРІБНО — ті самі `Φ`, `φ`.
Далі центрування: `S_jk = A_jk − m_k·Q_j − 2m_j·C_jk + 2m_j²·m_k`,
   `Q_j=E[a_j²]`, `C_jk=E[a_ja_k]`. ⛔пломба: при `j=k` дає `E[a³]−3mE[a²]+2m³`.

## ⛔ ТРИ ВАРІАНТИ — РОЗДІЛЯЮТЬ ДВІ ХВОРОБИ
  V0  `S` З ВИБІРКИ                     — контроль, має відтворити jjk.py
  VA  `S` з ФОРМУЛИ на ВИБІРКОВИХ μ,σ,ρ — ізолює похибку САМОЇ формули
  VB  `S` з ФОРМУЛИ на ЗАМИКАНИХ μ,σ,ρ  — реальний сценарій
Еталон завжди — `κ₃(h)` з вибірки. Метрика — не сама `S`, а `D+T` через неї.

🎯 ГЕЙТ: VB корел ≥0.85 І масштаб сталий у 25%.
📌 ПЛАНКА НИЗЬКА: нинішня діагональ дає корел +0.65 і масштаб 0.078.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ:
   V0 відтворить jjk.py побітово (та сама арифметика) — це пломба стенду.
   VA: корел 0.85–0.95, масштаб 0.85–1.05 (похибка лише від негаусовості `h`).
   VB: корел 0.75–0.90, масштаб 0.6–0.95 — гірше, бо додається похибка `m,C`.
   Головний ризик: `S` — ТРЕТІЙ момент, і гаусове наближення для `h` могло б
   саме тут бути найслабшим (пор. похибка форми `C` = 8.0% проти `m` = 0.19%).
   Шанс пройти гейт: **45%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/jjkclos.py [cuda:N] [NETS] [LOG2N] [KMAX]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import _phi, _Phi                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
KMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 4
NS = 1 << LOG2N
BATCH = 1 << 15
L_, W_ = 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in set(int(v) for v in _TT['seeds'])][:NETS]
print(f'{len(FS)} мереж, вибірка 2^{LOG2N} = {NS}, K={KMAX}, {DEV}\n',
      flush=True)

_FACT = [1.0]
for _q in range(1, 40):
    _FACT.append(_FACT[-1] * _q)


def mehler_A(t, ph, Ph, n):
    """A_n = E[ReLU(Z+t)·He_n(Z)] = He_{n−2}(−t)·φ(t) для n≥2.

    ⛔звірено з виписаними: A₂=φ · A₃=−tφ · A₄=(t²−1)φ · A₅=(3t−t³)φ.
    """
    if n == 0:
        return t * Ph + ph
    if n == 1:
        return Ph
    x = -t
    h0 = torch.ones_like(t)
    h1 = x
    if n == 2:
        return h0 * ph
    if n == 3:
        return h1 * ph
    for k in range(1, n - 2):
        h0, h1 = h1, x * h1 - float(k) * h0
    return h1 * ph


def s_from_gauss(mu, sg, cov, kmax):
    """`S_jk = E[ã_j²ã_k]` для a=ReLU(h), h гаусів із (mu, sg, cov).

    ⛔пломба всередині: діагональ мусить збігтися з `E[a³]−3mE[a²]+2m³`,
    порахованим із ТОЧНОЇ рекурсії обрізаних моментів `L_k`.
    """
    t = mu / sg
    ph, Ph = _phi(t), _Phi(t)
    # моменти активації з ТОЧНОЇ рекурсії L_k (одновимірні)
    L0 = Ph
    L1 = mu * Ph + sg * ph
    L2 = mu * L1 + sg * sg * L0
    m, Q = L1, L2
    # коваріація активацій через той самий ряд Мелера
    rho = cov / torch.outer(sg, sg)
    rho = rho.clamp(-1.0, 1.0)
    Cact = torch.zeros_like(rho)
    rp = torch.ones_like(rho)
    for n in range(kmax + 1):
        an = mehler_A(t, ph, Ph, n)
        Cact = Cact + (rp / _FACT[n]) * torch.outer(an, an)
        rp = rp * rho
    Cact = Cact * torch.outer(sg, sg)
    # A_jk = E[ReLU(h_j)²ReLU(h_k)]
    F0 = (t * t + 1.0) * Ph + t * ph
    Ajk = torch.zeros_like(rho)
    rp = torch.ones_like(rho)
    for n in range(kmax + 1):
        Fn = F0 if n == 0 else 2.0 * mehler_A(t, ph, Ph, n - 1)
        An = mehler_A(t, ph, Ph, n)
        Ajk = Ajk + (rp / _FACT[n]) * torch.outer(Fn, An)
        rp = rp * rho
    Ajk = Ajk * torch.outer(sg * sg, sg)
    S = (Ajk - torch.outer(Q, m) - 2.0 * torch.outer(m, torch.ones_like(m))
         * Cact + 2.0 * torch.outer(m * m, m))
    return S, m, Q


def closure_parts(W, dev):
    """Замикання: `mu, sg, cov` ПЕРЕДАКТИВАЦІЙ кожного шару."""
    n0 = W[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    MU, SG, CP = [], [], []
    for i in range(len(W)):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        MU.append(mp.clone()); SG.append(s.clone()); CP.append(cp.clone())
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        if i < len(W) - 1:
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp)*torch.outer(ph/s, ph/s)
            d = (mp*mp + s*s) * Ph + mp * s * ph - m * m
            C = C + torch.diag(d - C.diagonal())
    return MU, SG, CP


def sample_all(W, dev, ns, batch, seed, M):
    """`S` з вибірки · `κ₃(h)` еталон · ВИБІРКОВІ mu,sg,cov передактивацій."""
    S = [torch.zeros(W_, W_, dtype=torch.float64, device=dev)
         for _ in range(L_)]
    HH = [torch.zeros(W_, W_, dtype=torch.float64, device=dev)
          for _ in range(L_)]
    H1 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    H2 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    H3 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    Mf = [m.float() for m in M]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    done = 0
    while done < ns:
        b = min(batch, ns - done)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            ad = a - Mf[i]
            S[i] += ((ad * ad).T @ ad).double()
            h = a @ W[i]
            HH[i] += (h.T @ h).double()
            hd = h.double(); h2 = hd * hd
            H1[i] += hd.sum(0); H2[i] += h2.sum(0); H3[i] += (h2 * hd).sum(0)
            a = torch.relu(h)
        done += b
    OUT = []
    for i in range(L_):
        e1, e2, e3 = H1[i]/ns, H2[i]/ns, H3[i]/ns
        k3 = e3 - 3.0*e1*e2 + 2.0*e1**3
        cv = HH[i]/ns - torch.outer(e1, e1)
        OUT.append((S[i]/ns, k3, e1, (e2 - e1*e1).clamp_min(1e-20).sqrt(), cv))
    return OUT


def pass1_means(W, dev, ns, batch, seed):
    M = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    done = 0
    while done < ns:
        b = min(batch, ns - done)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            M[i] += a.double().sum(0)
            a = torch.relu(a @ W[i])
        done += b
    return [m / ns for m in M]


def dt(W, S):
    w = W.double(); w2 = w * w
    D = (w2 * w).T @ S.diagonal()
    return D + 3.0 * ((w2 * (S @ w)).sum(0) - D)


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = float(x.norm() * y.norm())
    return float((x*y).sum()) / d if d > 0 else float('nan')


ROWS, SEAL = [], []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        M = pass1_means(W, DEV, NS, BATCH, 7100 + n)
        SM = sample_all(W, DEV, NS, BATCH, 7100 + n, M)
        MUc, SGc, CPc = closure_parts(W, DEV)
        row, sl = [], []
        for i in range(1, L_):
            Ssmp, k3, mus, sgs, cvs = SM[i]
            # `S` — для АКТИВАЦІЙ на вході шару i, тобто ReLU(h) шару i−1
            SA, mA, QA = s_from_gauss(SM[i-1][2], SM[i-1][3], SM[i-1][4], KMAX)
            SB, mB, QB = s_from_gauss(MUc[i-1], SGc[i-1], CPc[i-1], KMAX)
            v0, vA, vB = dt(W[i], Ssmp), dt(W[i], SA), dt(W[i], SB)
            kn = max(float(k3.norm()), 1e-300)
            row.append((corr(v0, k3), corr(vA, k3), corr(vB, k3),
                        float(v0.norm())/kn, float(vA.norm())/kn,
                        float(vB.norm())/kn))
            # ⛔пломба формули: діагональ S проти E[a³]−3mE[a²]+2m³ з L_k
            mu, sg = MUc[i-1], SGc[i-1]
            t = mu/sg; ph, Ph = _phi(t), _Phi(t)
            L1 = mu*Ph + sg*ph; L2 = mu*L1 + sg*sg*Ph
            L3 = mu*L2 + 2.0*sg*sg*L1
            ref = L3 - 3.0*L1*L2 + 2.0*L1**3
            sl.append(float((SB.diagonal()-ref).norm()/ref.norm().clamp_min(1e-300)))
        ROWS.append(row); SEAL.append(sl)
        print(f'  … {n+1}/{len(FS)} seed {sd0} · ⛔пломба формули (діагональ '
              f'проти L_k): max {max(sl):.2e}', flush=True)
    del W
    torch.cuda.empty_cache()

A = np.array(ROWS)
np.save('/mnt/win_d/llm/White-Box/arena62/logs/jjkclos.npy', A)
print(f'\n⛔ПЛОМБА формули по всіх мережах і шарах: max '
      f'{np.array(SEAL).max():.3e}  (мусить бути ~1e-12)')
print(f'\n{"шар":>4s} {"корел V0":>9s} {"корел VA":>9s} {"корел VB":>9s} '
      f'{"масшт V0":>9s} {"масшт VA":>9s} {"масшт VB":>9s}')
print('─' * 64)
for i in range(A.shape[1]):
    ly = i + 1
    if ly <= 6 or ly % 4 == 0 or ly == L_ - 1:
        print(f'{ly:4d} {A[:, i, 0].mean():+9.4f} {A[:, i, 1].mean():+9.4f} '
              f'{A[:, i, 2].mean():+9.4f} {A[:, i, 3].mean():9.4f} '
              f'{A[:, i, 4].mean():9.4f} {A[:, i, 5].mean():9.4f}')
print('─' * 64)
for k, nm in ((0, 'V0 вибірка'), (1, 'VA формула+вибіркові μσρ'),
              (2, 'VB формула+ЗАМИКАННЯ')):
    c = A[:, :, k].mean()
    s = A[:, :, k + 3].mean(0)
    print(f'{nm:26s} корел {c:+.4f} · масштаб {s.mean():.4f} ± {s.std():.4f} '
          f'({s.std()/max(abs(s.mean()),1e-30)*100:.1f}%)')
print('\n🎯 ГЕЙТ: VB корел ≥0.85 І розкид масштабу <25%.'
      '\n📌 нинішня діагональ для порівняння: корел +0.6504, масштаб 0.0783.')
