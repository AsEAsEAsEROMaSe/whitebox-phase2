"""ТРИ ПЕРЕВІРКИ ВЛАСНОГО КОДУ Й КОНСТАНТ (С88, після зауваження ЮЗЕРА)

ЮЗЕР: «дивно, що куди не плюнь — завжди вирок, виведений з півшишки».
Перевірка показала: сім вироків дня проходили через константу перекладу
`(k+15.66)/16.66`, ЯКУ НІХТО НЕ МІРЯВ, і вона хибна в 1.5× (правильна 10.33 —
самоузгоджена з `_W_BLEND=0.0883`). Цей стенд перевіряє ще три місця.

## 🎯 ЩО ПЕРЕВІРЯЄТЬСЯ
 C1 `cop88`: «10% пар без розв'язку» — АРТЕФАКТ ЧИ ФАКТ? Скануємо
    `Cov_gauss(ρ)` на щільній сітці і рахуємо ЧИСЛО КОРЕНІВ. Якщо коренів
    2+ — бісекція некоректна і 10% завищені. Якщо 0 — факт.
 C2 `herm88`: `s` бралася з ПРЯМОГО МК, а `ρ` — з `C_h = WᵀC_prev W`.
    Це два шляхи до однієї величини. Міряємо розбіжність
    `|SH − √diag(C_h)| / SH`. Якщо &gt;1% — числа зміщені.
 C3 🔑ПРЯМИЙ ВИМІР КОНСТАНТИ ПЕРЕКЛАДУ. Незалежно міряємо:
      `V`  — дисперсію МК-оцінки з N семплами (як у сабміті),
      `E²` — MSE детермінованого замикання після `α`,
    і перевіряємо `MSE = (1−w)²V + w²E²` проти ВІДОМОГО 1.869e-06.
    Це пломба на ВІДОМЕ ГРЕЙДЕРНЕ ЧИСЛО, а не наша алгебра.

## ⛔ ПЛОМБИ
 P1 C3 мусить відтворити 1.869e-06 (наш sub97 при множнику 0.1002).
 P2 `w_opt = V/(V+E²)` мусить збігтися з `_W_BLEND = 0.0883` (грейдер).
 P3 C1: на шарі 1 (гаусів вхід) коренів мусить бути РІВНО 1 для всіх пар.
 P4 масштабування: `V ∝ 1/N` — міряємо при N і 4N.

## 📣 ПЕРЕДБАЧЕННЯ
 A. C1: більшість «без розв'язку» виявиться СПРАВЖНІМИ (0 коренів), бо
    `Cov_gauss` обмежена зверху значенням при `ρ=1`. Шанс **65%**.
 B. C2: розбіжність &lt;0.5% ⟹ `herm88` не зміщений. Шанс **70%**.
 C. 🎯C3: `1/r` вийде **10–12**, підтвердивши 10.33 і спростувавши 15.66.
    Шанс **60%**.
"""
import glob, math, sys
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
NSUB = 8206            # семплів у sub97 (mccal88: пілот 82.6%, c=2.90e6)
NREP = 64              # незалежних повторів для виміру дисперсії
assert DEV not in ('cuda:0', 'cuda:4')
print(__doc__.split('📣')[1], flush=True)

I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
KMAX = 12
WK = torch.tensor([1.0 / ((k + 1) * math.factorial(k)) for k in range(1, KMAX + 1)],
                  dtype=torch.float64, device=DEV)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
      if int(np.load(f)['seed']) in have][:NETS]
NF = len(FS)
print(f'{NF} мереж · істина 2^{LOG2N} · N_sub={NSUB} × {NREP} повторів · {DEV}\n',
      flush=True)


def herm(t, k):
    o = [torch.ones_like(t), t.clone()]
    for n in range(1, k - 1):
        o.append(t * o[n] - n * o[n - 1])
    return o[:k]


def cov_g(Pj, Pk, cj, ck, r):
    acc = r * (Pj * Pk); rp = r.clone()
    for k in range(1, KMAX + 1):
        rp = rp * r
        acc = acc + WK[k - 1] * rp * (cj[k - 1] * ck[k - 1])
    return acc


C1 = np.zeros((NF, 3))     # частка пар з 0 / 1 / ≥2 коренями
C2 = np.zeros((NF, 2))     # розбіжність σ: середня, макс
VMC = np.zeros((NF, 2))    # дисперсія МК при N і 4N
E2 = np.zeros(NF)
TRU = []
DET = []

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32),
                        device=DEV, dtype=torch.float64) for q in range(L_)]
    Wf = [w.float() for w in W32]
    with torch.no_grad():
        # ---------- істина ----------
        A1 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        AA = torch.zeros(W_, W_, dtype=torch.float64, device=DEV)
        S2 = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        Aprev = None
        g = torch.Generator(device=DEV); g.manual_seed(2100 + n); d = 0
        AAp = torch.zeros(L_, W_, W_, dtype=torch.float64, device=DEV)
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                h = (a @ Wf[q]).double()
                S2[q] += (h * h).sum(0)
                A1[q] += h.sum(0)
                aq = torch.clamp(h, min=0.0)
                AAp[q] += aq.T @ aq
                a = aq.float()
            d += b
        MH = A1 / NS
        SH = torch.sqrt(torch.clamp(S2 / NS - MH * MH, min=1e-30))
        AM = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        for q in range(L_):
            AM[q] = torch.diagonal(AAp[q]).clone() * 0
        # середні активацій
        A1a = torch.zeros(L_, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV); g.manual_seed(2100 + n); d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g, dtype=torch.float32)
            for q in range(L_):
                h = (a @ Wf[q]).double()
                aq = torch.clamp(h, min=0.0)
                A1a[q] += aq.sum(0)
                a = aq.float()
            d += b
        A1a /= NS
        TRU.append(A1a[L_ - 1].clone())

        # ---------- C2: дві оцінки σ ----------
        rel = []
        prevC = None
        for q in range(L_):
            Cq = AAp[q] / NS - torch.outer(A1a[q], A1a[q])
            if q > 0:
                Ch = W32[q].T @ (prevC @ W32[q])
                sd_ = torch.sqrt(torch.clamp(torch.diagonal(Ch), min=1e-30))
                rel.append(float(((sd_ - SH[q]).abs() / SH[q]).mean()))
            prevC = Cq
        C2[n, 0] = float(np.mean(rel)); C2[n, 1] = float(np.max(rel))

        # ---------- C1: число коренів Cov_gauss(ρ) ----------
        for q in (0, 20):
            if q == 0:
                Ch = W32[0].T @ W32[0]
                mh = torch.zeros(W_, dtype=torch.float64, device=DEV)
            else:
                Cp = AAp[q - 1] / NS - torch.outer(A1a[q - 1], A1a[q - 1])
                Ch = W32[q].T @ (Cp @ W32[q]); mh = A1a[q - 1] @ W32[q]
            s = torch.sqrt(torch.clamp(torch.diagonal(Ch), min=1e-30))
            t = mh / s
            P_, f_ = Phi(t), phi(t)
            cs = [h_ * f_ for h_ in herm(t, KMAX)]
            iu = torch.triu_indices(W_, W_, offset=1, device=DEV)[:, ::37]
            j0, j1 = iu[0], iu[1]
            Cq = AAp[q] / NS - torch.outer(A1a[q], A1a[q])
            tgt = Cq[j0, j1] / (s[j0] * s[j1])
            grid = torch.linspace(-0.999, 0.999, 401, dtype=torch.float64,
                                  device=DEV)
            vals = torch.stack([cov_g(P_[j0], P_[j1], [c[j0] for c in cs],
                                      [c[j1] for c in cs],
                                      torch.full_like(tgt, float(gg)))
                                for gg in grid])
            sgn = torch.sign(vals - tgt.unsqueeze(0))
            nroot = (sgn[1:] * sgn[:-1] < 0).sum(0)
            if q == 20:
                C1[n, 0] = float((nroot == 0).double().mean())
                C1[n, 1] = float((nroot == 1).double().mean())
                C1[n, 2] = float((nroot >= 2).double().mean())
            else:
                P3 = float((nroot == 1).double().mean())

        # ---------- детермінований ланцюг (для E²) ----------
        m = torch.zeros(W_, dtype=torch.float64, device=DEV)
        C = torch.eye(W_, dtype=torch.float64, device=DEV)
        for q in range(L_):
            cp = W32[q].T @ (C @ W32[q]); cp = 0.5 * (cp + cp.T)
            mp = m @ W32[q]
            s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
            t = mp / s; P_, f_ = Phi(t), phi(t)
            m = mp * P_ + s * f_
            Ca = cp * torch.outer(P_, P_) + 0.5 * (cp * cp) * torch.outer(f_ / s,
                                                                          f_ / s)
            dd = (mp * mp + s * s) * P_ + (mp * s) * f_ - m * m
            C = Ca + torch.diag(dd - torch.diagonal(Ca))
        DET.append(m.clone())

        # ---------- C3: дисперсія МК-оцінки при N_sub ----------
        for ii, Nn in enumerate((NSUB, 4 * NSUB)):
            acc = torch.zeros(NREP, W_, dtype=torch.float64, device=DEV)
            gg = torch.Generator(device=DEV); gg.manual_seed(31000 + n)
            for rp in range(NREP):
                z = torch.randn(Nn // 2, W_, device=DEV, generator=gg,
                                dtype=torch.float32)
                a = torch.cat([z, -z], 0)
                for q in range(L_):
                    a = torch.relu(a @ Wf[q])
                acc[rp] = a.double().mean(0)
            VMC[n, ii] = float(((acc - acc.mean(0)) ** 2).mean())
        del AAp
    print(f'  net{n}  C1 без коренів {C1[n,0]:.1%} · C2 σ {C2[n,0]:.3%} · '
          f'V {VMC[n,0]:.3e}', flush=True)

Y = torch.stack(TRU); D = torch.stack(DET)
al = float((Y * D).sum() / (D * D).sum())
E2v = float(((al * D - Y) ** 2).mean())
V = VMC[:, 0].mean()
r = V / E2v
w_opt = V / (V + E2v)
W_B = 0.0883
mse_model = (1 - W_B) ** 2 * V + W_B ** 2 * E2v

print(f'\n{"=" * 72}\n⛔ ПЛОМБИ')
print(f' P1  MSE(модель) = {mse_model:.4e}  проти ВІДОМОГО 1.869e-06  '
      f'⟹ {"✅" if abs(mse_model/1.869e-06-1)<0.35 else "⛔"} '
      f'({100*(mse_model/1.869e-06-1):+.1f}%)')
print(f' P2  w_opt = {w_opt:.4f}  проти _W_BLEND = {W_B}  '
      f'⟹ {"✅" if abs(w_opt-W_B)<0.02 else "⛔"}')
print(f' P3  шар1: частка пар з РІВНО 1 коренем = {P3:.2%}  (мусить ~100%)')
print(f' P4  V(N)/V(4N) = {VMC[:,0].mean()/max(VMC[:,1].mean(),1e-30):.2f}  '
      f'(мусить ~4.0)')

print(f'\n{"=" * 72}\n📊 C1: ЧИСЛО КОРЕНІВ `Cov_gauss(ρ) = Cov_true` (шар 21)')
print(f'   0 коренів (справді немає): {C1[:,0].mean():.2%}')
print(f'   1 корінь  (бісекція вірна): {C1[:,1].mean():.2%}')
print(f'  ≥2 корені (бісекція БРЕШЕ): {C1[:,2].mean():.2%}')

print(f'\n📊 C2: `|√diag(WᵀC_prevW) − σ_МК| / σ_МК` = {C2[:,0].mean():.4%} '
      f'(макс {C2[:,1].max():.4%})')

print(f'\n{"=" * 72}\n📊 C3: 🔑ПРЯМИЙ ВИМІР КОНСТАНТИ ПЕРЕКЛАДУ')
print(f'   V (дисперсія МК, N={NSUB}) = {V:.4e}')
print(f'   E² (замикання після α)     = {E2v:.4e}   (α = {al:.5f})')
print(f'   1/r = E²/V                 = {1/r:.2f}')
print(f'   агент казав 15.66 · моя арифметика дала 10.33')
print(f'\n📣 ВИРОК: переклад `виграш = (k·r+1)/(r+1)` ⟹')
for k in (1.07, 1.34, 2.0, 5.0, 31.2):
    print(f'   k={k:5.2f}: ×{(k*r+1)/(r+1):.4f}   '
          f'(за 15.66 було ×{(k+15.66)/16.66:.4f})')
