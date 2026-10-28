"""cum89c — ЧИ ВИЖИВАЄ κ₂₁-ПОПРАВКА, КОЛИ κ₂₁ БЕРЕТЬСЯ З ДЕШЕВОЇ МК?

════════════════ СТАН ПИТАННЯ ════════════════
`cum89b` (24 мережі, N=2^16, розділені сіди):
    база 0.9352%  ·  κ₃ k=1.006  ·  κ₂₁ k=1.499  ·  κ₂₁ при λ=0.75 **k=2.485**
    КОНТРОЛЬ: чужа κ₂₁ найкраща при λ=0.25 і дає k=0.973 (ГІРШЕ бази) ⟹
    відношення своя/чужа = 2.55 ⟹ це ІНФОРМАЦІЯ, не калібровка.
k=2.485 ⟹ у балі ×1.131 ⟹ 1.8676e-07 → 1.651e-07. Найбільше за багато сесій.

🚨 АЛЕ ЦЕ ОРАКУЛ: κ₂₁ узято з МК на 65536 семплах. У сабміті МК має ~5400
семплів, і витратити їх на κ₂₁ можна лише частково: два матмули на шар
(`hᵀh` і `(h²)ᵀh`) коштують 4·n²·N, тобто при N=512 це 1.5% B на 31 шар —
а в нас запас РІВНО НУЛЬ (стоїмо на підлозі 0.1·B).

════════════════ ЩО МІРЯЄМО ════════════════
`k(λ=0.75)` як функція `N_orc` — числа семплів, з яких оцінено κ₂₁ (істина
завжди з ОКРЕМОГО прогону на 2^16, сіди РОЗДІЛЕНІ).

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
κ₂₁ — ТРЕТІЙ момент, його відносний шум ~ `1/√N` із великим префактором.
Ставлю: k впаде нижче 1.2 вже при N_orc ≈ 2048 (шанс 60%), і гілка виявиться
нереалізовною. Якщо ж k>1.8 тримається до N_orc=1024 — гілка ЖИВА, і тоді
головним стає гейт ціни, а не точність.
⚠️ Окремо стежимо, чи не з'явиться k<1 на малих N — це означало б, що шумна
κ₂₁ АКТИВНО шкодить, і поправку треба вимикати, а не послаблювати.

════════════════ ПЛОМБА ════════════════
При N_orc = 2^16 мусимо відтворити k=2.485 ± 0.05 (те саме число, що в cum89b).
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
DT = torch.float32
SQ2PI = float(np.sqrt(2.0 * np.pi))
LAMS = (0.15, 0.25, 0.4, 0.55, 0.75, 1.0)
NORCS = (256, 512, 1024, 2048)
NTRUTH = 1 << 16

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · λ-скан {LAMS} · істина 2^16 · N_orc {NORCS}")


def phi(t):
    return torch.exp(-0.5 * t * t) / SQ2PI


def Phi(t):
    return 0.5 * (1.0 + torch.erf(t * 0.7071067811865476))


def truth_pass(ws, nsamp, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    done = 0
    while done < nsamp:
        b = min(1 << 13, nsamp - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(L):
            h = x @ ws[i]
            if i == L - 1:
                acc += torch.clamp(h, min=0.0).sum(0).double()
                break
            x = torch.clamp(h, min=0.0)
        done += b
    return acc / nsamp


def kappa_pass(ws, nsamp, seed):
    """Мішані κ₂₁ і маргінальні κ₃ з `nsamp` семплів. Повертає БЕЗРОЗМІРНІ."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    H1 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    H2 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    HH = [torch.zeros(n, n, dtype=torch.float64, device=DEV) for _ in range(L)]
    H2H = [torch.zeros(n, n, dtype=torch.float64, device=DEV) for _ in range(L)]
    done = 0
    while done < nsamp:
        b = min(1 << 13, nsamp - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(L - 1):
            h = x @ ws[i]
            hd = h.double()
            H1[i] += hd.sum(0)
            H2[i] += (hd * hd).sum(0)
            HH[i] += hd.T @ hd
            H2H[i] += (hd * hd).T @ hd
            x = torch.clamp(h, min=0.0)
        done += b
    out = [None] * L
    for i in range(L - 1):
        m1, m2 = H1[i] / nsamp, H2[i] / nsamp
        hh, h2h = HH[i] / nsamp, H2H[i] / nsamp
        k21 = (h2h - 2.0 * m1.view(-1, 1) * hh - m1.view(1, -1) * m2.view(-1, 1)
               + 2.0 * (m1 * m1).view(-1, 1) * m1.view(1, -1))
        v = torch.clamp(m2 - m1 * m1, min=1e-30)
        sd = torch.sqrt(v)
        out[i] = (k21 / torch.outer(v, sd)).to(DT)
    return out


def predict(ws, K21=None, lam=1.0):
    m = torch.zeros(n, dtype=DT, device=DEV)
    C = torch.eye(n, dtype=DT, device=DEV)
    for i in range(L - 1):
        w = ws[i]
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
        mp = m @ w
        t = mp / s
        P, f = Phi(t), phi(t)
        e1 = s * (t * P + f)
        e2 = (mp * mp + s * s) * P + (mp * s) * f
        var = torch.clamp(e2 - e1 * e1, min=1e-30)
        Ca = cp * torch.outer(P, P) + 0.5 * (cp * cp) * torch.outer(f / s, f / s)
        if K21 is not None and K21[i] is not None:
            k21 = K21[i]
            corr = torch.outer(P, f * 0.5) * k21.T + torch.outer(f * 0.5, P) * k21
            Ca = Ca + lam * corr * torch.outer(s, s)
            Ca = torch.nan_to_num(Ca, nan=0.0, posinf=0.0, neginf=0.0)
        Ca = Ca + torch.diag(var - torch.diagonal(Ca))
        m, C = e1, Ca
    w = ws[L - 1]
    cp = w.T @ (C @ w)
    s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
    mp = m @ w
    t = mp / s
    return torch.nan_to_num(s * (t * Phi(t) + phi(t)), nan=0.0,
                            posinf=0.0, neginf=0.0)


base, cols = [], {(N, l): [] for N in NORCS for l in LAMS}
t0 = time.time()
for j in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[j, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth_pass(ws, NTRUTH, 7000 + j).to(DT)
    den = float((tr * tr).mean())
    base.append(float(((predict(ws) - tr) ** 2).mean() / den) ** 0.5)
    for N in NORCS:
        K = kappa_pass(ws, N, 555000 + j)
        for l in LAMS:
            p = predict(ws, K21=K, lam=l)
            cols[(N, l)].append(float(((p - tr) ** 2).mean() / den) ** 0.5)
    if j % 8 == 0:
        print(f"  net{j:>3d} база {100 * base[-1]:.4f}%")

b = np.array(base)
print(f"\n  час {time.time() - t0:.0f} с")
print()
print("=" * 76)
print(f"λ-СКАН ПІД ШУМНОЮ κ₂₁ · {NNET} мереж (істина 2^16, сіди розділені)")
print("=" * 76)
print(f"  база = {100 * b.mean():.4f}%")
hdr = "  N_orc  ціна%B  " + "".join(f"λ={l:<5}" for l in LAMS)
print(hdr)
V0, E20, SAMP = 2.050e-06, 2.118e-05, 0.0838
best = (None, 1e9)
for N in NORCS:
    cost = 4.0 * n * n * N * (L - 1) / 2.72e11
    row = f"{N:>7d} {100 * cost:>6.2f}  "
    for l in LAMS:
        k = float((b ** 2).mean() / (np.array(cols[(N, l)]) ** 2).mean())
        row += f"{k:>7.3f}"
        # НЕТТО: κ₂₁ забирає бюджет у МК ⟹ V росте, E² падає
        Vn = V0 * SAMP / max(SAMP - cost, 1e-9)
        En = E20 / k
        mse = Vn * En / (Vn + En)
        if mse < best[1]:
            best = ((N, l, k), mse)
    print(row)
print()
print("  💰 НЕТТО У БАЛІ (κ₂₁ забирає бюджет у МК; поточний MSE = 1.863e-06):")
print(f"{'N_orc':>7s} {'λ':>6s} {'k':>7s} {'MSE_нетто':>12s} {'бал':>11s} {'проти рекорду':>15s}")
for N in NORCS:
    cost = 4.0 * n * n * N * (L - 1) / 2.72e11
    for l in LAMS:
        k = float((b ** 2).mean() / (np.array(cols[(N, l)]) ** 2).mean())
        Vn = V0 * SAMP / max(SAMP - cost, 1e-9)
        mse = Vn * (E20 / k) / (Vn + E20 / k)
        if l in (0.25, 0.55):
            print(f"{N:>7d} {l:>6} {k:>7.3f} {mse:>12.4e} {mse * 0.1:>11.4e}"
                  f" {'×' + format(1.8676e-07 / (mse * 0.1), '.4f'):>15s}")
(bn, bl, bk), bmse = best
print(f"\n  🎯 НАЙКРАЩА КОМБІНАЦІЯ: N_orc={bn} λ={bl} k={bk:.3f} ⟹ бал {bmse * 0.1:.4e}"
      f"  (×{1.8676e-07 / (bmse * 0.1):.4f} проти рекорду)")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/cum89c.npz", base=b,
         **{f"n{N}_l{l}": np.array(cols[(N, l)]) for N in NORCS for l in LAMS})
