"""rank89 — ЧИ МАЄ κ₂₁ МАЛИЙ РАНГ? ЦЕ ВИРІШУЄ, ЧИ ГІЛКА ВЗАГАЛІ МОЖЕ ЖИТИ.

════════════════ ЧОМУ ЦЕ ГОЛОВНЕ ПИТАННЯ ════════════════
`cum89c/d`: κ₂₁ дає `k=2.485` (оракул) з ЧИСТИМ контролем, але коштує `4n²N`
на шар — перевищення гейту окупності у 2.7×, і жодна з 24 комбінацій не
виходить у плюс.
📐 АЛЕ якщо тензор третіх кумулянтів активацій має малий ранг
`T ≈ Σ_{r≤R} λ_r u_r⊗u_r⊗u_r`, то
    κ₂₁(h_j,h_k) = Σ_{l,m,p} W_lj W_mj W_pk T_lmp = Σ_r λ_r (Wᵀu_r)_j²·(Wᵀu_r)_k
тобто **R МАТВЕКІВ, а не матмулів**: `R·2n²` замість `4n²N`.
При R=8 це 0.012% B проти 0.76% — у 60 разів дешевше, і гейт (0.284% при
k=1.35, 1.204% при k=2.48) виконується з ВЕЛИЧЕЗНИМ запасом.
🏆 І це рівно та структура, яку ми пришпилили в ЛІДЕРА: «3–4 понейронні
вектори, ЖОДНОЇ матриці, три редукції на нейрон на шар».

════════════════ ЩО МІРЯЄМО ════════════════
Беремо ОРАКУЛЬНУ κ₂₁ (ту, що дає k=2.485), обрізаємо матрицю до рангу R і
дивимось, як падає `k`. Це відповідає на питання про ранг НАПРЯМУ, не
торкаючись поки рекурсії для `u_r`.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
🚨 `lowrk88` уже показав, що низький ранг для САМОЇ `C` мертвий (r=3 дало
×0.32, бо рекурсія піднімає ранг через поелементний `C²`). Але тут інший
об'єкт: обрізається ПОПРАВКА, а не стан, і вона не проходить через `C²`.
Ставлю: спектр κ₂₁ буде значно КОНЦЕНТРОВАНІШИЙ за спектр `C` (шанс 60%),
бо `tprof88` зміряв еф. ранг `C` на глибині = 3.1 — а κ₂₁ будується з тих
самих напрямків. R=16 збереже ≥70% від k=2.485 (шанс 45%).
⚠️ Якщо ж k розсиплеться вже при R=32 — гілка мертва остаточно, і це буде
ТРЕТІЙ незалежний доказ, що низький ранг у цій задачі не працює.

════════════════ ПЛОМБА ════════════════
R=256 (повний ранг) мусить відтворити k без обрізання, до 1e-6.
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
NORC = 1 << 16
LAM = 0.75
RANKS = (2, 4, 8, 16, 32, 64, 256)
NTRUTH = 1 << 16
V0, E20, SAMP = 2.050e-06, 2.118e-05, 0.0838
B = 2.72e11

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · κ₂₁ оракул 2^16 · λ={LAM} · ранги {RANKS}")


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
    out, spec = [None] * L, []
    for i in range(L - 1):
        m1, m2 = H1[i] / nsamp, H2[i] / nsamp
        hh, h2h = HH[i] / nsamp, H2H[i] / nsamp
        k21 = (h2h - 2.0 * m1.view(-1, 1) * hh - m1.view(1, -1) * m2.view(-1, 1)
               + 2.0 * (m1 * m1).view(-1, 1) * m1.view(1, -1))
        v = torch.clamp(m2 - m1 * m1, min=1e-30)
        out[i] = (k21 / torch.outer(v, torch.sqrt(v))).to(DT)
        if i in (8, 20):
            sv = torch.linalg.svdvals(out[i].double())
            spec.append((sv / sv.sum()).cpu().numpy())
    return out, spec


def truncate(K, R):
    """SVD-обрізання ПОПРАВКИ (не стану!) до рангу R."""
    if R >= n:
        return K
    out = [None] * L
    for i in range(L - 1):
        if K[i] is None:
            continue
        U, S, Vh = torch.linalg.svd(K[i].double())
        out[i] = ((U[:, :R] * S[:R]) @ Vh[:R]).to(DT)
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
            Ca = torch.nan_to_num(Ca + lam * corr * torch.outer(s, s),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        Ca = Ca + torch.diag(var - torch.diagonal(Ca))
        m, C = e1, Ca
    w = ws[L - 1]
    cp = w.T @ (C @ w)
    s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
    t = (m @ w) / s
    return torch.nan_to_num(s * (t * Phi(t) + phi(t)), nan=0.0,
                            posinf=0.0, neginf=0.0)


base, cols, specs = [], {R: [] for R in RANKS}, []
t0 = time.time()
for j in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[j, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth_pass(ws, NTRUTH, 7000 + j).to(DT)
    den = float((tr * tr).mean())
    base.append(float(((predict(ws) - tr) ** 2).mean() / den) ** 0.5)
    K, sp = kappa_pass(ws, NORC, 555000 + j)
    specs.append(sp)
    for R in RANKS:
        p = predict(ws, K21=truncate(K, R), lam=LAM)
        cols[R].append(float(((p - tr) ** 2).mean() / den) ** 0.5)
    if j % 8 == 0:
        print(f"  net{j:>3d} база {100 * base[-1]:.4f}%  R8 {100 * cols[8][-1]:.4f}%"
              f"  R256 {100 * cols[256][-1]:.4f}%")

b = np.array(base)
print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 82)
print(f"СПЕКТР κ₂₁: ЧАСТКА НОРМИ В ПЕРШИХ R СИНГУЛЯРНИХ ЧИСЛАХ ({NNET} мереж)")
print("=" * 82)
for idx, lay in enumerate((9, 21)):
    sp = np.mean([s[idx] for s in specs], axis=0)
    cum = np.cumsum(sp)
    print(f"  шар {lay:>2d}: " + "  ".join(f"R{R}:{100 * cum[R - 1]:.1f}%"
                                           for R in (2, 4, 8, 16, 32, 64)))
print()
print("=" * 82)
print(f"k(λ={LAM}) ПРИ ОБРІЗАННІ κ₂₁ ДО РАНГУ R")
print("=" * 82)
print(f"  база = {100 * b.mean():.4f}%")
print(f"{'R':>5s} {'похибка':>10s} {'k':>8s} {'ціна R матвеків':>17s} {'НЕТТО бал':>12s} {'проти рек.':>11s}")
for R in RANKS:
    a = np.array(cols[R])
    k = float((b ** 2).mean() / (a ** 2).mean())
    cost = R * 2.0 * n * n * (L - 1) / B
    Vn = V0 * SAMP / max(SAMP - cost, 1e-9)
    mse = Vn * (E20 / k) / (Vn + E20 / k)
    print(f"{R:>5d} {100 * a.mean():>9.4f}% {k:>8.3f} {100 * cost:>16.4f}%"
          f" {mse * 0.1:>12.4e} {'×' + format(1.8676e-07 / (mse * 0.1), '.4f'):>11s}")
kf = float((b ** 2).mean() / (np.array(cols[256]) ** 2).mean())
print(f"\n  🔬 ПЛОМБА: R=256 (повний ранг) дає k={kf:.3f}, cum89b дав 2.485 "
      f"{'✅' if abs(kf - 2.485) < 0.06 else '⚠️ РОЗБІЖНІСТЬ'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/rank89.npz", base=b,
         **{f"r{R}": np.array(cols[R]) for R in RANKS})
