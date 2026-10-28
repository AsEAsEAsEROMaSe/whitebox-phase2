"""cum89d — ДВА ЗДЕШЕВЛЕННЯ κ₂₁, ЩОБ ВЛІЗТИ В ГЕЙТ ОКУПНОСТІ.

════════════════ ГЕЙТ, ВИВЕДЕНИЙ У С89 ════════════════
Ми стоїмо РІВНО на підлозі бюджету, тож будь-яка додаткова детермінована
робота ціною `c` забирає семпли в МК. Оскільки `MSE ≈ 0.912·V`, окупність
жорстка:
    k=1.35 → дозволено 0.284% B  ·  k=2.00 → 0.811%  ·  k=2.48 → 1.204%
`cum89c`: найкраще N_orc=256, λ=0.4, k=1.348, але ціна **0.76% B** ⟹
перевищення у **2.7×**, і нетто виходить ×0.947 (ГІРШЕ за рекорд).

════════════════ ДВА ЗДЕШЕВЛЕННЯ ════════════════
(1) ×0.5 — `κ₂₁ = E[h_j²h_k] − 2m_j·E[h_jh_k] − m_k·E[h_j²] + 2m_j²m_k`
    вимагає ДВОХ матмулів, але `E[h_jh_k] = cp_jk + m_j m_k` і `E[h_j²]`,
    `m_j` замикання дає АНАЛІТИЧНО. ⟹ МК потрібен лише для `(h²)ᵀh`.
    🚨 Ціна: аналітичні `m, cp` самі несуть похибку замикання (та, яку ми й
    виправляємо) ⟹ можливе самоотруєння. Саме це й міряємо.
(2) ×0.68…0.36 — застосовувати лише на ГЛИБОКИХ шарах: безрозмірний |κ₂₁|
    дорівнює 0.0044 на шарі 1 і 0.160 на шарі 21 ⟹ мілкі шари майже нічого
    не дають, але коштують стільки ж.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
(1) самоотруєння З'ЇСТЬ від третини до половини `k` (шанс 60%): аналітична
    `cp` завищена саме там, де κ₂₁ великий. Якщо k впаде нижче 1.2 —
    здешевлення (1) мертве і лишається тільки (2).
(2) обрізання до шарів ≥16 забере ~15% `k` (шанс 55%), бо `tprof88` показав,
    що еф. ранг падає 127.6→3.1 саме на глибині, тобто там і живе кореляція.
🎯 ЩОБ ГІЛКА ОЖИЛА, потрібно НЕТТО > ×1.00. Ставлю на це 30%.

════════════════ ПЛОМБИ ════════════════
P1  повний МК-варіант (обидва матмули, усі шари) мусить відтворити
    cum89c: N=256 λ=0.4 → k=1.348
P2  при START=0 і повній МК гілки (1) і (0) мусять збігтися до МК-шуму
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
NORC = 256
LAMS = (0.25, 0.4, 0.55)
STARTS = (0, 8, 16, 21)
NTRUTH = 1 << 16
V0, E20, SAMP = 2.050e-06, 2.118e-05, 0.0838
B = 2.72e11

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · N_orc={NORC} · λ {LAMS} · старти {STARTS}")


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


def closure_state(ws):
    """Аналітичні `m`, `cp` пошарово — те, чим замінюємо ДРУГИЙ матмул."""
    m = torch.zeros(n, dtype=DT, device=DEV)
    C = torch.eye(n, dtype=DT, device=DEV)
    MS, CPS = [], []
    for i in range(L - 1):
        w = ws[i]
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
        mp = m @ w
        MS.append(mp)
        CPS.append(cp)
        t = mp / s
        P, f = Phi(t), phi(t)
        e1 = s * (t * P + f)
        e2 = (mp * mp + s * s) * P + (mp * s) * f
        var = torch.clamp(e2 - e1 * e1, min=1e-30)
        Ca = cp * torch.outer(P, P) + 0.5 * (cp * cp) * torch.outer(f / s, f / s)
        Ca = Ca + torch.diag(var - torch.diagonal(Ca))
        m, C = e1, Ca
    return MS, CPS


def kappa_pass(ws, nsamp, seed, cheap, MS=None, CPS=None, start=0):
    """cheap=False — обидва матмули з МК; cheap=True — лише `(h²)ᵀh`."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    H1 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    H2 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    HH = [None] * L
    H2H = [torch.zeros(n, n, dtype=torch.float64, device=DEV) for _ in range(L)]
    if not cheap:
        HH = [torch.zeros(n, n, dtype=torch.float64, device=DEV) for _ in range(L)]
    done = 0
    while done < nsamp:
        b = min(1 << 13, nsamp - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(L - 1):
            h = x @ ws[i]
            if i >= start:
                hd = h.double()
                H2H[i] += (hd * hd).T @ hd
                if not cheap:
                    H1[i] += hd.sum(0)
                    H2[i] += (hd * hd).sum(0)
                    HH[i] += hd.T @ hd
            x = torch.clamp(h, min=0.0)
        done += b
    out = [None] * L
    for i in range(start, L - 1):
        h2h = H2H[i] / nsamp
        if cheap:
            m1 = MS[i].double()
            cpd = CPS[i].double()
            hh = cpd + torch.outer(m1, m1)
            m2 = torch.diagonal(cpd) + m1 * m1
            v = torch.clamp(torch.diagonal(cpd), min=1e-30)
        else:
            m1, m2 = H1[i] / nsamp, H2[i] / nsamp
            hh = HH[i] / nsamp
            v = torch.clamp(m2 - m1 * m1, min=1e-30)
        k21 = (h2h - 2.0 * m1.view(-1, 1) * hh - m1.view(1, -1) * m2.view(-1, 1)
               + 2.0 * (m1 * m1).view(-1, 1) * m1.view(1, -1))
        out[i] = (k21 / torch.outer(v, torch.sqrt(v))).to(DT)
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


KEYS = [(ch, st, l) for ch in (False, True) for st in STARTS for l in LAMS]
base, cols = [], {k: [] for k in KEYS}
t0 = time.time()
for j in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[j, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth_pass(ws, NTRUTH, 7000 + j).to(DT)
    den = float((tr * tr).mean())
    base.append(float(((predict(ws) - tr) ** 2).mean() / den) ** 0.5)
    MS, CPS = closure_state(ws)
    for ch in (False, True):
        for st in STARTS:
            K = kappa_pass(ws, NORC, 555000 + j, ch, MS, CPS, st)
            for l in LAMS:
                p = predict(ws, K21=K, lam=l)
                cols[(ch, st, l)].append(float(((p - tr) ** 2).mean() / den) ** 0.5)
    if j % 8 == 0:
        print(f"  net{j:>3d} база {100 * base[-1]:.4f}%")

b = np.array(base)
print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 88)
print(f"ДВА ЗДЕШЕВЛЕННЯ · {NNET} мереж · N_orc={NORC} · істина 2^16 · сіди розділені")
print("=" * 88)
print(f"  база = {100 * b.mean():.4f}%")
print(f"{'варіант':>34s} {'ціна%B':>8s} {'k':>7s} {'дозв.%B':>9s} {'НЕТТО бал':>12s} {'проти рек.':>11s}")
best = (None, 1e9)
for ch in (False, True):
    for st in STARTS:
        nlay = (L - 1) - st
        mm = 2.0 if ch else 4.0
        cost = mm * n * n * NORC * nlay / B
        for l in LAMS:
            k = float((b ** 2).mean() / (np.array(cols[(ch, st, l)]) ** 2).mean())
            Vn = V0 * SAMP / max(SAMP - cost, 1e-9)
            mse = Vn * (E20 / k) / (Vn + E20 / k)
            lo, hi = 0.0, SAMP * 0.9
            for _ in range(60):
                c = (lo + hi) / 2
                mm2 = V0 * SAMP / (SAMP - c) * (E20 / k) / (V0 * SAMP / (SAMP - c) + E20 / k)
                lo, hi = (c, hi) if mm2 < V0 * E20 / (V0 + E20) else (lo, c)
            nm = ("ДЕШЕВА (1 матмул)" if ch else "повна (2 матмули)") + f" шари≥{st} λ={l}"
            print(f"{nm:>34s} {100 * cost:>7.3f}% {k:>7.3f} {100 * lo:>8.3f}%"
                  f" {mse * 0.1:>12.4e} {'×' + format(1.8676e-07 / (mse * 0.1), '.4f'):>11s}")
            if mse < best[1]:
                best = ((ch, st, l, k, cost), mse)
(bch, bst, bl, bk, bc), bmse = best
print(f"\n  🎯 НАЙКРАЩЕ: {'дешева' if bch else 'повна'} шари≥{bst} λ={bl} k={bk:.3f} "
      f"ціна {100 * bc:.3f}%B ⟹ бал {bmse * 0.1:.4e} "
      f"(×{1.8676e-07 / (bmse * 0.1):.4f} проти рекорду 1.8676e-07)")
kp = float((b ** 2).mean() / (np.array(cols[(False, 0, 0.4)]) ** 2).mean())
print(f"  🔬 P1: повна/шари≥0/λ=0.4 дає k={kp:.3f}, cum89c дав 1.348  "
      f"{'✅' if abs(kp - 1.348) < 0.06 else '⚠️ РОЗБІЖНІСТЬ'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/cum89d.npz", base=b,
         **{f"c{int(ch)}_s{st}_l{l}": np.array(cols[(ch, st, l)]) for ch, st, l in KEYS})
