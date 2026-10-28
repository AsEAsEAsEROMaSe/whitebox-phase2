"""rank89b — ЧЕСНА ЦІНА: κ₂₁ ОТРИМУЄТЬСЯ РАНДОМІЗОВАНОЮ ПРОЄКЦІЄЮ, НЕ ОРАКУЛОМ.

════════════════ ЩО ВИПРАВЛЯЄМО ════════════════
`rank89` дав нетто ×1.10 — але там ціна враховувала лише ЗАСТОСУВАННЯ
низькорангової поправки (`R` матвеків), а сама κ₂₁ бралася з оракульної МК на
2^16 семплів. Ціна ОТРИМАННЯ не врахована ⟹ таблиця була неповна, і я це
кажу прямо, а не переказую результат як готовий.

════════════════ ЧЕСНА СХЕМА ════════════════
Якщо κ₂₁ має ранг ~R, її НЕ треба матеріалізувати як `n²`. Досить `R`
випадкових проєкцій (рандомізований ескіз):
    Y_r = E[δh²·(δh·g_r)]     — `R` векторів замість матриці `n×n`
Ціна: `4·R·n·N` замість `4·n²·N` ⟹ виграш `n/R` = 32× при R=8.
Плюс застосування `2·R·n²` на шар.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
Ескіз рангу R з N семплів несе ДВА шуми одразу — від скінченного N і від
випадкових напрямків. Ставлю: при N=2048, R=8 залишиться k = 1.4…1.7
(шанс 50%), і нетто буде ×1.03…1.06. Якщо ж k впаде нижче 1.25 — ескіз
не працює, і треба точні напрямки (степеневий метод), що дорожче.
🚨 Окремо перевіряю ПЛОМБУ: ескіз при R=256 і N=2^16 мусить відтворити
повну κ₂₁ (k=2.262).
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
LAM = 0.75
COMB = [(R, N, IT) for R in (4, 8, 16) for N in (1024, 2048) for IT in (1, 2, 3)]
NTRUTH = 1 << 16
V0, E20, SAMP = 2.050e-06, 2.118e-05, 0.0838
B = 2.72e11

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · λ={LAM} · (R,N) {COMB}")


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


def sketch_once(ws, nsamp, seed, G):
    """РАНДОМІЗОВАНИЙ ЕСКІЗ κ₂₁: `R` проєкцій замість матриці `n×n`.

    κ₂₁ ≈ Y·(GᵀΣG)⁻¹Gᵀ… — беремо простий варіант: Y = κ₂₁·G, а далі
    ортогоналізуємо Y і проєктуємо: κ₂₁ ≈ Q(Qᵀκ₂₁), де Qᵀκ₂₁ теж рахується
    з тих самих семплів як `E[(Qᵀδh²)·δh]` — ще R редукцій.
    """
    R = G[0].shape[1]
    Gd = [g.double() for g in G]
    g = torch.Generator(device=DEV).manual_seed(seed)
    S1 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    S2 = [torch.zeros(n, dtype=torch.float64, device=DEV) for _ in range(L)]
    AA = [torch.zeros(n, R, dtype=torch.float64, device=DEV) for _ in range(L)]
    BB = [torch.zeros(n, R, dtype=torch.float64, device=DEV) for _ in range(L)]
    done = 0
    while done < nsamp:
        b = min(1 << 13, nsamp - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(L - 1):
            h = x @ ws[i]
            hd = h.double()
            h2 = hd * hd
            pr = hd @ Gd[i]                  # (b,R) — R матвеків, НЕ матриця n²
            S1[i] += hd.sum(0)
            S2[i] += h2.sum(0)
            AA[i] += h2.T @ pr               # E[h_j²·p_r]
            BB[i] += hd.T @ pr               # E[h_j·p_r]
            x = torch.clamp(h, min=0.0)
        done += b
    # pinv(G) = (GᵀG)⁻¹Gᵀ — точне відновлення для матриці рангу ≤ R
    pinvG = [torch.linalg.pinv(g) for g in Gd]
    YS, VS = [None] * L, [None] * L
    for i in range(L - 1):
        m1, m2 = S1[i] / nsamp, S2[i] / nsamp
        A, Bm = AA[i] / nsamp, BB[i] / nsamp
        mg = m1 @ Gd[i]
        # κ₂₁G[j,r] = E[δh_j²(δh·g_r)]
        YS[i] = (A - 2.0 * m1.view(-1, 1) * Bm
                 + (m1 * m1).view(-1, 1) * mg.view(1, -1)
                 - (m2 - m1 * m1).view(-1, 1) * mg.view(1, -1))
        VS[i] = torch.clamp(m2 - m1 * m1, min=1e-30)
    return YS, VS, pinvG


def sketch_power(ws, nsamp, seed, R, gseed, n_iter):
    """СТЕПЕНЕВИЙ МЕТОД: `G` оновлюється з попереднього проходу.

    Випадковий `G` не ловить головні напрямки (k=1.10 проти 2.07 у SVD).
    Кожна ітерація — ще один прохід по тих самих семплах з `R` проєкціями.
    ⚠️ `G` тут СПІЛЬНИЙ для всіх шарів (інакше треба тримати L матриць);
    беремо ортобазу від СЕРЕДНЬОГО `Y` по глибоких шарах, де κ₂₁ великий.
    """
    gg = torch.Generator(device=DEV).manual_seed(gseed)
    # 🔑 G ОКРЕМИЙ НА КОЖЕН ШАР: головні напрямки κ₂₁ у шарів РІЗНІ, і спільна
    #    ортобаза (перша спроба) валила k до 0.87. Ціна та сама — R матвеків.
    G = [torch.randn(n, R, generator=gg, device=DEV, dtype=DT) for _ in range(L)]
    YS = VS = pinvG = None
    for it in range(n_iter):
        YS, VS, pinvG = sketch_once(ws, nsamp, seed, G)
        if it < n_iter - 1:
            G = []
            for i in range(L):
                if YS[i] is None:
                    G.append(torch.randn(n, R, generator=gg, device=DEV, dtype=DT))
                else:
                    Q, _ = torch.linalg.qr(YS[i])
                    G.append(Q.to(DT))
    out = [None] * L
    for i in range(L - 1):
        K = YS[i] @ pinvG[i]
        out[i] = (K / torch.outer(VS[i], torch.sqrt(VS[i]))).to(DT)
    return out


def predict(ws, K21=None, lam=1.0):
    m = torch.zeros(n, dtype=DT, device=DEV)
    C = torch.eye(n, dtype=DT, device=DEV)
    for i in range(L - 1):
        w = ws[i]
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
        t = (m @ w) / s
        mp = m @ w
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


base, cols = [], {c: [] for c in COMB}
t0 = time.time()
for j in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[j, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth_pass(ws, NTRUTH, 7000 + j).to(DT)
    den = float((tr * tr).mean())
    base.append(float(((predict(ws) - tr) ** 2).mean() / den) ** 0.5)
    for (R, N, IT) in COMB:
        K = sketch_power(ws, N, 555000 + j, R, 99000 + j, IT)
        p = predict(ws, K21=K, lam=LAM)
        cols[(R, N, IT)].append(float(((p - tr) ** 2).mean() / den) ** 0.5)
    if j % 8 == 0:
        print(f"  net{j:>3d} база {100 * base[-1]:.4f}%  "
              f"R8N1024it3 {100 * cols[(8, 1024, 3)][-1]:.4f}%")

b = np.array(base)
print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 86)
print(f"ЧЕСНА ЦІНА: ЕСКІЗ РАНГУ R З N СЕМПЛІВ · {NNET} мереж · λ={LAM}")
print("=" * 86)
print(f"  база = {100 * b.mean():.4f}%   (для порівняння: оракул повного рангу k=2.262)")
print(f"{'R':>4s} {'N':>6s} {'iter':>5s} {'похибка':>10s} {'k':>7s} {'ціна отрим.':>12s} "
      f"{'ціна заст.':>11s} {'РАЗОМ':>8s} {'НЕТТО бал':>12s} {'проти рек.':>11s}")
best = (None, 1e9)
for (R, N, IT) in COMB:
    a = np.array(cols[(R, N, IT)])
    k = float((b ** 2).mean() / (a ** 2).mean())
    c_get = IT * 4.0 * R * n * N * (L - 1) / B
    c_app = 2.0 * R * n * n * (L - 1) / B
    cost = c_get + c_app
    Vn = V0 * SAMP / max(SAMP - cost, 1e-9)
    mse = Vn * (E20 / k) / (Vn + E20 / k)
    print(f"{R:>4d} {N:>6d} {IT:>5d} {100 * a.mean():>9.4f}% {k:>7.3f} {100 * c_get:>11.4f}%"
          f" {100 * c_app:>10.4f}% {100 * cost:>7.4f}% {mse * 0.1:>12.4e}"
          f" {'×' + format(1.8676e-07 / (mse * 0.1), '.4f'):>11s}")
    if mse < best[1]:
        best = ((R, N, IT, k, cost), mse)
(bR, bN, bIT, bk, bc), bmse = best
print(f"\n  🎯 НАЙКРАЩЕ: R={bR} N={bN} iter={bIT} k={bk:.3f} ціна {100 * bc:.4f}%B ⟹ "
      f"бал {bmse * 0.1:.4e} (×{1.8676e-07 / (bmse * 0.1):.4f} проти рекорду)")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/rank89b.npz", base=b,
         **{f"r{R}_n{N}_i{IT}": np.array(cols[(R, N, IT)]) for R, N, IT in COMB})
