"""lam89 — λ-КАНАЛИ СТОЯТЬ ЛИШЕ НА ШАРАХ 1..8. А ГРОШІ В ГЛИБИНІ.

════════════════ ЧОМУ САМЕ ЦЕ ════════════════
`hyb89` зміряв: чиста МК 0.187–0.257%, замикання 0.935% ⟹ **МК у 5× точніша**,
і 91.2% балу — її дисперсія. Ми ж усі сесії полірували замикання (вага 8.83%).
Дисперсію МК зменшують λ-канали (`_LAM_S/A/Q/G`), і вони стоять **лише на
шарах 1..8** — у шапці sub97 прямо: «рівно як у sub87», тобто успадковано, а
не оптимізовано. При цьому С88 зміряла: шари 1..8 дають рівно ×1.000, а
9..32 — **×1.333** (гроші в глибині).

════════════════ ЩО МІРЯЄМО ════════════════
Скільки ШУМУ ВИХОДУ пояснюють ознаки на шарах 1..8 проти 1..32. Для кожної
мережі женемо `M` НЕЗАЛЕЖНИХ МК-прогонів по `N` семплів, збираємо шум
`y_r − ȳ` і ознаки (ŝ−s_cl, x̄−m_cl, q̂−q_cl, gate−P_cl) пошарово, і рахуємо
`R²` пуленої регресії. `1/(1−R²)` = у скільки разів падає дисперсія.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
Ставлю до першого числа:
  шари 1..8 дадуть `R²` ≈ 0.65…0.72 (це відтворить відомий виграш 3.1–3.7×),
  а 1..32 — `R²` ≈ 0.80…0.88 (шанс 45%), тобто дисперсія впаде ще в 1.6…2.4×.
  Шанс, що глибокі шари дадуть менше за 20% приросту: 35%.
⚠️ Проти цього: канали на глибоких шарах СИЛЬНО КОРЕЛЬОВАНІ між собою
(активації на глибині мають еф. ранг 3.1) ⟹ 24 нових ознаки можуть нести
менше інформації, ніж 8 старих. Саме тому міряю `R²`, а не число ознак.
🚨 І головне: `R²` на N прогонах — це IN-SAMPLE. Тому рахую **LOO по мережах**:
λ фітяться на всіх мережах, крім тестової.

ПЛОМБА: R²(шари 1..8) мусить дати падіння дисперсії 3.1–3.7× — відоме число.
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
MREP = int(sys.argv[3]) if len(sys.argv) > 3 else 24
NS = 4096
DT = torch.float32
SQ2PI = float(np.sqrt(2.0 * np.pi))

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · прогонів/мережу {MREP} · N={NS}")


def phi(t):
    return torch.exp(-0.5 * t * t) / SQ2PI


def Phi(t):
    return 0.5 * (1.0 + torch.erf(t * 0.7071067811865476))


def closure(ws):
    """Опори замикання ПОШАРОВО (s, m, q, P) — на ВСІ шари, не лише 8."""
    m = torch.zeros(n, dtype=DT, device=DEV)
    C = torch.eye(n, dtype=DT, device=DEV)
    S, M, Q, P_ = [], [], [], []
    for i in range(L - 1):
        w = ws[i]
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
        mp = m @ w
        t = mp / s
        Pp, f = Phi(t), phi(t)
        e1 = s * (t * Pp + f)
        e2 = (mp * mp + s * s) * Pp + (mp * s) * f
        var = torch.clamp(e2 - e1 * e1, min=1e-30)
        Ca = cp * torch.outer(Pp, Pp) + 0.5 * (cp * cp) * torch.outer(f / s, f / s)
        Ca = Ca + torch.diag(var - torch.diagonal(Ca))
        S.append(s)
        M.append(e1)
        Q.append(torch.diagonal(Ca) + e1 * e1)
        P_.append(Pp)
        m, C = e1, Ca
    return S, M, Q, P_


def one_run(ws, S, M, Q, P_, seed):
    """Один МК-прогін: вихід + ознаки (4 на шар), усереднені по нейронах."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(NS, n, generator=g, device=DEV, dtype=DT)
    x = torch.cat([z, -z], 0)
    feats = []
    for i in range(L):
        p = x @ ws[i]
        if i == L - 1:
            out = torch.clamp(p, min=0.0).mean(0)
            break
        mh = p.mean(0)
        sh = torch.sqrt(torch.clamp((p * p).mean(0) - mh * mh, min=1e-30))
        gate = (p > 0).to(DT).mean(0)
        x = torch.clamp(p, min=0.0)
        xm = x.mean(0)
        xq = (x * x).mean(0)
        # чотири канали, згорнуті в скаляри вагою φ(t) — як у сабміті
        w_ = phi(mh / sh)
        feats += [float((w_ * (sh - S[i])).mean()),
                  float((xm - M[i]).mean()),
                  float((xq - Q[i]).mean()),
                  float((gate - P_[i]).mean())]
    return float(out.mean()), np.array(feats)


rows = []
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    S, M, Q, P_ = closure(ws)
    ys, fs = [], []
    for r in range(MREP):
        y, f = one_run(ws, S, M, Q, P_, 77000 + 991 * q + r)
        ys.append(y)
        fs.append(f)
    ys = np.array(ys)
    fs = np.array(fs)
    rows.append((ys - ys.mean(), fs - fs.mean(0)))
    if q % 8 == 0:
        print(f"  net{q:>3d}  шум виходу std={ys.std():.3e}  ({time.time() - t0:.0f} с)")


def loo_r2(rows, nlay):
    """LOO по мережах: λ фітяться на всіх, крім тестової."""
    cols = 4 * nlay
    num = den = 0.0
    for i in range(len(rows)):
        A = np.concatenate([rows[j][1][:, :cols] for j in range(len(rows)) if j != i])
        b = np.concatenate([rows[j][0] for j in range(len(rows)) if j != i])
        lam, *_ = np.linalg.lstsq(A, b, rcond=1e-8)
        yi, Fi = rows[i]
        res = yi - Fi[:, :cols] @ lam
        num += float((res ** 2).sum())
        den += float((yi ** 2).sum())
    return 1.0 - num / den


print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 74)
print(f"СКІЛЬКИ ШУМУ ВИХОДУ ЗНІМАЮТЬ λ-КАНАЛИ · {NNET} мереж × {MREP} прогонів")
print("=" * 74)
print(f"{'шарів':>7s} {'ознак':>7s} {'LOO R²':>10s} {'дисперсія падає в':>20s} {'бал':>12s}")
base = None
for nl in (4, 8, 12, 16, 24, 31):
    r2 = loo_r2(rows, nl)
    drop = 1.0 / max(1.0 - r2, 1e-9)
    if nl == 8:
        base = drop
    print(f"{nl:>7d} {4 * nl:>7d} {r2:>10.4f} {drop:>19.3f}×"
          f" {1.8676e-07 * (base / drop if base else 1):>12.4e}")
r8, r31 = loo_r2(rows, 8), loo_r2(rows, 31)
d8, d31 = 1 / (1 - r8), 1 / (1 - r31)
print(f"\n  🔬 ПЛОМБА: шари 1..8 дають падіння ×{d8:.2f} (відоме 3.1–3.7×) "
      f"{'✅' if 2.6 < d8 < 4.4 else '⚠️ НЕ ВІДТВОРЮЄ'}")
print(f"  🎯 ГЛИБИНА: 1..31 дає ×{d31:.2f} ⟹ додатковий виграш ×{d31 / d8:.3f}"
      f" ⟹ бал {1.8676e-07 * d8 / d31:.4e}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/lam89.npz",
         Y=np.array([r[0] for r in rows]), F=np.array([r[1] for r in rows]))
