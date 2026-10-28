"""turbo94 — «МЕХАНІЧНИЙ АПАРАТ ТУРБІНИ»: чи є ВАЛ, що зчеплює 32 ступені?

════════════════ ЗВІДКИ (ідея ЮЗЕРА, «турбіна-2», Сесія 15) ════════════════
`[ЦИТАТА MEMORY_3.md:131]` «деталі поміряли але не СИМУЛЮВАЛИ: ми міряємо 32
ступені НЕЗАЛЕЖНО, а вони зобов'язані бути узгодженими (один потік)» …
`[ЦИТАТА там само]` «ідея фільтра уздовж глибини НЕ закрита — треба справжню
ПОСЛІДОВНУ СТРУКТУРУ (Кальман), не фічі-в-купу» (r116 пробував фічі — нейтраль).

Наш сабміт має 4 λ-родини × 8 шарів = **32 незалежні скаляри**, підігнані
регресією. Це і є «поміряли, але не симулювали».

════════════════ ЩО САМЕ ПЕРЕВІРЯЄМО ════════════════
Похибка замикання МУСИТЬ підкорятись власній рекурсії:
```
e_{L+1} = G_L·e_L + inj_L ,   G_L = якобіан кроку замикання (ВІДОМИЙ з ваг)
```
Якщо `G` справді передбачає похибку наступного шару — вал існує, і по ньому
можна фільтрувати (Кальман), об'єднуючи 31 шумне спостереження ОДНОГО стану.
Якщо `G` не передбачає — ступені розчеплені, і незалежні λ були правильні.

📐 `G` для похибки `δm` (лінеаризація кроку `m ← mᵀW`, потім relu-замикання):
```
δm_a = Φ(t)·δm_p + φ(t)·δs        ← ТОЧНО перший порядок (С66, вже в сабміті)
δm_p = Wᵀ·δm_a(попередній)
```
⟹ оператор `G_L = diag(Φ_L)·W_Lᵀ` (по каналу `m`) — ЧИСТО з ваг, без даних.

════════════════ ЩО МІРЯЄМО ════════════════
```
e_L = m_замик(L) − m_істина(L)        похибка замикання на шарі L (ВЕКТОР 256)
pred_{L+1} = G_L · e_L                 передбачення валом
cos(pred, e_{L+1}) і масштаб           ← ЦЕ І Є ВІДПОВІДЬ
```
```
⛔ КОНТРОЛЬ «ЧУЖА»: G з ІНШОЇ мережі ⟹ cos мусить впасти до ~0
⛔ КОНТРОЛЬ «БЕЗ ГЕЙТА»: G = Wᵀ без diag(Φ) ⟹ якщо не гірше, гейт зайвий
🔒 ПЛОМБА: на шарі 1 вхід РІВНО гаусів ⟹ e_1 ≈ 0 (тільки шум)
```
🔑 РІШУЧЕ ПИТАННЯ: скільки з `e_{L+1}` пояснює вал (`R²`), а скільки —
СВІЖА ін'єкція. Вал сильний ⟹ фільтр уздовж глибини має сенс; вал слабкий ⟹
кожен шар справді самостійний і λ-схема була правильною.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) cos(вал) > 0.7 на глибині (бо Φ→1 ⟹ майже лінійно)   шанс 65%
 (б) масштаб вала ≈ 1.0 ± 0.2                              шанс 55%
 (в) «без гейта» буде ПОМІТНО гірше (гейт несе інформацію)  шанс 50%
 (г) R² вала на глибині > 0.5 ⟹ фільтр окупний             шанс 45%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NSAMP = 1 << LOGN
CH = 1 << 15
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i]
            got += 1


def Phi(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def phi(x):
    return torch.exp(-0.5 * x * x) * S2PI


def truth_ma(W, seed):
    """Істинні `E[a]` активацій КОЖНОГО шару (те, що замикання й передбачає)."""
    n = W[0].shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    ma = []
    X = None
    for L in range(len(W)):
        s1 = torch.zeros(n, dtype=torch.float64, device=DEV)
        buf = []
        for off in range(0, NSAMP, CH):
            b = min(CH, NSAMP - off)
            h = (torch.randn(b, n, generator=g, device=DEV) @ W[0]
                 if L == 0 else X[off:off + b] @ W[L])
            buf.append(torch.relu(h))
            s1 += buf[-1].double().sum(0)
        ma.append(s1 / NSAMP)
        X = torch.cat(buf, 0)
        del buf
    return ma


def closure_trace(W, K=24):
    """Замикання Мелера + збережені `Φ`,`φ`,`s`,`m_a` кожного шару."""
    n = W[0].shape[0]
    m = torch.zeros(n, dtype=torch.float64, device=DEV)
    C = (W[0].double().T @ W[0].double())
    MA, PH, FI, SS = [], [], [], []
    for L in range(len(W)):
        s = C.diagonal().clamp_min(1e-30).sqrt()
        t = m / s
        P, f = Phi(t), phi(t)
        ma = s * (f + t * P)
        MA.append(ma); PH.append(P); FI.append(f); SS.append(s)
        if L == len(W) - 1:
            break
        rho = (C / torch.outer(s, s)).clamp(-1.0, 1.0)
        a1 = s * P
        cov = torch.outer(a1, a1) * rho
        Rn = rho.clone()
        Hpre, Hm = torch.zeros_like(t), torch.ones_like(t)
        fact, sf = 2.0, s * f
        for nn in range(2, K + 1):
            Rn = Rn * rho
            an = sf * Hm / fact
            cov = cov + torch.outer(an, an) * Rn * fact
            Hpre, Hm = Hm, (-t) * Hm - float(nn - 2) * Hpre
            fact *= (nn + 1)
        Wd = W[L + 1].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)
    return MA, PH, FI, SS


def cs(p, q):
    p, q = p.double(), q.double()
    nn = (p.norm() * q.norm()).clamp_min(1e-300)
    return float((p * q).sum() / nn), float((p * q).sum()
                                            / (p * p).sum().clamp_min(1e-300))


print(f"turbo94 · {DEV} · {NNET} мереж · N=2^{LOGN}", flush=True)
t0 = time.time()
CO, SC, CX, CN, R2 = [], [], [], [], []
prevG = None
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = truth_ma(W, 980000 + 157 * q)
        MA, PH, FI, SS = closure_trace(W)
        e = [MA[L] - tru[L] for L in range(32)]
        co, sc, cx, cn, r2 = [], [], [], [], []
        G = []
        for L in range(31):
            Wd = W[L + 1].double()
            # вал: δm_a(L+1) = Φ(L+1) ⊙ (Wᵀ·δm_a(L))
            G.append((PH[L + 1], Wd))
            pred = PH[L + 1] * (e[L] @ Wd)
            c, s_ = cs(pred, e[L + 1])
            co.append(c); sc.append(s_)
            r2.append(1.0 - float(((e[L + 1] - pred) ** 2).sum()
                                  / (e[L + 1] ** 2).sum().clamp_min(1e-300)))
            # ⛔ без гейта
            cn.append(cs(e[L] @ Wd, e[L + 1])[0])
            # ⛔ чужа: Φ і W з ІНШОЇ мережі
            if prevG is not None and L < len(prevG):
                pf, pw = prevG[L]
                cx.append(cs(pf * (e[L] @ pw), e[L + 1])[0])
        prevG = G
        CO.append(co); SC.append(sc); CN.append(cn); R2.append(r2)
        if cx:
            CX.append(cx)
        print(f"  net{q} ‖e₁‖/‖m₁‖={float(e[1].norm()/MA[1].norm()):.5f} "
              f"({time.time()-t0:.0f} с)", flush=True)

C_ = np.array(CO); S_ = np.array(SC); N_ = np.array(CN); R_ = np.array(R2)
X_ = np.array(CX) if CX else None
print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 76)
print("ВАЛ ТУРБІНИ: чи передбачає G=diag(Φ)·Wᵀ похибку наступного ступеня")
print("=" * 76)
print(f"{'шар':>5s}{'cos ВАЛ':>11s}{'масштаб':>10s}{'R²':>9s}"
      f"{'⛔без гейта':>13s}{'⛔ЧУЖА':>10s}")
for L in (0, 1, 3, 6, 10, 14, 18, 22, 26, 29):
    xv = X_[:, L].mean() if X_ is not None and L < X_.shape[1] else float('nan')
    print(f"{L+2:>5d}{C_[:,L].mean():>11.4f}{S_[:,L].mean():>10.4f}"
          f"{R_[:,L].mean():>9.4f}{N_[:,L].mean():>13.4f}{xv:>10.4f}")
print(f"\n📣 ВАЛ: cos={C_.mean():.4f} масшт={S_.mean():.4f} R²={R_.mean():.4f}")
print(f"   глибина (17-31): cos={C_[:,15:].mean():.4f} "
      f"R²={R_[:,15:].mean():.4f} · без гейта cos={N_[:,15:].mean():.4f}")
if X_ is not None:
    print(f"⛔ЧУЖА: cos={X_.mean():.4f} "
          f"{'✅ вал СВІЙ' if abs(X_.mean()) < 0.3 else '🚨 порожній контроль'}")
print("\n🔑 ЧИТАТИ: R² високий ⟹ ступені ЗЧЕПЛЕНІ, фільтр уздовж глибини має що")
print("   об'єднувати. R² низький ⟹ турбіна розсипана, незалежні λ були праві.")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/turbo94.npz",
         co=C_, sc=S_, cn=N_, r2=R_, cx=X_ if X_ is not None else np.zeros(0))
