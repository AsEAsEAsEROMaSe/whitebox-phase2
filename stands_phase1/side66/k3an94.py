"""k3an94 — κ₃ ПЕРЕАКТИВАЦІЙ, ОБЧИСЛЕНИЙ ЗІ СТАНУ (m,C), А НЕ ПЕРЕНЕСЕНИЙ.

════════════════ ДІРКА, ЯКУ ЦЕ ЗАКРИВАЄ ════════════════
`[ЦИТАТА finding-edgeworth-cumulants-are-the-live-branch]` Пам'ять тримає РІВНО
дві опції для κ₃ у ланцюгу, і обидві мертві:
```
діагональ  κ₃(p_α) ≈ Σ_j W_jα³·κ₃(a_j)   ⟹ cos 0.649, масштаб 0.077 (×13 замало)
точний     тензор 256³ на шар            ⟹ 2.75e11 = 10.1× бюджету          ⛔
```
🔑 ТРЕТЯ ОПЦІЯ НЕ РОЗГЛЯДАЛАСЬ: **κ₃ взагалі НЕ НЕСТИ.** Якщо `h` гаусова з
`(m,C)` — а це вже наше припущення — то `a = relu(h)` має κ₃ У ЗАМКНЕНОМУ
ВИГЛЯДІ (Мелер), і згортка з `W³` робиться АНАЛІТИЧНО, без тензора.

════════════════ ВИВЕДЕННЯ ════════════════
`h_j = m_j + s_j·u_j`, `u` стандартні гаусові з кореляцією `ρ_jk`.
`a_j = relu(h_j) = Σ_n a_{j,n}·He_n(u_j)`, де (Мелер, конвенція `E[He_nHe_m] =
δ_nm·n!·ρ^n`):
```
a_{j,0} = s_j·M(t_j)   a_{j,1} = s_j·Φ(t_j)   a_{j,n≥2} = s_j·φ(t_j)·He_{n−2}(−t_j)/n!
```
⟹ `a_{j,2} = s_j·φ(t_j)/2`.
Третій змішаний кумулянт трьох гаусових-функцій: найнижчий ненульовий порядок —
`(n,m,p)` = перестановки `(1,1,2)`, бо `E[He₁He₁He₂] = 2ρ_jl·ρ_kl`
(Іссерліс: `E[u_ju_ku_l²] = ρ_jk + 2ρ_jlρ_kl`, мінус `E[u_ju_k]·1`). ⟹
```
κ₃(a_j,a_k,a_l) ≈ 2[a₁ⱼa₁ₖa₂ₗ ρ_jlρ_kl + a₁ⱼa₁ₗa₂ₖ ρ_jkρ_lk + a₁ₖa₁ₗa₂ⱼ ρ_kjρ_lj]
```
Три доданки під симетричною згорткою `Σ_jkl W_jW_kW_l` рівні ⟹ множник 3:
```
κ₃(p_α) = 6·Σ_l W_lα·a₂_l·[Σ_j ρ_lj·W_jα·a₁_j]²  =  6·Σ_l W_lα·a₂_l·G[l,α]²
G = ρ @ (a₁ ⊙ W)                       ← ОДИН матмул 256³ = 3.36e7 на шар
```
💰 ЦІНА: 3.36e7 × 31 = 1.04e9 = **0.38% B**. Проти 10.1×B у точного тензора.
🔑 І це НЕ «незалежність нейронів»: перехресні члени `ρ_jl·ρ_kl` — саме те, що
діагональ викидала, і саме тому вона занижувала в 13×.

════════════════ ЧОМУ ЦЕ ВАРТО МІРЯТИ (ВАГА) ════════════════
`[ЦИТАТА field.py, 24 мережі, ТОЧНІ m,s]` маргінальні кумулянти при точних
моментах: гаус **1.116%** → +κ₃ **0.2027%** → +κ₄ **0.0496%** → +κ₅ **0.0256%**.
`[ЦИТАТА ladder87]` `L5` (оракул m,σ,γ) = 0.0162% ⟹ MSE 2.32e-08 ⟹ бал 2.32e-09.
`[ЦИТАТА борд]` топ-1 = 4.00e-10, топ-3 = 1.00e-09. ⟹ **клас накриває ціль.**

════════════════ ПЛОМБИ ════════════════
P1 🔒 ДІАГОНАЛЬНЕ наближення мусить відтворити ВІДОМІ `cos ≈ 0.649`,
   `масштаб ≈ 0.077` (`k3prop.py`, С78). Не відтворило ⟹ стенд не той об'єкт.
P2 🔒 На шарі 0 вхід РІВНО гаусів ⟹ `κ₃(h_0) ≡ 0` з точністю шуму. Це пломба
   на сам вимірювач κ₃ (він мусить давати ~0 там, де істина 0).
P3 🔒 ПЛОМБА МАСШТАБУВАННЯ: прогін на `N` і `N/4` — `cos` мусить СТОЯТИ, а не
   рости (інакше міряю шум).
P4 ⛔ КОНТРОЛЬ «ЧУЖА»: та сама формула з `ρ` ІНШОЇ мережі. Мусить дати cos ≈ 0.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) провідний член дасть cos > 0.90                       шанс 60%
 (б) масштаб потрапить у 0.6…1.4 (проти 0.077 діагоналі)   шанс 45%
 (в) масштаб буде СИСТЕМАТИЧНО < 1 (наступний порядок ρ³
     того самого знака ⟹ провідний недобирає)              шанс 55%
 (г) cos спадатиме з глибиною (ρ росте ⟹ ряд по ρ гірший)  шанс 70%
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
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 19
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


def phi(x):
    return torch.exp(-0.5 * x * x) * S2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def moments(W, seed):
    """Істинні (m, C, κ₃) ПЕРЕАКТИВАЦІЙ кожного шару + κ₃ АКТИВАЦІЙ."""
    n = W[0].shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    mh, Ch, k3h, k3a = [], [], [], []
    X = None
    for L in range(len(W)):
        # ── прохід чанками: два проходи, щоб центрувати чесно ──
        s1 = torch.zeros(n, dtype=torch.float64, device=DEV)
        buf = []
        for off in range(0, NSAMP, CH):
            b = min(CH, NSAMP - off)
            if L == 0:
                z = torch.randn(b, n, generator=g, device=DEV)
                h = z @ W[0]
            else:
                h = X[off:off + b] @ W[L]
            s1 += h.double().sum(0)
            buf.append(h)
        m = s1 / NSAMP
        mf = m.float()
        X = None                      # звільнити попередній шар ДО cat нижче
        s2 = torch.zeros(n, n, dtype=torch.float64, device=DEV)
        s3 = torch.zeros(n, dtype=torch.float64, device=DEV)
        sa1 = torch.zeros(n, dtype=torch.float64, device=DEV)
        for i, h in enumerate(buf):
            d = (h - mf).double()
            s2 += d.T @ d
            s3 += (d * d * d).sum(0)
            buf[i] = torch.relu(h)
            sa1 += buf[i].double().sum(0)
        mh.append(m)
        Ch.append(s2 / NSAMP)
        k3h.append(s3 / NSAMP)
        # κ₃ активацій (для контролю «діагональ») — другим проходом, без fp64-копії
        ma = (sa1 / NSAMP).float()
        s3a = torch.zeros(n, dtype=torch.float64, device=DEV)
        for h in buf:
            d = (h - ma).double()
            s3a += (d * d * d).sum(0)
        k3a.append(s3a / NSAMP)
        X = torch.cat(buf, 0)
        del buf
    return mh, Ch, k3h, k3a, X


def predict_k3(m, C, W):
    """Провідний член: κ₃(p_α) = 6·Σ_l W_lα·a₂_l·G[l,α]², G = ρ @ (a₁⊙W)."""
    s = C.diagonal().clamp_min(1e-30).sqrt()
    t = m / s
    a1 = s * Phi(t)
    a2 = s * phi(t) * 0.5
    rho = C / torch.outer(s, s)
    G = rho @ (a1.unsqueeze(1) * W)
    return 6.0 * ((W * a2.unsqueeze(1)) * G * G).sum(0)


def relu_mom(t, s):
    """E[a], E[a²], E[a³] для a=relu(m+s·ε), t=m/s. Усе через (t,s)."""
    P, f = Phi(t), phi(t)
    e1 = s * (f + t * P)
    e2 = s * s * ((1.0 + t * t) * P + t * f)
    e3 = s ** 3 * ((t ** 3 + 3.0 * t) * P + (t * t + 2.0) * f)
    return e1, e2, e3


def predict_k3_r1(m, C, W, K=24):
    """🆕 УМОВНО-ГАУСОВА СУМІШ ПО ОДНОМУ СПІЛЬНОМУ ФАКТОРУ (ранг-1).

    `ρ ≈ q qᵀ + diag(1−q²)` ⟹ умовно на `ξ` компоненти НЕЗАЛЕЖНІ ⟹ закон
    повних кумулянтів дає ТОЧНО:
    ```
    μ₃(p) = ⟨Σ_j W³·μ₃(a_j|ξ)⟩ + 3·Cov_ξ(A,B) + μ₃_ξ(A)
    A(ξ) = Wᵀμ(ξ)   B(ξ) = (W²)ᵀv(ξ)
    ```
    🔑 Перший доданок — це РІВНО діагональне наближення, але УМОВНЕ; уся маса,
    якої діагоналі бракувало в 13×, сидить у другому й третьому.
    """
    s = C.diagonal().clamp_min(1e-30).sqrt()
    rho = C / torch.outer(s, s)
    ev, V = torch.linalg.eigh(rho.double())
    q = (V[:, -1] * ev[-1].clamp_min(0).sqrt()).clamp(-0.995, 0.995).float()
    sc = s * (1.0 - q * q).clamp_min(1e-12).sqrt()          # умовна σ
    # вузли Гаусса–Ерміта по ξ
    x, w = np.polynomial.hermite_e.hermegauss(K)
    x = torch.as_tensor(x, dtype=torch.float32, device=m.device)
    w = torch.as_tensor(w / np.sqrt(2 * np.pi), dtype=torch.float64,
                        device=m.device)
    W2, W3 = W * W, W * W * W
    A = torch.empty(K, W.shape[1], dtype=torch.float64, device=m.device)
    B = torch.empty_like(A)
    Cc = torch.empty_like(A)
    for i in range(K):
        mc = m + s * q * x[i]
        t = mc / sc
        e1, e2, e3 = relu_mom(t, sc)
        v = (e2 - e1 * e1).clamp_min(0)
        k3 = e3 - 3.0 * e1 * e2 + 2.0 * e1 ** 3
        A[i] = (e1.double() @ W.double())
        B[i] = (v.double() @ W2.double())
        Cc[i] = (k3.double() @ W3.double())
    mA = (w[:, None] * A).sum(0)
    mB = (w[:, None] * B).sum(0)
    mC = (w[:, None] * Cc).sum(0)
    dA = A - mA
    cov = (w[:, None] * dA * (B - mB)).sum(0)
    k3A = (w[:, None] * dA ** 3).sum(0)
    return mC + 3.0 * cov + k3A


def cosscale(p, q):
    p = p.double(); q = q.double()
    c = float((p * q).sum() / (p.norm() * q.norm()).clamp_min(1e-300))
    sc = float((p * q).sum() / (p * p).sum().clamp_min(1e-300))
    return c, sc


print(f"k3an94 · dev={DEV} · {NNET} мереж · N=2^{LOGN}={NSAMP}", flush=True)
t0 = time.time()
COS, SCA, COSD, SCAD, COSX, CR1, SR1 = [], [], [], [], [], [], []

with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        mh, Ch, k3h, k3a, _ = moments(W, 940000 + 137 * q)
        p2 = float(k3h[0].abs().mean() / Ch[0].diagonal().mean() ** 1.5)
        print(f"  net{q}: 🔒P2 |κ₃(h₀)|/σ³ = {p2:.3e} "
              f"{'✅' if p2 < 0.01 else '🚨 вимірювач κ₃ бракований'} "
              f"({time.time()-t0:.0f} с)", flush=True)
        cs, sc, cd, sd, cx = [], [], [], [], []
        c1, s1v = [], []
        for L in range(31):
            tru = k3h[L + 1]
            pre = predict_k3(mh[L], Ch[L], W[L + 1]).double()
            c, s_ = cosscale(pre, tru)
            cs.append(c); sc.append(s_)
            # ⛔ контроль-діагональ: має відтворити 0.649 / 0.077
            dia = ((W[L + 1].double() ** 3) * k3a[L].unsqueeze(1)).sum(0)
            c2, s2_ = cosscale(dia, tru)
            cd.append(c2); sd.append(s2_)
            # 🆕 ранг-1 умовно-гаусова суміш
            pr1 = predict_k3_r1(mh[L].float(), Ch[L].float(), W[L + 1])
            c3, s3_ = cosscale(pr1, tru)
            c1.append(c3); s1v.append(s3_)
            # ⛔ контроль «чужа»: ρ з ІНШОЇ мережі (зсув шару на 7)
            if L >= 7:
                px = predict_k3(mh[L - 7], Ch[L - 7], W[L + 1]).double()
                cx.append(cosscale(px, tru)[0])
        COS.append(cs); SCA.append(sc); COSD.append(cd); SCAD.append(sd)
        CR1.append(c1); SR1.append(s1v)
        COSX.append(cx)

C_ = np.array(COS); S_ = np.array(SCA)
CD = np.array(COSD); SD = np.array(SCAD); CX = np.array(COSX)
R1 = np.array(CR1); RS = np.array(SR1)
print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 90)
print("κ₃(p) ОБЧИСЛЕНИЙ ЗІ СТАНУ (провідний член ρ²) ПРОТИ ІСТИНИ")
print("=" * 90)
print(f"{'шар':>5s}{'cos Мелер':>11s}{'масшт':>9s}"
      f"{'🆕cos РАНГ1':>13s}{'🆕масшт':>10s}"
      f"{'cos діаг':>10s}{'масшт':>9s}{'cos ЧУЖА':>10s}")
for L in (0, 1, 3, 6, 10, 14, 18, 22, 26, 30):
    xv = CX[:, L - 7].mean() if L >= 7 else float('nan')
    print(f"{L+1:>5d}{C_[:,L].mean():>11.4f}{S_[:,L].mean():>9.3f}"
          f"{R1[:,L].mean():>13.4f}{RS[:,L].mean():>10.3f}"
          f"{CD[:,L].mean():>10.4f}{SD[:,L].mean():>9.2f}{xv:>10.4f}")
print(f"\n🔒P1 діагональ: cos={CD.mean():.4f} (очік 0.649) · "
      f"масштаб={SD.mean():.4f} (очік 0.077) "
      f"{'✅ стенд той самий об єкт' if abs(CD.mean()-0.649) < 0.12 else '🚨 РОЗБІЖНІСТЬ'}")
print(f"⛔P4 чужа: cos={CX.mean():.4f} {'✅' if abs(CX.mean()) < 0.3 else '🚨 порожній контроль'}")
print(f"\n📣 Мелер ρ²: cos={C_.mean():.4f} масшт={S_.mean():.3f} · "
      f"🆕РАНГ-1: cos={R1.mean():.4f} масшт={RS.mean():.3f}")
print(f"   глибина (шари 20-31): Мелер {C_[:,19:].mean():.4f} · "
      f"РАНГ-1 {R1[:,19:].mean():.4f} · діаг {CD[:,19:].mean():.4f} · "
      f"чужа {CX[:,12:].mean():.4f}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/k3an94.npz",
         cos=C_, sca=S_, cosd=CD, scad=SD, cosx=CX, r1=R1, rs=RS)
