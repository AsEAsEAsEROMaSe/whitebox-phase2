"""k3src94 — ЗВІДКИ БЕРЕТЬСЯ κ₃ ПЕРЕАКТИВАЦІЙ: НАРОДЖУЄТЬСЯ ЧИ ПЕРЕНОСИТЬСЯ?

════════════════ ЩО ЦЕ РОЗДІЛЯЄ ════════════════
`[ЗМІРЯНО k3an94, 2 мережі × 2¹⁷]` ТРИ незалежні формули, що виводять κ₃(p')
з припущення «`h` гаусова», усі недобирають на ПОРЯДОК:
```
            шар 1   шар 11   шар 31        масштаб = у скільки разів ЗАНИЖУЮТЬ
Мелер ρ²    1.10     3.21     −2.10
ранг-1      1.23     9.72      2.90
діагональ   1.26    17.07     22.78        (відомі 0.077 з С78 ⟺ 13×) ✅пломба
```
🔑 Спільне в усіх трьох — припущення «`h` гаусова». Воно ХИБНЕ: `γ₃(h)` росте
0 → 0.434 вглиб. ⟹ у κ₃(p') є ДВА джерела, і формули бачать лише перше:
```
(1) НАРОДЖЕННЯ: relu робить `a` негаусовою навіть із гаусової `h`     ← рахують
(2) ПЕРЕНЕСЕННЯ: вже наявна негаусовість `h` проходить крізь relu     ← НЕ рахують
```
При `t = m/s → 3.0` маємо `Φ(t) → 0.9987`, тобто relu діє на хвіст майже як
ТОТОЖНІСТЬ ⟹ джерело (2) мусить бути майже ЛІНІЙНИМ перенесенням.

════════════════ ВИМІР ════════════════
Лінеаризований перенос (БЕЗ жодної нової негаусовості, лише наявна):
```
u = (h − m)/s          y_α = Σ_j W_jα·a₁_j·u_j ,  a₁_j = s_j·Φ(t_j)
κ₃^перенос(p'_α) = μ₃(y_α)          ← рахується З СЕМПЛІВ, один матмул
```
Якщо `cos(κ₃^перенос, κ₃^істина) > 0.9` і масштаб ≈ 1 ⟹ **κ₃ самовідтворюється,
його ТРЕБА нести, і жодна формула «з (m,C)» його не замінить.**
І тоді єдиний живий носій — **CP-ранг**: лінійне перетворення ЗБЕРІГАЄ CP-ранг
(`κ₃ = Σ_r λ_r u_r^{⊗3}` ⟹ `κ₃' = Σ_r λ_r (Wᵀu_r)^{⊗3}`), а діагональ, яка нам
і потрібна, — це `Σ_r λ_r·((Wᵀu_r)_α)³`, тобто **r матвеків на шар**.

════════════════ ПЛОМБИ ════════════════
P1 🔒 шар 0→1: `h₀` РІВНО гаусова ⟹ перенос мусить дати ~0, а істина ≠ 0
   (там працює лише народження). Якщо перенос там великий — формула хибна.
P2 🔒 ПЛОМБА МАСШТАБУВАННЯ: 2¹⁷ проти 2¹⁹ — cos і масштаб мусять СТОЯТИ.
P3 ⛔ КОНТРОЛЬ «ЧУЖА»: `a₁` з іншого шару. cos мусить впасти.
P4 🔒 СУМА: `перенос + народження(ранг-1)` мусить дати масштаб БЛИЖЧЕ до 1,
   ніж кожен окремо — інакше розклад на два джерела неправильний.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) перенос дасть cos > 0.90 на шарах ≥10                  шанс 75%
 (б) масштаб переносу впаде в 1.0…2.0 (проти 10…20)         шанс 60%
 (в) на шарі 1 перенос дасть ~0 (пломба P1)                 шанс 90%
 (г) частка переносу РОСТЕ з глибиною (бо Φ(t)→1)           шанс 80%
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
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 17
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


def mu3(chunks, mean):
    acc = torch.zeros(mean.shape[0], dtype=torch.float64, device=DEV)
    for c in chunks:
        d = (c - mean).double()
        acc += (d * d * d).sum(0)
    return acc / NSAMP


def cosscale(p, q):
    p, q = p.double(), q.double()
    c = float((p * q).sum() / (p.norm() * q.norm()).clamp_min(1e-300))
    return c, float((p * q).sum() / (p * p).sum().clamp_min(1e-300))


print(f"k3src94 · dev={DEV} · {NNET} мереж · N=2^{LOGN}", flush=True)
t0 = time.time()
CT, ST, CX, FRAC = [], [], [], []
with torch.no_grad():
    for qn, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        n = 256
        g = torch.Generator(device=DEV).manual_seed(950000 + 211 * qn)
        H = None                          # передактивації поточного шару
        ct, st, cx, fr = [], [], [], []
        a1_prev = None
        for L in range(32):
            hb = []
            s1 = torch.zeros(n, dtype=torch.float64, device=DEV)
            for off in range(0, NSAMP, CH):
                b = min(CH, NSAMP - off)
                if L == 0:
                    h = torch.randn(b, n, generator=g, device=DEV) @ W[0]
                else:
                    h = torch.relu(H[off:off + b]) @ W[L]
                s1 += h.double().sum(0)
                hb.append(h)
            m = (s1 / NSAMP)
            mf = m.float()
            v = torch.zeros(n, dtype=torch.float64, device=DEV)
            for h in hb:
                d = (h - mf).double()
                v += (d * d).sum(0)
            s = (v / NSAMP).clamp_min(1e-30).sqrt()
            k3_true = mu3(hb, mf)
            if L > 0:
                # ── ЛІНЕАРИЗОВАНИЙ ПЕРЕНОС із попереднього шару ──
                yb, sy = [], torch.zeros(n, dtype=torch.float64, device=DEV)
                for off in range(0, NSAMP, CH):
                    b = min(CH, NSAMP - off)
                    u = (H[off:off + b] - m_prev.float()) / s_prev.float()
                    y = u @ (a1_prev.unsqueeze(1) * W[L])
                    sy += y.double().sum(0)
                    yb.append(y)
                my = (sy / NSAMP).float()
                k3_tr = mu3(yb, my)
                c, sc = cosscale(k3_tr, k3_true)
                ct.append(c); st.append(sc)
                fr.append(float(k3_tr.norm() / k3_true.norm()))
                # ⛔ ЧУЖА: a₁ перевернутий по індексах
                yb2, sy2 = [], torch.zeros(n, dtype=torch.float64, device=DEV)
                for off in range(0, NSAMP, CH):
                    b = min(CH, NSAMP - off)
                    u = (H[off:off + b] - m_prev.float()) / s_prev.float()
                    y = u @ (a1_prev.flip(0).unsqueeze(1) * W[L])
                    sy2 += y.double().sum(0)
                    yb2.append(y)
                cx.append(cosscale(mu3(yb2, (sy2 / NSAMP).float()),
                                   k3_true)[0])
                del yb, yb2
            m_prev, s_prev = m, s
            a1_prev = (s * Phi(m / s)).float()
            H = torch.cat(hb, 0)
            del hb
        CT.append(ct); ST.append(st); CX.append(cx); FRAC.append(fr)
        print(f"  net{qn} готово ({time.time()-t0:.0f} с)", flush=True)

C_ = np.array(CT); S_ = np.array(ST); X_ = np.array(CX); F_ = np.array(FRAC)
print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 80)
print("κ₃(p') = ПЕРЕНЕСЕНА негаусовість `h` (лінеаризовано) ПРОТИ ІСТИНИ")
print("=" * 80)
print(f"{'шар':>5s}{'cos ПЕРЕНОС':>14s}{'масштаб':>11s}"
      f"{'‖перен‖/‖іст‖':>16s}{'cos ЧУЖА':>11s}")
for L in (0, 1, 3, 6, 10, 14, 18, 22, 26, 30):
    print(f"{L+1:>5d}{C_[:,L].mean():>14.4f}{S_[:,L].mean():>11.4f}"
          f"{F_[:,L].mean():>16.4f}{X_[:,L].mean():>11.4f}")
print(f"\n🔒P1 шар1 (вхід РІВНО гаусів): ‖перенос‖/‖істина‖ = {F_[:,0].mean():.4f}"
      f" {'✅ народження домінує там, де перенести нічого' if F_[:,0].mean() < 0.5 else '🚨 формула хибна'}")
print(f"⛔P3 чужа: cos={X_.mean():.4f} vs своя {C_.mean():.4f}")
print(f"\n📣 ПЕРЕНОС: cos={C_.mean():.4f} масшт={S_.mean():.4f} · "
      f"глибина(20-31) cos={C_[:,19:].mean():.4f} масшт={S_[:,19:].mean():.4f}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/k3src94.npz",
         ct=C_, st=S_, cx=X_, fr=F_)
