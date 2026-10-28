"""newalg95d — ДЕ САМЕ СИДИТЬ ПОХИБКА σ=1.8%, І ЧИ БЕРЕ ЇЇ ОДНОВИМІРНИЙ ЕДЖВОРТ

════════════════ ЗВІДКИ ════════════════
`[ЗМІРЯНО newalg95c, 4 мережі × 2¹⁸]` Мелер K=24 з ТОЧНИМИ (μ,σ,ρ):
   шар 30: σ 1.798% · діаг Cov 1.029% · позадіаг 4.081% · E[relu] 0.139%
   ⟹ ДРУГИЙ момент гірший за перший у 13×
`[ЗМІРЯНО там само, КОНТРОЛЬ]` вхід СФЕРА замість гауса: σ 1.790% — БЕЗ ЗМІН
   ⟹ ⛔ гіпотезу «1.8% = радіальна дисперсія ‖z‖» СПРОСТОВАНО (збіг 1.66% був
      випадковий; контроль урятував від висновку на збігу)
`[ЗМІРЯНО newalg95]` 12.3% нейронів мають |t|<1 на шарі 31, медіана |t|=3.50

════════════════ ГІПОТЕЗА ════════════════
При |t|≳3 ReLU майже не діє ⟹ σ(h)≈σ(p) незалежно від форми. Отже вся похибка
σ мусить сидіти на нейронах з МАЛИМ |t| — там, де ReLU справді нелінійний.
І якщо так — її бере ОДНОВИМІРНИЙ Еджворт по (γ₃,γ₄) кожного нейрона окремо,
без жодних матриць і тензорів.

════════════════ ЩО МІРЯЄМО (шари 7,15,23,30) ════════════════
 D1  розбивка похибки σ за квантилями |t| — де саме вона живе
 D2  чотири оцінки Var(ReLU(p_j)) з ТОЧНИМИ одновимірними моментами:
       (i)   гаусова          (μ,σ)
       (ii)  Еджворт +γ₃      (μ,σ,γ₃)
       (iii) Еджворт +γ₃+γ₄   (μ,σ,γ₃,γ₄)
       (iv)  ЕМПІРИЧНА густина p_j (гістограма) — межа одновимірного підходу
     проти істини Var(relu(p_j)) з семплів
 D3  те саме для E[ReLU] (для довідки — мусить лишитись майже точним)
🔒 ПЛОМБА: при γ₃=γ₄=0 схема (ii)/(iii) мусить ЗБІГТИСЬ із гаусовою до 1e-6
🔒 ПЛОМБА: (iv) на гаусових даних мусить дати похибку ≈ шум сітки, не 1.8%

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) 🔑 понад 70% похибки σ сидить у нейронах з |t| нижче медіани   шанс 70%
 (б) Еджворт +γ₃ зніме >40% похибки σ                              шанс 50%
 (в) Еджворт +γ₃+γ₄ зніме >70%                                     шанс 40%
 (г) 🔑 емпірична 1-D густина зніме >90% ⟹ проблема ОДНОВИМІРНА     шанс 45%
 (д) якщо (г) провалиться — похибка у СПІЛЬНОМУ розподілі, і жодна
     покоординатна поправка її не візьме                           шанс 55%
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
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
GAUSS_CTRL = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NSAMP = 1 << LOGN
CH = 1 << 14
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
PROBE = [7, 15, 23, 30]
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
NY = 4001
YMAX = 9.0


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


def gauss_relu(t):
    """(E[relu(t+Y)], E[relu(t+Y)²]) для Y~N(0,1), у одиницях σ."""
    P, f = Phi(t), phi(t)
    m1 = f + t * P
    m2 = (1.0 + t * t) * P + t * f
    return m1, m2


def edge_relu(t, g3, g4, y, wq):
    """Ті самі моменти, але густина Еджворта 1+γ₃He₃/6+γ₄He₄/24 (сітка y)."""
    # y: (NY,), t/g3/g4: (D,)  →  працюємо (D, NY) чанками по нейронах
    he3 = y ** 3 - 3.0 * y
    he4 = y ** 4 - 6.0 * y * y + 3.0
    base = phi(y) * wq
    D = t.shape[0]
    m1 = torch.zeros(D, dtype=y.dtype, device=y.device)
    m2 = torch.zeros(D, dtype=y.dtype, device=y.device)
    for o in range(0, D, 64):
        tt = t[o:o + 64].unsqueeze(1)
        dens = base * (1.0 + g3[o:o + 64].unsqueeze(1) * he3 / 6.0
                       + g4[o:o + 64].unsqueeze(1) * he4 / 24.0)
        dens = dens.clamp_min(0.0)
        dens = dens / dens.sum(1, keepdim=True)
        r = (tt + y).clamp_min(0.0)
        m1[o:o + 64] = (dens * r).sum(1)
        m2[o:o + 64] = (dens * r * r).sum(1)
    return m1, m2


def analyse(h, Wn, n, y, wq):
    dev = h.device
    z64 = dict(dtype=torch.float64, device=dev)
    D = h.shape[1]
    s1 = torch.zeros(D, **z64); s2 = torch.zeros(D, **z64)
    s3 = torch.zeros(D, **z64); s4 = torch.zeros(D, **z64)
    r1 = torch.zeros(D, **z64); r2 = torch.zeros(D, **z64)
    for o in range(0, n, CH):
        pb = h[o:o + CH].double() @ Wn
        rb = pb.clamp_min(0.0)
        s1 += pb.sum(0); s2 += (pb * pb).sum(0)
        r1 += rb.sum(0); r2 += (rb * rb).sum(0)
    mu = s1 / n
    var = (s2 / n - mu * mu).clamp_min(1e-300)
    sd = var.sqrt()
    for o in range(0, n, CH):
        c = (h[o:o + CH].double() @ Wn - mu) / sd
        s3 += (c ** 3).sum(0); s4 += (c ** 4).sum(0)
    g3 = s3 / n
    g4 = s4 / n - 3.0
    t = mu / sd
    # істина
    m_true = r1 / n
    v_true = (r2 / n - m_true * m_true).clamp_min(1e-300)
    sig_true = v_true.sqrt()
    # (i) гаусова
    a1, a2 = gauss_relu(t)
    m_g = sd * a1
    sig_g = (sd * sd * (a2 - a1 * a1)).clamp_min(1e-300).sqrt()
    # (ii) Еджворт лише γ₃
    z0 = torch.zeros_like(g3)
    b1, b2 = edge_relu(t, g3, z0, y, wq)
    m_e3 = sd * b1
    sig_e3 = (sd * sd * (b2 - b1 * b1)).clamp_min(1e-300).sqrt()
    # (iii) Еджворт γ₃+γ₄
    c1_, c2_ = edge_relu(t, g3, g4, y, wq)
    m_e4 = sd * c1_
    sig_e4 = (sd * sd * (c2_ - c1_ * c1_)).clamp_min(1e-300).sqrt()
    return dict(t=t, sig_true=sig_true, m_true=m_true, sd=sd,
                sig_g=sig_g, sig_e3=sig_e3, sig_e4=sig_e4,
                m_g=m_g, m_e3=m_e3, m_e4=m_e4, g3=g3, g4=g4)


def rel(a, b):
    return float((a - b).norm() / b.norm())


print(f"newalg95d · {DEV} · {NNET} мереж · N=2^{LOGN} · GAUSS_CTRL={GAUSS_CTRL}",
      flush=True)
t0 = time.time()
y = torch.linspace(-YMAX, YMAX, NY, dtype=torch.float64, device=DEV)
wq = float(2 * YMAX / (NY - 1))
ROW, BUCK = [], []
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        g = torch.Generator(device=DEV).manual_seed(953000 + 151 * q)
        X = torch.randn(NSAMP, 256, generator=g, device=DEV)
        row, buck = {}, {}
        for L in range(32):
            X = torch.relu(X @ W[L])
            if L in PROBE and L + 1 < 32:
                Wn = W[L + 1].double()
                Xa = X
                if GAUSS_CTRL:
                    # 🔒 ПЛОМБА: підмінити h на ГАУСІВ з тими самими (m,Cov)
                    md = torch.zeros(256, dtype=torch.float64, device=DEV)
                    acc = torch.zeros(256, 256, dtype=torch.float64, device=DEV)
                    for o in range(0, NSAMP, CH):
                        hb = X[o:o + CH].double()
                        md += hb.sum(0); acc += hb.T @ hb
                    md /= NSAMP
                    Cv = acc / NSAMP - torch.outer(md, md)
                    Lc = torch.linalg.cholesky(
                        Cv + 1e-10 * torch.eye(256, dtype=Cv.dtype, device=DEV))
                    gg = torch.Generator(device=DEV).manual_seed(7 + q)
                    Xa = (torch.randn(NSAMP, 256, generator=gg, device=DEV,
                                      dtype=torch.float64) @ Lc.T + md).float()
                a = analyse(Xa, Wn, NSAMP, y, wq)
                row[L] = (rel(a['sig_g'], a['sig_true']),
                          rel(a['sig_e3'], a['sig_true']),
                          rel(a['sig_e4'], a['sig_true']),
                          rel(a['m_g'], a['m_true']),
                          rel(a['m_e3'], a['m_true']),
                          rel(a['m_e4'], a['m_true']),
                          float(a['g3'].abs().mean()),
                          float(a['g4'].abs().mean()))
                # D1 — розбивка похибки σ за квантилями |t|
                at = a['t'].abs()
                order = torch.argsort(at)
                err = (a['sig_g'] - a['sig_true']) ** 2
                tot = float(err.sum())
                bs = []
                for k in range(4):
                    idx = order[k * 64:(k + 1) * 64]
                    bs.append((float(err[idx].sum() / max(tot, 1e-300)),
                               float(at[idx].mean())))
                buck[L] = bs
        ROW.append(row); BUCK.append(buck)
        print(f"  net{q} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 84)
print("D2 — ЧИ БЕРЕ ОДНОВИМІРНИЙ ЕДЖВОРТ ПОХИБКУ σ (усе з ТОЧНИМИ 1-D моментами)")
print("=" * 84)
print(f"{'шар':>4s}{'σ гаус':>10s}{'σ +γ₃':>10s}{'σ +γ₃γ₄':>10s}"
      f"{'×γ₃':>7s}{'×γ₃γ₄':>8s}{'|γ₃|':>8s}{'|γ₄|':>8s}")
for L in PROBE:
    a = np.mean([r[L] for r in ROW if L in r], axis=0)
    print(f"{L:>4d}{a[0]:>10.5f}{a[1]:>10.5f}{a[2]:>10.5f}"
          f"{a[0]/max(a[1],1e-12):>7.2f}{a[0]/max(a[2],1e-12):>8.2f}"
          f"{a[6]:>8.4f}{a[7]:>8.4f}")

print("\nD3 — той самий розклад для E[relu] (для довідки)")
for L in PROBE:
    a = np.mean([r[L] for r in ROW if L in r], axis=0)
    print(f"  шар {L:>2d}: гаус {a[3]:.5f} · +γ₃ {a[4]:.5f} · +γ₃γ₄ {a[5]:.5f} "
          f"· ×γ₃γ₄ {a[3]/max(a[5],1e-12):.2f}")

print("\n" + "=" * 84)
print("D1 — ДЕ ЖИВЕ ПОХИБКА σ: частка Σ(Δσ)² по квартилях |t| (64 нейрони кожен)")
print("=" * 84)
print(f"{'шар':>4s}{'Q1 (найм|t|)':>14s}{'Q2':>10s}{'Q3':>10s}"
      f"{'Q4 (найб|t|)':>14s}   середні |t|")
for L in PROBE:
    bs = np.mean([[x[0] for x in b[L]] for b in BUCK if L in b], axis=0)
    ts = np.mean([[x[1] for x in b[L]] for b in BUCK if L in b], axis=0)
    print(f"{L:>4d}{bs[0]:>14.4f}{bs[1]:>10.4f}{bs[2]:>10.4f}{bs[3]:>14.4f}"
          f"   {ts[0]:.2f}/{ts[1]:.2f}/{ts[2]:.2f}/{ts[3]:.2f}")

print("\n🔑 ЧИТАТИ: якщо ×γ₃γ₄ велике — похибка ОДНОВИМІРНА і береться дешево")
print("   (γ₃,γ₄ покоординатно = вектори, не тензори). Якщо ×≈1 — вона у")
print("   СПІЛЬНОМУ розподілі, і покоординатні поправки мертві за побудовою.")
