"""mix95 — СУМІШ K ГАУСІВ ЗАМІСТЬ ОДНОГО: чи падає похибка замикання як K?

════════════════ ЗВІДКИ (сторінка сабміту ЛІДЕРА dpskv5 #324969, 08.08) ═══════
```
BUDGET USED 9.59%  ·  MEAN EFFECTIVE COMPUTE 2.61e10  ·  HEADROOM 10×
final MSE 3.628e-9 · vs sampling 1782× (найкраща MLP 5051×)
розкид між мережами ×10 (1.28e-10…1.23e-9) · IQR 2.21…4.75e-10
ALL-LAYERS MSE 0.7437 · GROWTH L0→L31 5.36× · STEEPEST LAYER L1 7.4e8×
```
🔑 ЛІДЕР НА ТІЙ САМІЙ ПІДЛОЗІ (множник 0.1), різниця ЛИШЕ в MSE — ×513 від нас.
🔑 `2.61e10 / 1.66e9 (усе наше замикання) = ×15.7` ⟹ його метод коштує як
   ШІСТНАДЦЯТЬ наших замикань. Це не дешевий трюк, а чесно дорога схема.
🔑 `STEEPEST L1 = 7.4e8×` при all-layer MSE 0.7437 ⟹ він рахує ТОЧНО лише L0,
   решту шарів не заповнює ⟹ all-layer не є діагностикою методу.

════════════════ ГІПОТЕЗА ════════════════
×15.7 — це рівно ціна суміші з ~16 гаусових компонент. Суміш:
 · апроксимує розподіл із γ₃=0.38 там, де Еджворт РОЗХОДИТЬСЯ
 · всередині компоненти дисперсія менша ⟹ ReLU лінійніший ⟹ замикання точніше
 · відповідає геометрії: мережа КУСКОВО-ЛІНІЙНА, отже кусково-гаусове
   представлення — природний об'єкт, а не наближення рядом

════════════════ ЩО МІРЯЄМО (шари 7,15,23,30, ОДИН крок) ════════════════
Розділяємо ІСТИННІ семпли h на K кластерів за квантилями проєкції на напрямок v,
для КОЖНОГО кластера беремо свої (m_k, C_k), застосовуємо Мелер, зважено
складаємо перший і другий моменти. Порівнюємо з K=1 (звичайне замикання).
 напрямки v: (m) середнє · (e) топ-власний вектор Cov · (t) нейрон з мін |t|
🔑 РІШУЧЕ: чи падає похибка МОНОТОННО з K і як швидко (K^-1? K^-2?)
⛔ КОНТРОЛЬ «ЧУЖИЙ ПОДІЛ»: межі кластерів із ІНШОЇ мережі
🔒 ПЛОМБА: K=1 мусить ТОЧНО відтворити звичайний Мелер (до 1e-12)

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) 🔑 K=16 знизить похибку σ у >3×                            шанс 45%
 (б) падіння буде швидшим за K^-1                               шанс 35%
 (в) напрямок (m) буде найкращим із трьох                       шанс 40%
 (г) позадіагональ Cov покращиться СИЛЬНІШЕ за діагональ        шанс 50%
     (бо саме вона несе 4.08% проти 1.03%)
 (д) чужий поділ дасть ×1.0                                     шанс 75%
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
NSAMP = 1 << LOGN
CH = 1 << 14
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
PROBE = [7, 15, 23, 30]
KLIST = [1, 2, 4, 8, 16, 32]
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
MK = 24


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


def mehler(mu, s, rho):
    t = mu / s
    P, f = Phi(t), phi(t)
    ma = s * (f + t * P)
    a1 = s * P
    cov = torch.outer(a1, a1) * rho
    Rn = rho.clone()
    Hpre, Hm = torch.zeros_like(t), torch.ones_like(t)
    fact, sf = 2.0, s * f
    for nn in range(2, MK + 1):
        Rn = Rn * rho
        an = sf * Hm / fact
        cov = cov + torch.outer(an, an) * Rn * fact
        Hpre, Hm = Hm, (-t) * Hm - float(nn - 2) * Hpre
        fact *= (nn + 1)
    return ma, cov


def moments_of(h, Wn, idx):
    """(m, C) передактивації на підмножині семплів idx."""
    D = Wn.shape[1]
    z64 = dict(dtype=torch.float64, device=h.device)
    s1 = torch.zeros(D, **z64)
    s2 = torch.zeros(D, D, **z64)
    n = idx.numel()
    for o in range(0, n, CH):
        pb = h[idx[o:o + CH]].double() @ Wn
        s1 += pb.sum(0)
        s2 += pb.T @ pb
    mu = s1 / n
    return mu, s2 / n - torch.outer(mu, mu)


def truth_of(h, Wn):
    D = Wn.shape[1]
    z64 = dict(dtype=torch.float64, device=h.device)
    r1 = torch.zeros(D, **z64)
    r2 = torch.zeros(D, D, **z64)
    n = h.shape[0]
    for o in range(0, n, CH):
        rb = (h[o:o + CH].double() @ Wn).clamp_min(0.0)
        r1 += rb.sum(0)
        r2 += rb.T @ rb
    m = r1 / n
    return m, r2 / n - torch.outer(m, m)


def mixture(h, Wn, proj, K, bounds=None):
    """Суміш K гаусів: поділ за квантилями proj, Мелер у кожній компоненті."""
    n = h.shape[0]
    D = Wn.shape[1]
    z64 = dict(dtype=torch.float64, device=h.device)
    if K == 1:
        parts = [torch.arange(n, device=h.device)]
    else:
        if bounds is None:
            q = torch.linspace(0, 1, K + 1, dtype=torch.float64,
                               device=h.device)[1:-1]
            bounds = torch.quantile(proj, q)
        b = torch.bucketize(proj, bounds)
        parts = [torch.nonzero(b == k, as_tuple=True)[0] for k in range(K)]
    M = torch.zeros(D, **z64)
    C = torch.zeros(D, D, **z64)
    MS, WS = [], []
    for idx in parts:
        if idx.numel() < 8:
            continue
        w = idx.numel() / n
        mu, Cp = moments_of(h, Wn, idx)
        s = Cp.diagonal().clamp_min(1e-30).sqrt()
        rho = (Cp / torch.outer(s, s)).clamp(-1.0, 1.0)
        mk, ck = mehler(mu, s, rho)
        M += w * mk
        C += w * ck
        MS.append(mk); WS.append(w)
    MS = torch.stack(MS)
    WS = torch.tensor(WS, **z64)
    dm = MS - M
    C = C + (dm.T * WS) @ dm
    return M, C, (bounds if K > 1 else None)


def rel(a, b):
    return float((a - b).norm() / b.norm())


print(f"mix95 · {DEV} · {NNET} мереж · N=2^{LOGN} · K={KLIST}", flush=True)
t0 = time.time()
RES = []
prevB = {}
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        g = torch.Generator(device=DEV).manual_seed(956000 + 167 * q)
        X = torch.randn(NSAMP, 256, generator=g, device=DEV)
        row = {}
        for L in range(32):
            X = torch.relu(X @ W[L])
            if L in PROBE and L + 1 < 32:
                Wn = W[L + 1].double()
                m_t, C_t = truth_of(X, Wn)
                sig_t = C_t.diagonal().clamp_min(1e-300).sqrt()
                off = ~torch.eye(256, dtype=torch.bool, device=DEV)
                # напрямки поділу
                mh = torch.zeros(256, dtype=torch.float64, device=DEV)
                for o in range(0, NSAMP, CH):
                    mh += X[o:o + CH].double().sum(0)
                mh /= NSAMP
                u_m = mh / mh.norm()
                mu0, Cp0 = moments_of(X, Wn,
                                      torch.arange(NSAMP, device=DEV))
                t0v = mu0 / Cp0.diagonal().clamp_min(1e-30).sqrt()
                jmin = int(t0v.abs().argmin())
                PR = {}
                pm = torch.empty(NSAMP, dtype=torch.float64, device=DEV)
                pt = torch.empty(NSAMP, dtype=torch.float64, device=DEV)
                for o in range(0, NSAMP, CH):
                    hb = X[o:o + CH].double()
                    pm[o:o + CH] = hb @ u_m
                    pt[o:o + CH] = hb @ Wn[:, jmin]
                PR['m'] = pm
                PR['t'] = pt
                ent = {}
                for tag, pr in PR.items():
                    for K in KLIST:
                        M, C, bd = mixture(X, Wn, pr, K)
                        sg = C.diagonal().clamp_min(1e-300).sqrt()
                        ent[(tag, K)] = (rel(M, m_t), rel(sg, sig_t),
                                         rel(C.diagonal(), C_t.diagonal()),
                                         rel(C[off], C_t[off]))
                        if K == 16:
                            if (L, tag) in prevB:
                                Mx, Cx, _ = mixture(X, Wn, pr, K,
                                                    bounds=prevB[(L, tag)])
                                sx = Cx.diagonal().clamp_min(1e-300).sqrt()
                                ent[(tag, 'X')] = (rel(Mx, m_t),
                                                   rel(sx, sig_t),
                                                   rel(Cx.diagonal(),
                                                       C_t.diagonal()),
                                                   rel(Cx[off], C_t[off]))
                            prevB[(L, tag)] = bd
                row[L] = ent
        RES.append(row)
        print(f"  net{q} ({time.time()-t0:.0f} с)", flush=True)


def avg(L, key, i):
    v = [r[L][key][i] for r in RES if L in r and key in r[L]]
    return float(np.mean(v)) if v else float('nan')


print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 86)
print("СУМІШ K ГАУСІВ — ПОХИБКА ОДНОГО КРОКУ (K=1 — звичайне замикання)")
print("=" * 86)
for tag, nm in (('m', 'поділ за проєкцією на СЕРЕДНЄ'),
                ('t', 'поділ за нейроном з МІН |t|')):
    print(f"\n── {nm} ──")
    print(f"{'шар':>4s}{'K':>4s}{'E[relu]':>10s}{'σ':>10s}{'діаг':>10s}"
          f"{'поза':>10s}{'×σ vs K=1':>11s}{'×поза':>9s}")
    for L in PROBE:
        b_s = avg(L, (tag, 1), 1)
        b_o = avg(L, (tag, 1), 3)
        for K in KLIST:
            k = (tag, K)
            print(f"{L if K == KLIST[0] else '':>4}{K:>4d}"
                  f"{avg(L,k,0):>10.5f}{avg(L,k,1):>10.5f}"
                  f"{avg(L,k,2):>10.5f}{avg(L,k,3):>10.5f}"
                  f"{b_s/max(avg(L,k,1),1e-12):>11.2f}"
                  f"{b_o/max(avg(L,k,3),1e-12):>9.2f}")
        kx = (tag, 'X')
        if not np.isnan(avg(L, kx, 1)):
            print(f"{'':>4}{'⛔X':>4s}{avg(L,kx,0):>10.5f}{avg(L,kx,1):>10.5f}"
                  f"{avg(L,kx,2):>10.5f}{avg(L,kx,3):>10.5f}"
                  f"{b_s/max(avg(L,kx,1),1e-12):>11.2f}"
                  f"{b_o/max(avg(L,kx,3),1e-12):>9.2f}  чужий поділ")
print("\n🔑 ЧИТАТИ: якщо ×σ росте з K і при K=16 дає >3× — суміш ЖИВА, і її ціна")
print("   (×16 замикання = 2.66e10) РІВНО збігається з бюджетом лідера 2.61e10.")
