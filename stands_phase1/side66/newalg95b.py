"""newalg95b — ДЕ ЖИВЕ γ₃=0.38, І ЧИ ЙОГО МОЖНА НЕСТИ ТОЧНО ОДНОВИМІРНО

════════════════ ЗВІДКИ (прямий наслідок `newalg95`) ════════════════
`[ЗМІРЯНО newalg95, 4 мережі × 2¹⁷]`
  γ₃(повна)=0.3802 · γ₃(лише A, t>0)=0.3801 · γ₃(лише B)=0.7243 · Var(B)/Var=0.017
  ⟹ розклад «гаусове ядро + рідкі спрацювання» МЕРТВИЙ: B несе 1.7% і 0% γ₃
`[ЗМІРЯНО там само]` cos(h(z),h(z′)) = 0.9741 на шарі 31 (теорія 3π/ℓ → 0.954)
  ⟹ h_ℓ = промінь + 2.6% флуктуації ⟹ 256 доданків суми КОЛІНЕАРНІ ⟹ ЦГТ не
  гасить γ₃ (мусила б у √256=16 разів за шар) ⟹ γ₃ насичується на 0.38

════════════════ ГІПОТЕЗА, ЯКУ ПЕРЕВІРЯЄМО ════════════════
Уся негаусовість — властивість ОДНОГО скаляра, а не 256-вимірного розподілу:
    h_ℓ = m + R·û + e ,   û = m/‖m‖ ,   R = û·(h−m) — НЕГАУСІВ скаляр
                                        e ⊥ û      — гаусів залишок (255 вимірів)
    p_{ℓ+1,a} = μ_a + R·c_a + g_a ,  c = Wᵀû ,  g = Wᵀe
    E[ReLU(p_a)] = E_R[ σ_g·ψ((μ_a + R·c_a)/σ_g) ] ,  ψ(t)=φ(t)+tΦ(t)   ТОЧНО
Тоді густина `R` (одновимірна, на сітці) несе ВСЮ негаусовість без обрізання.

🔑 ЕКОНОМІЯ ПАМ'ЯТІ (перша версія впала на OOM): ортогональну частину `e`
НЕ МАТЕРІАЛІЗУЄМО — `e·W = (p − μ) − R⊗c`, а `e·v = h·v − m·v − R·(û·v)`.

════════════════ ЩО МІРЯЄМО (на шарах 7,15,23,30 → передактивація ℓ+1) ═══════
 S1  γ₃,γ₄ проєкції на û проти γ₃,γ₄ проєкції на топ-ПК ортогонального залишку
     🔑 якщо γ₃(û) >> γ₃(ПК⊥) — негаусовість справді ОДНОВИМІРНА
 S2  частка дисперсії p_{ℓ+1}, яку несе канал R
 S3  🔑🔑 ПРЯМИЙ ВИГРАШ на ОДИН крок, три оцінки E[ReLU(p_a)]:
       (0) істина          mean(relu(p_a))
       (i) ГАУСОВА база    s_a·ψ(μ_a/s_a)      ← що робить будь-яке замикання
      (ii) НОВА rank-1     mean_R[ σ_g·ψ((μ_a+R c_a)/σ_g) ]
     (iii) НОВА rank-2     те саме, плюс топ-ПК залишку
     ⛔КОНТРОЛЬ «чужий R»: R з ІНШОЇ мережі ⟹ виграш мусить ЗНИКНУТИ
 S4  ПЛОМБА МАСШТАБУВАННЯ: N та N/4 ⟹ шум падає вдвічі, систематика стоїть

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) γ₃ проєкції на û буде > 0.5 (більша за покоординатну 0.38)     шанс 60%
 (б) γ₃ проєкції на топ-ПК⊥ буде < 0.15                            шанс 45%
 (в) 🔑 rank-1 зніме >50% похибки гаусової бази на шарі 30          шанс 40%
 (г) rank-2 додасть ще помітно понад rank-1                        шанс 35%
 (д) чужий R дасть ×1.0 (виграш зникне)                            шанс 80%
 (е) канал R нестиме 60…95% дисперсії p_{ℓ+1}                      шанс 55%
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


def psi(t):
    """E[ReLU(N(t,1))] = φ(t) + t·Φ(t)."""
    return torch.exp(-0.5 * t * t) * S2PI + t * Phi(t)


def gam(x):
    """γ₃, γ₄ скаляра (1-D тензор)."""
    c = x - x.mean()
    v = (c * c).mean().clamp_min(1e-300)
    return float((c ** 3).mean() / v ** 1.5), float((c ** 4).mean() / v ** 2 - 3)


def evaluate(h, Wn, n, Rext=None):
    """h: (N,256) fp32 на GPU. Усе важке — чанками, у fp64 лише акумулятори."""
    dev, D = h.device, h.shape[1]
    z64 = dict(dtype=torch.float64, device=dev)
    # ── прохід 1: m, mu, sd, truth
    sm = torch.zeros(D, **z64)
    s1v = torch.zeros(D, **z64)
    s2v = torch.zeros(D, **z64)
    tru = torch.zeros(D, **z64)
    for o in range(0, n, CH):
        hb = h[o:o + CH].double()
        pb = hb @ Wn
        sm += hb.sum(0)
        s1v += pb.sum(0)
        s2v += (pb * pb).sum(0)
        tru += pb.clamp_min(0.0).sum(0)
    m = sm / n
    mu = s1v / n
    sd = (s2v / n - mu * mu).clamp_min(1e-300).sqrt()
    tru = tru / n
    base = sd * psi(mu / sd)
    u = m / m.norm().clamp_min(1e-300)
    c1 = u @ Wn
    mu_u = float(m @ u)

    # ── R (скаляр) без матеріалізації e
    R = torch.empty(n, **z64)
    for o in range(0, n, CH):
        R[o:o + CH] = h[o:o + CH].double() @ u - mu_u
    varR = float((R * R).mean())

    # ── степеневий метод на Cov(e), e = hc − R⊗u ; ec·v = hc·v − R·(u·v)
    v2 = torch.randn(D, **z64)
    v2 /= v2.norm()
    for _ in range(10):
        acc = torch.zeros(D, **z64)
        uv = float(u @ v2)
        for o in range(0, n, CH):
            hb = h[o:o + CH].double() - m
            y = hb @ v2 - R[o:o + CH] * uv
            acc += hb.T @ y - u * float(R[o:o + CH] @ y)
        v2 = acc / acc.norm().clamp_min(1e-300)
    c2 = v2 @ Wn
    uv = float(u @ v2)
    R2 = torch.empty(n, **z64)
    for o in range(0, n, CH):
        hb = h[o:o + CH].double() - m
        R2[o:o + CH] = hb @ v2 - R[o:o + CH] * uv

    # ── дисперсії залишків g1 = (p−μ) − R c1 ,  g2 = g1 − R2 c2
    a1 = torch.zeros(D, **z64)
    a2 = torch.zeros(D, **z64)
    for o in range(0, n, CH):
        pb = h[o:o + CH].double() @ Wn - mu
        g1 = pb - torch.outer(R[o:o + CH], c1)
        a1 += (g1 * g1).sum(0)
        g1 -= torch.outer(R2[o:o + CH], c2)
        a2 += (g1 * g1).sum(0)
    s_g1 = (a1 / n).clamp_min(1e-300).sqrt()
    s_g2 = (a2 / n).clamp_min(1e-300).sqrt()

    # ── оцінки rank-1 / rank-2 (чанками по семплах)
    Ru = R if Rext is None else (Rext[:n] - Rext[:n].mean())
    e1 = torch.zeros(D, **z64)
    e2 = torch.zeros(D, **z64)
    for o in range(0, n, CH):
        arg = (mu + torch.outer(Ru[o:o + CH], c1)) / s_g1
        e1 += psi(arg).sum(0)
        arg = (mu + torch.outer(Ru[o:o + CH], c1)
               + torch.outer(R2[o:o + CH], c2)) / s_g2
        e2 += psi(arg).sum(0)
    r1 = s_g1 * (e1 / n)
    r2 = s_g2 * (e2 / n)

    def rel(x):
        return float((x - tru).norm() / tru.norm())
    fracR = float(varR * (c1 * c1).sum() / (sd * sd).sum())
    return (rel(base), rel(r1), rel(r2), fracR), R, R2


print(f"newalg95b · {DEV} · {NNET} мереж · N=2^{LOGN}", flush=True)
t0 = time.time()
RES, GAM = [], []
prevR = {}
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        g = torch.Generator(device=DEV).manual_seed(951000 + 137 * q)
        X = torch.randn(NSAMP, 256, generator=g, device=DEV)
        row, gm = {}, {}
        for L in range(32):
            X = torch.relu(X @ W[L])
            if L in PROBE and L + 1 < 32:
                Wn = W[L + 1].double()
                a, R, R2 = evaluate(X, Wn, NSAMP)
                a4, _, _ = evaluate(X, Wn, NSAMP // 4)
                rx = float('nan')
                if L in prevR and prevR[L].numel() >= NSAMP:
                    ax, _, _ = evaluate(X, Wn, NSAMP, Rext=prevR[L])
                    rx = ax[1]
                row[L] = a + (rx, a4[0], a4[1])
                g3u, g4u = gam(R)
                g3e, g4e = gam(R2)
                gm[L] = (g3u, g4u, g3e, g4e)
                prevR[L] = R
                del R, R2
                torch.cuda.empty_cache()
        RES.append(row)
        GAM.append(gm)
        print(f"  net{q} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 80)
print("S1 — ДЕ ЖИВЕ НЕГАУСОВІСТЬ: проєкція на û проти проєкції на топ-ПК⊥")
print("=" * 80)
print(f"{'шар':>4s}{'γ₃(û)':>11s}{'γ₄(û)':>11s}{'γ₃(ПК⊥)':>11s}{'γ₄(ПК⊥)':>11s}")
for L in PROBE:
    a = np.mean([GAM[q][L] for q in range(len(GAM))], axis=0)
    print(f"{L:>4d}{a[0]:>11.4f}{a[1]:>11.4f}{a[2]:>11.4f}{a[3]:>11.4f}")

print("\n" + "=" * 80)
print("S3 — ПРЯМИЙ ВИГРАШ НА ОДИН КРОК (віднос. похибка E[ReLU(p)] по вектору)")
print("=" * 80)
print(f"{'шар':>4s}{'гаусова':>11s}{'rank-1':>11s}{'rank-2':>11s}"
      f"{'×r1':>8s}{'×r2':>8s}{'⛔чужий R':>11s}{'частка R':>10s}")
for L in PROBE:
    a = np.mean([RES[q][L] for q in range(len(RES))], axis=0)
    xf = a[0] / a[4] if a[4] == a[4] and a[4] > 0 else float('nan')
    print(f"{L:>4d}{a[0]:>11.5f}{a[1]:>11.5f}{a[2]:>11.5f}"
          f"{a[0]/max(a[1],1e-12):>8.2f}{a[0]/max(a[2],1e-12):>8.2f}"
          f"{xf:>11.2f}{a[3]:>10.4f}")

print("\n" + "=" * 80)
print("S4 — ПЛОМБА МАСШТАБУВАННЯ: N проти N/4 (шум падає вдвічі, систематика — ні)")
print("=" * 80)
for L in PROBE:
    a = np.mean([RES[q][L] for q in range(len(RES))], axis=0)
    print(f"  шар {L:>2d}: гаусова N={a[0]:.5f} N/4={a[5]:.5f} · "
          f"rank-1 N={a[1]:.5f} N/4={a[6]:.5f}")

print("\n🔑 ЧИТАТИ: γ₃(û) >> γ₃(ПК⊥) ⟹ негаусовість ОДНОВИМІРНА ⟹ густина на")
print("   сітці несе її ТОЧНО. ×r1 — прямий множник у похибці ОДНОГО кроку.")
