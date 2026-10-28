"""cv95 — КОНТРОЛЬНА ЗМІННА З ТОЧНИМ СЕРЕДНІМ: чи дає вал ×100…×1000 редукції?

════════════════ ЗВІДКИ ════════════════
`[ЦИТАТА сторінка лідера #324969]` MEAN EFFECTIVE COMPUTE 2.61e10 (= ×15.7
   нашого замикання), final MSE 3.628e-9 ⟹ віднос. похибка **0.0064%**,
   vs sampling **1782×** (найкраща MLP 5051×), BUDGET USED 9.59% = ПІДЛОГА.
🔑 НАША похибка замикання 0.85%. **(0.85%)² = 0.0072% ≈ 0.0064%** — це рівно
   КВАДРАТИЧНЕ уточнення нашого рівня. Такий закон дає саме контрольна змінна:
   `Var(f − g) = Var(f)·(1 − corr²)`, і щоб мати ×565 треба corr = 0.99912.
`[ЗМІРЯНО accum95c]` суміш ×1.39, але ціна 4.9% B проти дозволених 0.35% ⛔
`[ЗМІРЯНО newalg95]` cos(h(z),h(z′)) = 0.974 ⟹ мережа сильно колінеарна
`[ЗМІРЯНО С94 + accum95c]` ВАЛ `A = Π diag(Φ_ℓ)·W_ℓ` — усереднений якобіан
   ∂h_31/∂z; R²=0.943 на переносі похибки; напрямок вала — ЄДИНИЙ, що дав
   виграш у суміші (×1.39 проти rand ×1.01)

════════════════ КОНСТРУКЦІЯ (усе в межах сендбоксу, дешево) ════════════════
```
g_a(z) = ReLU( (A z)·w_a )        A — ДЕТЕРМІНОВАНА, з ваг і Φ рекурсії
  ⟹ (Az)·w_a ~ N(0, ‖Aᵀw_a‖²) ТОЧНО (лінійна від гауса!)
  ⟹ E[g_a] = ‖Aᵀw_a‖ · φ(0) = ‖Aᵀw_a‖/√(2π)      ← АНАЛІТИЧНО, БЕЗ семплів
f_a(z) = ReLU( h_31(z)·w_a )       те, що ми оцінюємо МК
оцінка:  Ê[f] = E[g] (точно) + mean_i[ f(z_i) − g(z_i) ]
редукція дисперсії = 1/(1 − corr²(f,g))
```
Ціна `A`: 31 матмул 256³ = **1.04e9 = 0.38% B**. Ціна `g` на семплі: один
матвек `z·A` (1.31e5) + проєкція ⟹ ×1.03 до ціни семпла. Практично безкоштовно.

════════════════ ЩО МІРЯЄМО ════════════════
 C1  R² лінійної апроксимації h_31(z) ≈ A z (по всіх 256 координатах)
 C2  🔑 corr(f_a, g_a) покоординатно і РЕДУКЦІЯ ДИСПЕРСІЇ 1/(1−corr²)
 C3  те саме для оптимального A_opt = Cov(h_31, z) (найкраща лінійна) — СТЕЛЯ
 C4  чи точна аналітична формула E[g] (пломба: проти МК на g)
 C5  ⛔КОНТРОЛЬ: A з ІНШОЇ мережі ⟹ редукція мусить зникнути
 C6  редукція для суми з коефіцієнтом: β = Cov(f,g)/Var(g) (оптимальний CV)

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) R² вала буде 0.90…0.99                                    шанс 55%
 (б) 🔑 редукція дисперсії вала буде ×5…×50                     шанс 45%
 (в) оптимальний лінійний A_opt дасть ×20…×300                  шанс 40%
 (г) 🚨 щоб зрівнятись із лідером треба ×565 — цього НЕ буде     шанс 70%
 (д) пломба E[g] аналітична = МК зійдеться до 1e-3              шанс 90%
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
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NSAMP = 1 << LOGN
CH = 1 << 14
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
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
        fm = np.asarray(t.column("final_means").combine_chunks()
                        .flatten()).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i]
            got += 1


def Phi(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def phi(x):
    return torch.exp(-0.5 * x * x) * S2PI


def shaft(W):
    """A: z ↦ передактивація шару 31 у лінеаризації. A = W₁·diag(Φ₁)·W₂···"""
    m = torch.zeros(256, dtype=torch.float64, device=W[0].device)
    C = W[0].double().T @ W[0].double()
    A = W[0].double()
    for L in range(32):
        s = C.diagonal().clamp_min(1e-30).sqrt()
        t = m / s
        P, f = Phi(t), phi(t)
        ma = s * (f + t * P)
        if L == 31:
            break
        rho = (C / torch.outer(s, s)).clamp(-1.0, 1.0)
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
        Wd = W[L + 1].double()
        A = (A * P) @ Wd
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)
    return A


print(f"cv95 · {DEV} · {NNET} мереж · N=2^{LOGN}", flush=True)
t0 = time.time()
OUT = []
prevA = None
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        z64 = dict(dtype=torch.float64, device=DEV)
        A = shaft(W)                       # (256 вхід, 256 вихід)
        g = torch.Generator(device=DEV).manual_seed(960000 + 179 * q)
        # ── один прохід: збираємо f (=relu(p_32)), g_shaft, g_opt-статистики
        sf1 = torch.zeros(256, **z64); sf2 = torch.zeros(256, **z64)
        sg1 = torch.zeros(256, **z64); sg2 = torch.zeros(256, **z64)
        sfg = torch.zeros(256, **z64)
        sx1 = torch.zeros(256, **z64); sx2 = torch.zeros(256, **z64)
        sfx = torch.zeros(256, **z64)
        # для A_opt = Cov(p_32, z): накопичуємо zᵀp
        szp = torch.zeros(256, 256, **z64)
        sp1 = torch.zeros(256, **z64)
        spp = torch.zeros(256, **z64)
        gy1 = torch.zeros(256, **z64); gy2 = torch.zeros(256, **z64)
        sfy = torch.zeros(256, **z64)
        chunks = []
        for o in range(0, NSAMP, CH):
            z = torch.randn(CH, 256, generator=g, device=DEV)
            h = z
            for L in range(31):
                h = torch.relu(h @ W[L])
            p = (h @ W[31]).double()             # передактивація шару 31
            f = p.clamp_min(0.0)
            zd = z.double()
            gv = (zd @ A).clamp_min(0.0)         # вал як CV
            sf1 += f.sum(0); sf2 += (f * f).sum(0)
            sg1 += gv.sum(0); sg2 += (gv * gv).sum(0)
            sfg += (f * gv).sum(0)
            sp1 += p.sum(0); spp += (p * p).sum(0)
            szp += zd.T @ p
            if prevA is not None:
                gx = (zd @ prevA).clamp_min(0.0)
                sx1 += gx.sum(0); sx2 += (gx * gx).sum(0)
                sfx += (f * gx).sum(0)
            chunks.append((z, p, f))
        n = NSAMP
        Aopt = szp / n                            # Cov(p, z) бо Cov(z)=I
        for (z, p, f) in chunks:
            gy = (z.double() @ Aopt).clamp_min(0.0)
            gy1 += gy.sum(0); gy2 += (gy * gy).sum(0)
            sfy += (f * gy).sum(0)
        del chunks

        def stats(s1, s2, sc):
            mg = s1 / n
            vg = (s2 / n - mg * mg).clamp_min(1e-300)
            cf = sc / n - (sf1 / n) * mg
            return mg, vg, cf
        mf = sf1 / n
        vf = (sf2 / n - mf * mf).clamp_min(1e-300)
        res = {}
        for tag, (s1, s2, sc, Amat) in {
                'shaft': (sg1, sg2, sfg, A),
                'opt': (gy1, gy2, sfy, Aopt),
                'foreign': (sx1, sx2, sfx, prevA)}.items():
            if Amat is None:
                continue
            mg, vg, cf = stats(s1, s2, sc)
            corr = cf / (vf * vg).clamp_min(1e-300).sqrt()
            beta = cf / vg
            var_cv = (vf - beta * cf).clamp_min(1e-300)      # Var(f − βg)
            # 🔒 пломба: аналітичне E[g] = ‖Aᵀw‖/√(2π) — тут Amat уже = A·W? ні:
            #    gv = relu(z·Amat), тобто σ_a = ‖Amat[:,a]‖
            sig = Amat.norm(dim=0)
            mg_an = sig * S2PI
            res[tag] = (float(corr.mean()),
                        float((vf.sum() / var_cv.sum())),
                        float(((mg_an - mg).abs() / mg.abs()
                               .clamp_min(1e-300)).mean()),
                        float(beta.mean()))
        # R² лінійної апроксимації p_32 ≈ z·A
        mp = sp1 / n
        vp = (spp / n - mp * mp).clamp_min(1e-300)
        r2_sh = float(1.0 - ((vp - 2 * (A * Aopt).sum(0) + (A * A).sum(0))
                             ).clamp_min(0).sum() / vp.sum())
        r2_op = float(((Aopt * Aopt).sum(0)).sum() / vp.sum())
        OUT.append((res, r2_sh, r2_op))
        prevA = A
        print(f"  net{q}: R²(вал)={r2_sh:.5f} R²(опт)={r2_op:.5f} "
              f"редукція вал ×{res['shaft'][1]:.1f} опт ×{res['opt'][1]:.1f} "
              f"({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 80)
print("КОНТРОЛЬНА ЗМІННА — РЕДУКЦІЯ ДИСПЕРСІЇ МК (потрібно ×565 щоб зрівнятись)")
print("=" * 80)
print(f"{'варіант':>10s}{'corr':>10s}{'редукція ×':>13s}"
      f"{'β серед.':>11s}{'пломба E[g]':>14s}")
for tag in ('shaft', 'opt', 'foreign'):
    v = [o[0][tag] for o in OUT if tag in o[0]]
    if not v:
        continue
    a = np.mean(v, axis=0)
    print(f"{tag:>10s}{a[0]:>10.5f}{a[1]:>13.2f}{a[3]:>11.4f}{a[2]:>14.2e}")
print(f"\nR² лінійної апроксимації p_32(z): вал "
      f"{np.mean([o[1] for o in OUT]):.5f} · оптимальна "
      f"{np.mean([o[2] for o in OUT]):.5f}")
print("\n🔑 ЧИТАТИ: редукція = 1/(1−corr²). Наш поточний 'vs sampling' = 2.6×,")
print("   у лідера 1782×. Якщо вал дає ×20+ — це найбільший стрибок за проєкт.")
