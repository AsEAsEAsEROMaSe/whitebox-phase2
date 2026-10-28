"""sigstep94 — ІН'ЄКЦІЯ ОДНОГО КРОКУ В `σ` І `m` ПРИ ТОЧНОМУ ВХОДІ.

════════════════ ЧОМУ САМЕ ЦЕ ════════════════
`[ЦИТАТА logs/r538.log, 16 мереж × 2e6 пар]` maxent проти Еджворта: **×1.37 обидва**
у накопиченому прогоні (M≈B до 4-го знака), але з ІСТИННОЮ `C` — ×77.87 проти
×18.74. ⟹ маргіналь НЕ вузьке місце, **усе в `C`**.
`[ЦИТАТА ladder87]` `σ` псована на 3% ⟹ ×176; `γ` на 3% ⟹ ×1.2.
`[ЦИТАТА ladder87 траса]` похибка `σ`: 0.099% (шар 1) → 14.887% (шар 31), тобто
**×1.17 за шар** — експоненційне підсилення.

🔑 ЧОГО НЕ ЗМІРЯНО НІДЕ: скільки з цих 14.887% — це ПІДСИЛЕННЯ вхідної похибки,
а скільки — СВІЖА ІН'ЄКЦІЯ кроку. Це різні хвороби з різними ліками:
```
підсилення домінує ⟹ лікувати стійкість рекурсії (демпфування, нерухома точка)
ін'єкція домінує   ⟹ лікувати САМ КРОК (негаусова поправка до Cov)
```
Вимір: дати кроку ТОЧНІ `(m,C)` з великої МК і зміряти похибку на ВИХОДІ кроку.
Крок — ПОВНИЙ ряд Мелера (K=24), тобто ТОЧНИЙ для гаусової `h`; усе, що
лишиться, — рівно ціна негаусовості.

════════════════ ПЛОМБИ ════════════════
P1 🔒 ШАР 0→1: вхід `z` РІВНО гаусів ⟹ крок мусить бути ТОЧНИЙ. Похибка там
   мусить сидіти на шумі МК. Не сидить ⟹ ряд Мелера обрізаний закоротко
   або в кроці баг. Це найсильніша пломба стенду.
P2 🔒 ПЛОМБА МАСШТАБУВАННЯ: 2¹⁷ проти 2¹⁹ — ін'єкція мусить СТОЯТИ, шум ділиться.
P3 🔒 Мелер K=8 проти K=24: якщо різниця не зникає — ряд не збігся.
P4 📐 РОЗКЛАД: ін'єкція = спільний МАСШТАБ (усі нейрони однаково) + РОЗКИД.
   Масштаб `_ALPHA` з'їдає безкоштовно ⟹ лікувати треба лише розкид.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) P1 пройде: похибка кроку на шарі 1 < 0.05%                шанс 85%
 (б) ін'єкція σ на крок буде 0.3…1.5% і РОСТИМЕ з глибиною     шанс 60%
 (в) ≥50% ін'єкції виявиться спільним масштабом                шанс 55%
 (г) ін'єкція `m` буде в 5–10× менша за ін'єкцію `σ`           шанс 70%
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
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
KMEL = int(sys.argv[4]) if len(sys.argv) > 4 else 24
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


def mel_step(m, C, W, K=KMEL):
    """ТОЧНИЙ (для гаусової h) крок замикання: (m,C) → (m',C') наступного шару.

    `Cov(a)_jk = Σ_{n≥1} n!·a_n[j]·a_n[k]·ρ_jk^n`, `a_n` — коефіцієнти Ерміта
    для `relu`. Ряд СУМУЄТЬСЯ, не обрізається на 2 — тобто це не «наше
    замикання», а його ТОЧНА для гаусової h версія.
    """
    s = C.diagonal().clamp_min(1e-30).sqrt()
    t = m / s
    P, f = Phi(t), phi(t)
    ma = s * (f + t * P)                       # a_0 = E[relu]
    rho = (C / torch.outer(s, s)).clamp(-1.0, 1.0)
    a1 = s * P
    cov = torch.outer(a1, a1) * rho            # n=1
    Rn = rho.clone()
    Hpre = torch.zeros_like(t)                 # He_{-1} (не використовується)
    Hm = torch.ones_like(t)                    # He_0(−t)
    fact = 2.0                                 # n! для n=2
    sf = s * f
    for n in range(2, K + 1):
        Rn = Rn * rho
        an = sf * Hm / fact
        cov = cov + torch.outer(an, an) * Rn * fact
        Hpre, Hm = Hm, (-t) * Hm - float(n - 2) * Hpre
        fact *= (n + 1)
    Wd = W.double()
    return ma.double() @ Wd, Wd.T @ (cov.double() @ Wd), cov.double()


def dcov3(m, C, T):
    """ΔCov³_jk = ½[(φ_j/s_j)·Φ_k·T_jk + Φ_j·(φ_k/s_k)·T_kj] — провідна
    поправка до Cov(relu) від ТРЕТІХ ЗМІШАНИХ кумулянтів `h`.
    🔑 Зрізи `T_jk = κ₃(h_j,h_j,h_k)` з CP-форми `κ₃=Σ_r λ_r u_r^{⊗3}` дають
    `T = Σ_r λ_r (u_r∘u_r)u_rᵀ` — МАТРИЦЯ РАНГУ r ⟹ r матвеків, не тензор.
    """
    s = C.diagonal().clamp_min(1e-30).sqrt()
    t = m / s
    P, f = Phi(t), phi(t)
    A = (f / s).double()
    B = P.double()
    return 0.5 * (torch.outer(A, B) * T + torch.outer(B, A) * T.T)


def truth(W, seed):
    n = W[0].shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    mh, Ch, Th = [], [], []
    X = None
    for L in range(len(W)):
        s1 = torch.zeros(n, dtype=torch.float64, device=DEV)
        buf = []
        for off in range(0, NSAMP, CH):
            b = min(CH, NSAMP - off)
            h = (torch.randn(b, n, generator=g, device=DEV) @ W[0]
                 if L == 0 else X[off:off + b] @ W[L])
            s1 += h.double().sum(0)
            buf.append(h)
        m = s1 / NSAMP
        mf = m.float()
        X = None
        s2 = torch.zeros(n, n, dtype=torch.float64, device=DEV)
        s3 = torch.zeros(n, n, dtype=torch.float64, device=DEV)
        for i, h in enumerate(buf):
            d = (h - mf).double()
            s2 += d.T @ d
            s3 += (d * d).T @ d          # T_jk = Σ h̃_j²·h̃_k
            buf[i] = torch.relu(h)
        mh.append(m)
        Ch.append(s2 / NSAMP)
        Th.append(s3 / NSAMP)
        X = torch.cat(buf, 0)
        del buf
    return mh, Ch, Th


print(f"sigstep94 · {DEV} · {NNET} мереж · N=2^{LOGN} · Мелер K={KMEL}",
      flush=True)
t0 = time.time()
ES, EM, SCA, SPR, EB = [], [], [], [], []
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        mh, Ch, Th = truth(W, 960000 + 173 * q)
        es, em, sca, spr, eb = [], [], [], [], []
        for L in range(31):
            mp, Cp, cova = mel_step(mh[L].float(), Ch[L].float(), W[L + 1])
            Wd = W[L + 1].double()
            d3 = dcov3(mh[L].float(), Ch[L].float(), Th[L])
            Cb = Wd.T @ ((cova + d3) @ Wd)
            st = Ch[L + 1].diagonal().clamp_min(1e-30).sqrt()
            sp = Cp.diagonal().clamp_min(1e-30).sqrt()
            r = sp / st                                   # відношення по нейронах
            k = float(r.mean())                           # спільний МАСШТАБ
            es.append(float((r - 1.0).abs().mean()))       # повна ін'єкція σ
            sca.append(abs(k - 1.0))                       # її масштабна частина
            spr.append(float((r / k - 1.0).abs().mean()))  # РОЗКИД (α не з'їсть)
            mt = mh[L + 1]
            em.append(float((mp - mt).abs().mean()
                            / mt.abs().mean().clamp_min(1e-30)))
            rb = Cb.diagonal().clamp_min(1e-30).sqrt() / st
            eb.append(float((rb / rb.mean() - 1.0).abs().mean()))
        ES.append(es); EM.append(em); SCA.append(sca); SPR.append(spr)
        EB.append(eb)
        print(f"  net{q} ({time.time()-t0:.0f} с)", flush=True)

E = np.array(ES) * 100; M_ = np.array(EM) * 100
A = np.array(SCA) * 100; S = np.array(SPR) * 100
B_ = np.array(EB) * 100
print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 78)
print("ІН'ЄКЦІЯ ОДНОГО КРОКУ ПРИ ТОЧНОМУ ВХОДІ (%, Мелер = точний для гауса)")
print("=" * 78)
print(f"{'шар':>5s}{'ін.σ повна':>12s}{'МАСШТАБ':>10s}"
      f"{'РОЗКИД A':>11s}{'🆕РОЗКИД B(κ₃)':>16s}{'B/A':>8s}{'ін.m':>9s}")
for L in (0, 1, 2, 4, 7, 11, 15, 19, 23, 27, 30):
    print(f"{L+1:>5d}{E[:,L].mean():>12.4f}{A[:,L].mean():>10.4f}"
          f"{S[:,L].mean():>11.4f}{B_[:,L].mean():>16.4f}"
          f"{S[:,L].mean()/max(B_[:,L].mean(),1e-12):>8.2f}{M_[:,L].mean():>9.4f}")
print(f"\n🔒P1 шар 1 (вхід РІВНО гаусів ⟹ крок мусить бути ТОЧНИЙ): "
      f"ін.σ = {E[:,0].mean():.4f}% · ін.m = {M_[:,0].mean():.4f}% "
      f"{'✅ крок чистий' if E[:,0].mean() < 0.05 else '🚨 БАГ або ряд не збігся'}")
print(f"📐P4 частка МАСШТАБУ в ін'єкції σ: {A.mean()/max(E.mean(),1e-12):.3f} "
      f"⟹ РОЗКИД {S.mean():.4f}% — саме він і є ціллю")
print(f"\n📣 середня ін'єкція за крок: σ {E.mean():.4f}% · m {M_.mean():.4f}%")
print(f"🆕 РОЗКИД σ: A(гаус) {S.mean():.4f}% → B(+оракульна κ₃) {B_.mean():.4f}%"
      f"  ⟹ ×{S.mean()/max(B_.mean(),1e-12):.2f}")
print(f"   глибина 20-31: A {S[:,19:].mean():.4f}% → B {B_[:,19:].mean():.4f}%"
      f"  ⟹ ×{S[:,19:].mean()/max(B_[:,19:].mean(),1e-12):.2f}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/sigstep94.npz",
         es=E, em=M_, sca=A, spr=S, eb=B_)
