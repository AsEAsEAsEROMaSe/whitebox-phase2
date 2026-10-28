"""accum96 — НЕГАУСОВІСТЬ ЯК РОЗЩЕПЛЕННЯ УЗДОВЖ `û`: ДІАГОНАЛЬ І ПОЗАДІАГОНАЛЬ РАЗОМ

════════════════ ЧОМУ САМЕ ЦЕ (ланцюг С95, усе ЗМІРЯНО) ════════════════
`[ЗМІРЯНО newalg95c, крок із ТОЧНИМИ (μ,σ,ρ)]`
    E[relu] 0.139% · σ 1.802% · діагональ Cov 1.03% · **позадіагональ 4.08%**
`[ЗМІРЯНО accum95]` 1-D Еджворт: σ ×4.88, E[relu] ×4.57 на КРОЦІ,
    але НАКОПИЧЕНО **×0.83** ⟹ ГІРШЕ бази.
🔑 МЕХАНІЗМ ПРОВАЛУ (гейт узгодженості `C`): accum95 виправляв ЛИШЕ діагональ
    (`v_new` через Еджворт), а кореляції лишав Мелеровими ⟹ поправка розбила
    баланс похибок, замість того щоб його переписати.
🔑 ⟹ ПОТРІБЕН ОБ'ЄКТ, ЩО ЧІПАЄ ОБИДВІ ЧАСТИНИ ОДНИМ РУХОМ.

`[ЗМІРЯНО newalg95b]` негаусовість ОДНОВИМІРНА: γ₃(û)=0.4268 проти
    γ₃(ПК⊥)=−0.0490, розрив ×8.7   `[ЗМІРЯНО newalg95]` γ₃ насичується на 0.38
    і НЕ гасне ЦГТ ⟹ доданки суми КОЛІНЕАРНІ (мережа = промінь, cos@31=0.974)
⟹ 📐 ЯКЩО негаусовість живе В ОДНОМУ напрямку `û`, то УВЕСЬ третій кумулянт
   передактивації ≈ РАНГ-1: `κ₃ ≈ λ·(Wû)^⊗3`. Тоді точний спосіб узяти його —
   НЕ ряд Еджворта на маргіналях, а **умовність по проєкції**:
```
   ξ = ûᵀ(a−m)/σ_u  — скаляр, НЕсиметричний
   при ФІКСОВАНОМУ ξ вектор `a` рівно ГАУСІВ:  a|ξ ~ N(m + ξ·d , C − dd ᵀ),
       d = Cû/σ_u  ⟹ Мелер усередині компоненти ТОЧНИЙ
   E[·] = Σ_q w_q · Мелер(m + μ_q d , C + (v_q−1)dd ᵀ)
```
🔑 Це поправляє `m`, діагональ І ПОЗАДІАГОНАЛЬ ОДНИМ об'єктом ⟹ **проходить
   гейт узгодженості `C`**, чого не робила ЖОДНА поправка С88–С95.
`[ЗМІРЯНО mix95]` на ОДНОМУ кроці таке розщеплення дало позадіаг ×2.73
   (насичення K=4) ⟹ механізм працює, питання ЛИШЕ в накопиченні.

════════════════ ВУЗЛИ: ЗАМКНЕНА ФОРМА, БЕЗ КВАДРАТУР ════════════════
Густина проєкції — Еджворт `p(x) = φ(x)[1 + (γ₃/6)He₃ + (γ₄/24)He₄]`.
Ділимо вісь на K гаусових квантильних кошиків і беремо ТОЧНІ моменти `p`
на кожному, через `∫_a^b He_n φ = He_{n−1}(a)φ(a) − He_{n−1}(b)φ(b)`:
```
w   = ΔΦ + (γ₃/6)J₃ + (γ₄/24)J₄
M₁  = (φ(a)−φ(b)) + (γ₃/6)(J₄+3J₂) + (γ₄/24)(J₅+4J₃)
M₂  = [ΔΦ + aφ(a) − bφ(b)] + (γ₃/6)(J₅+7J₃+6J₁) + (γ₄/24)(J₆+9J₄+12J₂)
```
При `γ=0` це РІВНО зрізаний гаус ⟹ M0 ізолює ефект Єнсена, M4/M8 — сам γ.

════════════════ ВАРІАНТИ ════════════════
 A   чистий Мелер K=24                         🔒ПЛОМБА: мусить дати 0.847121%
 M0  розщеплення K=4, γ=0                      ізолює ефект Єнсена
 M4  розщеплення K=4, ОРАКУЛЬНІ γ₃,γ₄          ← ГОЛОВНИЙ
 M8  розщеплення K=8, оракульні                чи насичується по K
 MW  K=4, γ з РОБОЧОЇ вибірки 12288            реалістичний метод
 MF  ⛔K=4, ЧУЖІ γ (з попередньої мережі)       ← ВИРІШАЛЬНИЙ КОНТРОЛЬ
 L50/L150  сила поправки λ=0.5 / 1.5           оптимум мусить бути при λ>0

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ, протокол п.6) ════════════════
 (а) M0 (сам поділ, γ=0) дасть ×1.05…×1.20                     шанс 75%
     [опора: accum95d ℓ₀=0,K=4 = ×1.14]
 (б) M4 поб'є M0 щонайменше в ×1.5                             шанс 40%
     [це і є питання дня: чи виживе УЗГОДЖЕНА поправка там, де
      маргінальна дала ×0.83]
 (в) MF (чужі γ) дасть <×1.05 проти M0                         шанс 70%
 (г) M8 ≈ M4 (насичення на K=4, як у mix95)                    шанс 65%
 (д) M4 дотягне до потрібних ×34 у MSE                         шанс 5%
     [чесно: потрібне 0.85%→0.145%; жодна гілка ще не давала >×1.4]
 (е) λ-оптимум M4 буде при λ≥0.75                              шанс 55%

Ціль — `final_means` з датасету (істина 1e9 семплів).
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
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NSAMP = 1 << LOGN
NWORK = 12288
CH = 1 << 14
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
MK = 24
EDGE = 8.0


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


# ─────────────────── γ ПРОЄКЦІЇ НА НАПРЯМОК СЕРЕДНЬОГО ───────────────────
def proj_cumulants(W, seed, n, dev):
    """γ₃,γ₄ СКАЛЯРА ξ = ûᵀ(a−m)/σ_u для передактивації кожного шару."""
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(n, 256, generator=g, device=dev)
    G3, G4 = [], []
    for L in range(32):
        Wd = W[L]
        s1 = torch.zeros(256, dtype=torch.float64, device=dev)
        for o in range(0, n, CH):
            s1 += (X[o:o + CH] @ Wd).double().sum(0)
        mu = s1 / n
        nrm = mu.norm()
        u = mu / nrm.clamp_min(1e-300) if float(nrm) > 0 else None
        if u is None:                       # шар 0: m ≡ 0, гаус ТОЧНИЙ
            G3.append(0.0)
            G4.append(0.0)
        else:
            uf = u.to(X.dtype)
            k2 = k3 = k4 = 0.0
            for o in range(0, n, CH):
                c = ((X[o:o + CH] @ Wd).double() - mu) @ u
                k2 += float((c * c).sum())
                k3 += float((c ** 3).sum())
                k4 += float((c ** 4).sum())
            v = k2 / n
            sd = v ** 0.5
            G3.append(k3 / n / sd ** 3)
            G4.append(k4 / n / v ** 2 - 3.0)
            del uf
        X = torch.relu(X @ Wd)
    return G3, G4


# ─────────────────── ВУЗЛИ РОЗЩЕПЛЕННЯ (замкнена форма) ───────────────────
def he(n, x):
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    hm, h = np.ones_like(x), x
    for k in range(2, n + 1):
        hm, h = h, x * h - (k - 1) * hm
    return h


def nodes_raw(K, g3, g4):
    """(w, μ, v) для K квантильних кошиків під Еджвортовою густиною."""
    from math import erf, sqrt
    q = np.array([k / K for k in range(K + 1)])
    ed = np.empty(K + 1)
    ed[0], ed[-1] = -EDGE, EDGE
    for k in range(1, K):                       # Φ⁻¹ бісекцією (60 кроків)
        lo, hi = -EDGE, EDGE
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < q[k]:
                lo = mid
            else:
                hi = mid
        ed[k] = 0.5 * (lo + hi)
    a, b = ed[:-1], ed[1:]
    fa = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    fb = np.exp(-0.5 * b * b) / np.sqrt(2 * np.pi)
    Pa = np.array([0.5 * (1.0 + erf(v / sqrt(2.0))) for v in a])
    Pb = np.array([0.5 * (1.0 + erf(v / sqrt(2.0))) for v in b])
    dP = Pb - Pa

    def J(n):                                   # ∫_a^b He_n φ
        return he(n - 1, a) * fa - he(n - 1, b) * fb

    J1, J2, J3, J4, J5, J6 = (J(n) for n in (1, 2, 3, 4, 5, 6))
    c3, c4 = g3 / 6.0, g4 / 24.0
    w = dP + c3 * J3 + c4 * J4
    M1 = (fa - fb) + c3 * (J4 + 3 * J2) + c4 * (J5 + 4 * J3)
    M2 = (dP + a * fa - b * fb) + c3 * (J5 + 7 * J3 + 6 * J1) \
        + c4 * (J6 + 9 * J4 + 12 * J2)
    w = np.maximum(w, 1e-9)
    w = w / w.sum()
    mu = M1 / w
    vv = np.maximum(M2 / w - mu * mu, 1e-6)
    return w, mu, vv


def mom(w, mu, v):
    """Моменти 1..4 самої СУМІШІ гаусів N(μ_q, v_q) — не зрізаних кошиків."""
    m1 = float((w * mu).sum())
    m2 = float((w * (v + mu ** 2)).sum())
    m3 = float((w * (mu ** 3 + 3 * mu * v)).sum())
    m4 = float((w * (mu ** 4 + 6 * mu ** 2 * v + 3 * v ** 2)).sum())
    return m1, m2, m3, m4


_CACHE = {}


def nodes(K, g3, g4, lam):
    """Вузли, чия СУМІШ має РІВНО задані (γ₃,γ₄).

    🚨 Прямий поділ на кошики цього НЕ дає: сама дискретизація вносить
    ексцес −0.42 при K=4 (пломба показала m4=2.5835 замість 3.0), тобто
    БІЛЬШЕ за справжній γ₄=0.23 і з ПРОТИЛЕЖНИМ знаком. Тому підбираємо
    вхідні (c3,c4) Ньютоном так, щоб ВИХІДНІ моменти збіглись із ціллю.
    ⟹ при γ=0 суміш стає точним гаусом ⟹ M0 ізолює ЛИШЕ ефект Єнсена.
    """
    t3, t4 = lam * g3, 3.0 + lam * g4
    key = (K, round(t3, 6), round(t4, 6))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    c3, c4 = t3, t4 - 3.0
    for _ in range(6):
        w, mu, v = nodes_raw(K, c3, c4)
        _, _, m3, m4 = mom(w, mu, v)
        r3, r4 = m3 - t3, m4 - t4
        if abs(r3) < 1e-10 and abs(r4) < 1e-10:
            break
        h = 1e-4
        w1, u1, v1 = nodes_raw(K, c3 + h, c4)
        w2, u2, v2 = nodes_raw(K, c3, c4 + h)
        _, _, a3, a4 = mom(w1, u1, v1)
        _, _, b3, b4 = mom(w2, u2, v2)
        j11, j21 = (a3 - m3) / h, (a4 - m4) / h
        j12, j22 = (b3 - m3) / h, (b4 - m4) / h
        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-12:
            break
        c3 -= (j22 * r3 - j12 * r4) / det
        c4 -= (-j21 * r3 + j11 * r4) / det
    out = nodes_raw(K, c3, c4)
    _CACHE[key] = out
    return out


def node_seal(K, g3, g4):
    """ПЛОМБА вузлів: моменти суміші мусять відтворити 0,1,γ₃,3+γ₄."""
    return mom(*nodes(K, g3, g4, 1.0))


# ─────────────────── КРОК ЗАМИКАННЯ ───────────────────
def mehler(m, C):
    s = C.diagonal().clamp_min(1e-30).sqrt()
    t = m / s
    P, f = Phi(t), phi(t)
    ma = s * (P * t + f)
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
    return ma, cov


def step_mix(m, C, w, mu, vv):
    """Розщеплення уздовж û = m/‖m‖, Мелер у КОЖНІЙ компоненті, злиття."""
    nrm = m.norm()
    if float(nrm) <= 0.0:
        return mehler(m, C)
    u = m / nrm
    Cu = C @ u
    su = (u @ Cu).clamp_min(1e-30).sqrt()
    d = Cu / su
    dd = torch.outer(d, d)
    mo = torch.zeros_like(m)
    So = torch.zeros_like(C)
    for q in range(len(w)):
        mq = m + float(mu[q]) * d
        Cq = C + (float(vv[q]) - 1.0) * dd
        aq, cq = mehler(mq, Cq)
        wq = float(w[q])
        mo = mo + wq * aq
        So = So + wq * (cq + torch.outer(aq, aq))
    return mo, So - torch.outer(mo, mo)


def run(W, G3, G4, K=0, lam=1.0):
    m = torch.zeros(256, dtype=torch.float64, device=W[0].device)
    C = W[0].double().T @ W[0].double()
    for L in range(32):
        if K == 0:
            ma, cov = mehler(m, C)
        else:
            g3 = 0.0 if G3 is None else float(G3[L])
            g4 = 0.0 if G4 is None else float(G4[L])
            w, mu, vv = nodes(K, g3, g4, lam)
            ma, cov = step_mix(m, C, w, mu, vv)
        if L == 31:
            return ma
        Wd = W[L + 1].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)


# ═══════════════════════ ПРОГІН ═══════════════════════
print(f"accum96 · {DEV} · {NNET} мереж · оракул N=2^{LOGN} · робоча {NWORK}",
      flush=True)
print("ПЛОМБА ВУЗЛІВ (мусить бути 0, 1, γ₃, 3+γ₄):")
for K in (4, 8):
    for gg3, gg4 in ((0.0, 0.0), (0.38, 0.36)):
        m1, m2, m3, m4 = node_seal(K, gg3, gg4)
        print(f"  K={K} γ=({gg3},{gg4}): m1={m1:+.2e} m2={m2:.6f} "
              f"m3={m3:+.5f} (ціль {gg3:+.2f})  m4={m4:.5f} "
              f"(ціль {3+gg4:.2f})", flush=True)

t0 = time.time()
KEYS = ['A', 'M0', 'M4', 'M8', 'MW', 'MF', 'L50', 'L150']
ERR = {k: [] for k in KEYS}
prev = None
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        G3o, G4o = proj_cumulants(W, 960000 + 149 * q, NSAMP, DEV)
        G3w, G4w = proj_cumulants(W, 983000 + 167 * q, NWORK, DEV)
        if q == 0:
            print("  γ₃ проєкції по шарах (оракул): "
                  + " ".join(f"{G3o[i]:.3f}" for i in (0, 1, 4, 8, 16, 24, 31)),
                  flush=True)
            print("  γ₄ проєкції по шарах (оракул): "
                  + " ".join(f"{G4o[i]:.3f}" for i in (0, 1, 4, 8, 16, 24, 31)),
                  flush=True)

        def er(x):
            return float((x - tru).norm() / tru.norm())
        ERR['A'].append(er(run(W, None, None)))
        ERR['M0'].append(er(run(W, None, None, K=4)))
        ERR['M4'].append(er(run(W, G3o, G4o, K=4)))
        ERR['M8'].append(er(run(W, G3o, G4o, K=8)))
        ERR['MW'].append(er(run(W, G3w, G4w, K=4)))
        ERR['L50'].append(er(run(W, G3o, G4o, K=4, lam=0.5)))
        ERR['L150'].append(er(run(W, G3o, G4o, K=4, lam=1.5)))
        if prev is not None:
            ERR['MF'].append(er(run(W, prev[0], prev[1], K=4)))
        prev = (G3o, G4o)
        if q % 4 == 3 or q == 0:
            print(f"  net{q}: A={ERR['A'][-1]*100:.4f}% M0={ERR['M0'][-1]*100:.4f}%"
                  f" M4={ERR['M4'][-1]*100:.4f}% M8={ERR['M8'][-1]*100:.4f}%"
                  f" ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 78)
print("НАКОПИЧЕНА ПОХИБКА ПРОТИ `final_means` (істина 1e9 семплів)")
print("=" * 78)
mse = {k: float(np.mean(np.array(v) ** 2)) for k, v in ERR.items() if v}
base = float(np.mean(ERR['A']))
names = {'A': '🔒чистий Мелер K=24 (ПЛОМБА 0.847121%)',
         'M0': 'розщеплення K=4, γ=0 (ефект Єнсена)',
         'M4': '🔥K=4, ОРАКУЛЬНІ γ₃,γ₄ — ГОЛОВНИЙ',
         'M8': 'K=8, оракульні (насичення по K?)',
         'MW': '🟢K=4, γ з робочої вибірки 12288',
         'MF': '⛔K=4, ЧУЖІ γ — ВИРІШАЛЬНИЙ КОНТРОЛЬ',
         'L50': 'K=4, λ=0.50', 'L150': 'K=4, λ=1.50'}
print(f"{'вар':>5s}{'віднос.похибка':>16s}{'MSE ×A':>9s}{'MSE ×M0':>10s}  опис")
for k in ['A', 'M0', 'M4', 'M8', 'MW', 'MF', 'L50', 'L150']:
    if not ERR[k]:
        continue
    r = float(np.mean(ERR[k]))
    print(f"{k:>5s}{r*100:>15.4f}%{mse['A']/mse[k]:>9.2f}"
          f"{mse['M0']/mse[k]:>10.2f}  {names[k]}")
print("\n🔑 ЧИТАТИ:")
print("   M4 проти M0 — це САМ γ (не ефект поділу). M4 проти MF — інформація")
print("   проти калібровки. Потрібне для життя: MSE ×34 проти A (0.85%→0.145%).")
print("   λ-скан: своя поправка мусить мати оптимум при λ>0 (контроль С89).")
