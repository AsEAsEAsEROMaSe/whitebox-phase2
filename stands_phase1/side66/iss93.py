"""iss93 — КОНТРОЛЬНА ЗМІННА 4-ГО ПОРЯДКУ ПО `z` ІЗ `b ≡ 0` ЗА ІССЕРЛІСОМ.

════════════════ ЧОМУ САМЕ 4-Й ПОРЯДОК ════════════════
`[ЗМІРЯНО varq93 + пломба a₀]` Антитетика `(z,−z)` робить емпіричне середнє
БУДЬ-ЯКОЇ непарної функції `z` рівним 0 ТОЧНО; вибілювання робить точним
середнє БУДЬ-ЯКОЇ квадратичної форми. Пломба: на `a₀` теорія `½‖W_j‖²` і
вимір збіглись до **1.245e-08**, `Var(Q̄₀) = 4.4e-08` ⟹ `g₀ = 5.5e+05`.
```
⟹ порядки 1, 2, 3 накриті ТОЧНО. ПЕРШИЙ ВІЛЬНИЙ — 4. Між ними НЕМА нічого.
```

════════════════ ЧОМУ НЕ ПОВТОРЮЄМО МЕРТВЕ ════════════════
🚨 `[ЦИТАТА CLAUDE.md С89]` «конуси `f(z)=|z|f(u)` точно (пломба 5.5e-07), але
поверх вибілювання **НУЛЬ**» ⟹ **ІЗОТРОПНА частина 4-го порядку вже з'їдена.**
`‖z‖⁴` — радіальний, тобто вже мертвий. Живою лишається лише АНІЗОТРОПНА
частина, а її напрямки мусять прийти з ГЛИБИНИ мережі, бо з He-init `W₀`
жоден напрям не виділений.

════════════════ ВИВЕДЕННЯ НАПРЯМКІВ ════════════════
Похибка нашої оцінки для вихідного нейрона `j` іде вздовж ефективного
лінійного шляху від `z` до цього нейрона:
```
M = (whiten·W₀) · G₁W₁ · G₂W₂ · … · G₃₁W₃₁ ,   G_i = diag(частка відкритих)
```
`M_j` — стовпець `j`. Регресор із ТОТОЖНО нульовим середнім (Іссерліс,
`E[(u·z)⁴] = 3‖u‖⁴` для `z~N(0,I)`):
```
c_j = ⟨(M_j·z)⁴⟩ − 3‖M_j‖⁴          E[c_j] = 0 ТОЧНО ⟹ b ≡ 0 ⟹ τ ≡ 0
```
🔑 Механізму, що вбив `λ` (`τ = N·b²/σ_Q²`), тут НЕ ІСНУЄ ЗА ПОБУДОВОЮ.
🔑 І це ПО РЕГРЕСОРУ НА КОЖЕН ВИХІДНИЙ НЕЙРОН — структура збігається з ціллю.

════════════════ ЩО ПОРІВНЮЄТЬСЯ ════════════════
```
C_path  c_j із M_j (шлях до нейрона j)              ← виведений
C_iso   c = ⟨‖z‖⁴⟩ − n(n+2)  (один на мережу)       ← ⛔мусить дати НУЛЬ (С89)
C_rnd   M_j замінено на ВИПАДКОВІ напрямки          ← ⛔контроль «чужа»
C_w0    M_j = стовпці whiten·W₀ (без глибини)       ← чи потрібна глибина
```

════════════════ ПЛОМБИ ════════════════
P1 `E[c_j] = 0`: середнє `c_j` по багатьох розіграшах мусить бути ≪ його std.
P2 `C_iso` мусить дати `R² ≈ 0` — інакше суперечність із С89 і стенд бракований.
P3 `C_rnd` мусить дати `R² ≈ 0`.
P4 ЦІНА рахується явно у флопах.
P5 `R²` — LOO ПО МЕРЕЖАХ (одиниця незалежності — мережа).

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) C_iso дасть R² < 0.01 (підтвердить С89)                  шанс 85%
 (б) C_path дасть R² 0.02…0.15                                шанс 45%
 (в) C_path поб'є C_w0 щонайменше вдвічі (глибина потрібна)    шанс 55%
 (г) підсумковий виграш на балі < ×1.10                        шанс 70%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

sys.path.insert(0, "/mnt/win_d/llm/White-Box/side66")
import sub91_port as P                                        # noqa: E402

P.W_BLEND = 0.0883
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        fm = np.asarray(t.column("final_means").combine_chunks().flatten()
                        ).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i].astype(np.float64)
            got += 1


def run(wc, npair, seed, gr):
    """Конвеєр сабміту + регресори 4-го порядку. Повертає (ours, {ім'я: c})."""
    g = torch.Generator().manual_seed(seed)
    n = wc[0].shape[0]
    z = torch.randn(npair, n, generator=g)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    w0 = wh @ wc[0]
    y = z @ w0
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    denom = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / denom - sg0 * P.INV_SQRT_2PI
    M = w0.clone()                     # ефективний шлях z → поточний шар
    for i in range(1, len(wc)):
        p = x @ wc[i]
        gate = (p[:P.GATE_N] > 0).to(p.dtype).sum(0).double() / P.GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        M = ((M.double() * 1.0) @ wd * gate).float()
        x = torch.relu(p)
        if i == len(wc) - 1:
            last = x.double().sum(0) / denom
    ours = last - v
    # ── регресори 4-го порядку, усі з ТОЧНО нульовим середнім ──
    zd = z.double()
    out = {}
    for nm, U in (('path', M.double()), ('w0', w0.double()),
                  ('rnd', torch.randn(n, M.shape[1], generator=gr,
                                      dtype=torch.float64))):
        yy = zd @ U                                  # (npair, m)
        # 🚨 НЕ ‖U_j‖²: вибілювання сидить УСЕРЕДИНІ U ⟹ ⟨y²⟩ = U_jᵀ·cov·U_j.
        #    Емпіричний 4-й кумулянт ⟨y⁴⟩−3⟨y²⟩² нульовий за побудовою для
        #    БУДЬ-ЯКОГО фіксованого u і сам себе нормує. Пломба P1 це спіймала.
        s2 = yy.pow(2).mean(0)
        out[nm] = (yy.pow(4).mean(0) - 3.0 * s2 * s2) / s2.clamp_min(1e-30) ** 2
    r2 = (zd * zd).sum(1)
    out['iso'] = ((r2.pow(2).mean() - n * (n + 2)) / (n * (n + 2))
                  ).expand(M.shape[1]).clone()
    return ours, out


NAMES = ('path', 'w0', 'rnd', 'iso')
OURS, TR, DEEP, REG = [], [], [], {k: [] for k in NAMES}
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i])) for i in range(32)]
        wc, npair, _ = P.cut_budget91(W, "cpu")
        gr = torch.Generator().manual_seed(88000 + q)
        ours, reg = run(wc, npair, 700000 + 53 * q, gr)
        _, _, _, _, m_deep = P.closure91(wc, P.KSIG, "cpu", kdeep=len(wc))
        OURS.append(ours.numpy()); DEEP.append(m_deep.numpy())
        TR.append(truth)
        for k in NAMES:
            REG[k].append(reg[k].numpy())
        if q % 20 == 0:
            print(f"  net{q:>4d} n_pairs={npair} ({time.time()-t0:.0f} с)",
                  flush=True)

O = np.array(OURS); T = np.array(TR); D = np.array(DEEP)
R = {k: np.array(REG[k]) for k in NAMES}
Q = len(T)
print(f"\n  час {time.time()-t0:.0f} с · {Q} мереж\n")
W_B, AL = P.W_BLEND, P.ALPHA
pred = (1 - W_B) * O + W_B * (1 + AL) * D
E = pred - T                                    # похибка, яку зменшуємо
base = float((E * E).mean())
print("=" * 88)
print(f"РЕГРЕСОРИ 4-ГО ПОРЯДКУ ПО `z` · {Q} мереж · LOO по мережах")
print("=" * 88)
print(f"  база (сабміт) MSE = {base:.6e}")
for k in NAMES:
    c = R[k]
    print(f"  ⛔P1 E[c_{k}] = {c.mean():+.4e} проти std {c.std():.4e} "
          f"⟹ {abs(c.mean())/max(c.std(),1e-300):.4f} σ")


def loo(c):
    """Один спільний λ, LOO по мережах. c: (Q,256)."""
    num = (c * E).sum(1)
    den = (c * c).sum(1)
    err = np.zeros_like(E)
    for q in range(Q):
        m = np.arange(Q) != q
        lam = num[m].sum() / max(den[m].sum(), 1e-300)
        err[q] = E[q] - lam * c[q]
    return float((err * err).mean())


print(f"\n{'регресор':>10s}{'MSE':>14s}{'бал ×':>10s}{'R²':>10s}{'вирок':>22s}")
for k in NAMES:
    v = loo(R[k] * (1 - W_B))
    r2 = 1 - v / base
    ok = ('⛔мусить бути 0' if k in ('rnd', 'iso') else
          '✅ живий' if v < base else '⛔мертвий')
    print(f"{k:>10s}{v:>14.6e}{base/v:>10.4f}{r2:>10.4f}{ok:>22s}")
print(f"\n  ціна: M будується 31 матмулом 256³ = 1.04e9 = "
      f"{100*1.04e9/2.72e11:.3f}% B · z@M = 2·N·256² ≈ "
      f"{100*2*7300*65536/2.72e11:.3f}% B")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/iss93.npz",
         O=O, T=T, D=D, **{f"r_{k}": R[k] for k in NAMES})
