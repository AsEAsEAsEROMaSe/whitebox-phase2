"""bcross93 — ЧИ ЛЯГАЄ ПЕРЕТИН `b_ℓ = σ_Q,ℓ/√N` НА ℓ≈8? ПЛОМБА ТЕОРІЇ `λ*`.

════════════════ ЩО ПЕРЕВІРЯЄТЬСЯ ════════════════
АРХІВ вивів для оцінювача з контрольною змінною `m̂ = X̄ − λ(Q̄ − q_замик)`,
де `q_замик = q_істина + b`:
```
MSE(λ) = (σ_X² − 2λ·Cov + λ²σ_Q²)/N + λ²b²
λ* = Cov/(σ_Q² + N·b²)        τ ≡ N·b²/σ_Q²
канал живий ⟺ b_ℓ < σ_Q,ℓ/√N
```
🚨 Якщо це вірно, то `_KSIG = 8` у сабміті — не цінове обмеження (як я
записав у С92 і на чому втратив сабміт), а **саме цей перетин**.
📏 ПЕРЕДБАЧЕННЯ АРХІВУ: при `N ≈ 8300` (стільки семплів у сабміті) перетин
мусить лягти на `ℓ ≈ 8`.

════════════════ ЯК ════════════════
`q_ℓ = E[a_ℓ²]` понейронно. `b_ℓ = q_замик − q_іст` (детерміноване зміщення),
`σ_Q,ℓ² = Var_семпл(a_ℓ²)` ⟹ шум вибіркового `Q̄` є `σ_Q/√N`.
Істину беру ВЕЛИКОЮ МК (окремий сід), бо офіційний датасет дає `all_layer_means`,
а не `E[a²]`.

════════════════ ПЛОМБИ ════════════════
P1 масштабування: `b_ℓ` мусить СТОЯТИ при N_істини ×4, `σ_Q,ℓ` теж стоїть
   (це властивості розподілу), а от ОЦІНКА `b` шумить як 1/√N_іст ⟹
   беру N_іст = 65536 проти робочих 8300, тобто шум оцінки `b` у 2.8× менший
   за саму величину, яку порівнюємо.
P2 шар 1: замикання там майже точне ⟹ `b_1/(σ_Q/√N)` мусить бути ≪ 1.
P3 монотонність `τ_ℓ` — з виведення `τ ∝ b²`, а `b` росте вглиб.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) перетин ляже в діапазоні ℓ = 5..14                    шанс 55%
 (б) τ на шарі 31 буде > 100                               шанс 70%
 (в) відношення b/(σ_Q/√N) монотонне по глибині            шанс 60%
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

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
NWORK = 8300                       # стільки семплів реально в сабміті
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
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i]
            got += 1


B, SQ, RT = [], [], []
t0 = time.time()
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i])) for i in range(32)]
        wc, npair, _ = P.cut_budget91(W, "cpu")
        depth = len(wc)
        # 🚨 closure91 віддає (s_o, m_o, q_o, p_o, m_deep) — треба ТРЕТІЙ.
        #    Перший прогін узяв s_o і дав 79% «похибки» на шарі 1, де замикання
        #    ТОЧНЕ за виведенням (mp=0 ⟹ q = 0.5·s² тотожно). Пломба P2 і
        #    спіймала «не той об'єкт».
        _, _, q_cl, _, _ = P.closure91(wc, depth - 1, "cpu", kdeep=depth)
        g = torch.Generator().manual_seed(510000 + 37 * q)
        # ⛔ вибілювання, як у сабміті: вхід має ОДИНИЧНУ емпіричну коваріацію
        z = torch.randn(NTRUE, 256, generator=g)
        cov = (z.T @ z) / NTRUE
        ev, V = torch.linalg.eigh(cov.double())
        x = (z.double() @ ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T)).float()
        bb, ss = [], []
        for i in range(depth):
            h = x @ wc[i]
            x = torch.relu(h)
            if i == 0:
                continue
            a2 = x.double().pow(2)
            qt = a2.mean(0)                       # істинне E[a²]
            sq = a2.std(0)                        # σ_Q понейронно
            bb.append(float((q_cl[i - 1] - qt).abs().mean()))
            ss.append(float(sq.mean()))
        B.append(bb); SQ.append(ss)
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

B, SQ = np.array(B), np.array(SQ)
print(f"\n  час {time.time()-t0:.0f} с · {len(B)} мереж · N_іст={NTRUE} · "
      f"N_роб={NWORK}\n")
noise = SQ / np.sqrt(NWORK)
rat = B / noise
tau = NWORK * B ** 2 / SQ ** 2
print("=" * 84)
print(f"ПЕРЕТИН `b_ℓ = σ_Q,ℓ/√N` · передбачення АРХІВУ: ℓ ≈ 8")
print("=" * 84)
print(f"{'шар':>5s}{'b_ℓ':>13s}{'σ_Q/√N':>13s}{'b/(σ_Q/√N)':>14s}{'τ=N b²/σ_Q²':>15s}")
for i in (0, 2, 4, 6, 7, 8, 9, 11, 15, 19, 23, 29):
    if i >= B.shape[1]:
        continue
    print(f"{i+1:>5d}{B[:,i].mean():>13.5e}{noise[:,i].mean():>13.5e}"
          f"{rat[:,i].mean():>14.4f}{tau[:,i].mean():>15.3f}")
m = rat.mean(0)
cr = next((i + 1 for i in range(len(m)) if m[i] >= 1.0), None)
print(f"\n📣 ПЕРЕТИН (перший шар із b ≥ σ_Q/√N): ℓ = {cr}")
print(f"   передбачення АРХІВУ ℓ≈8 · сабміт має _KSIG=8 "
      f"{'✅ ТЕОРІЮ ЗАПЕЧАТАНО' if cr and 5 <= cr <= 14 else '🚨 НЕ СХОДИТЬСЯ'}")
print(f"⛔P2 шар 1: b/(σ_Q/√N) = {m[0]:.4f} "
      f"{'✅ ≪1' if m[0] < 0.5 else '🚨'}")
print(f"⛔P3 монотонність відношення: "
      f"{'✅' if np.all(np.diff(m[:20]) > -0.05) else '⚠️ немонотонне'}")
print(f"\n   τ на шарі 31 = {tau[:,-1].mean():.1f} ⟹ λ* там менше за "
      f"оптимальне у {1+tau[:,-1].mean():.0f}× ⟹ канал сам зануляється")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/bcross93.npz", B=B, SQ=SQ)
