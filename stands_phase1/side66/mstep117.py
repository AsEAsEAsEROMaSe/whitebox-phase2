"""С117 — ПОХИБКА КРОКУ КОВАРІАЦІЇ В ІЗОЛЯЦІЇ (аналог res117 для Мелера).

[res117] крок СЕРЕДНЬОГО при істинних моментах виправляється на 99.9% через
γ₃+γ₄, а в рекурсії дає лише ×1.19-1.23 ⟹ решта похибки мусить бути в
переносі `C`. Але ПОХИБКУ САМОГО КРОКУ `C` не міряв ніхто — усі виміри були
або накопиченими, або через вихід.

Тут: беремо ІСТИННУ cp_l (банк повних коваріацій), робимо РІВНО наш крок
   Cov(h_l) ≈ cp·Φ⊗Φ + ½cp²·(φ/s)⊗(φ/s), діагональ = точна dd
   cp_{l+1} = W_{l+1}ᵀ · Cov(h_l) · W_{l+1}
і порівнюємо з ІСТИННОЮ cp_{l+1}.

🔒 ПЛОМБА: на шарі 0 `p` гаусів ТОЧНО ⟹ похибка кроку мусить бути на рівні
   шуму банку (1/√N ≈ 0.07%), а не структурною.

🔮 M1 (70%) похибка ОДНОГО кроку cp ≈ 0.3–1.5% ⟹ накопичення ×2–5 до 5.9%
🔮 M2 (60%) похибка ДІАГОНАЛІ кроку менша за похибку позадіагоналі
🔮 M3 (50%) частка похибки, яку бере точна `dd`, велика (діагональ уже точна)
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def mehler_step(cp, mp, w_next, exact_dd=True):
    """наш крок: cp_l (передактивації) → cp_{l+1}"""
    s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
    t = mp/s; ph = phi(t); Ph = Phi(t)
    m = mp*Ph + s*ph
    dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
    Ch = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
    if exact_dd: Ch = Ch + np.diag(dd - np.diag(Ch))
    return w_next.T @ Ch @ w_next, m

def rel(a, b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
FS = sorted(glob.glob("/home/trim/wbcov/cov_*.npz"))
lg("мереж %d" % len(FS))
ACC = np.zeros((D-1, 6)); CNT = 0
for f in FS:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    for L in range(D-1):
        cpT = d["cp_%d" % L].astype(np.float64); mpT = d["mp_%d" % L]
        nxT = d["cp_%d" % (L+1)].astype(np.float64)
        pr, mpred = mehler_step(cpT, mpT, W[L+1], True)
        pr0, _ = mehler_step(cpT, mpT, W[L+1], False)
        sp = np.sqrt(np.maximum(np.diag(pr), 1e-30)); st = np.sqrt(np.maximum(np.diag(nxT), 1e-30))
        Rp = pr/np.outer(sp, sp); Rt = nxT/np.outer(st, st)
        ACC[L] += [100*rel(pr, nxT), 100*rel(sp, st), 100*rel(Rp, Rt),
                   100*rel(pr0, nxT),
                   100*rel(mpred, d["relu_%d" % L]),
                   100*rel(np.diag(pr), np.diag(nxT))]
    CNT += 1
    lg("%-20s готово" % nm)
ACC /= CNT
print("\n── похибка ОДНОГО кроку від ІСТИННОЇ cp_l до cp_{l+1} (%%, %d мереж) ──" % CNT)
print("%4s %9s %9s %9s %10s %9s %9s" % ("l→l+1", "cp повна", "sd", "корел R",
      "cp без dd", "m кроку", "diag cp"))
for L in range(D-1):
    print("%5d %9.4f %9.4f %9.4f %10.4f %9.4f %9.4f"
          % (L, ACC[L,0], ACC[L,1], ACC[L,2], ACC[L,3], ACC[L,4], ACC[L,5]))
lg("ГОТОВО")
print("DONE_MSTEP")
