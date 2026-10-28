"""С117 — ЧОМУ ІДЕАЛЬНИЙ КРОК ДАЄ ×1.23, А НЕ ×30.

ФАКТИ, ЩО СТВОРИЛИ СУПЕРЕЧНІСТЬ:
  [res117]  R²(γ₃+γ₄ пояснює залишок кроку) = 0.999 ⟹ амплітуда залишку ×31
  [mstep117] накопичена похибка mp₁₅ 0.209% = 0.058%·√14 ⟹ це СУМА кроків
  [res117]  але в рекурсії (шари 9..15) точний δ дає лише ×1.2298 у MSE

ДВІ ВЕРСІЇ:
  A) діапазон: правились 7 шарів із 15 ⟹ стеля ×1.9, не ×30
  B) неузгодженість: правимо `m`, лишаємо гаусовим `E[ReLU²]` ⟹ `dd` псується

Тут перевіряється (A): скан ДІАПАЗОНУ шарів при точному δ.
(B) — у `dd117.py`.

🔮 F1 (60%) LAY=усі дає ×1.9…×2.5, тобто (A) пояснює частину, але не все
🔮 F2 (30%) LAY=усі дає ≥×5 ⟹ (A) пояснює все, і треба просто ширший діапазон
🔮 F3 (40%) є ОПТИМАЛЬНИЙ діапазон, і він НЕ «усі» ⟹ живе (B) або баланс
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)

def closure(W, dl=None, c=0.0, LAY=()):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if dl is not None and i in LAY and c != 0.0: m = m + c*s*dl[i]
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    dl = {}
    for L in range(D):
        mp, sd = d["mp_%d" % L], d["sd_%d" % L]
        t = mp/np.maximum(sd, 1e-300); ph = phi(t); Ph = Phi(t)
        dl[L] = (d["relu_%d" % L] - (mp*Ph + sd*ph))/np.maximum(sd, 1e-300)
    B.append(dict(name=nm, W=W, T=d["relu_15"], dl=dl))
lg("мереж %d" % len(B))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
lg("база %.6e · adj %.4e" % (BASE, sc(BASE)))

RANGES = [("15",       (15,)),
          ("14..15",   tuple(range(14, 16))),
          ("12..15",   tuple(range(12, 16))),
          ("9..15",    tuple(range(9, 16))),
          ("5..15",    tuple(range(5, 16))),
          ("1..15",    tuple(range(1, 16))),
          ("0..15",    tuple(range(0, 16)))]
CS = [0.40, 0.60, 0.80, 1.00, 1.20]
print("\n── ТОЧНИЙ δ, скан ДІАПАЗОНУ шарів (×база у MSE) ──")
print("%9s | %s" % ("шари", " ".join("c=%.2f" % c for c in CS)))
for nm, LAY in RANGES:
    row = [BASE/float(np.mean([cerr(closure(b["W"], b["dl"], c, LAY), b["T"]) for b in B]))
           for c in CS]
    print("%9s | %s   найкраще %6.4f" % (nm, " ".join("%6.4f" % x for x in row), max(row)))
    sys.stdout.flush()
lg("ГОТОВО")
print("DONE_FULL117")
