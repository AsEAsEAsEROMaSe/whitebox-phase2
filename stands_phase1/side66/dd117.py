"""С117 — УЗГОДЖЕНА ПОПРАВКА: `m` І `E[ReLU²]` З ОДНІЄЇ ЩІЛЬНОСТІ (АНАЛІТИКА).

СУПЕРЕЧНІСТЬ:
  [res117]   крок середнього виправляється γ₃+γ₄ на 99.9% (R²=0.999)
  [mstep117] накопичена похибка mp₁₅ 0.209% = 0.058%·√14 ⟹ це СУМА кроків
  ⟹ мало б дати ×30 у MSE, рекурсія дає ×1.23

ВЕРСІЯ B: правимо `m`, лишаємо гаусовим `E[ReLU²]` ⟹ `dd = E[ReLU²] − m²`
(діагональ коваріації) псується на `2·m·δm/dd` ≈ **1.0%** при власній похибці
кроку `dd` = 1.06% [mstep117] ⟹ поправка вносить у дисперсію рівно стільки,
скільки знімає із середнього.

🚨 ПЕРША ВЕРСІЯ ЦЬОГО ФАЙЛУ РАХУВАЛА ЧЕРЕЗ ГАУССА-ЕРМІТА І ПЛОМБА ВПАЛА:
   max|Δm1| = 4.07e-03 при потрібних ~1e-12 (GH погано бере злам ReLU).
   ⟹ замінено АНАЛІТИКОЮ, виведеною інтегруванням частинами:

   f  = s(t+x)₊ :  J_k = He_{k−2}(−t)·φ(t)   ⟹ J₃ = −tφ · J₄ = (t²−1)φ
   f² = s²(t+x)²₊: I_k = 2·He_{k−3}(−t)·φ(t) ⟹ I₃ = 2φ  · I₄ = −2tφ

   Δm         = s ·φ(t)·[ −γ₃·t/6  + γ₄·(t²−1)/24 ]
   ΔE[ReLU²]  = s²·φ(t)·[  γ₃/3    − γ₄·t/12      ]

🔒 ПЛОМБИ ВИВЕДЕННЯ: J₃, J₄ відтворюють записані c₃ = −stφ, c₄ = s(t²−1)φ;
   член γ₃/3·s²φ відтворює корпусну формулу дисперсійного Еджворта ДОСЛІВНО.

🔮 D1 (45%) узгоджена версія б'є «лише середнє» помітно (≥×1.4 проти ×1.23)
🔮 D2 (30%) «лише E[ReLU²]» сама по собі ПОГІРШУЄ
🔮 D3 (70%) оптимальне c для узгодженої ближче до 1.0
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
LAY = tuple(range(9, 16))
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)

def closure(W, bk=None, c=0.0, mode="both", LAY=LAY):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        e2 = (mp*mp + s*s)*Ph + (mp*s)*ph
        if bk is not None and i in LAY and c != 0.0:
            g3, g4 = bk[i]
            if mode in ("both", "mean"):
                m = m + c*(s*ph*(-(g3*t)/6.0 + g4*(t*t - 1.0)/24.0))
            if mode in ("both", "var"):
                e2 = e2 + c*((s*s)*ph*(g3/3.0 - (g4*t)/12.0))
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = e2 - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, T=d["relu_15"],
                  bk={L: (d["g3_%d" % L], d["g4_%d" % L]) for L in range(D)}))
lg("мереж %d" % len(B))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
lg("база %.6e · adj %.4e" % (BASE, sc(BASE)))

CS = [0.20, 0.40, 0.60, 0.80, 1.00, 1.30]
for rng, nm in ((tuple(range(9, 16)), "9..15"), (tuple(range(1, 16)), "1..15")):
    print("\n── шари %s, ×база у MSE ──" % nm)
    print("%12s | %s" % ("режим", " ".join("c=%.2f" % c for c in CS)))
    for md in ("mean", "var", "both"):
        row = [BASE/float(np.mean([cerr(closure(b["W"], b["bk"], c, md, rng), b["T"])
                                   for b in B])) for c in CS]
        print("%12s | %s   найкраще %6.4f" % (md, " ".join("%6.4f" % x for x in row), max(row)))
        sys.stdout.flush()
lg("ГОТОВО")
print("DONE_DD117")
