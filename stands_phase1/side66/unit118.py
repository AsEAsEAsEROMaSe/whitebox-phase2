import sys, os, numpy as np
sys.path.insert(0, "/mnt/win_d/llm/White-Box-P2"); sys.path.insert(0, "/mnt/win_d/llm/White-Box-P2/side66")
os.environ["NETS"] = "regina-johnston"; os.environ["SEAL"] = "0"
import pyarrow.parquet as pq
import flopscope.numpy as fnp
import estimator_p2_k3 as E
NET = "regina-johnston"
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]; idx = NAMES.index(NET)
W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(16, 1024, 1024)
Kt = np.load("/home/trim/wblogs/sub118_regina-johnston.npz")["P3f_1"]
def R2s(K, T): off = ~np.eye(1024, dtype=bool); return (1-((T-K)[off]**2).sum()/(T[off]**2).sum(), (K*T).sum()/(K*K).sum())
# --- стенд (numpy, без прунінгу) ---
from math import erf, factorial
phi = lambda t: np.exp(-0.5*t*t)/np.sqrt(2*np.pi); Phi = lambda t: 0.5*(1+np.vectorize(erf)(t/np.sqrt(2)))
mp = np.zeros(1024); Sp = W[0].T @ W[0]; s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s)
ph = phi(t); Ph = Phi(t)
exec(open("rec118h.py").read().split("import pyarrow")[0].split("class Source")[0])  # hcoef, He, mm, FL
src_code = "class Source" + open("rec118h.py").read().split("class Source")[1].split("def core_contract")[0]
PRH = 0.0; PRC = 0.0; exec(src_code)
S = Source(mp, Sp, 2); m1 = mp*Ph + s*ph
S.push(W[1]); Ks = S.kab()
print("стенд        R²off %.3f scale %.3f" % R2s(Ks, Kt))
# --- оцінювач ---
c = E._hcoef(fnp.asarray(s), fnp.asarray(t), 6, fnp.asarray(ph), fnp.asarray(Ph))
cs = hcoef(s, t, 6)
print("hcoef max rel diff:", max(float(np.abs(np.asarray(c[k]) - cs[k]).max()/np.abs(cs[k]).max()) for k in range(7)))
for prh, prc in [(0.0, 0.0), (0.05, 0.03)]:
    E._PRH, E._PRC = prh, prc
    src = E._Source(fnp.asarray(s), fnp.asarray(t), fnp.asarray(rho), c, fnp.asarray(ph))
    src.push(fnp.asarray(W[1]).astype(fnp.float32))
    Ke = np.asarray(src.kab(), np.float64)
    print("оцінювач PRH=%.2f PRC=%.2f  R²off %.3f scale %.3f · |I|=%s" % ((prh, prc) + R2s(Ke, Kt) + (None if src.I is None else int(np.asarray(src.I).sum()),)))
# --- _mm проти matmul ---
A = np.random.default_rng(0).standard_normal((1024, 1024)).astype(np.float32); B = np.random.default_rng(1).standard_normal((1024, 1024)).astype(np.float32)
for sh in [(1024, 1024), (1024, 512), (512, 1024)]:
    a = fnp.asarray(A[:sh[0], :sh[1]]); b = fnp.asarray(B[:sh[1], :sh[0]])
    d = np.asarray(E._mm(a, b), np.float64) - np.asarray(a, np.float64) @ np.asarray(b, np.float64)
    print("_mm %s: max|Δ|/max|AB| %.2e" % (sh, np.abs(d).max()/np.abs(np.asarray(a, np.float64) @ np.asarray(b, np.float64)).max()))
print("DONE_UNIT")
