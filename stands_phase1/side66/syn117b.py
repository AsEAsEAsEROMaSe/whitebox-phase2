"""С117 — СИНЕРГІЯ mp↔s, ВИПРАВЛЕНИЙ ПРИЛАД.

🚨 syn117 БУВ ЗЛАМАНИЙ: міряв corr(log mp, log s) з РІВНОЮ вагою по нейронах,
а підлога шуму банку для `log mp` дорівнює `1/(t√N)` і вибухає при малих `t`.
Ознака поломки була видна прямо в таблиці: e_mp = 1.8…5.5% у логарифмічній
мірі проти 0.10…0.21% в енергетичній ⟹ ×20 ⟹ мірялась НЕ похибка, а шум.
📏 ЗАКОН: якщо дві міри ТІЄЇ САМОЇ похибки розходяться на порядок — це не
   «різні статистики», це поломка приладу; шукати підлогу.

Тут: (1) енергетична вага `mp²`, (2) відбір `t > 1` (де log добре зумовлений),
(3) підлога рахується і друкується поруч.

🔮 S1' (55%) при енергетичній вазі corr стає ПОМІТНО додатною (>0.3)
🔮 S2' (45%) якщо ні — синергія ×8.4 має інший механізм, і його треба шукати
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def state(W):
    m = np.zeros(NW); C = np.eye(NW); ST = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        ST.append((mp.copy(), s.copy()))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return ST

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wbcov/cov_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"]); N = int(d["n"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, N=N,
                  mp=[d["mp_%d" % L] for L in range(D)],
                  sd=[np.sqrt(np.maximum(np.diag(d["cp_%d" % L].astype(np.float64)), 1e-30))
                      for L in range(D)]))
lg("мереж %d" % len(B))
STS = [state(b["W"]) for b in B]
lg("стани пораховано")

print("\n── ЕНЕРГЕТИЧНО зважена кореляція похибок mp і s (%d мереж) ──" % len(B))
print("%4s %8s %8s %9s %8s %8s %8s" % ("шар", "e_mp%", "e_s%", "підл_mp%",
      "corr_w", "нахил", "N(t>1)"))
for L in range(1, D):
    EM, ES, FL, CR, SL, NN = [], [], [], [], [], []
    for k, b in enumerate(B):
        mpo, so = STS[k][L]; mpt, st = b["mp"][L], b["sd"][L]
        tt = mpt/st
        sel = (tt > 1.0) & (mpo > 1e-12)
        if sel.sum() < 64: continue
        u = np.log(mpo[sel]/mpt[sel]); v = np.log(so[sel]/st[sel])
        wgt = mpt[sel]**2; wgt = wgt/wgt.sum()
        um = u - (wgt*u).sum(); vm = v - (wgt*v).sum()
        suu = float((wgt*um*um).sum()); svv = float((wgt*vm*vm).sum())
        CR.append(float((wgt*um*vm).sum())/np.sqrt(suu*svv))
        SL.append(float((wgt*um*vm).sum())/svv)
        EM.append(100*np.sqrt(suu)); ES.append(100*np.sqrt(svv))
        # підлога банку на log mp, зважена так само
        FL.append(100*np.sqrt(float((wgt*(st[sel]/(mpt[sel]*np.sqrt(b["N"])))**2).sum())))
        NN.append(int(sel.sum()))
    print("%4d %8.4f %8.4f %9.4f %8.4f %8.4f %8.0f"
          % (L, np.mean(EM), np.mean(ES), np.mean(FL), np.mean(CR), np.mean(SL), np.mean(NN)))
    sys.stdout.flush()
lg("ГОТОВО")
print("DONE_SYN2")
