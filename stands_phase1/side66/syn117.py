"""С117 — МЕХАНІЗМ СИНЕРГІЇ ×8.4 МІЖ КАНАЛАМИ mp І sd.

[С116] лише mp точне ×1.685 · лише sd точне ×1.160 · ОБИДВА ×16.524.
Така супермультиплікативність має рівно одне просте пояснення:
похибки `mp` і `s` КОРЕЛЬОВАНІ по нейронах, а `t = mp/s` тоді майже правильне;
косинусна метрика знімає спільний масштаб ⟹ корельована частина БЕЗКОШТОВНА,
а часткове виправлення її РОЗРИВАЄ і робить похибку реальною.

🔮 S1 (75%) corr(log(mp_o/mp_t), log(s_o/s_t)) > 0.6 на глибоких шарах
🔮 S2 (60%) нахил регресії ≈ 1 ⟹ це справді спільний множник
🔮 S3 (50%) залишок log(mp) після зняття log(s) МЕНШИЙ за сиру похибку mp у ≥2×
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
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, mp=[d["mp_%d" % L] for L in range(D)],
                  sd=[d["sd_%d" % L] for L in range(D)]))
lg("мереж %d" % len(B))

STS = [state(b["W"]) for b in B]
lg("стани пораховано")
print("\n── похибки mp і s: чи це СПІЛЬНИЙ МНОЖНИК? (%d мереж) ──" % len(B))
print("%4s %8s %8s %8s %8s %9s %9s" % ("шар", "e_mp%", "e_s%", "corr", "нахил",
      "залишок%", "×виграш"))
for L in range(1, D):          # шар 0 вироджений: mp ≡ 0
    cs, sl, em, es, rr = [], [], [], [], []
    for b in B:
        mpo, so = STS[B.index(b)][L]
        mpt, st = b["mp"][L], b["sd"][L]
        ok = (mpt > 1e-12) & (mpo > 1e-12)
        if ok.sum() < 32: continue                # log лише для додатних mp
        u = np.log(mpo[ok]/mpt[ok]); v = np.log(so/st)[ok]
        u = u - u.mean(); v = v - v.mean()
        c = float(u @ v)/np.sqrt(float(u @ u)*float(v @ v))
        beta = float(u @ v)/float(v @ v)
        res = u - beta*v
        cs.append(c); sl.append(beta)
        em.append(100*np.sqrt(np.mean(u*u))); es.append(100*np.sqrt(np.mean(v*v)))
        rr.append(100*np.sqrt(np.mean(res*res)))
    print("%4d %8.4f %8.4f %8.4f %8.4f %9.4f %9.3f"
          % (L, np.mean(em), np.mean(es), np.mean(cs), np.mean(sl),
             np.mean(rr), np.mean(em)/max(np.mean(rr), 1e-12)))
    sys.stdout.flush()
lg("ГОТОВО")
print("DONE_SYN")
