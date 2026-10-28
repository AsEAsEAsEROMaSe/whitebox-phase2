"""С117 — ПАРАДОКС c*=0.4 У ТІЙ САМІЙ МЕТРИЦІ, ДЕ ЙОГО ЗМІРЯНО.

[par117] у СИРІЙ метриці β=+0.24 ⟹ c*=1.24, а стенди в КОСИНУСНІЙ дають 0.4.
Розбіжність = масштаб: ‖e_state‖ учетверо більша за ‖e_ng‖ і майже вся —
спільний множник, який косинусна метрика знімає.

Тут на шарі 15 БЕЗ рекурсії: x(c) = m_ours + c·s·δ, похибка = cerr(x, T).
Скан c дрібно; три варіанти множника при δ: s_ours · s_true · 1 (δ·s_true=−e_ng).
Плюс контроль: та сама процедура з ВИПАДКОВИМ вектором тієї ж норми.

🔮 Q1 (70%) у косинусній метриці c*≈0.4 відтворюється (інакше стенди брехали)
🔮 Q2 (50%) при s_true замість s_ours c* зсувається ≤0.1 ⟹ масштаб s не винен
🔮 Q3 (60%) після зняття масштабу β_⊥ ≈ −0.6 ⟹ механізм живе В ФОРМІ, не в нормі
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)
def G(mp, s):
    t = mp/s; return mp*Phi(t) + s*phi(t)

def state(W):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: return mp, s, m
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    L = D-1
    B.append(dict(name=nm, W=W, T=d["relu_%d" % L], mp=d["mp_%d" % L], sd=d["sd_%d" % L]))
lg("мереж %d" % len(B))
STS = [state(b["W"]) for b in B]
lg("стани пораховано")

CS = np.arange(0.0, 1.31, 0.1)
rs = np.random.RandomState(4117)
rows = {"s_ours·δ": [], "s_true·δ": [], "🔒випадк.": []}
beta_perp = []
for k, b in enumerate(B):
    mpo, so, mo = STS[k]; mpt, st, T = b["mp"], b["sd"], b["T"]
    delta = (T - G(mpt, st))/st                     # точний безрозмірний залишок
    d1, d2 = so*delta, st*delta
    rnd = rs.standard_normal(NW); rnd *= np.linalg.norm(d1)/np.linalg.norm(rnd)
    for key, dv in (("s_ours·δ", d1), ("s_true·δ", d2), ("🔒випадк.", rnd)):
        rows[key].append([cerr(mo + c*dv, T) for c in CS])
    # β після зняття масштабу: проєкція ортогонально до T
    u = T/np.linalg.norm(T)
    def perp(v): return v - (v @ u)*u
    e_st = perp(G(mpo, so) - G(mpt, st)); e_ng = perp(G(mpt, st) - T)
    beta_perp.append(float(e_st @ e_ng)/float(e_ng @ e_ng))
base = float(np.mean([r[0] for r in rows["s_ours·δ"]]))
print("\n── шар 15, БЕЗ рекурсії, косинусна MSE, ×база ──")
print("%10s | %s" % ("δ·", " ".join("%5.1f" % c for c in CS)))
for key, R in rows.items():
    R = np.mean(np.array(R), axis=0)
    ratio = base/R
    print("%10s | %s   c*=%.1f ×%.4f" % (key, " ".join("%5.3f" % x for x in ratio),
          CS[int(np.argmax(ratio))], ratio.max()))
print("\nβ_⊥ (масштаб знято) = %.3f ± %.3f ⟹ c* = 1+β_⊥ = %.3f"
      % (np.mean(beta_perp), np.std(beta_perp)/np.sqrt(len(beta_perp)), 1 + np.mean(beta_perp)))
lg("ГОТОВО")
print("DONE_PAR117B")
