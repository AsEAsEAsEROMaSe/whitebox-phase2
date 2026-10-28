"""С117 — ПАРАДОКС c*=0.4: ЧИ ПОХИБКА СТАНУ ВЖЕ РОБИТЬ РОБОТУ γ₃?

Факт: ТОЧНА поправка δ (R²=0.999) хоче вагу c*≈0.4 навіть на ОДНОМУ шарі 15,
де подальшого переносу нема [full117: LAY={15} c*≤0.40]. Отже причина не в
накопиченні, а в тому, що δ(істинний стан) не є правильною поправкою до
НАШОГО стану.

Розклад на шарі L, понейронно:
   e_ours  = m_ours − m_true                       (наша похибка)
   e_state = G(mp_o, s_o) − G(mp_t, s_t)           (від хибного стану)
   e_ng    = G(mp_t, s_t) − m_true = −s_t·δ        (негаусовість кроку)
   e_ours  = e_state + e_ng  (для чистого гаусового кроку)
Оптимальна вага поправки −e_ng:  c* = 1 + β,  β = (e_state·e_ng)/(e_ng·e_ng)
⟹ c*=0.4 ⟺ β = −0.6 ⟺ ПОХИБКА СТАНУ АНТИКОРЕЛЬОВАНА з негаусовістю і вже
   компенсує 60% її.

🔮 P1 (60%) β ∈ [−0.75, −0.45] на шарі 15 ⟹ механізм підтверджено
🔮 P2 (50%) β монотонно росте за модулем з глибиною (стан дрейфує у бік γ₃)
🔮 P3 (55%) розклад β по каналах: частка від `s` ≫ частки від `mp`
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def G(mp, s):
    t = mp/s; return mp*Phi(t) + s*phi(t)

def state(W, lam):
    m = np.zeros(NW); C = np.eye(NW); ST = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if lam != 0.0: m = m - lam*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        ST.append((mp.copy(), s.copy(), m.copy()))
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
                  sd=[d["sd_%d" % L] for L in range(D)], relu=[d["relu_%d" % L] for L in range(D)]))
lg("мереж %d" % len(B))

for lam, tag in ((0.0, "СИРЕ замикання"), (LAM_FEAT, "з shape_corr (LAM=%.2f)" % LAM_FEAT)):
    STS = [state(b["W"], lam) for b in B]
    print("\n── %s: β = проєкція похибки СТАНУ на негаусовість ──" % tag)
    print("%4s %8s %8s %8s | %8s %8s %8s | %8s" % ("шар", "‖e_st‖", "‖e_ng‖", "‖e_our‖",
          "β", "β_mp", "β_s", "c*=1+β"))
    for L in (1, 4, 8, 11, 13, 15):
        acc = np.zeros(6)
        for k, b in enumerate(B):
            mpo, so, mo = STS[k][L]; mpt, st, mt = b["mp"][L], b["sd"][L], b["relu"][L]
            gt = G(mpt, st)
            e_state = G(mpo, so) - gt
            e_ng = gt - mt
            e_our = mo - mt
            # розклад стану по каналах: лише mp хибне / лише s хибне
            e_mp = G(mpo, st) - gt
            e_s = G(mpt, so) - gt
            nn = float(e_ng @ e_ng)
            acc += [np.linalg.norm(e_state), np.linalg.norm(e_ng), np.linalg.norm(e_our),
                    float(e_state @ e_ng)/nn, float(e_mp @ e_ng)/nn, float(e_s @ e_ng)/nn]
        acc /= len(B)
        print("%4d %8.2e %8.2e %8.2e | %8.3f %8.3f %8.3f | %8.3f"
              % (L, acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], 1.0 + acc[3]))
        sys.stdout.flush()
lg("ГОТОВО")
print("DONE_PAR117")
