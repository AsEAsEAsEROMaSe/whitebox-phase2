"""ЩО САМЕ ВИПРАВЛЯЄ γ₃ — ЗСУВ ПРОФІЛЮ ПОХИБКИ ПІСЛЯ ЇЇ ПРИКЛАДАННЯ.

ПОСТАНОВКА ЮЗЕРА: «зрозуміти як саме на рівні чисел вони витягають, і далі
через мат суть виводити дешевші». Стенд не питає «чи стало краще» (вже знаємо,
×1.13) — він питає, ДЕ поділась похибка і що лишилось.

Три виміри, яких корпус не робив:
 A. профіль похибки (mp, sd) по 16 шарах ДО і ПІСЛЯ γ₃@9..15
 B. оракульні сходи «вкинути істинні (mp,sd) на шарі K» ДО і ПІСЛЯ γ₃
    ⟹ якщо стеля просіла — γ₃ і оракул беруть ОДНЕ, якщо ні — РІЗНЕ
 C. розклад залишку за смугами |t| ДО і ПІСЛЯ

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 V1 (60%) після γ₃ похибка mp₁₅ ПАДАЄ (а не росте, як було при вкиданні `s`)
 V2 (45%) стеля оракула mpC@15 просідає менш ніж на 20% ⟹ канали РІЗНІ
 V3 (55%) залишок після γ₃ зсувається до нейронів із МАЛИМ |t| (там ряд
          Еджворта найсильніший, отже він там і спрацював)
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
LAY = tuple(range(9, 16)); C3 = 0.40
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T), a*x - T
rel = lambda a, b: 100*np.linalg.norm(a-b)/np.linalg.norm(b)

def closure(W, g3=None, c3=0.0, inj=(), mpT=None, sdT=None):
    m = np.zeros(NW); C = np.eye(NW); MP = []; SD = []; TT = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20))
        if i in inj and sdT is not None:
            g = np.maximum(sdT[i], 1e-20)/s; cp_ = cp_*np.outer(g, g); s = np.maximum(sdT[i], 1e-20)
        mp = m @ w
        if i in inj and mpT is not None: mp = mpT[i]
        MP.append(mp.copy()); SD.append(s.copy())
        t = mp/s; TT.append(t.copy()); ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if g3 is not None and i in LAY and c3 != 0.0:
            m = m - c3*s*(g3[i]/6.0)*(t*ph)
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m, np.stack(MP), np.stack(SD), np.stack(TT)

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, T=d["relu_15"],
                  mp=np.stack([d["mp_%d" % L] for L in range(D)]),
                  sd=np.stack([d["sd_%d" % L] for L in range(D)]),
                  g3=np.stack([d["g3_%d" % L] for L in range(D)])))
lg("банків %d" % len(B))
ADJ0 = None
def score(cs):
    v = float(np.mean(cs)); return v
def adjof(v, base): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/base))))*MULT

# ── A. профіль ДО і ПІСЛЯ ──
PA = np.zeros((len(B), D, 2)); PB = np.zeros((len(B), D, 2)); cA = []; cB = []
for k, b in enumerate(B):
    m0, mp0, sd0, _ = closure(b["W"])
    m1, mp1, sd1, _ = closure(b["W"], b["g3"], C3)
    cA.append(cerr(m0, b["T"])[0]); cB.append(cerr(m1, b["T"])[0])
    for L in range(D):
        PA[k, L] = [rel(mp0[L], b["mp"][L]), rel(sd0[L], b["sd"][L])]
        PB[k, L] = [rel(mp1[L], b["mp"][L]), rel(sd1[L], b["sd"][L])]
BASE = score(cA); WITH = score(cB)
lg("база %.6e · з γ₃ %.6e ⟹ ×%.4f" % (BASE, WITH, adjof(BASE, BASE)/adjof(WITH, BASE)))
A, Bp = PA.mean(0), PB.mean(0)
print("\n── A. ПРОФІЛЬ ПОХИБКИ ДО / ПІСЛЯ γ₃@9..15 ──")
print("%4s %10s %10s %8s | %10s %10s %8s" % ("шар","mp до %","mp після","×","sd до %","sd після","×"))
for L in range(D):
    print("%4d %9.4f%% %9.4f%% %8.3f | %9.4f%% %9.4f%% %8.3f"
          % (L, A[L,0], Bp[L,0], A[L,0]/max(Bp[L,0],1e-12),
             A[L,1], Bp[L,1], A[L,1]/max(Bp[L,1],1e-12)))

# ── B. оракульні сходи ДО і ПІСЛЯ ──
print("\n── B. ОРАКУЛ «істинні (mp,sd) на шарі K» ДО / ПІСЛЯ γ₃ ──")
print("%6s %12s %12s %10s %10s" % ("K","без γ₃","з γ₃","×без","×з"))
for K in (8, 12, 14, 15):
    o0 = [cerr(closure(b["W"], inj=(K,), mpT=b["mp"], sdT=b["sd"])[0], b["T"])[0] for b in B]
    o1 = [cerr(closure(b["W"], b["g3"], C3, inj=(K,), mpT=b["mp"], sdT=b["sd"])[0], b["T"])[0] for b in B]
    v0, v1 = score(o0), score(o1)
    print("%6d %12.5e %12.5e %10.3f %10.3f" % (K, v0, v1, BASE/v0, WITH/v1))
    lg("  K=%d" % K)

# ── C. залишок за смугами |t| на шарі 15 ──
print("\n── C. ЗАЛИШОК ЗА СМУГАМИ |t| НА ШАРІ 15 ──")
EDG = [0.0, 1.0, 2.0, 3.0, 4.0, 99.0]
accA = np.zeros(len(EDG)-1); accB = np.zeros(len(EDG)-1); cnt = np.zeros(len(EDG)-1)
for b in B:
    m0, _, _, tt0 = closure(b["W"]); m1, _, _, _ = closure(b["W"], b["g3"], C3)
    _, r0 = cerr(m0, b["T"]); _, r1 = cerr(m1, b["T"])
    t = np.abs(tt0[15])
    for j in range(len(EDG)-1):
        msk = (t >= EDG[j]) & (t < EDG[j+1])
        accA[j] += float((r0[msk]**2).sum()); accB[j] += float((r1[msk]**2).sum()); cnt[j] += msk.sum()
print("%12s %10s %14s %14s %8s" % ("смуга |t|","нейронів","‖залишок‖² до","після","×"))
for j in range(len(EDG)-1):
    print("%5.1f–%-6.1f %10d %14.5e %14.5e %8.3f"
          % (EDG[j], EDG[j+1], cnt[j], accA[j], accB[j], accA[j]/max(accB[j],1e-30)))
print("частка похибки, знята γ₃, по смугах: %s"
      % " ".join("%.1f%%" % (100*(accA[j]-accB[j])/max(accA.sum()-accB.sum(),1e-30)) for j in range(len(EDG)-1)))
lg("ГОТОВО")
