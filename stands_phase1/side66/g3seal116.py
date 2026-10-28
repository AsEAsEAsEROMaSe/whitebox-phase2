"""🔒 ПЛОМБА НА СПІЛЬНИЙ ШУМ: γ₃ з половини A, ЦІЛЬ із половини B.

[g3sh116] γ₃@9..15 дала ×1.13, але γ₃ і ціль бралися з ОДНОГО прогону банку.
С115-к спалилась рівно на цьому («оракул із того самого прогону, що й ціль,
ЗАВИЩУЄ СЕБЕ»). Банк gamhalf має дві НЕЗАЛЕЖНІ половини по 2²² семплів.

Режими: shared = γ₃(A) проти цілі A  ·  split = γ₃(A) проти цілі B (і навпаки).
Різниця shared−split і Є артефактом спільного шуму.
🔮 S1 (55%) split лишається ≥×1.05 · S2 (30%) обвалюється до ≤×1.02
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

def closure(W, g3=None, c=0.0):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if g3 is not None and i in LAY and c != 0.0:
            m = m - c*s*(g3[i]/6.0)*t*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamhalf_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W,
                  g3=[np.stack([d["h%d_g3_%d" % (h, L)] for L in range(D)]) for h in (0, 1)],
                  T=[d["h%d_relu_15" % h] for h in (0, 1)],
                  m=closure(W)))
lg("банків %d" % len(B))
ADJ0 = (1.0/(1.0/V_MC + 1.0/ANCH_E2))*MULT
CS = [0.25, 0.40, 0.50, 0.60, 0.75]

# 🔒 шум цілі між половинами
dab = float(np.mean([np.linalg.norm(b["T"][0]-b["T"][1])/np.linalg.norm(b["T"][1]) for b in B]))
print("🔒 A-vs-B на relu₁₅: %.4f%% ⟹ шум однієї половини %.4f%%" % (100*dab, 100*dab/np.sqrt(2)))
print("🔒 corr(γ₃_A, γ₃_B) ш15 = %.4f (мусить бути ~1: сигнал реальний, шум малий)"
      % float(np.mean([np.corrcoef(b["g3"][0][15], b["g3"][1][15])[0,1] for b in B])))

for tag, gi, ti in (("🔒 shared  γ₃(A) → ціль A", 0, 0),
                    ("   split   γ₃(A) → ціль B", 0, 1),
                    ("   split   γ₃(B) → ціль A", 1, 0)):
    base = float(np.mean([cerr(b["m"], b["T"][ti]) for b in B]))
    def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/base))))*MULT
    row = []
    for c in CS:
        v = float(np.mean([cerr(closure(b["W"], b["g3"][gi], c), b["T"][ti]) for b in B]))
        row.append(ADJ0/sc(v))
    print("%-28s | %s  ⟹ ×%.4f при c=%.2f"
          % (tag, " ".join("%6.4f" % x for x in row), max(row), CS[int(np.argmax(row))]))
    lg("  %s" % tag.strip())
lg("ГОТОВО")
