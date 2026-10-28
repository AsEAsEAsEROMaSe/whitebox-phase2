"""γ₃ НА МІЛКИХ ШАРАХ — там, де НАШІ МОМЕНТИ ЩЕ ТОЧНІ, А ПЕРЕДАЧА НАЙБІЛЬША.

ЧОМУ ЦЕ НЕ ПОВТОР `prop116`: той міряв γ₃@8 і глибше (×0.96, шкодить). Мілкі
шари не пробувались ЖОДНОГО РАЗУ, а саме там збігаються три умови:
  [gam116b, після віднімання підлоги приладу] похибка mp: ш1 0.009% · ш4 0.115%
      ⟹ на мілких моменти МАЙЖЕ ТОЧНІ — умова, яку корпус називає ворітьми γ
      («γ вмикається лише на точних (m,C)», С101 edg ×1.02 проти edgm ×22.8)
  [stransfer116] коефіцієнт передачі до виходу: ш2 0.500 · ш4 0.339 · ш15 0.070
  [gam116b] γ₃ там МАЛА (sd 0.038 на ш1) ⟹ ряд Еджворта збігається найкраще

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  R1 (55%) γ₃@1..4 дає ≥×1.05 (на відміну від ×0.96 для γ₃@8+)
  R2 (30%) ≥×1.15
  R3 (70%) ефект монотонно спадає з початковим шаром: @1..4 > @5..8 > @9..15
🔒 КОНТРОЛІ: чужа γ₃ · перемішана γ₃ · скан `c` з вершиною параболи
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT, TRUE2 = 3.3478454e-06, 1.4355576e-06, 0.1, 0.909290
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)
rel = lambda a, b: 100*np.linalg.norm(a-b)/np.linalg.norm(b)

def closure(W, g3=None, lay=(), c=0.0):
    m = np.zeros(NW); C = np.eye(NW); MP = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w; MP.append(mp.copy())
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if g3 is not None and i in lay and c != 0.0:
            m = m - c*s*(g3[i]/6.0)*t*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m, np.stack(MP)

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, g3=np.stack([d["g3_%d" % L] for L in range(D)]),
                  T=d["relu_15"]))
lg("банків %d" % len(B))
BASE = float(np.mean([cerr(closure(b["W"])[0], b["T"]) for b in B]))
ADJ0 = (1.0/(1.0/V_MC + 1.0/ANCH_E2))*MULT
lg("база %.6e" % BASE)
def sc(c_):
    e2 = c_*(ANCH_E2/BASE); return (1.0/(1.0/V_MC + 1.0/e2))*MULT

def ev(lay, c, mode="own"):
    ce = []
    for k, b in enumerate(B):
        if mode == "own":     g = b["g3"]
        elif mode == "foreign": g = B[(k+1) % len(B)]["g3"]
        else:
            rs = np.random.RandomState(4242+k); g = np.stack([rs.permutation(b["g3"][L]) for L in range(D)])
        ce.append(cerr(closure(b["W"], g, lay, c)[0], b["T"]))
    return float(np.mean(ce))

SETS = [("γ₃@1..4", tuple(range(1,5))), ("γ₃@1..7", tuple(range(1,8))),
        ("γ₃@5..8", tuple(range(5,9))), ("γ₃@9..15", tuple(range(9,16))),
        ("γ₃@ALL", tuple(range(D)))]
GRID = [-0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5]
print("\n%-12s %8s %12s %10s %8s" % ("набір","c","косинусна","adj","×база"))
BESTS = {}
for tag, lay in SETS:
    best = (BASE, 0.0)
    for c in GRID:
        v = ev(lay, c)
        if v < best[0]: best = (v, c)
        print("%-12s %8.2f %12.5e %10.4e  ×%.4f" % (tag, c, v, sc(v), ADJ0/sc(v)))
    BESTS[tag] = best
    print("   ⟹ найкраще c=%.2f ⟹ ×%.4f" % (best[1], ADJ0/sc(best[0])))

tag = min(BESTS, key=lambda z: BESTS[z][0]); v, c = BESTS[tag]
lay = dict(SETS)[tag]
print("\n🔒 КОНТРОЛІ на %s (c=%.2f, ×%.4f)" % (tag, c, ADJ0/sc(v)))
for md in ("foreign", "shuffle"):
    vv = ev(lay, c, md); print("   🔒 %-8s %12.5e ⟹ ×%.4f" % (md, vv, ADJ0/sc(vv)))
lg("ГОТОВО")
