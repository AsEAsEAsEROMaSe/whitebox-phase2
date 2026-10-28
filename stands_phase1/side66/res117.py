"""С117 — ЕМПІРИЧНИЙ ЗАЛИШОК КРОКУ: ЧИ Є ІНФОРМАЦІЯ ПОНАД γ₃?

МОТИВ (не гучна назва, а число). Наш крок рахує E[ReLU(p)] за гаусовою
формулою `mp·Φ(t) + s·φ(t)`. γ₃-поправка — це ЛИШЕ ТРЕТІЙ ЧЛЕН ряду цієї
похибки. Але банк дає `relu_L` ПРЯМО ⟹ повний залишок

    Δ_j = relu_j − [mp_j Φ(t_j) + sd_j φ(t_j)]        (при ІСТИННИХ mp,sd)
    δ_j = Δ_j / sd_j                                   (безрозмірний)

вимірюється БЕЗ жодної моделі. γ₃-член є його наближенням:
    δ_edg = −(γ₃/6)·t·φ + (γ₄/24)·(t²−1)·φ

ПИТАННЯ, ЯКЕ ЖОДНА СЕСІЯ НЕ СТАВИЛА: яку ЧАСТКУ δ бере δ_edg, і чи має
залишок `δ − δ_edg` виграш у рекурсії. Якщо має — існує метод, що витягає
з тих самих семплів БІЛЬШЕ, ніж третій кумулянт, за ту саму ціну.

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 P1 (70%) γ₃-член пояснює 60–90% ‖δ‖ на глибоких шарах (бо |t|≈2.2, ряд збігається)
 P2 (45%) повний δ дає у рекурсії БІЛЬШЕ за γ₃ (×1.15…×1.35 проти ×1.13)
 P3 (30%) залишок δ−δ_edg САМ ПО СОБІ дає ≥×1.03 ⟹ є що витягати понад Едж.
 P4 (75%) оптимальне c для повного δ БЛИЖЧЕ ДО 1, ніж 0.40 для γ₃
          (бо δ — точна величина, а γ₃-член її лише частина)
 P5 (60%) контроль «чужа δ» ламає (<1.0), «переміш» ламає
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
    """dl: dict шар→δ (безрозмірний, додається як +c*s*δ)"""
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if dl is not None and i in LAY and c != 0.0:
            m = m + c*s*dl[i]
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
    b = dict(name=nm, W=W, T=d["relu_15"], n=int(d["n"]))
    dfull, dedg, dg3 = {}, {}, {}
    for L in range(D):
        mp, sd = d["mp_%d" % L], d["sd_%d" % L]
        g3, g4, rl = d["g3_%d" % L], d["g4_%d" % L], d["relu_%d" % L]
        t = mp/np.maximum(sd, 1e-300); ph = phi(t); Ph = Phi(t)
        gauss = mp*Ph + sd*ph
        dfull[L] = (rl - gauss)/np.maximum(sd, 1e-300)          # ТОЧНИЙ залишок
        dg3[L] = -(g3/6.0)*t*ph                                  # третій член
        dedg[L] = dg3[L] + (g4/24.0)*(t*t - 1.0)*ph              # + четвертий
    b["full"], b["edg"], b["g3"] = dfull, dedg, dg3
    b["res"] = {L: dfull[L] - dedg[L] for L in range(D)}         # понад Еджворта
    b["res3"] = {L: dfull[L] - dg3[L] for L in range(D)}         # понад γ₃
    B.append(b)
lg("банків %d · семплів у банку %d" % (len(B), B[0]["n"]))

# ── 1. РОЗКЛАД: скільки ‖δ‖ бере кожен член ──────────────────────────────
print("\n── розклад залишку кроку δ = Δ/sd (24 мережі, RMS по нейронах) ──")
print("%4s %10s %10s %10s %8s %8s %8s" % ("шар", "‖δ‖", "‖δ_g3‖", "‖δ_edg‖",
      "cos g3", "R² g3", "R² edg"))
for L in range(D):
    nf = np.mean([np.sqrt(np.mean(b["full"][L]**2)) for b in B])
    n3 = np.mean([np.sqrt(np.mean(b["g3"][L]**2)) for b in B])
    ne = np.mean([np.sqrt(np.mean(b["edg"][L]**2)) for b in B])
    cs, r3, re = [], [], []
    for b in B:
        f, g, e = b["full"][L], b["g3"][L], b["edg"][L]
        ff = float(f @ f)
        if ff <= 0: continue
        cs.append(float(f @ g)/np.sqrt(ff*max(float(g @ g), 1e-300)))
        r3.append(1.0 - float((f-g) @ (f-g))/ff)
        re.append(1.0 - float((f-e) @ (f-e))/ff)
    print("%4d %10.3e %10.3e %10.3e %8.4f %8.4f %8.4f"
          % (L, nf, n3, ne, np.mean(cs), np.mean(r3), np.mean(re)))

# ── 2. РЕКУРСІЯ: скан c для кожного варіанта δ ───────────────────────────
LAY = tuple(range(9, 16))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
ADJ0 = sc(BASE)
lg("база %.6e · adj %.4e" % (BASE, ADJ0))

rs = np.random.RandomState(7717)
def variant(b, kind):
    if kind == "g3":   return b["g3"]
    if kind == "edg":  return b["edg"]
    if kind == "full": return b["full"]
    if kind == "res":  return b["res"]
    if kind == "res3": return b["res3"]
    if kind == "foreign":                       # δ іншої мережі
        o = B[(B.index(b)+7) % len(B)]; return o["full"]
    if kind == "shuf":
        return {L: b["full"][L][rs.permutation(NW)] for L in range(D)}
    if kind == "const":                         # пошарова СТАЛА (S=0)
        return {L: np.full(NW, b["full"][L].mean()) for L in range(D)}
    raise ValueError(kind)

CS = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20]
KINDS = ["g3", "edg", "full", "res", "res3", "foreign", "shuf", "const"]
print("\n── рекурсія, шари 9..15, скан c (×база у КОСИНУСНІЙ похибці) ──")
print("%9s | %s" % ("δ", " ".join("c=%.2f" % c for c in CS)))
best = {}
for k in KINDS:
    row, dls = [], [variant(b, k) for b in B]
    for c in CS:
        v = float(np.mean([cerr(closure(b["W"], dl, c, LAY), b["T"])
                           for b, dl in zip(B, dls)]))
        row.append(BASE/v)
    best[k] = (CS[int(np.argmax(row))], max(row))
    print("%9s | %s" % (k, " ".join("%6.4f" % x for x in row)))
    sys.stdout.flush()

print("\n%9s %8s %8s %12s" % ("δ", "c*", "×база", "adj"))
for k in KINDS:
    c, r = best[k]
    print("%9s %8.2f %8.4f %12.4e" % (k, c, r, sc(BASE/r)))
lg("ГОТОВО")
print("DONE_RES117")
