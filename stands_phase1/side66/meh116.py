"""МЕЛЕР K=3,4 У КРОЦІ КОВАРІАЦІЇ — ЧИ ЖИВЕ ВІН У РЕЖИМІ ФАЗИ 2.

ВИВЕДЕННЯ (не вгадування). Для `f(x) = s·(t+x)₊` коефіцієнти Ерміта
`c_k = E[f(X)He_k(X)]` беруться інтегруванням частинами; крайовий член
зникає, бо при `x=−t` маємо `t+x=0`:
    c_0 = s·ψ(t),  ψ = φ+tΦ
    c_1 = s·Φ(t)
    c_2 = s·φ(t)
    c_3 = s·(−t·φ(t))
    c_4 = s·((t²−1)·φ(t))
Мелер: `Cov(f_i,f_j) = Σ_{k≥1} (ρ^k/k!)·c_k[i]·c_k[j]`, `ρ = cp_ij/(s_i s_j)`.
Код нині має рівно k=1,2:
    C = cp·Φ⊗Φ + (1/2)·cp²·(φ/s)⊗(φ/s)
Отже k=3 і k=4 у тих самих змінних:
    += (1/6)·cp³·(tφ/s²)⊗(tφ/s²)
    += (1/24)·cp⁴·((t²−1)φ/s³)⊗((t²−1)φ/s³)
🔒 ДІАГОНАЛЬ у коді перезаписується ТОЧНОЮ формулою ⟹ члени чіпають лише
   позадіагональ, і ряд там збігається як `ρ^k/k!`.

ЧОМУ ЦЕ НЕ ПОВТОР С100 («квадратура vs Мелер = 0.1% ⟹ мертво»): той вимір
зроблено при `|t|` мед **3.50**, де ряд Еджворта РОЗХОДИВСЯ. У фазі 2
`|t|` мед **2.21** і С116 зміряла `‖c₃‖/‖гаус‖ = 0.0005` ⟹ ряд ЗБІГАЄТЬСЯ.
Це рівно та зміна режиму, що воскресила γ₃.
🔑 І воно діє НА ВСІХ ШАРАХ ОДНОЧАСНО ⟹ проходить закон lad116 («часткове
   виправлення структурно від'ємне»).

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 Z1 (45%) K=3 з фітованим коефіцієнтом дає ≥×1.02
 Z2 (20%) ≥×1.05
 Z3 (65%) оптимальний коефіцієнт СТИСНЕНИЙ (<1), як усі попередні
 ⚠️ СТЕЛЯ ВІДОМА НАПЕРЕД: канал `C` з ПОВНИМ оракулом = +23% [С115]
    ⟹ Мелер може взяти лише частину цього
🔒 ПЛОМБА: при коефіцієнті 0 мусить відтворитись база БІТ У БІТ
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

def closure(W, c3=0.0, c4=0.0, g3=None, cg=0.0):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if g3 is not None and cg != 0.0 and 9 <= i <= 15:
            m = m - cg*s*(g3[i]/6.0)*(t*ph)
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        cp2 = cp_*cp_
        C = cp_*np.outer(Ph, Ph) + 0.5*cp2*np.outer(ph/s, ph/s)
        if c3 != 0.0:
            u3 = (t*ph)/(s*s)
            C = C + (c3/6.0)*(cp2*cp_)*np.outer(u3, u3)
        if c4 != 0.0:
            u4 = ((t*t - 1.0)*ph)/(s*s*s)
            C = C + (c4/24.0)*(cp2*cp2)*np.outer(u4, u4)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, T=d["relu_15"],
                  g3=np.stack([d["g3_%d" % L] for L in range(D)])))
lg("банків %d" % len(B))
BASE = float(np.mean([cerr(closure(b["W"]), b["T"]) for b in B]))
ADJ0 = (1.0/(1.0/V_MC + 1.0/ANCH_E2))*MULT
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE))))*MULT
lg("база %.6e" % BASE)
print("🔒 ПЛОМБА c3=c4=0: %.6e проти бази %.6e" % (
    float(np.mean([cerr(closure(b["W"], 0.0, 0.0), b["T"]) for b in B])), BASE))

print("\n── МЕЛЕР K=3 (скан коефіцієнта), БЕЗ γ₃ ──")
print("%8s %14s %10s" % ("c3", "косинусна", "×база"))
best3 = (0.0, 1.0)
for c in (-1.0, -0.5, 0.25, 0.5, 1.0, 2.0, 4.0):
    v = float(np.mean([cerr(closure(b["W"], c, 0.0), b["T"]) for b in B]))
    r = ADJ0/sc(v); print("%8.2f %14.5e %10.4f" % (c, v, r), flush=True)
    if r > best3[1]: best3 = (c, r)
print("⟹ найкраще c3=%.2f ⟹ ×%.4f" % best3)

print("\n── + МЕЛЕР K=4 при найкращому c3 ──")
for c in (-1.0, 0.5, 1.0, 2.0):
    v = float(np.mean([cerr(closure(b["W"], best3[0], c), b["T"]) for b in B]))
    print("%8.2f %14.5e %10.4f" % (c, v, ADJ0/sc(v)), flush=True)

print("\n── РАЗОМ ІЗ γ₃@9..15 (c=0.40): чи МНОЖАТЬСЯ канали ──")
vg = float(np.mean([cerr(closure(b["W"], 0.0, 0.0, b["g3"], 0.40), b["T"]) for b in B]))
print("  сама γ₃            %14.5e  ×%.4f" % (vg, ADJ0/sc(vg)), flush=True)
for c in (best3[0], best3[0]*2):
    v = float(np.mean([cerr(closure(b["W"], c, 0.0, b["g3"], 0.40), b["T"]) for b in B]))
    print("  γ₃ + K=3 c3=%.2f  %14.5e  ×%.4f" % (c, v, ADJ0/sc(v)), flush=True)
lg("ГОТОВО")
