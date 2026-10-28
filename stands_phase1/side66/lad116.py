"""ЧИ СПРАВДІ СЕРЕДНІ ШАРИ МАРНІ — ЧИ ЦЕ АРТЕФАКТ ПІДІГНАНИХ КОНСТАНТ?

[shift116 §B] вкидання ІСТИННИХ (mp,sd) на шарі K дало:
    K=8 ×0.541 · K=12 ×0.381 · K=14 ×0.637 · K=15 ×9.532
Істина В СЕРЕДИНІ ШКОДИТЬ у 2–3×. Два прочитання:
  (H1) «збалансована система похибок» — фізика мережі;
  (H2) АРТЕФАКТ: `_KAPPA`, `_LAM_FEAT`, `_FW` підбирались НА ВЛАСНІЙ траєкторії.
       Вкидання істини зсуває стан із неї, підігнані константи перестають
       підходити (закон С100 «константи не переносяться між конфігураціями»).
Косинусна метрика знімає МАСШТАБ ⟹ `_ALPHA` тут ні до чого. Але `_KAPPA` і
`_LAM_FEAT` діють на ФОРМУ і лишаються неузгодженими.

РОЗДІЛЯЄМО: та сама драбина на СИРОМУ замиканні (LAM_FEAT=0, KAPPA=1).
зникає шкода ⟹ H2, і «приз лише на останньому шарі» — хибний висновок.
лишається   ⟹ H1, найсильніший вимір гасіння в проєкті.

🔮 Y1 (55%) на сирому середні шари ПЕРЕСТАЮТЬ шкодити (× > 1.0)
🔮 Y2 (40%) шкодять і там ⟹ гасіння справжнє
🔒 ПЛОМБА: K=15 мусить лишитись сильно додатним В ОБОХ режимах — там
   траєкторія до шару 15 не чіпається взагалі.
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

def closure(W, inj=(), mpT=None, sdT=None, lam=None, kap=None):
    lam = LAM_FEAT if lam is None else lam
    kap = KAPPA if kap is None else kap
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*kap
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20))
        if i in inj and sdT is not None:
            g = np.maximum(sdT[i], 1e-20)/s
            cp_ = cp_*np.outer(g, g); s = np.maximum(sdT[i], 1e-20)
        mp = m @ w
        if i in inj and mpT is not None: mp = mpT[i]
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if lam != 0.0: m = m - lam*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamall_*.npz"))[:16]:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    B.append(dict(name=nm, W=W, T=d["relu_15"],
                  mp=np.stack([d["mp_%d" % L] for L in range(D)]),
                  sd=np.stack([d["sd_%d" % L] for L in range(D)])))
lg("банків %d" % len(B))
for tag, lam, kap in (("ПОВНИЙ (lam=0.35, kap=0.9999)", None, None),
                      ("СИРИЙ  (lam=0, kap=1.0)", 0.0, 1.0)):
    base = float(np.mean([cerr(closure(b["W"], lam=lam, kap=kap), b["T"]) for b in B]))
    print("\n── %s ·  база %.5e ──" % (tag, base), flush=True)
    print("%8s %14s %10s" % ("K", "косинусна", "×база"), flush=True)
    for K in (4, 8, 12, 14, 15):
        v = float(np.mean([cerr(closure(b["W"], inj=(K,), mpT=b["mp"], sdT=b["sd"],
                                        lam=lam, kap=kap), b["T"]) for b in B]))
        print("%8d %14.5e %10.3f" % (K, v, base/v), flush=True); lg("  K=%d" % K)
    for K in (12, 14):
        v = float(np.mean([cerr(closure(b["W"], inj=tuple(range(K, D)), mpT=b["mp"],
                                        sdT=b["sd"], lam=lam, kap=kap), b["T"]) for b in B]))
        print("%5d..15 %14.5e %10.3f" % (K, v, base/v), flush=True)
lg("ГОТОВО")
