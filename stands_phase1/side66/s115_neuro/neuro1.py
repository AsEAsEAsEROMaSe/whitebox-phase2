"""С115-е: ПОНЕЙРОННИЙ РОЗКЛАД ПОХИБКИ ЗАМИКАННЯ, фаза 2 (1024x16).

Питання ЮЗЕРА: чому точність саме така і де зони руху.
Істина — `all_layer_means` із датасету (400k семплів), НЕЗАЛЕЖНА від нашого коду.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (щоб не тлумачити заднім числом):
  P1 похибка `m` росте з глибиною і НАСИЧУЄТЬСЯ (закон L^0.28)      70%
  P2 після зняття масштабу (косинусна) лишиться 40-70% сирої        60%
  P3 похибка ІЗОТРОПНА по нейронах: топ-8 несуть ~= сигнал топ-8    65%
  P4 |t| велике (медіана ~3.5) ⟹ E[relu] ~ m, форма майже не видна  75%
"""
import numpy as np, pyarrow.parquet as pq, sys, json, time

PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

# ─────────── константи оцінювача (з estimator_p2_BEST_v11.py) ───────────
KAPPA, LAM_FEAT, FEAT_N0, KSIG = 0.9999, 0.35, 256.0, 8
import ast as _ast
src = open("/mnt/win_d/llm/White-Box-P2/estimator_p2_BEST_v11.py").read()
_tree = _ast.parse(src); ns = {}
for _n in _tree.body:
    if isinstance(_n, _ast.Assign) and len(_n.targets) == 1 and isinstance(_n.targets[0], _ast.Name):
        try: ns[_n.targets[0].id] = _ast.literal_eval(_n.value)
        except Exception: pass
FW = np.array(ns["_FW"], dtype=np.float64)
KAPPA, LAM_FEAT, FEAT_N0, KSIG = ns["_KAPPA"], ns["_LAM_FEAT"], ns["_FEAT_N0"], ns["_KSIG"]
log("константи: KAPPA=%.6g LAM_FEAT=%.4g KSIG=%d FW[%d]" % (KAPPA, LAM_FEAT, KSIG, len(FW)))

INV_SQRT_2PI = 0.3989422804014327
def phi(x): return np.exp(-0.5*x*x)*INV_SQRT_2PI
from math import erf as _erf
def Phi(x): return 0.5*(1.0+np.vectorize(_erf)(x/np.sqrt(2.0)))

def closure(W, use_feat=True, use_kappa=True):
    """рекурсія замикання. Повертає m,s,t по шарах (список масивів 1024)."""
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n)
    MS, SS, TS, VV = [], [], [], []
    for i in range(d):
        w = W[i]
        cp = w.T @ C @ w
        if use_kappa and i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if use_feat and LAM_FEAT != 0.0:
            m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy()); SS.append(s.copy()); TS.append(t.copy()); VV.append(dd.copy())
        if i == d-1: break
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS, SS, TS, VV

def shape_corr(cpm, s, t, Ph, ph, mn, depth_frac):
    n = s.shape[0]; inv = 1.0/s
    R = cpm*np.outer(inv, inv); R = R - np.diag(np.diag(R))
    Rs = R*R; sc = FEAT_N0/n
    R2 = Rs.sum(1)*sc; R3 = (Rs*R).sum(1)*sc; R4 = (Rs*Rs).sum(1)*sc
    A1 = np.abs(R).sum(1)*sc
    lPR = np.log(np.maximum(R2*R2/np.maximum(R4, 1e-30), 1e-30))
    sn = s/s.mean(); tp = t*ph; hp = (t*t-1.0)*ph
    y = (FW[0]+FW[1]*t+FW[2]*Ph+FW[3]*ph+FW[4]*tp+FW[5]*hp+FW[6]*R2
         + FW[7]*R3+FW[8]*R4+FW[9]*A1+FW[10]*lPR+FW[11]*sn+FW[12]*depth_frac
         + FW[13]*(t*R2)+FW[14]*(ph*R2)+FW[15]*(hp*R2))
    corr = y*s
    g = mn/max(np.sqrt((mn*mn).sum()), 1e-30)
    return corr - (corr*g).sum()*g

# ─────────── дані ───────────
tb = pq.read_table(PQ)
log("parquet: %d мереж" % tb.num_rows)
res = []
for k in range(min(NNET, tb.num_rows)):
    nm = tb["mlp_name"][k].as_py()
    W = np.asarray(tb["weights"][k].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][k].as_py(), dtype=np.float64).reshape(16, 1024)
    av = tb["avg_variance"][k].as_py()
    MS, SS, TS, VV = closure(W)
    MS0, _, _, _ = closure(W, use_feat=False, use_kappa=False)
    row = {"name": nm, "avg_variance": av, "layers": []}
    for i in range(16):
        m, s, t, tr = MS[i], SS[i], TS[i], T[i]
        nz = float(np.sqrt(np.maximum(VV[i],0).sum()/400000.0))
        e = m - tr
        nt = np.sqrt((tr*tr).sum())
        a = (m@tr)/max((m*m).sum(), 1e-300)          # оптимальний масштаб
        ecos = a*m - tr
        e0 = MS0[i] - tr
        a0 = (MS0[i]@tr)/max((MS0[i]*MS0[i]).sum(), 1e-300)
        row["layers"].append(dict(
            i=i, noise=float(nz/nt), raw=float(np.sqrt((e*e).sum())/nt), cos=float(np.sqrt((ecos*ecos).sum())/nt),
            raw0=float(np.sqrt((e0*e0).sum())/nt),
            cos0=float(np.sqrt(((a0*MS0[i]-tr)**2).sum())/nt),
            scale=float(1.0/a),
            t_med=float(np.median(np.abs(t))), t_lo=float(np.percentile(np.abs(t), 5)),
            frac_t3=float((np.abs(t) < 3).mean()),
            s_med=float(np.median(s)), m_med=float(np.median(m)),
            # концентрація: частка ‖e‖² у топ-8 нейронах, і те саме для сигналу
            e_top8=float(np.sort(ecos*ecos)[-8:].sum()/max((ecos*ecos).sum(), 1e-300)),
            T_top8=float(np.sort(tr*tr)[-8:].sum()/max((tr*tr).sum(), 1e-300)),
            corr_e_t=float(np.corrcoef(np.abs(ecos), np.abs(t))[0, 1]),
            corr_e_s=float(np.corrcoef(np.abs(ecos), s)[0, 1]),
            corr_e_m=float(np.corrcoef(np.abs(ecos), np.abs(tr))[0, 1]),
        ))
    res.append(row); log("%2d %-22s ш0 %.4f%%  ш8 %.4f%%  ш15 %.4f%% (cos)" %
        (k, nm, row["layers"][0]["cos"]*100, row["layers"][8]["cos"]*100, row["layers"][15]["cos"]*100))
json.dump(res, open("/home/trim/wblogs/neuro1.json", "w"))
log("ГОТОВО -> /home/trim/wblogs/neuro1.json")
