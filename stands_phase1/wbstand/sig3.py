"""С115-ж СТАДІЯ 3: чи є резидуал `C_true − cp_our` РАНГ-1 ВЗДОВЖ СЕРЕДНЬОГО?

Гіпотеза виведена з фіту стадії 1, а не вгадана:
    δ log s = 0.0012·t²  ⟹  δ(s²) = 0.0024·mp²  ⟹  δC_jk = 0.0024·mp_j·mp_k
Незалежний доказ ЗЗОВНІ: чужий write-up (skye) зміряв «ранг резидуала M2 = 2-5,
топ-мода вирівняна з μ̂, cos²≈0.6» — ми цю величину НІКОЛИ не міряли.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 cos(Δ, outer(mp,mp)) > 0.3 на глибоких шарах                       60%
  P2 фітований c у межах ×2 від передбаченого 0.0024                    55%
  P3 ранг-1 знімає >30% енергії Δ                                       50%
  P4 топ-власний вектор Δ має cos² з mp̂ > 0.3                           55%
"""
import numpy as np, pyarrow.parquet as pq, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def cp_chain(W):
    """повертає НАШІ cp (передактиваційна коваріація) по шарах."""
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); CP = []; MP = []
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        CP.append(cp.copy()); MP.append(mp.copy())
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        if i == d-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return CP, MP

tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
LAYS = [2, 5, 8, 11, 14, 15]
res = {L: [] for L in LAYS}
for f in range(6):
    d = np.load("/home/trim/wblogs/truthC_%02d.npz" % f)
    nm = str(d["name"]); k = NAMES.get(nm)
    if k is None: log("НЕМА %s" % nm); continue
    W = np.asarray(tb["weights"][k].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    CP, MP = cp_chain(W)
    Ct = d["C"]; mt = d["mp"]
    for L in LAYS:
        A = Ct[L].astype(np.float64); B = CP[L]
        D = A - B
        u = MP[L]                                   # НАШЕ mp (відоме даром)
        U = np.outer(u, u); nU2 = float((U*U).sum())
        c = float((D*U).sum()/nU2)
        eD = float((D*D).sum())
        rem = float(((D - c*U)**2).sum())
        cosU = float((D*U).sum()/np.sqrt(eD*nU2))
        # спектр резидуала
        ev, evec = np.linalg.eigh(D)
        o = np.argsort(-np.abs(ev)); ev = ev[o]; evec = evec[:, o]
        uh = u/np.linalg.norm(u)
        cos2 = float((evec[:, 0] @ uh)**2)
        pr = float((np.abs(ev).sum()**2)/max((ev*ev).sum(), 1e-300))
        top1 = float(ev[0]**2/max((ev*ev).sum(), 1e-300))
        res[L].append((c, cosU, 1.0-rem/eD, cos2, pr, top1,
                       eD/float((A*A).sum())))
    log("%d %s" % (f, nm))

print("\n%-3s %10s %8s %8s %8s %8s %8s %8s" %
      ("ш", "c фіт", "cos(Δ,U)", "знято%", "cos2 e1", "PR(Δ)", "top1%", "‖Δ‖/‖C‖"))
for L in LAYS:
    a = np.array(res[L])
    print("%-3d %10.6f %8.4f %8.4f %8.4f %8.1f %8.4f %8.4f" %
          (L, np.median(a[:,0]), np.median(a[:,1]), np.median(a[:,2]),
           np.median(a[:,3]), np.median(a[:,4]), np.median(a[:,5]), np.median(a[:,6])))
json.dump({str(L): [list(map(float,r)) for r in res[L]] for L in LAYS},
          open("/home/trim/wblogs/sig3.json", "w"))
log("ГОТОВО")
