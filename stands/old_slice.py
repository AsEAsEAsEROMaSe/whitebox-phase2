"""Чи переноситься СТАРИЙ ярус (джерела віку ≥ A) як n²-об'єкт (зріз D21 + D3) без ніг?
Правило зрізу (≥2 збіжні індекси): D21' = (T∘T) D21 Tᵀ + 2 (T∘(T D21ᵀ)) Tᵀ + (T∘T) diag(D3) Tᵀ,
D3'_i = 3·rowsum((T∘T) D21 ∘ T) + (T∘T∘T) D3,  T = W_{l+1} diag(w1_l) (перенос ніг у ланцюгу).
Порівняння: правило на зрізі джерел віку ≥A на шарі l проти точного зрізу тих самих джерел на l+1.
Передбачення ДО прогону: R² ~0.80 для A=1 (відтворює slice-only на істині) і НЕ росте з A (позазрізова частина
не залежить від віку) ⟹ шанс, що старий ярус закривається зрізом, ~15%. Якщо R²(A≥6) > 0.98, двері відкриті."""
import os, sys, numpy as np, torch, time
sys.argv = [sys.argv[0]] + (sys.argv[1:] if len(sys.argv) > 1 else ["0"])
import lean_oracle as lo, lean_k3_aug_gpu as lk
dumps = [int(x) for x in sys.argv[1].split(",")]
AGES = tuple(range(1, 15))
REC = []
_lin, _wick = lk.K3Blocks.linear_, lk.K3Blocks.wick_
def lin(self, W): REC.append(("W", W.detach().clone())); _lin(self, W)
def wick(self, w1): REC.append(("w1", w1.detach().clone())); _wick(self, w1)
lk.K3Blocks.linear_, lk.K3Blocks.wick_ = lin, wick
class Probe:
    def __init__(self): self.S = {}
    def __call__(self, l, K3, D3, D21):
        for A in AGES:
            sub = [s for s in K3.sources if s.get("age", 0) >= A]
            if not sub: continue
            D3o, D21o = K3.dslices(subset=sub)
            D21o = D21o.detach().double().clone(); D21o.fill_diagonal_(0.0)
            self.S[(l, A)] = (D3o.detach().double().clone(), D21o, len(sub))
def r2(x, y):
    a = float((x*y).sum()/(x*x).sum()); res = y - a*x
    return 1 - float((res*res).sum()/(y*y).sum()), a, float(res.norm()/y.norm())
statics = None
for i in dumps:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]
    if statics is None: statics = lk.build_static_coefs(w.shape[1])
    lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; REC.clear(); t0 = time.perf_counter()
    fit = lo.FitC(); lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, g_hook=fit)
    REC.clear(); pr = Probe(); lk.K3_PROBE = pr
    LT = {"rres": 16, "d21": 16}
    pred = lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef),
                   g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc=LT)
    lk.K3_PROBE = None
    Ws = [r[1] for r in REC if r[0] == "W"]; w1s = [r[1] for r in REC if r[0] == "w1"]
    print("dump%d: %d W, %d w1 recorded, t=%.0fs" % (i, len(Ws), len(w1s), time.perf_counter()-t0), flush=True)
    # T_l: probe(l) -> wick(w1_l) -> linear(W_{l+1}) -> probe(l+1). W list index: W_0 used before probe 0.
    n = w.shape[1]; m = ~torch.eye(n, dtype=torch.bool, device=Ws[0].device)
    acc = {}
    for l in range(1, 15):
        T = (Ws[l] * w1s[l-1][None, :]).double()   # exact: align_test.py rel err 0 (wick not called at layer 0)
        TT = T*T
        for A in AGES:
            if (l, A) not in pr.S or (l+1, A+1) not in pr.S: continue
            D3o, D21o, k = pr.S[(l, A)]; D3n, D21n, kn = pr.S[(l+1, A+1)]
            assert k == kn, (l, A, k, kn)
            pred21 = TT @ D21o @ T.T + 2 * (T * (T @ D21o.T)) @ T.T + (TT * D3o[None, :]) @ T.T
            pred3 = 3 * ((TT @ D21o) * T).sum(1) + (TT * T) @ D3o
            r21, a21, e21 = r2(pred21[m], D21n[m]); r3, a3, e3 = r2(pred3, D3n)
            # контроль: лише перший член (чистий діагональний перенос)
            r1, _, _ = r2((TT @ D21o @ T.T)[m], D21n[m])
            acc.setdefault(A, []).append((r21, a21, e21, r3, a3, e3, r1))
            if l in (4, 8, 12, 14) and A in (1, 2, 4, 6, 8, 10):
                print("  l=%2d A=%2d k=%2d | D21: R² %.4f a %.3f rel-res %.3f (1-член R² %.3f) | D3: R² %.4f a %.3f rel-res %.3f" % (l, A, k, r21, a21, e21, r1, r3, a3, e3), flush=True)
    for A in AGES:
        if A in acc:
            v = np.array(acc[A]); print("dump%d A=%2d СЕРЕДНЄ по шарах (%d): D21 R² %.4f a %.3f res %.3f | D3 R² %.4f res %.3f | 1-член %.3f" % (i, A, len(v), v[:,0].mean(), v[:,1].mean(), v[:,2].mean(), v[:,3].mean(), v[:,5].mean(), v[:,6].mean()), flush=True)
