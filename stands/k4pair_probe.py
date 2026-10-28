"""13.10.2026: нев'язка his κ₄(p_i, шар l) ~ перенос ПАРНИХ кумулянтів передактивацій шару l−1 (банк, істинні):
T22 = Σ_{j≠k} W_ji² W_ki² c2_j c2_k κ₂₂(p)_jk · T31 = Σ_{j≠k} W_ji³ W_ki c3_j c1_k κ₃₁(p)_jk + сим · T4 = Σ_j W_ji⁴ κ₄(h_j) (банк post)
c1=Φ(t), c2=φ(t)/σ, c3=−tφ(t)/σ² (перший порядок діаграм). Контроль: T22g = Σ W²W² c2c2 · 2C² (гаусова частина, має бути вже в his),
перемішаний: перестановка нейронів шару l−1 у M (без W). Передбачення: R² нев'язки на T22+T31 ≥ 0.5 на глибині (шанс 35%)."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
from scipy.stats import norm
dumps = [int(x) for x in os.environ.get("DUMPS", "0,1").split(",")]; dev = "cuda"
def r2(X, y):
    X = np.column_stack(X); c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c; return 1 - (r @ r) / (y @ y), c
def diagT(A, M, B):   # diag(Aᵀ M B) = Σ_jk A_ji M_jk B_ki
    return torch.einsum("ji,jk,ki->i", A, M, B)
for I in dumps:
    d = np.load(lo.DUMP.format(I)); w = d["weights"]; kp = np.load(lo.BANK.format(I)); n = w.shape[1]
    st = lk.build_static_coefs(n); rec = {}
    class H:
        def __call__(self, l, wk4_row, wk4_mat, wk431, live):
            if wk4_row is not None: rec[l] = wk4_row.detach().double().cpu().numpy()
            return wk4_row, wk4_mat, wk431
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None
    lk.lean_k3_predict(w, dtype=torch.float32, statics=st, k4_aug=True, g_hook=fit)
    lk.WK_HOOK = H()
    lk.lean_k3_predict(w, dtype=torch.float32, statics=st, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    lk.WK_HOOK = None
    print(f"=== дамп {I}")
    for l in range(2, 15):
        a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l]; E3 = kp["pre_m3"][l]; E4 = kp["pre_m4"][l]
        v = E2 - a*a; k4t = E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v; res = k4t - rec[l]
        k22, k31, k4p, mu21 = lo.true_pair(kp, l-1)
        ap = kp["pre_mean"][l-1].astype(np.float64); vp = kp["pre_m2"][l-1] - ap*ap; sp = np.sqrt(vp); t = ap/sp
        C = kp["pre_M11"][l-1].astype(np.float64) - np.outer(ap, ap)
        c1 = norm.cdf(t); c2 = norm.pdf(t)/sp; c3 = -t*norm.pdf(t)/sp**2
        pm = kp["mean"][l-1].astype(np.float64); p2 = kp["post_m2"][l-1]; p3 = kp["post_m3"][l-1]; p4 = kp["post_m4"][l-1]
        vh = p2 - pm*pm; k4h = p4 - 4*pm*p3 + 6*pm*pm*p2 - 3*pm**4 - 3*vh*vh
        W = torch.tensor(w[l], dtype=torch.float64, device=dev)       # p_i = Σ_j W[j,i] h_j (орієнтація h@W перевірена в k4diag_probe)
        W2 = W*W; W3 = W2*W
        T = lambda M: diagT(W2, torch.tensor(M, device=dev), W2).cpu().numpy()
        off = ~np.eye(n, dtype=bool)
        M22 = np.where(off, np.outer(c2, c2)*k22, 0.0); M22g = np.where(off, np.outer(c2, c2)*2*C*C, 0.0)
        M31 = np.where(off, np.outer(c3, c1)*k31, 0.0)
        T22 = T(M22); T22g = T(M22g)
        T31 = (diagT(W3, torch.tensor(M31, device=dev), W) + diagT(W, torch.tensor(M31.T, device=dev), W3)).cpu().numpy()
        T4 = (W2*W2).T.cpu().numpy() @ k4h
        perm = np.random.default_rng(l).permutation(n)
        T22p = T(M22[perm][:, perm]); T31p = (diagT(W3, torch.tensor(M31[perm][:, perm], device=dev), W) + diagT(W, torch.tensor(M31[perm][:, perm].T, device=dev), W3)).cpu().numpy()
        s = lambda X: r2(X, res)
        rA, cA = s([T22]); rB, cB = s([T31]); rAB, cAB = s([T22, T31]); rAll, _ = s([T22, T31, T4]); rG, _ = s([T22g]); rP, _ = s([T22p, T31p]); rW, _ = s([T22, T31, rec[l], v*v])
        print(f"l={l:2d} ε {np.linalg.norm(res)/np.linalg.norm(k4t):.3f} | R² T22 {rA:.3f} (c {cA[0]:+.2f}) · T31 {rB:.3f} (c {cB[0]:+.2f}) · T22+T31 {rAB:.3f} (c {cAB[0]:+.2f},{cAB[1]:+.2f}) · +T4 {rAll:.3f} · +his+var² {rW:.3f} | 🔒 гаус T22g {rG:.3f} · переміш {rP:.3f} | ‖T22‖,‖T31‖/‖κ₄‖ {np.linalg.norm(T22)/np.linalg.norm(k4t):.3f},{np.linalg.norm(T31)/np.linalg.norm(k4t):.3f}", flush=True)
