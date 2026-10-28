"""13.10.2026: нев'язка his κ₄(p_i) проти істини (банк) і кандидати-носії за n²:
T4 = Σ_j W_ji⁴ κ₄(h_j)_true (перенос власного ексцесу post), T4c = те саме з центрованим κ₄(h) мінус гаусова частина (κ₄h − κ₄h_gauss(state)),
T3 = Σ_j W_ji³ … ні (не той порядок). Також регресія нев'язки на his κ₄, var², κ₄(h)-перенос.
Передбачення: R² нев'язки на T4 ≥ 0.5 на глибині (шанс 40%); якщо так, носій = n² на шар."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dumps = [int(x) for x in os.environ.get("DUMPS", "0,1,2,3").split(",")]
def r2(X, y):
    X = np.column_stack(X); c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c; return 1 - (r @ r) / (y @ y), c
for I in dumps:
    d = np.load(lo.DUMP.format(I)); w = d["weights"]; kp = np.load(lo.BANK.format(I)); n = w.shape[1]
    st = lk.build_static_coefs(n); rec = {}
    class H:
        def __call__(self, l, wk4_row, wk4_mat, wk431, live):
            if wk4_row is not None: rec[l] = dict(k4=wk4_row.detach().double().cpu().numpy(), var=live["var"].detach().double().cpu().numpy(), mu=live["mu"].detach().double().cpu().numpy())
            return wk4_row, wk4_mat, wk431
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None
    lk.lean_k3_predict(w, dtype=torch.float32, statics=st, k4_aug=True, g_hook=fit)
    lk.WK_HOOK = H()
    lk.lean_k3_predict(w, dtype=torch.float32, statics=st, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    print(f"=== дамп {I}  шари з κ₄: {sorted(rec)}")
    for l in sorted(rec):
        if l < 1 or l >= 15: continue
        a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l]; E3 = kp["pre_m3"][l]; E4 = kp["pre_m4"][l]
        v = E2 - a*a; k4t = E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v          # істинна κ₄(p_l)
        his = rec[l]["k4"]; res = k4t - his
        # post шару l−1 (h, вхід у p_l = h @ W_l?): перевірити орієнтацію ваг через var: var(p_l) = Σ_j W_ji² var(h_j)
        pm = kp["mean"][l-1].astype(np.float64); p2 = kp["post_m2"][l-1]; p3 = kp["post_m3"][l-1]; p4 = kp["post_m4"][l-1]
        vh = p2 - pm*pm; k3h = p3 - 3*pm*p2 + 2*pm**3; k4h = p4 - 4*pm*p3 + 6*pm*pm*p2 - 3*pm**4 - 3*vh*vh
        W = w[l].astype(np.float64)
        vA = (W**2).T @ vh; vB = (W**2) @ vh
        orient = "h@W" if np.corrcoef(vA, v)[0,1] > np.corrcoef(vB, v)[0,1] else "W@h"
        Wn = W if orient == "h@W" else W.T                    # p_i = Σ_j Wn[j,i] h_j
        T4 = (Wn**4).T @ k4h                                  # перенос власного ексцесу
        T4g = (Wn**4).T @ (3*vh*vh)                           # масштабний контроль (гаусова частина)
        r_h, _ = r2([his], k4t); r_s, _ = r2([his, v*v], k4t)
        r_res_T4, c1 = r2([T4], res); r_res_T4v, c2 = r2([T4, his, v*v], res)
        r_res_T4g, _ = r2([T4g], res)
        rr = np.random.default_rng(l); perm = rr.permutation(n)
        r_res_perm, _ = r2([(Wn**4).T @ k4h[perm]], res)
        print(f"l={l:2d} {orient} corr(var) {max(np.corrcoef(vA,v)[0,1],np.corrcoef(vB,v)[0,1]):.4f} | his~κ₄ R² {r_h:.3f} (+var² {r_s:.3f}) ε {np.linalg.norm(res)/np.linalg.norm(k4t):.3f} | нев'язка~T4 R² {r_res_T4:.3f} coef {c1[0]:+.3f} · ~T4+his+var² {r_res_T4v:.3f} · ~T4гаус {r_res_T4g:.3f} · 🔒переміш {r_res_perm:.3f} | ‖T4‖/‖κ₄‖ {np.linalg.norm(T4)/np.linalg.norm(k4t):.3f}")
