"""Носій κ₃₁ «народження з κ₃⊗κ₃ + перенос»: T_ab = f(a)·κ₂₁(h_{l-1})_ab, f = κ₃(h_a)/var(h_a) (пост-ReLU, з ланцюга),
κ₃₁(p_l) = (W∘W∘W) T Wᵀ + 3 (W∘W∘(W Tᵀ)) Wᵀ.  Пломба: якщо форма вірна, R² високий і масштаб сталий по шарах/мережах.
Режими: rec — лише R²/масштаб; inj — wk431 := s_l·модель (s_l з таблиці SCALE або з фіту на істині, ORACLE=1)."""
import os, sys, time, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_k3_aug_gpu as lk
from lean_oracle import true_pair, r2, FitC, Replay, approx_pre, zd   # noqa (lean_oracle виконає свій main? ні: захист нижче)
DEV = lk.DEV; dtype = torch.float32
DUMP = "/home/trim/wb2data/dumps_p2/mlp_{:04d}.npz"; BANK = "/home/trim/wb2data/kp_p2/mlp_{:05d}.npz"
dumps = [int(x) for x in sys.argv[1].split(",")]; mode = sys.argv[2]
ORACLE = os.environ.get("ORACLE", "0") == "1"; SCALE = float(os.environ.get("SCALE", "1.0")); MIX = float(os.environ.get("MIX", "0"))
POST = {}
class GH:   # g_hook: replay λ і зберегти пост-ReLU κ₃ зрізки
    def __init__(self, coef): self.coef = coef
    def __call__(self, l, G, live):
        lk.REGEN_STATE["lam"] = float(self.coef[l]); lk.REGEN_STATE["g"] = torch.diagonal(G).clone()
        POST[l] = (live["K21"].detach().clone(), live["K3v"].detach().clone(), live["var"].detach().clone(), live["C"].detach().clone())
        return torch.diag(torch.diagonal(G)) + self.coef[l]*zd(live["C"])
class WK:
    def __init__(self, kp, Ws): self.kp = kp; self.Ws = Ws; self.lines = []; self.scales = []
    def __call__(self, l, wk4_row, wk4_mat, wk431, live):
        if wk431 is None or l < 1 or (l-1) not in POST: return wk4_row, wk4_mat, wk431
        K21, K3v, var, C = POST[l-1]; W = self.Ws[l]
        f = K3v/var; T = f[:, None]*K21; T = T.clone(); T.fill_diagonal_(0.0)
        W2 = W*W; W3 = W2*W
        model = W3 @ T @ W.T + 3*(W2*(W @ T.T)) @ W.T
        model.fill_diagonal_(0.0)
        k22, k31, k4, _ = true_pair(self.kp, l); offm = ~np.eye(1024, dtype=bool)
        mn = model.detach().cpu().double().numpy(); ra, sa = r2(mn[offm], k31[offm])
        # діагональна частина хаба: T_aaaa = κ₄(h)? ні — лише перевірка внеску власного κ₃²: f·K3v
        w3 = wk431.detach().cpu().double().numpy(); rb, sb = r2(w3[offm], k31[offm])
        # комбінація модель + memoryless D21·D3/var
        D21 = live["D21"].detach(); D3 = live["D3"].detach(); v = live["var"].detach()
        mm = (D21*(D3/v)[:, None]); mm = mm.clone(); mm.fill_diagonal_(0.0); mmn = mm.detach().cpu().double().numpy()
        X = np.stack([mn[offm], mmn[offm], w3[offm]], 1); y = k31[offm]; sol = np.linalg.lstsq(X, y, rcond=None)[0]; rc = 1 - ((y - X@sol)**2).sum()/(y*y).sum()
        self.lines.append("l=%2d модель R² %.3f масштаб %.3f | λC (його) R² %.3f | модель+D21·D3/v+λC R² %.3f coef %s" % (l, ra, sa, rb, rc, ["%.3g" % c for c in sol]))
        self.scales.append(sa)
        if mode == "inj":
            s_ = sa if ORACLE else SCALE
            new = (s_*model).to(wk431.dtype)
            if MIX > 0: new = new + MIX*mm.to(wk431.dtype)
            wk431 = new
        return wk4_row, wk4_mat, wk431
statics = None; mses = []; allsc = []
for i in dumps:
    d = np.load(DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(BANK.format(i))
    if statics is None: statics = lk.build_static_coefs(w.shape[1])
    Ws = [torch.from_numpy(x.astype(np.float64).T).contiguous().to(dtype).to(DEV) for x in w]
    lk.REGEN_STATE.clear(); lk.WK_HOOK = None; POST.clear()
    fit = FitC(); lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, g_hook=fit)
    hook = WK(kp, Ws); lk.WK_HOOK = hook; POST.clear()
    pred = lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=GH(fit.coef),
                              g_feed_hook=approx_pre, g_use_hook=approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True,
                              leg_trunc={"rres": 16, "d21": 16}).detach().cpu().double().numpy()
    mse = float(np.mean((pred[-1] - gt)**2)); mses.append(mse); allsc.append(hook.scales)
    print("[tb/%s] dump%d mse_final=%.4e" % (mode, i, mse), flush=True)
    for ln in hook.lines: print("   ", ln, flush=True)
print("[tb/%s] MEAN mse_final=%.4e over %s" % (mode, float(np.mean(mses)), dumps), flush=True)
sc = np.array(allsc); print("масштаб по шарах (сер ± sd між мережами):", " ".join("%.2f±%.2f" % (m, s) for m, s in zip(sc.mean(0), sc.std(0))))
