"""13.10.2026: пломби симетрій ланцюга на дампі 0: (1) z→−z ≡ W₀→−W₀ (точна симетрія задачі); (2) перестановка нейронів шару 5
(стовпці W₅ і рядки W₆) ⟹ вихід незмінний; (3) контроль: −W₁ (НЕ симетрія, бо ReLU) мусить змінити вихід."""
import numpy as np, torch, sys
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
d = np.load(lo.DUMP.format(0)); w = d["weights"].copy(); gt = d["final_means"].astype(np.float64); n = w.shape[1]
st = lk.build_static_coefs(n); REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
def run(W):
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; torch.manual_seed(0)
    lk.lean_k3_predict(W, dtype=torch.float32, statics=st, k4_aug=True, g_hook=fit); torch.manual_seed(0)
    return lk.lean_k3_predict(W, dtype=torch.float32, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG).detach().double().cpu().numpy()
p0 = run(w); mse = lambda p: float(((p[-1]-gt)**2).mean()); rel = lambda a, b: float(np.abs(a-b).max()/np.abs(b).max())
w1 = w.copy(); w1[0] = -w1[0]; p1 = run(w1)
w2 = w.copy(); perm = np.random.default_rng(1).permutation(n); w2[5] = w2[5][:, perm]; w2[6] = w2[6][perm, :]; p2 = run(w2)
w3 = w.copy(); w3[1] = -w3[1]; p3 = run(w3)
w4 = w.copy(); w4[0] = w4[0][:, perm]; w4[1] = w4[1][perm, :]; p4 = run(w4)   # перестановка на шарі 0..1
print(f"база MSE {mse(p0):.6e}")
print(f"(1) W₀→−W₀:       MSE {mse(p1):.6e}  max|Δвихід|/max|вихід| {rel(p1[-1],p0[-1]):.2e}  (усі шари {rel(p1,p0):.2e})")
print(f"(2) перестановка ш.5: MSE {mse(p2):.6e}  max|Δ| {rel(p2[-1],p0[-1]):.2e}  шар 5 після зворотної перестановки {rel(p2[5][np.argsort(perm)], p0[5]):.2e}")
print(f"(2b) перестановка ш.0: MSE {mse(p4):.6e}  max|Δ| {rel(p4[-1],p0[-1]):.2e}")
print(f"(3) контроль W₁→−W₁ (не симетрія): MSE {mse(p3):.6e}  max|Δ| {rel(p3[-1],p0[-1]):.2e}")
