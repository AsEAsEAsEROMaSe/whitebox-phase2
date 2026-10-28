"""Щільна G у dense+wk22: G_off = λ_l·C_off + top-r(eig) залишку. Чи вистачає малого r, щоб узяти ×1.09 (regen→dense).
Передбачення ДО прогону: r=0 ≈ 2.08e-8 (= d+regen), r=16 бере ≥60% розриву до 1.86e-8; шанс 35% (504aldo: R² λC 0.985 на глибині ⟹ залишок малий, але може бути ізотропним)."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32; R = {}
RS = [int(x) for x in os.environ.get("RS", "0,8,16,32,64,-1").split(",")]
class LowG:
    def __init__(self, r): self.r = r; self.fr = []
    def __call__(self, l, G, live):
        C = lo.zd(live["C"]); Go = lo.zd(G); lam = float((C*Go).sum()/(C*C).sum()); Rm = Go - lam*C
        if self.r < 0: return G
        out = torch.diag(torch.diagonal(G)) + lam*C
        if self.r > 0:
            ev, U = torch.linalg.eigh(0.5*(Rm+Rm.T)); idx = torch.argsort(ev.abs(), descending=True)[:self.r]
            Rr = (U[:, idx]*ev[idx]) @ U[:, idx].T; out = out + lo.zd(Rr)
            self.fr.append(float((Rr*Rr).sum()/(Rm*Rm).sum()))
        return out
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None
    for r in RS:
        lk.REGEN_STATE.clear(); h = LowG(r)
        if os.environ.get("CHAIN", "dw") == "dw":
            ex = eval(os.environ.get("EXTRA", "{}"))
            p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, no_wk22_fac=True, g_hook=h, **ex)
        else:   # reg-прапорці, але k211 рахується, G з гака скрізь (без approx_pre на feed/use)
            kw = dict(k4_aug=True, aug_no_k211=False, shadow_B2=True, no_wk22_fac=True, aug_no_31=os.environ.get("NO31", "1") == "1", leg_trunc={"rres": 16, "d21": 16})
            p = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=h, **kw)
        R.setdefault(r, []).append(float(np.mean((p[-1].detach().cpu().double().numpy() - gt)**2)))
        if h.fr and i == 0: print("  r=%d частка енергії залишку по шарах:" % r, " ".join("%.2f" % x for x in h.fr), flush=True)
    print("dump", i, " | ".join("r%d %.4e" % (r, R[r][-1]) for r in RS), flush=True)
for r in RS: print("MEAN r=%d %.4e" % (r, np.mean(R[r])), flush=True)
