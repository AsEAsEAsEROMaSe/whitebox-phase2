"""Шлях reg → D у lean (8 дампів): які проміжні конфіги реалізовні і скільки кожен дає. d21 = ранг ніг D21 (як R_FB у v29).
Передбачення ДО прогону: reg64 ≈ 2.14e-8; reg64−B2 ≈ 2.2 (тінь B2 у reg корисна); dreg64 (dense+регенерація) ≈ 2.05; D64 1.937 (зміряно)."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32; R = {}
class LowG:
    def __init__(self, r): self.r = r
    def __call__(self, l, G, live):
        C = lo.zd(live["C"]); Go = lo.zd(G); lam = float((C*Go).sum()/(C*C).sum()); Rm = Go - lam*C
        out = torch.diag(torch.diagonal(G)) + lam*C
        if self.r > 0:
            ev, U = torch.linalg.eigh(0.5*(Rm+Rm.T)); idx = torch.argsort(ev.abs(), descending=True)[:self.r]
            out = out + lo.zd((U[:, idx]*ev[idx]) @ U[:, idx].T)
        return out
def T(r): return {"rres": 16, "d21": r}
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True)
CF = {  # (прапорці, режим G): "regen" = Replay+approx_pre, ціле число = LowG(r)
    "reg16": (dict(REG, leg_trunc=T(16)), "regen"), "reg64": (dict(REG, leg_trunc=T(64)), "regen"),
    "reg64-B2": (dict(REG, shadow_B2=False, leg_trunc=T(64)), "regen"), "reg64-B2+31": (dict(REG, shadow_B2=False, aug_no_31=False, leg_trunc=T(64)), "regen"),
    "dreg64": (dict(k4_aug=True, no_wk22_fac=True, leg_trunc=T(64)), "regen"), "dreg16": (dict(k4_aug=True, no_wk22_fac=True, leg_trunc=T(16)), "regen"),
    "D64": (dict(k4_aug=True, no_wk22_fac=True, leg_trunc=T(64)), 96), "D64-31": (dict(k4_aug=True, no_wk22_fac=True, aug_no_31=True, leg_trunc=T(64)), 96)}
DK = dict(k4_aug=True, no_wk22_fac=True, leg_trunc=T(64))
for ag, ar in (("v29", [(4, 384), (7, 224)]), ("r256", [(4, 256), (7, 128)]), ("a3", [(3, 384), (6, 224)]), ("a2r512", [(2, 512), (5, 256)])):
    CF["D64@" + ag] = (dict(DK, age_rank=ar), 96); CF["reg64@" + ag] = (dict(REG, leg_trunc=T(64), age_rank=ar), "regen")
CF["D64g48"] = (DK, 48); CF["D64g32"] = (DK, 32)
only = os.environ.get("CFS"); CF = {k: v for k, v in CF.items() if not only or k in only.split(",")}
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for nm, (kw, gm) in CF.items():
        lk.REGEN_STATE.clear()
        hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre) if gm == "regen" else dict(g_hook=LowG(gm))
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **kw)
        R.setdefault(nm, []).append(float(np.mean((p[-1].detach().cpu().double().numpy() - gt)**2)))
    print("dump", i, " | ".join("%s %.4e" % (k, R[k][-1]) for k in CF), flush=True)
for k in CF: print("MEAN %-12s %.4e" % (k, np.mean(R[k])), flush=True)
