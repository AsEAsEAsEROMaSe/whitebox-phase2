"""D64 (dense+wk22, ноги D21 64, rres 16): який шматок G_off несе цінність низькорангового залишку.
A = G повна → λC + r96(залишку) [= D64 1.937e-8]; B = G без контракції по джерелах (comb) → те саме;
C = G без народження-частини (birth); R = регенерація (λC, діагональ щільна, r=0).
Передбачення ДО прогону: B гірший за A (comb несе залишок), але не до R; шанс B≈A: 20%."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32; R = {}; lk.GPARTS = {}
class H:
    def __init__(self, r, drop): self.r = r; self.drop = drop
    def __call__(self, l, G, live):
        Gm = G
        for k in self.drop:
            v = lk.GPARTS.get(k)
            if v is not None: Gm = Gm - lo.zd(v)
        C = lo.zd(live["C"]); Go = lo.zd(Gm); lam = float((C*Go).sum()/(C*C).sum()); Rm = Go - lam*C
        out = torch.diag(torch.diagonal(G)) + lam*C
        if self.r > 0:
            ev, U = torch.linalg.eigh(0.5*(Rm+Rm.T)); idx = torch.argsort(ev.abs(), descending=True)[:self.r]
            out = out + lo.zd((U[:, idx]*ev[idx]) @ U[:, idx].T)
        return out
V = {"A": (96, ()), "B": (96, ("comb",)), "C": (96, ("birth",)), "BC": (96, ("comb", "birth")), "R0": (0, ())}
only = os.environ.get("VS"); V = {k: v for k, v in V.items() if not only or k in only.split(",")}
KW = dict(k4_aug=True, no_wk22_fac=True, leg_trunc={"rres": 16, "d21": 64})
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None
    for nm, (r, dr) in V.items():
        lk.REGEN_STATE.clear()
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=H(r, dr), **KW)
        R.setdefault(nm, []).append(float(np.mean((p[-1].detach().cpu().double().numpy() - gt)**2)))
    print("dump", i, " | ".join("%s %.4e" % (k, R[k][-1]) for k in V), flush=True)
for k in V: print("MEAN %-3s %.4e" % (k, np.mean(R[k])), flush=True)
