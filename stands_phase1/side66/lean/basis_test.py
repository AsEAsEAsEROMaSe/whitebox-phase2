"""Базис для старих джерел (age>=4): його (eigh Грама ніг) проти власного базису коваріації стану C (пост-ReLU) проти змішаного."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_k3_aug_gpu as lk
from lean_oracle import FitC, Replay, approx_pre, zd
DEV = lk.DEV; dtype = torch.float32; DUMP = "/home/trim/wb2data/dumps_p2/mlp_{:04d}.npz"
dumps = [int(x) for x in sys.argv[1].split(",")]; BASIS = sys.argv[2]; RANK = int(sys.argv[3])
LAST = {}
class GH:
    def __init__(self, coef): self.coef = coef
    def __call__(self, l, G, live):
        lk.REGEN_STATE["lam"] = float(self.coef[l]); lk.REGEN_STATE["g"] = torch.diagonal(G).clone(); LAST["C"] = live["C"].detach()
        return torch.diag(torch.diagonal(G)) + self.coef[l]*zd(live["C"])
def q_cov(G, rank):
    _, V = torch.linalg.eigh(LAST["C"].double()); return V[:, -rank:].to(G.dtype).contiguous()
def q_mix(G, rank):   # Грам ніг, зважений коваріацією: C^{1/2} G C^{1/2}? простіше: G + α·C·tr(G)/tr(C)
    C = LAST["C"].to(G.dtype); Gm = G + C*(torch.trace(G)/torch.trace(C)); _, V = torch.linalg.eigh(Gm.double()); return V[:, -rank:].to(G.dtype).contiguous()
def q_rand(G, rank):
    g = torch.Generator(device="cpu").manual_seed(5); Y = torch.randn(G.shape[0], rank, generator=g, dtype=torch.float64).to(G.device); Q, _ = torch.linalg.qr(Y); return Q.to(G.dtype)
lk.CONFINE_Q_FN = {"his": None, "cov": q_cov, "mix": q_mix, "rand": q_rand}[BASIS]
statics = None; mses = []
for i in dumps:
    d = np.load(DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
    if statics is None: statics = lk.build_static_coefs(w.shape[1])
    lk.REGEN_STATE.clear(); fit = FitC(); lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, g_hook=fit)
    pred = lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=GH(fit.coef),
                              g_feed_hook=approx_pre, g_use_hook=approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True,
                              leg_trunc={"rres": 16, "d21": 16}, age_rank=[(4, RANK)] if RANK < 1024 else None).detach().cpu().double().numpy()
    mse = float(np.mean((pred[-1]-gt)**2)); mses.append(mse); print("[%s r=%d] dump%d %.4e" % (BASIS, RANK, i, mse), flush=True)
print("[%s r=%d] MEAN %.4e" % (BASIS, RANK, float(np.mean(mses))), flush=True)
