"""16.10.2026, юзер: «усі методи, які знаємо, щоб побачити межу точності взагалі, і порівняти з МК за ціною».
Члени (lean GPU fp32): reg (≈продакшн v34), dense (несене ядро κ4), wk22 (dense без факторизації wk22), g96 (wk22 + G = λC + ранг 96
залишку, регуляризація), simple, nok4, nofb, k211, regen, B2, no31, trunc (варіанти 23.09). Аргумент: номер дампу.
Флопи: torch FlopCounterMode (mm/bmm/einsum; SVD/eigh/qr не враховано) лише на дампі 0 (форми не залежать від мережі).
Вихід: /home/trim/wb2data/frontier_{i}.npz (передбачення членів, gt, avg_var), /home/trim/wblogs/frontier_flops.json.
Аналіз (LOO-суміш, МК за ціною): dig/frontier_an.py.
Передбачення ДО прогону: найкращий детермінований (LOO-суміш усіх) ≈1.55–1.65e-8 (×1.33–1.42 від reg 2.20e-8); МК за ціною суміші
≈5e-7…1e-6, тобто детермінований у ≥30× точніший за рівну ціну; МК наздожене суміш лише за ≥50 бюджетів грейдера."""
import os, sys, time, json, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
from torch.utils.flop_counter import FlopCounterMode
i = int(sys.argv[1]); dt = torch.float32
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(lo.BANK.format(i))
avg_var = float(np.mean(kp["post_m2"][-1] - kp["mean"][-1].astype(np.float64)**2))
st = lk.build_static_coefs(w.shape[1]); out = {"gt": gt, "avg_var": avg_var, "bank_mean": kp["mean"][-1]}
FL = {}
class LowG:   # з glow.py
    def __init__(self, r): self.r = r
    def __call__(self, l, G, live):
        C = lo.zd(live["C"]); Go = lo.zd(G); lam = float((C*Go).sum()/(C*C).sum()); Rm = Go - lam*C
        out_ = torch.diag(torch.diagonal(G)) + lam*C
        ev, U = torch.linalg.eigh(0.5*(Rm+Rm.T)); idx = torch.argsort(ev.abs(), descending=True)[:self.r]
        return out_ + lo.zd((U[:, idx]*ev[idx]) @ U[:, idx].T)
def run(tag, pre=None, **kw):
    def go():
        lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None; lk.C_HOOK = None
        extra = pre() if pre is not None else {}
        return lk.lean_k3_predict(w, dtype=dt, statics=st, **extra, **kw)[-1].detach().cpu().double().numpy()
    t0 = time.perf_counter()
    if i == 0:
        with FlopCounterMode(display=False) as fc: p = go()
        FL[tag] = int(fc.get_total_flops())
    else:
        p = go()
    out[tag] = p
    print("dump%d %-6s mse %.4e t %.0fs%s" % (i, tag, np.mean((p-gt)**2), time.perf_counter()-t0,
          (" flops %.3e" % FL[tag]) if tag in FL else ""), flush=True)
def regen_pre():   # пас 1 (фіт λ) входить у ціну reg/regen
    fit = lo.FitC(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    return dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
run("reg", regen_pre, k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
run("dense", k4_aug=True)
run("wk22", k4_aug=True, no_wk22_fac=True)
run("g96", lambda: dict(g_hook=LowG(96)), k4_aug=True, no_wk22_fac=True)
run("simple")
run("nok4", no_k4=True)
run("nofb", k4_aug=True, no_k3_feedback=True)
run("k211", k4_aug=True, aug_no_k211=True)
run("regen", regen_pre, k4_aug=True)
run("B2", k4_aug=True, shadow_B2=True)
run("no31", k4_aug=True, aug_no_31=True)
run("trunc", k4_aug=True, leg_trunc={"rres": 16, "d21": 16})
np.savez("/home/trim/wb2data/frontier_%d.npz" % i, **out)
if i == 0: json.dump(FL, open("/home/trim/wblogs/frontier_flops.json", "w"), indent=1)
print("DONE_FRONTIER", i, flush=True)
