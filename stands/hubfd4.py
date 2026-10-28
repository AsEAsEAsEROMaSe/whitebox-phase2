"""12.10.2026: стеля оракула знаків: викинути РАЗОМ усі шкідливі (Δ<0) стовпці з вибірки 64 на шарі, і на всіх трьох шарах разом; контроль: стільки ж випадкових."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; I = int(os.environ.get("DUMP", "0")); T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] d{I} {m}", flush=True)
d = np.load(lo.DUMP.format(I)); w = d["weights"]; gt = torch.tensor(d["all_layer_means"][-1], dtype=torch.float64, device="cuda"); n = w.shape[1]
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.HUB_MASK = None; lk.K3_PROBE_POST = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
def run(sets):
    def f(l):
        if l not in sets: return None
        m = torch.ones(n, device="cuda"); m[torch.as_tensor(np.asarray(sets[l]), device="cuda")] = 0.0; return m
    lk.REGEN_STATE.clear(); lk.HUB_MASK = f; torch.set_grad_enabled(False)
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
    return float(((p[-1].double() - gt) ** 2).mean())
base = run({}); log(f"base {base:.6e}")
fd = np.load(f"/home/trim/wb2data/hubfd3_{I}.npz"); rng = np.random.default_rng(5)
allneg = {}; allrnd = {}
for lf in (3, 6, 12):
    cols = fd[f"cols_{lf}"]; dl = fd[f"fd_{lf}"]; neg = cols[dl < 0]; allneg[lf] = neg
    rnd = rng.choice(cols, len(neg), replace=False); allrnd[lf] = rnd
    a = run({lf: neg}) - base; b = run({lf: rnd}) - base
    log(f"L{lf}: шкідливих {len(neg)}/64, разом Δ {a/base*100:+.2f}% (Σ поодинці {dl[dl<0].sum()/base*100:+.2f}%); випадкові {len(neg)}: {b/base*100:+.2f}%")
a = run(allneg) - base; b = run(allrnd) - base
log(f"усі три шари: шкідливі разом {a/base*100:+.2f}%, випадкові разом {b/base*100:+.2f}%")
print("DONE_HUBFD4")
