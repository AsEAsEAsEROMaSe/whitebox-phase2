"""12.10.2026: пломби до FD-важливості стовпців хабу (дамп 0, шар LF): (1) лінійність: Δ(маска 0.5) проти Δ(0)/2 на 16 стовпцях;
(2) адитивність і реальність «корисних» стовпців: викинути разом 8 найкорисніших (Δ<0), 8 найшкідливіших, 8 випадкових; порівняти з Σ Δ;
(3) ад'юнкт проти FD на тих самих стовпцях (corr), якщо є hubadj_adj_0.npz з g_LF."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; I = int(os.environ.get("DUMP", "0")); LF = int(os.environ.get("LF", "3"))
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
d = np.load(lo.DUMP.format(I)); w = d["weights"]; gt = torch.tensor(d["all_layer_means"][-1], dtype=torch.float64, device="cuda"); n = w.shape[1]
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.HUB_MASK = None; lk.K3_PROBE_POST = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
def run(mask_fn):
    lk.REGEN_STATE.clear(); lk.HUB_MASK = mask_fn; torch.set_grad_enabled(False)
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
    return float(((p[-1].double() - gt) ** 2).mean())
def mk(cols, val):
    def f(l):
        if l != LF: return None
        m = torch.ones(n, device="cuda"); m[torch.as_tensor(np.asarray(cols), device="cuda")] = val; return m
    return f
base = run(None); log(f"base {base:.6e}")
fd = np.load(f"/home/trim/wb2data/hubadj_fd_{I}.npz"); dl = fd[f"fd_{LF}"]; cols = fd[f"cols_{LF}"]
log(f"FD L{LF}: {len(cols)} стовпців, Δ: min {dl.min():+.3e} max {dl.max():+.3e}, Σ {dl.sum():+.3e}")
# (1) лінійність на 16 стовпцях (8 найкорисніших, 8 найшкідливіших)
o = np.argsort(dl); sel = np.concatenate([o[:8], o[-8:]])
half = np.array([run(mk([cols[i]], 0.5)) - base for i in sel])
log("лінійність Δ(0.5)/Δ(0) по 16: " + " ".join(f"{h/dl[i]:.2f}" for h, i in zip(half, sel)))
log(f"  медіана {np.median(half/dl[sel]):.3f} (лінійно = 0.50), corr {np.corrcoef(half, dl[sel])[0,1]:.3f}")
# (2) групи
for name, idx in (("8 корисних (Δ<0)", o[:8]), ("8 шкідливих (Δ>0)", o[-8:]), ("8 випадкових", np.random.default_rng(7).choice(len(cols), 8, replace=False))):
    g = run(mk(cols[idx], 0.0)) - base
    log(f"група {name}: разом Δ {g:+.3e} ({g/base*100:+.2f}%), Σ поодинці {dl[idx].sum():+.3e}, відношення {g/dl[idx].sum():.2f}")
# (3) ад'юнкт проти FD
try:
    adj = np.load(f"/home/trim/wb2data/hubadj_adj_{I}.npz")
    if f"g_{LF}" in adj:
        g = adj[f"g_{LF}"]; pred = -g[cols]
        log(f"ад'юнкт L{LF}: corr(FD Δ, -g) = {np.corrcoef(dl, pred)[0,1]:.3f}, corr(|Δ|,|g|) = {np.corrcoef(np.abs(dl), np.abs(pred))[0,1]:.3f}, масштаб Σ|Δ|/Σ|g| = {np.abs(dl).sum()/np.abs(pred).sum():.2f}")
except Exception as e: log(f"ад'юнкт нема: {e}")
print("DONE_HUBFD2")
