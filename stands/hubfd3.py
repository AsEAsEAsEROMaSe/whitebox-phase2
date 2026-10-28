"""12.10.2026: FD-важливість стовпців хабу + ознаки стовпця при народженні (чи передбачувана між мережами).
DUMP, LAYERS, NCOL. Зберігає Δ_j, ознаки: mu, var, t=mu/sd, норми стовпців X1,Y1,X2,Y2,X3,Y3,M новонародженого."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; I = int(os.environ.get("DUMP", "0")); LAYERS = [int(x) for x in os.environ.get("LAYERS", "3,6,12").split(",")]; NCOL = int(os.environ.get("NCOL", "64"))
OUT = f"/home/trim/wb2data/hubfd3_{I}.npz"; T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] d{I} {m}", flush=True)
d = np.load(lo.DUMP.format(I)); w = d["weights"]; gt = torch.tensor(d["all_layer_means"][-1], dtype=torch.float64, device="cuda"); n = w.shape[1]
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.HUB_MASK = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
feats = {}; live_rec = {}
def probe(l, K3):
    s = K3.sources[-1] if K3.sources else None
    if s is None or s.get("P") is not None or l in feats: return
    f = {}
    for k in ("X1", "Y1", "X2", "Y2", "X3", "Y3", "M"):
        f[k] = s[k].double().norm(dim=0).cpu().numpy() if s.get(k) is not None else np.zeros(n)
    feats[l] = f
def hook(l, D3, D21, live):
    if l not in live_rec:
        live_rec[l] = dict(mu=live["mu"].double().cpu().numpy(), var=live["var"].double().cpu().numpy())
    return D3, D21, None
def run(mask_fn):
    lk.REGEN_STATE.clear(); lk.HUB_MASK = mask_fn; lk.K3_PROBE_POST = probe; lk.D21_HOOK = hook; torch.set_grad_enabled(False)
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
    return float(((p[-1].double() - gt) ** 2).mean())
base = run(None); log(f"base {base:.6e}")
res = dict(base=base, n=n)
rng = np.random.default_rng(100 + I)
for lf in LAYERS:
    cols = rng.choice(n, NCOL, replace=False); dl = []
    for c in cols:
        def mfn(l, c=c):
            if l != lf: return None
            m = torch.ones(n, device="cuda"); m[c] = 0.0; return m
        dl.append(run(mfn) - base)
    dl = np.array(dl); res[f"fd_{lf}"] = dl; res[f"cols_{lf}"] = cols
    for k, v in feats[lf].items(): res[f"{k}_{lf}"] = v
    res[f"mu_{lf}"] = live_rec[lf]["mu"]; res[f"var_{lf}"] = live_rec[lf]["var"]
    np.savez(OUT, **res)
    t = live_rec[lf]["mu"][cols] / np.sqrt(live_rec[lf]["var"][cols])
    log(f"L{lf}: Σ Δ {dl.sum():+.3e} ({dl.sum()/base*100:+.1f}% за {NCOL} стовпців) sign+ {int((dl>0).sum())}/{NCOL} corr(Δ,t) {np.corrcoef(dl,t)[0,1]:+.3f} corr(Δ,|t|) {np.corrcoef(dl,np.abs(t))[0,1]:+.3f} corr(|Δ|,|t|) {np.corrcoef(np.abs(dl),np.abs(t))[0,1]:+.3f} corr(Δ,‖M‖) {np.corrcoef(dl,feats[lf]['M'][cols])[0,1]:+.3f}")
log(f"saved {OUT}"); print("DONE_HUBFD3")
