"""13.10.2026: сліди стовпців хабу. Ті самі 64 стовпці × 3 шари, що в hubfd3_{I}, але зберігаємо pred без стовпця (64×n),
pred бази і істину ⟹ офлайн: f_j = pred_base − pred_j; частка ‖e‖² у спані {f_j}; оракул МНК-перезваження; Грам.
Передбачення до прогону: частка в спані ≥ 50% (шанс 50%), контроль 64 випадкових напрямків ≈ 64/1024 = 6%.
Пломба: Δ_j з hubfd3 = (‖f_j‖² − 2⟨e,f_j⟩)/n ТОЧНО."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; I = int(os.environ.get("DUMP", "0")); T0 = time.time()
OUT = f"/home/trim/wb2data/hubfd5_{I}.npz"
def log(m): print(f"[{time.time()-T0:7.1f}s] d{I} {m}", flush=True)
d = np.load(lo.DUMP.format(I)); w = d["weights"]; gt = torch.tensor(d["all_layer_means"][-1], dtype=torch.float64, device="cuda"); n = w.shape[1]
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.HUB_MASK = None; lk.K3_PROBE_POST = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
def run(mask_fn):
    lk.REGEN_STATE.clear(); lk.HUB_MASK = mask_fn; torch.set_grad_enabled(False)
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
    return p[-1].double().cpu().numpy()
fd = np.load(f"/home/trim/wb2data/hubfd3_{I}.npz")
pb = run(None); gtn = gt.cpu().numpy(); base = float(((pb - gtn) ** 2).mean()); log(f"base {base:.6e} (hubfd3 base {float(fd['base']):.6e})")
res = dict(base=base, n=n, pred_base=pb, gt=gtn)
for lf in (3, 6, 12):
    cols = fd[f"cols_{lf}"]; P = np.zeros((len(cols), n))
    for i, c in enumerate(cols):
        def mfn(l, c=c):
            if l != lf: return None
            m = torch.ones(n, device="cuda"); m[c] = 0.0; return m
        P[i] = run(mfn)
    res[f"cols_{lf}"] = cols; res[f"pred_{lf}"] = P
    dl = ((P - gtn) ** 2).mean(1) - base; old = fd[f"fd_{lf}"]
    e = pb - gtn; F = pb[None, :] - P; ident = ((F ** 2).sum(1) - 2 * F @ e) / n
    np.savez(OUT, **res)
    log(f"L{lf}: пломба Δ проти hubfd3 max|diff|/|Δ| {np.max(np.abs(dl-old))/np.mean(np.abs(old)):.2e}; тотожність max|diff| {np.max(np.abs(dl-ident))/np.mean(np.abs(dl)):.2e}; sign+ {(dl>0).sum()}/{len(cols)}")
log(f"saved {OUT}"); print("DONE_HUBFD5")
