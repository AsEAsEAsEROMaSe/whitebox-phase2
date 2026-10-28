"""12.10.2026: важливість стовпців хабу новонароджених джерел = ад'юнкт (dL/d m_j) проти енергії ε_j; FD на ранніх шарах.
Питання: чи спектр важливості крутіший за спектр енергії (рано ε плоска: 53% у топ-½). Критерій: топ-½ частка |g| ≥ 0.75 рано ⟹ є що оптимізувати.
MODE=adj: маски з grad на шарах >= L0 (пам'ять ~1.7 ГБ/шар). MODE=fd: зануляти по одному стовпцю на шарах FD_LAYERS, NCOL випадкових стовпців."""
import os, sys, time, json, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32
MODE = os.environ.get("MODE", "adj"); L0 = int(os.environ.get("L0", "11")); I = int(os.environ.get("DUMP", "0"))
FD_LAYERS = [int(x) for x in os.environ.get("FD_LAYERS", "3,6,12").split(",")]; NCOL = int(os.environ.get("NCOL", "48"))
OUT = f"/home/trim/wb2data/hubadj_{MODE}_{I}.npz"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
d = np.load(lo.DUMP.format(I)); w = d["weights"]; gt = torch.tensor(d["all_layer_means"][-1], dtype=torch.float64, device="cuda")
n = w.shape[1]
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.HUB_MASK = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
energy = {}
def probe(l, K3):
    s = K3.sources[-1] if K3.sources else None
    if s is None or s.get("P") is not None: return
    e = torch.zeros(n, dtype=torch.float64, device="cuda")
    for xk, yk in K3.PAIRS:
        if s.get(xk) is not None and s.get(yk) is not None:
            e += (s[xk].double().norm(dim=0) * s[yk].double().norm(dim=0))
    if s.get("M") is not None: e += s["M"].double().norm(dim=0)
    energy[l] = e.detach().cpu().numpy()
def run(mask_fn, grad_from=None):
    lk.REGEN_STATE.clear(); lk.HUB_MASK = mask_fn; lk.K3_PROBE_POST = probe
    def hook(l, D3, D21, live):
        if grad_from is not None and l == grad_from: torch.set_grad_enabled(True)
        return D3, D21, None
    torch.set_grad_enabled(False); lk.D21_HOOK = hook
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
    loss = ((p[-1].double() - gt) ** 2).mean()
    return loss
base = float(run(None)); log(f"base loss {base:.6e} (dump {I})")
res = dict(base=base, n=n)
if MODE == "adj":
    masks = {}
    def mfn(l):
        if l < L0: return None
        m = torch.ones(n, device="cuda", requires_grad=True); masks[l] = m; return m
    loss = run(mfn, grad_from=L0)
    with torch.enable_grad(): loss.backward()
    log(f"adj loss {float(loss):.6e} layers {sorted(masks)}")
    for l, m in sorted(masks.items()):
        if m.grad is None: continue
        g = m.grad.detach().cpu().numpy().astype(np.float64); e = energy.get(l)
        res[f"g_{l}"] = g; res[f"e_{l}"] = e
        ag = np.abs(g); sg = np.sort(ag)[::-1]; se = np.sort(e)[::-1]
        top_g = sg[:n//2].sum()/sg.sum(); top_e = se[:n//2].sum()/se.sum()
        # частка |g| у топ-½ за ЕНЕРГІЄЮ (що бере нинішнє правило)
        oe = np.argsort(-e); top_g_by_e = ag[oe[:n//2]].sum()/ag.sum()
        corr = np.corrcoef(np.log(ag+1e-30), np.log(e+1e-30))[0,1]
        # перший порядок: втрата від викидання половини за g проти за e
        og = np.argsort(-ag)
        log(f"L{l}: top½|g|={top_g:.3f} top½e={top_e:.3f} |g| у топ½(e)={top_g_by_e:.3f} corr(log|g|,log e)={corr:.3f} "
            f"sum g={g.sum():+.3e} sum|g|={ag.sum():.3e} pred ΔL drop bottom½(g)={-g[og[n//2:]].sum():+.3e} drop bottom½(e)={-g[oe[n//2:]].sum():+.3e} (base {base:.3e})")
else:
    rng = np.random.default_rng(I)
    for lf in FD_LAYERS:
        cols = rng.choice(n, NCOL, replace=False); dl = []
        for c in cols:
            def mfn(l, c=c):
                if l != lf: return None
                m = torch.ones(n, device="cuda"); m[c] = 0.0; return m
            dl.append(float(run(mfn)) - base)
        dl = np.array(dl); e = energy[lf][cols]
        res[f"fd_{lf}"] = dl; res[f"cols_{lf}"] = cols; res[f"e_{lf}"] = energy[lf]
        ad = np.abs(dl); sd = np.sort(ad)[::-1]; se = np.sort(e)[::-1]
        log(f"FD L{lf}: ncol {NCOL} mean Δ {dl.mean():+.3e} (base {base:.3e}) sign+ {int((dl>0).sum())} top½|Δ|={sd[:NCOL//2].sum()/sd.sum():.3f} top½e={se[:NCOL//2].sum()/se.sum():.3f} "
            f"corr(log|Δ|,log e)={np.corrcoef(np.log(ad+1e-30), np.log(e+1e-30))[0,1]:.3f} CV|Δ|={ad.std()/ad.mean():.2f} CVe={e.std()/e.mean():.2f}")
np.savez(OUT, **res); log(f"saved {OUT}"); print("DONE_HUBADJ")
