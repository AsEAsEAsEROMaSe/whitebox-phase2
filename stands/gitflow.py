"""Тест «математичної роздільності вистачить» (юзер 14.10): ноги джерел κ₃ проєктуються на базис потоку U_l (вхідні напрямки T_{l→15},
пост-кадр шару l) рангу r_q(l) для q ∈ {99, 99.9, 99.99}% енергії; окремо молоді (вік ≤2 при проєкції, тобто доставки на віках 1–3) і старі;
контроль: випадковий ортонормований базис того ж рангу. Передбачення ДО прогону: all9999 ×1.3–3 MSE (65%), young9999 ×1.3–3, old9999 ≈ ×1.0–1.1,
rand9999 значно гірше за all9999 (базис потоку не випадковий)."""
import os, sys, numpy as np, torch
_argv = sys.argv; sys.argv = sys.argv[:1]
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
sys.argv = _argv
dump = int(sys.argv[1]); dt = torch.float32; DEV = lk.DEV
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
z = np.load("/home/trim/wb2data/gitlog_%d.npz" % dump); mu = z["mu"]; var = z["var"]; ml = list(z["mu_layers"])
d = np.load(lo.DUMP.format(dump)); w = d["weights"]; gt = d["final_means"].astype(np.float64); n = w.shape[1]; w64 = w.astype(np.float64)
Phi = lambda t: 0.5*(1+torch.erf(torch.as_tensor(t)/np.sqrt(2))).numpy()
M = np.eye(n); U = {}; R = {}
for l in range(14, -1, -1):
    k = l+1; t = mu[ml.index(k)]/np.sqrt(var[ml.index(k)]); M = w64[k] @ (Phi(t)[:, None]*M)   # M_l = T_{l→15}ᵀ (вхід × вихід)
    u, s, vh = np.linalg.svd(M); e = s**2; c = np.cumsum(e)/e.sum()
    U[l] = torch.as_tensor(u, dtype=dt, device=DEV); R[l] = {q: int(np.searchsorted(c, q))+1 for q in (0.99, 0.999, 0.9999)}
print("r99/r999/r9999 по шарах:", {l: (R[l][0.99], R[l][0.999], R[l][0.9999]) for l in sorted(R)}, flush=True)
g = torch.Generator(device="cpu").manual_seed(0); RAND = {l: torch.linalg.qr(torch.randn(n, n, generator=g))[0].to(dt).to(DEV) for l in range(15)}
MODE = {"rule": None}; LAYER = [0]; _lin = lk.K3Blocks.linear_
def lin(self, W):
    l = LAYER[0]          # джерела зараз у пост-кадрі шару l, перенос у шар l+1
    rule = MODE["rule"]
    if rule is not None:
        for s in self.sources:
            r_, basis = rule(l, s.get("age", 0))
            if r_ is None or r_ >= n: continue
            Q = basis[l][:, :r_]
            for key in self.KEYS:
                if s.get(key) is not None: s[key] = Q @ (Q.T @ s[key])
            s["P"] = (Q @ Q.T) if s["P"] is None else Q @ (Q.T @ s["P"])
    LAYER[0] += 1; _lin(self, W)
lk.K3Blocks.linear_ = lin
def mk(q, who, basis=U):
    def rule(l, age):
        if who == "young" and age > 2: return None, None
        if who == "old" and age <= 2: return None, None
        return R[l][q], basis
    return rule
def prodV(l, age): return (224 if age >= 6 else 384 if age >= 3 else None), U
cfgs = [("base", None), ("all99", mk(0.99, "all")), ("all999", mk(0.999, "all")), ("all9999", mk(0.9999, "all")),
        ("young9999", mk(0.9999, "young")), ("young999", mk(0.999, "young")), ("old9999", mk(0.9999, "old")), ("old99", mk(0.99, "old")),
        ("prodV", prodV), ("rand9999", mk(0.9999, "all", RAND)), ("rand_old99", mk(0.99, "old", RAND))]
if os.environ.get("CFGS"): cfgs = [c for c in cfgs if c[0] in os.environ["CFGS"].split(",")]
st = lk.build_static_coefs(n); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None
MODE["rule"] = None; LAYER[0] = 0; fit = lo.FitC(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
base = None
for name, rule in cfgs:
    MODE["rule"] = rule; LAYER[0] = 0; lk.REGEN_STATE.clear()
    try:
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG)[-1].detach().cpu().double().numpy()
        mse = float(np.mean((p-gt)**2)); base = mse if base is None else base
        print("dump%d %-11s mse %.4e  ×%.3f до base" % (dump, name, mse, mse/base), flush=True)
    except Exception as e: print("dump%d %-11s ERR %s %s" % (dump, name, type(e).__name__, str(e)[:150]), flush=True)
print("DONE_GITFLOW", flush=True)
