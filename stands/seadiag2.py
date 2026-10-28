"""01.10: пломба вписування парної функції: pair_delta_slab у ланцюзі (блоки шару 12) проти офлайн Δ з blocksS + pairfit на S³;
і напрямок: cos(Δ_pairs на S³, істинна поправка 0.5Σ w w w2 κ₄_true) та cos із залишком моря κ₃ (tru − cp)."""
import os, sys, numpy as np, torch
_a = sys.argv; sys.argv = sys.argv[:1]
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import lean_oracle as lo; lk = lo.lk; import sea4, d4terms
sys.argv = _a; torch.set_num_threads(1); dt = torch.float32; DEV = lk.DEV; m = 32
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
i, ls = int(sys.argv[1]), int(sys.argv[2])
d = np.load(lo.DUMP.format(i)); w = d["weights"]; n = w.shape[1]
rng = np.random.default_rng(11 + i); S = np.sort(rng.choice(n, m, replace=False)); St = torch.tensor(S, device=DEV)
D = np.load("/home/trim/wb2data/brc_%d_%d.npz" % (i, ls)); Nh = float(D["Nh"])
g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m); C2 = R2 - torch.outer(e, e)
C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K3_PROBE = None; lk.K_POST_HOOK = None; lk.K3_PROBE_POST = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
pfn = sea4.PairFn("/home/trim/wb2data/pairfit_pairs_%d.npz" % i, DEV)
S4 = sea4.Sea4(n, DEV, window=6, pairfn=pfn, use_S=False); lk.D4 = S4
class Stop(Exception): pass
RES = {}
def cp_on_S(K3):
    T = torch.zeros((m, m, m), dtype=torch.float64, device=DEV)
    for s in K3.sources:
        P = s["P"]; Ps = P[St].double()
        for xk, yk in (("X1", "Y1"), ("X3", "Y3")):
            if s.get(xk) is not None: T += torch.einsum("aJ,bJ,cJ->abc", s[xk][St].double(), Ps, s[yk][St].double())
        if s.get("M") is not None: T += torch.einsum("aJ,bJ,cJ->abc", s["M"][St].double(), Ps, Ps)
    return sum(T.permute(*p) for p in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]) / 6.0
def probe(l, K3, D3, D21):
    S4.stash(l, K3, D21)
    if l == ls: RES.update(cp=cp_on_S(K3), D3=D3.clone(), D21=D21.clone(), dtrans=(S4.D[St][:, St][:, :, St].double() / S4.SC) if S4.D is not None else None)
def d21hook(l, D3, D21, live):
    wk = lk.WickCache(live["mu"], live["var"]); S4.set_w(l, {k: wk(k, 1) for k in (1, 2, 3)}, lk.zero_diag_(live["C"].clone()), mu=live["mu"].clone())
    if l == ls: RES.update(mu=live["mu"].clone(), var=live["var"].clone(), C=lk.zero_diag_(live["C"].clone())); raise Stop()
    return D3, D21, None
lk.K3_PROBE = probe; lk.D21_HOOK = d21hook; lk.REGEN_STATE.clear()
try: lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
except Stop: print('STOP at ls', flush=True)
print('AFTER_PREDICT keys', sorted(RES.keys()), flush=True)
# Δ_pairs у ланцюзі на шарі ls для рядків S (слаби по рядках S окремо)
blocks = dict(var=RES["var"], C=RES["C"], D3=RES["D3"], D21=RES["D21"]); wick = lk.WickCache(RES["mu"], RES["var"])
Dp = torch.zeros((m, m, m), dtype=torch.float64, device=DEV)
for j, a_ in enumerate(S.tolist()):
    sl = slice(a_, a_ + 1); dn = sea4.pair_delta_slab(pfn, sl, n, blocks, RES["mu"], wick, DEV, sub=1)
    Dp[j] = dn[0][St][:, St].double()
# офлайн: ті самі ознаки з blocksS + pairfit (numpy), три положення подвоєної осі
B = np.load("/home/trim/wb2data/blocksS_%d_%d.npz" % (i, ls)); Z = np.load("/home/trim/wb2data/pairfit_pairs_%d.npz" % i)
s = np.sqrt(B["var"]); C = B["C"] / np.outer(s, s); D21 = B["D21"] / (s[:, None]**2 * s[None, :]); D3 = B["D3"] / s**3; mu = B["mu"] / s
w1 = wick(1, 1)[St].double().cpu().numpy(); w2 = wick(2, 1)[St].double().cpu().numpy()
def f(h, x, y):
    F = np.array([mu[h], mu[x] + mu[y], mu[x] * mu[y], C[h, x] + C[h, y], C[h, x] * C[h, y], C[x, y], D21[h, x] + D21[h, y], D21[x, h] + D21[y, h], D21[x, y] + D21[y, x], D21[h, x] * D21[h, y],
                  0, 0, 0, 0, 0, D3[h], D3[x] + D3[y], 0, 0, C[h, x] * C[x, y] + C[h, y] * C[x, y], C[h, x] * C[h, y] * C[x, y]])
    Fs = (F - Z["mean"]) / Z["scale"]; out = float(Z["intercept"])
    for t, pw in enumerate(Z["powers"]): out += Z["coef"][t] * np.prod(Fs ** pw)
    return out * s[h]**2 * s[x] * s[y]
Do = np.zeros((m, m, m))
for a_ in range(m):
    for b in range(m):
        for c in range(m):
            if len({a_, b, c}) < 3: continue
            Do[a_, b, c] = 0.5 * (w2[a_] * w1[b] * w1[c] * f(a_, b, c) + w1[a_] * w2[b] * w1[c] * f(b, a_, c) + w1[a_] * w1[b] * w2[c] * f(c, a_, b))
Do = torch.tensor(Do, device=DEV)
ar = torch.arange(m); A_, B_, C_ = torch.meshgrid(ar, ar, ar, indexing="ij"); sea = ((A_ != B_) & (A_ != C_) & (B_ != C_)).to(DEV)
x, y = Dp[sea], Do[sea]
print("пломба вписування: ‖chain−offline‖/‖offline‖ = %.3e  cos %.6f  ‖chain‖ %.3e ‖offline‖ %.3e" % ((x - y).norm() / y.norm(), (x * y).sum() / x.norm() / y.norm(), x.norm(), y.norm()), flush=True)
# істинна поправка з κ₄_true (той самий контур) і залишок моря
w1t = torch.tensor(w1, device=DEV); w2t = torch.tensor(w2, device=DEV); K = k4.to(DEV)
Dt = 0.5 * (w2t[:, None, None] * w1t[None, :, None] * w1t[None, None, :] * torch.stack([K[a, a] for a in range(m)])
            + w1t[:, None, None] * w2t[None, :, None] * w1t[None, None, :] * torch.stack([K[:, b, b, :] for b in range(m)]).permute(1, 0, 2)
            + w1t[:, None, None] * w1t[None, :, None] * w2t[None, None, :] * torch.stack([K[:, :, c, c] for c in range(m)]).permute(1, 2, 0))
t = Dt[sea]; r = (C3.to(DEV) - RES["cp"])[sea]
print("Δ_pairs проти істинної поправки: cos %+.3f ‖Δ‖/‖Δ_true‖ %.3f | проти залишку моря κ₃ (tru−cp): cos %+.3f ‖Δ‖/‖res‖ %.3f; Δ_true проти залишку: cos %+.3f ‖Δ_true‖/‖res‖ %.3f" % (
    (x * t).sum() / x.norm() / t.norm(), x.norm() / t.norm(), (x * r).sum() / x.norm() / r.norm(), x.norm() / r.norm(), (t * r).sum() / t.norm() / r.norm(), t.norm() / r.norm()), flush=True)
if RES["dtrans"] is not None:
    dd = RES["dtrans"][sea]; print("перенесена Δ (D на шарі ls) проти залишку: cos %+.3f ‖D‖/‖res‖ %.3f" % ((dd * r).sum() / dd.norm() / r.norm(), dd.norm() / r.norm()), flush=True)
print("DONE_SEADIAG2", flush=True)
