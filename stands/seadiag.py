"""Діагностика 30.09: чи Δ (пам'ять κ₄ → народження κ₃) лягає в напрямку похибки моря κ₃(p_ls) на 32 нейронах (істина brc).
На K3_PROBE l=ls: κ₃_chain(S³) = Σ_src Sym(X_S⊗P_S⊗Y_S) (+M,X3), Δ_S = D4.D[S,S,S]; порівняння з істинною κ₃(p_ls) на морі та (2,1)."""
import os, sys, time, numpy as np, torch
_a = sys.argv; sys.argv = sys.argv[:1]
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import lean_oracle as lo; lk = lo.lk; import sea4
sys.argv = _a; torch.set_num_threads(2)
dt = torch.float32; DEV = lk.DEV; m = 32
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
i = int(sys.argv[1]); ls = int(sys.argv[2]); WINDOW = int(os.environ.get("WINDOW", "6"))
D = np.load("/home/trim/wb2data/brc_%d_%d.npz" % (i, ls)); Nh = float(D["Nh"])
g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m)
C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
d = np.load(lo.DUMP.format(i)); w = d["weights"]; n = w.shape[1]
rng = np.random.default_rng(11 + i); S = torch.tensor(np.sort(rng.choice(n, m, replace=False)), device=DEV)
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.D4 = None; lk.K3_PROBE = None; lk.K3_INJ = None; lk.K_POST_HOOK = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
S4 = sea4.Sea4(n, DEV, window=WINDOW); lk.D4 = S4
class Stop(Exception): pass
RES = {}
def cp_on_S(K3):
    T = torch.zeros((m, m, m), dtype=torch.float64, device=DEV)
    for s in K3.sources:
        P = s["P"]; assert P is not None and s.get("Q") is None
        Ps = P[S].double()
        for xk, yk in (("X1", "Y1"), ("X3", "Y3")):
            if s.get(xk) is not None:
                T += torch.einsum("aJ,bJ,cJ->abc", s[xk][S].double(), Ps, s[yk][S].double())
        if s.get("M") is not None:
            T += torch.einsum("aJ,bJ,cJ->abc", s["M"][S].double(), Ps, Ps)
    return sum(T.permute(*p) for p in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]) / 6.0
def probe(l, K3, D3, D21):
    S4.stash(l, K3, D21)
    if l == ls:
        RES["cp"] = cp_on_S(K3)
        RES["delta"] = (S4.D[S][:, S][:, :, S].double() / S4.SC) if S4.D is not None else torch.zeros_like(RES["cp"])
        raise Stop()
def d21hook(l, D3, D21, live):
    wk = lk.WickCache(live["mu"], live["var"]); S4.set_w(l, {k: wk(k, 1) for k in (1, 2, 3)}, lk.zero_diag_(live["C"].clone()))
    return D3, D21, None
lk.K3_PROBE = probe; lk.D21_HOOK = d21hook; lk.REGEN_STATE.clear(); t0 = time.time()
try:
    lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
except Stop:
    pass
cp, delta = RES["cp"], RES["delta"]; tru = C3.to(DEV)
ar = torch.arange(m); A_, B_, C_ = torch.meshgrid(ar, ar, ar, indexing="ij")
sea = ((A_ != B_) & (A_ != C_) & (B_ != C_)).to(DEV)
d21m = (A_ != B_).to(DEV)
def rep(name, msk, get):
    t = get(tru)[msk]; c = get(cp)[msk]; dl = get(delta)[msk]; r = t - c
    print("  %-10s ‖tru‖ %.3e  похибка ланцюга ‖tru−cp‖/‖tru‖ %.4f  з Δ %.4f  cos(Δ, tru−cp) %+.3f  ‖Δ‖/‖tru−cp‖ %.3f  c*=%.2f" % (
        name, t.norm(), r.norm() / t.norm(), (r - dl).norm() / t.norm(), (dl * r).sum() / dl.norm() / r.norm(), dl.norm() / r.norm(), (dl * r).sum() / (dl * dl).sum()), flush=True)
print("dump %d l=%d (%.0f с)" % (i, ls, time.time() - t0), flush=True)
rep("море", sea, lambda T: T)
rep("зріз (2,1)", d21m, lambda T: torch.stack([T[x, x] for x in range(m)]))
print("DONE_SEADIAG", flush=True)
