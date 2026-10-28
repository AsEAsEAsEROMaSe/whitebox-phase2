"""Похибка МОРЯ (1,1,1) стану κ₃ ланцюга (стенд 26.09): на 32 нейронах S (як у birthref, brc_<i>_<ls>.npz, N=2²⁶)
щільно відновлюємо κ₃ ланцюга з CP-джерел (+Δ у d4) на шарі ls: ПЕРЕД wick (pre, проти істинної κ₃(p)) і ПІСЛЯ народження
(post, проти істинної κ₃(h)). Віднос. похибка за зрізами (1,1,1) x<y<z · (2,1) · (3). Пломба: зріз (2,1) з T_S проти K3.dslices()[S,S].
Передбачення ДО: pre-море ε ≈ 5–8% (як D21), post-море − wick·pre ≈ 3% (народження ланцюга), d4 знімає з народження ~половину,
але pre-море майже не міняється (накопичене) — шанс 50%."""
import os, sys, time, itertools, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk; import d4dense
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
torch.set_default_dtype(torch.float32)
dt = torch.float32
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
modes = os.environ.get("MODES", "base").split(",")
PERMS = list(itertools.permutations(range(3)))
W1 = {}
_orig_wick = lk.K3Blocks.wick_
def _wick_stash(self, w1):
    W1["w1"] = w1.clone(); return _orig_wick(self, w1)
lk.K3Blocks.wick_ = _wick_stash
def sym(T): return sum(T.permute(*p) for p in PERMS) / 6.0
def dense_S(K3, S, D4=None, only=None):
    T = None
    for s in K3.sources:
        if only == "old" and s["P"] is None: continue
        if only == "birth" and s["P"] is not None: continue
        P = s["P"]; Q = s.get("Q")
        def leg(k):
            v = s.get(k)
            if v is None: return None
            v = (Q @ v) if Q is not None else v
            return v[S].double()
        if P is None: PS = torch.eye(P_n, dtype=torch.float64, device=S.device)[S] if False else EYE[S]
        else: PS = ((Q @ P) if Q is not None else P)[S].double()
        for xk, yk in K3.PAIRS:
            X = leg(xk)
            if X is not None:
                t = torch.einsum("aj,bj,cj->abc", X, PS, leg(yk)); T = t if T is None else T + t
        M = leg("M")
        if M is not None:
            t = torch.einsum("aj,bj,cj->abc", M, PS, PS); T = t if T is None else T + t
    T = sym(T) if T is not None else torch.zeros((len(S),)*3, dtype=torch.float64, device=S.device)
    if D4 is not None and only is None and D4.D is not None:
        T = T + D4.D[S][:, S][:, :, S].double() / D4.SC
    return T
m = 32
iu = torch.tensor([[x, y, z] for x in range(m) for y in range(m) for z in range(m) if x < y < z]); p21 = torch.tensor([[x, x, y] for x in range(m) for y in range(m) if x != y]); p3 = torch.tensor([[x, x, x] for x in range(m)])
sel = lambda T, ix: T[ix[:, 0], ix[:, 1], ix[:, 2]]
def errs(T, R):
    return tuple(float((sel(T, ix) - sel(R, ix)).norm() / sel(R, ix).norm()) for ix in (iu, p21, p3))
def truth(fn):
    D = np.load(fn); Nh = float(D["Nh"]); g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
    e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m)
    C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
    hm = g_("h1"); H2 = g_("h2"); H3 = g_("h3").reshape(m, m, m)
    K3h = H3 - (H2[:, :, None]*hm[None, None, :] + H2[:, None, :]*hm[None, :, None] + H2[None, :, :]*hm[:, None, None]) + 2*hm[:, None, None]*hm[None, :, None]*hm[None, None, :]
    R4 = g_("q4").reshape(m, m, m, m); C2 = R2 - torch.outer(e, e)
    k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(pp)) for pp in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
    return C3, K3h, e, C2, k4
for arg in sys.argv[1].split(","):
    i, ls = [int(x) for x in arg.split(":")]
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); n = w.shape[1]
    rng = np.random.default_rng(11 + i); S = torch.tensor(np.sort(rng.choice(n, m, replace=False)), device=lk.DEV); EYE = torch.eye(n, dtype=torch.float64, device=lk.DEV)
    C3, K3h, e_t, C2_t, k4_t = truth("/home/trim/wb2data/brc_%d_%d.npz" % (i, ls)); C3 = C3.to(lk.DEV); K3h = K3h.to(lk.DEV)
    kpb = np.load(lo.BANK.format(i)); a_t = torch.tensor(kpb["pre_mean"][ls].astype(np.float64))[S.cpu()]; mean_t = (a_t + e_t).to(lk.DEV); C2_t = C2_t.to(lk.DEV)
    k4_t = k4_t.to(lk.DEV); ar = torch.arange(m, device=lk.DEV)
    A, B, Cc, Dd = torch.meshgrid(ar, ar, ar, ar, indexing="ij"); ndist = ((A != B).int() + (A != Cc).int() + (A != Dd).int() + (B != Cc).int() + (B != Dd).int() + (Cc != Dd).int())
    nd = torch.stack([A, B, Cc, Dd], -1); ndistinct = torch.stack([(nd == nd[..., k:k+1]).any(-1) for k in range(4)], -1).all(-1)   # заглушка
    # кількість різних індексів у (a,b,c,d)
    ndistinct = 1 + (B != A).int() + ((Cc != A) & (Cc != B)).int() + ((Dd != A) & (Dd != B) & (Dd != Cc)).int()
    K4V = dict(full=k4_t, zero=torch.zeros_like(k4_t), n2=k4_t * (ndistinct <= 2).double(), n2tri=k4_t * (ndistinct <= 3).double())
    def ref3(mean, C2, T3, k4=None):
        torch.set_default_dtype(torch.float64)
        tw = {1: mean.double(), 2: C2.double(), 3: T3.double()}
        if k4 is not None: tw[4] = k4.double()
        try: return relu_kprop(DSTower.from_tower(tw), k_max=3)[3].to_tensor().to(lk.DEV)
        finally: torch.set_default_dtype(torch.float32)
    for nm4, k4v in K4V.items():
        print("dump %d шар %d референс k_max=3 з ІСТИННИХ входів, κ₄=%-5s проти істини: (1,1,1) %.4f (2,1) %.4f (3) %.4f" % (i, ls, nm4, *errs(ref3(mean_t, C2_t, C3, k4v), K3h)), flush=True)
    st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K3_PROBE = None; lk.K3_PROBE_POST = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for nm in modes:
        lk.REGEN_STATE.clear(); t0 = time.time(); out = {}
        lk.D4 = d4dense.Dense4(n, lk.DEV) if nm == "d4" else None
        def pre(l, K3, D3, D21):
            if l != ls: return
            T = dense_S(K3, S, lk.D4); out["pre"] = errs(T, C3)
            d21S = D21[S][:, S].double(); tS = T[torch.arange(m), torch.arange(m), :]; tS = tS - torch.diag(torch.diagonal(tS))
            out["seal"] = float((d21S - tS).norm() / d21S.norm())
            out["Tcp"] = dense_S(K3, S, None); out["Dpre"] = (lk.D4.D[S][:, S][:, :, S].double() / lk.D4.SC) if (lk.D4 is not None and lk.D4.D is not None) else 0.0
        def post(l, K3):
            if l != ls: return
            T = dense_S(K3, S, lk.D4); out["post"] = errs(T, K3h)
            # народження ланцюга на S = джерела віку 0 (+Δ_new немає окремо: Δ у d4 містить і перенесене); істинне = K3h − w1³∘κ₃(p)
            w1 = W1["w1"][S].double(); Told = dense_S(K3, S, None, only="old"); Tb = dense_S(K3, S, None, only="birth")
            if lk.D4 is not None and lk.D4.D is not None:
                Tb = Tb + lk.D4.D[S][:, S][:, :, S].double() / lk.D4.SC - out["Dpre"] * (w1[:, None, None] * w1[None, :, None] * w1[None, None, :])
            true_b = K3h - C3 * (w1[:, None, None] * w1[None, :, None] * w1[None, None, :])
            out["wickseal"] = float((Told - out["Tcp"] * (w1[:, None, None] * w1[None, :, None] * w1[None, None, :])).norm() / Told.norm())
            out["birth"] = tuple(float((sel(Tb, ix) - sel(true_b, ix)).norm() / sel(K3h, ix).norm()) for ix in (iu, p21, p3))
            out["birth_rel"] = tuple(float((sel(Tb, ix) - sel(true_b, ix)).norm() / sel(true_b, ix).norm()) for ix in (iu, p21, p3))
            out["bnorm"] = tuple(float(sel(true_b, ix).norm() / sel(K3h, ix).norm()) for ix in (iu, p21, p3))
            Kref = ref3(out["mu"], out["C"], out["Tcp"] + (out["Dpre"] if lk.D4 is not None else 0.0), k4_t)
            out["ref_chain_in"] = errs(Kref, K3h); out["chain_vs_ref"] = errs(T, Kref)
            out["in_err"] = (float((out["mu"] - mean_t).norm() / mean_t.norm()), float((out["C"] - C2_t).norm() / C2_t.norm()))
            Kref2 = ref3(out["mu"], out["C"], out["Tcp"] + (out["Dpre"] if lk.D4 is not None else 0.0), K4V["n2"])
            out["chain_vs_ref_n2"] = errs(T, Kref2); out["ref_n2_chain_in"] = errs(Kref2, K3h)
            # перенос ЛИШЕ зрізу (2,1)+(3) пост-стану через W_{l+1}: яку частку D21(l+1) він дає
            D3h, D21h = K3.dslices(); Mh = 3.0 * D21h.T.clone(); Mh.fill_diagonal_(0.0); Mh = Mh + torch.diag(D3h)
            tmp = lk.K3Blocks(); tmp.sources = [dict(X1=None, Y1=None, X2=None, Y2=None, X3=None, Y3=None, X4=None, Y4=None, M=Mh, P=torch.eye(n, dtype=D21h.dtype, device=D21h.device), age=0)]
            tmp.linear_(torch.as_tensor(w[ls + 1].T.copy(), dtype=D21h.dtype, device=D21h.device)); out["carried"] = tmp.dslices()
        def pre_next(l, K3, D3, D21):
            if l != ls + 1 or "carried" not in out: return
            d3c, d21c = out["carried"]; x = d21c.double(); y = D21.double(); a = float((x*y).sum()/(x*x).sum())
            out["carry"] = (float(x.norm()/y.norm()), 1 - float(((y - a*x)**2).sum()/(y*y).sum()), a, float(d3c.double().norm()/D3.double().norm()))
            # частка народження: post − (post-стан без народження) недоступна прямо; міряємо post-помилку і pre-помилку окремо
        def pre_all(l, K3, D3, D21): pre(l, K3, D3, D21); pre_next(l, K3, D3, D21)
        def stash(l, D3, D21, live):
            if l == ls: out["mu"] = live["mu"][S].double().clone(); out["C"] = live["C"][S][:, S].double().clone()
            return D3, D21, None
        lk.D21_HOOK = stash
        lk.K3_PROBE = pre_all; lk.K3_PROBE_POST = post
        hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
        mse = float(np.mean((p[-1].detach().cpu().double().numpy() - gt)**2))
        print("dump %d шар %d %-5s mse %.4e (%.0f с) | пломба (2,1) %.2e | pre ε (1,1,1) %.4f (2,1) %.4f (3) %.4f | post ε (1,1,1) %.4f (2,1) %.4f (3) %.4f" % (
            i, ls, nm, mse, time.time() - t0, out["seal"], *out["pre"], *out["post"]), flush=True)
        print("   народження ланцюга на S: похибка/‖K3h‖ (1,1,1) %.4f (2,1) %.4f (3) %.4f | /‖істинне народж.‖ %.3f %.3f %.3f | ‖істинне народж.‖/‖K3h‖ %.3f %.3f %.3f | пломба wick %.1e" % (*out["birth"], *out["birth_rel"], *out["bnorm"], out["wickseal"]), flush=True)
        print("   референс k_max=3 з входів ЛАНЦЮГА (κ₄ істинна) проти істини: (1,1,1) %.4f (2,1) %.4f (3) %.4f | ланцюг проти цього референсу: %.4f %.4f %.4f | похибка входів mu %.4f C %.4f" % (*out["ref_chain_in"], *out["chain_vs_ref"], *out["in_err"]), flush=True)
        print("   референс k_max=3 з входів ЛАНЦЮГА, κ₄ лише n²: проти істини %.4f %.4f %.4f | ланцюг проти нього: %.4f %.4f %.4f" % (*out["ref_n2_chain_in"], *out["chain_vs_ref_n2"]), flush=True)
        print("   перенос лише зрізу у D21(l+1): ‖carried‖/‖D21‖ %.3f  R² %.3f нахил %.3f | D3 %.3f" % out["carry"], flush=True)
        lk.D4 = None; lk.K3_PROBE = None; lk.K3_PROBE_POST = None; lk.D21_HOOK = None; torch.cuda.empty_cache()
