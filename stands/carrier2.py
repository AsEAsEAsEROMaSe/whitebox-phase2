"""30.09 (ніч): повний носій пам'яті κ₄(p_ls) на S=32: Σ_{s<ls} T_{ls←s}·κ₄(h_s)·T·T·T, де κ₄(h_s) = усі k_max=4 діаграми
з n²-блоків ланцюга на шарі s (ключі (2,1,1) з конверсією, n²-зрізи, море; машинерія k4step з масками збігів)
+ sea3-носій (κ₃-хаб + ребро, seacarry npz, без масок). Ціль: істинна κ₄(p_ls) (brc): море, зріз (2,1,1), (3,1).
Передбачення: (2,1,1) cos 0.77 (лише sea3) → ≥0.9 з усіма класами (шанс 45%)."""
import os, sys, time, numpy as np, torch
_a = sys.argv; sys.argv = sys.argv[:1]
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import lean_oracle as lo; lk = lo.lk; import k4step as ks; import k4eng
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
sys.argv = _a; torch.set_num_threads(2)
m = 32; dt = torch.float32; dev = lk.DEV
i = int(sys.argv[1]); ls = int(sys.argv[2]); WINDOW = int(os.environ.get("WINDOW", "11"))
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
d = np.load(lo.DUMP.format(i)); w = d["weights"]; n = w.shape[1]
rng = np.random.default_rng(11 + i); S = torch.tensor(np.sort(rng.choice(n, m, replace=False)), device=dev)
Db = np.load("/home/trim/wb2data/brc_%d_%d.npz" % (i, ls)); Nh = float(Db["Nh"])
g_ = lambda k: (torch.tensor(Db["0_" + k]) + torch.tensor(Db["1_" + k])) / (2 * Nh); gh = lambda h, k: torch.tensor(Db["%d_%s" % (h, k)]) / Nh
def k4_of(g):
    e = g("q1"); R2 = g("q2"); R4 = g("q4").reshape(m, m, m, m); C2 = R2 - torch.outer(e, e)
    return R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
k4_t = k4_of(g_).to(dev); k4_a = k4_of(lambda k: gh(0, k)).to(dev); k4_b = k4_of(lambda k: gh(1, k)).to(dev)
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K3_PROBE = None; lk.K3_PROBE_POST = None; lk.K_POST_HOOK = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
LAY = {}; RES = {}
class Stop(Exception): pass
h16 = lambda t: None if t is None else t.half().clone()
def d21hook(l, D3, D21, live):
    if 1 <= l < ls: LAY.setdefault(l, {}).update(mu=live["mu"].clone(), var=live["var"].clone(), C=h16(live["C"]), D3=D3.clone(), D21=h16(D21))
    return D3, D21, None
def wkhook(l, wk4_row, wk4_mat, wk431, live):
    if 1 <= l < ls: LAY.setdefault(l, {}).update(K4=wk4_row.clone(), K22=h16(wk4_mat), K31=h16(wk431))
    return wk4_row, wk4_mat, wk431
def kpost(l, K, mu, C):
    if 1 <= l < ls: LAY.setdefault(l, {}).update(mu_h=mu.clone(), C_h=h16(C), K22h=h16(K[(2, 2)]), K31h=h16(K[(3, 1)]), K4h=K[(4,)].clone())
def k3post(l, K3):
    if 1 <= l < ls:
        trip = []
        for s in K3.sources:
            P = s["P"]; assert s.get("Q") is None
            Pd = None if P is None else h16(P)
            for xk, yk in K3.PAIRS:
                if s.get(xk) is not None: trip.append((h16(s[xk]), Pd, h16(s[yk])))
            if s["M"] is not None: trip.append((h16(s["M"]), Pd, Pd))
        D3h, D21h = K3.dslices(); LAY[l].update(trip=trip, D3h=D3h.clone(), D21h=h16(D21h))
def probe(l, K3, D3, D21):
    if l == ls:
        born = {ls - s.get("age", 0): s["P"] for s in K3.sources if s["P"] is not None}
        RES["T"] = {s: born[s][S].float().clone() for s in born if s in LAY}
        raise Stop()
lk.D21_HOOK = d21hook; lk.WK_HOOK = wkhook; lk.K_POST_HOOK = kpost; lk.K3_PROBE_POST = k3post; lk.K3_PROBE = probe; lk.REGEN_STATE.clear(); t0 = time.time()
try:
    lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG)
except Stop:
    pass
lk.D21_HOOK = None; lk.WK_HOOK = None; lk.K_POST_HOOK = None; lk.K3_PROBE_POST = None; lk.K3_PROBE = None
print("dump %d ls %d: ланцюг %.0f с; шарів пам'яті %d" % (i, ls, time.time() - t0, len(RES["T"])), flush=True)
T4 = GATI(4); diag211 = ks.computable((2, 1, 1), T4); diag1111 = ks.computable((1, 1, 1, 1), T4)
f32 = lambda t: t.float()
eyeI = torch.eye(n, dtype=dt, device=dev)
parts = {}
for s in sorted(RES["T"]):
    if s < ls - WINDOW: continue
    L = LAY[s]; WS = RES["T"][s]; t1 = time.time()
    wick = lk.WickCache(L["mu"], L["var"])
    Coff = f32(L["C"]).clone(); Coff.fill_diagonal_(0.0); D21 = f32(L["D21"]).clone(); D21.fill_diagonal_(0.0)
    K22 = f32(L["K22"]).clone(); K22.fill_diagonal_(0.0); K31 = f32(L["K31"]).clone() if L["K31"] is not None else torch.zeros_like(K22); K31.fill_diagonal_(0.0)
    B = dict(var=L["var"], D3=L["D3"], K4=L["K4"], C=Coff, D21=D21, K22=K22, K31=K31)
    trip = [tuple(eyeI if t_ is None else f32(t_) for t_ in tr) for tr in L["trip"]]
    with torch.no_grad():
        S211 = ks.slice_sum((2, 1, 1), diag211, B, wick, WS, n)
        conv = ks.conversion_211(L["mu_h"], f32(L["C_h"]), f32(L["D21h"]), L["D3h"], trip, WS, n)
        K22h = f32(L["K22h"]).clone(); K22h.fill_diagonal_(0.0); K31h = f32(L["K31h"]).clone(); K31h.fill_diagonal_(0.0)
        Sn2 = ks.n2_slices(dict(K22=K22h, K31=K31h, K4=L["K4h"]), WS, n)
        S1111 = ks.slice_sum((1, 1, 1, 1), diag1111, B, wick, WS, n)
    parts[s] = dict(k211=(S211 + conv).double(), n2=Sn2.double(), sea=S1111.double())
    print("  s=%d: %.0f с" % (s, time.time() - t1), flush=True)
    del trip; torch.cuda.empty_cache()
car = None
cf = "/home/trim/wb2data/seacarry_%d_%d.npz" % (i, ls)
if os.path.exists(cf):
    car = torch.tensor(np.load(cf)["carrier"], device=dev).double()
ar = torch.arange(m, device=dev); A_, B_, C_, D_ = torch.meshgrid(ar, ar, ar, ar, indexing="ij")
sea = (A_ != B_) & (A_ != C_) & (A_ != D_) & (B_ != C_) & (B_ != D_) & (C_ != D_)
a3, b3, c3 = torch.meshgrid(ar, ar, ar, indexing="ij"); m3 = (a3 != b3) & (a3 != c3) & (b3 != c3)
tri = lambda T: T[ar, ar][m3]
s31 = lambda T: torch.stack([T[x, x, x] for x in range(m)])[(a3[:, :, 0] != b3[:, :, 0])]
truth = k4_t.double(); noise = {nm: float((get(k4_a.double()) - get(k4_b.double())).norm() / 2 / get(truth).norm()) for nm, get in (("море", lambda T: T[sea]), ("(2,1,1)", tri), ("(3,1)", s31))}
print("шум істини:", {k: "%.3f" % v for k, v in noise.items()}, flush=True)
def rep(name, X):
    out = []
    for nm, get in (("море", lambda T: T[sea]), ("(2,1,1)", tri), ("(3,1)", s31)):
        y = get(truth); x = get(X)
        out.append("%s cos %+.3f ‖x‖/‖y‖ %.3f res %.3f" % (nm, (x * y).sum() / x.norm() / y.norm(), x.norm() / y.norm(), ((y - x) ** 2).sum() / (y ** 2).sum()))
    print("  %-34s %s" % (name, " | ".join(out)), flush=True)
tot = {k: sum(parts[s][k] for s in parts) for k in ("k211", "n2", "sea")}
rep("n²-класи: (2,1,1)-ключ Σs", tot["k211"]); rep("n²-класи: n²-зрізи Σs", tot["n2"]); rep("n²-класи: море Σs", tot["sea"])
alln2 = tot["k211"] + tot["n2"] + tot["sea"]; rep("n²-класи разом", alln2)
if car is not None:
    rep("sea3-носій (seacarry)", car); rep("n²-класи + sea3", alln2 + car)
    for k in ("k211", "n2", "sea"): rep("sea3 + лише %s" % k, car + tot[k])
    # найкращі скаляри на 4 компоненти для зрізу (2,1,1) (діагностика стелі)
    Xm = torch.stack([tri(car), tri(tot["k211"]), tri(tot["n2"]), tot["sea"][ar, ar][m3]], 1); y = tri(truth)
    coef = torch.linalg.lstsq(Xm, y[:, None]).solution[:, 0]; pred = Xm @ coef
    print("  МНК 4 компоненти на (2,1,1): R² %.3f coef %s" % (1 - float(((y - pred) ** 2).sum() / (y ** 2).sum()), " ".join("%.2f" % c for c in coef)), flush=True)
    for s in sorted(parts, reverse=True):
        cum = car + sum(parts[t][k] for t in parts if t >= s for k in ("k211", "n2", "sea"))
        rep("sea3 + n²-класи s>=%d" % s, cum)
torch.save({k: v.cpu() for k, v in tot.items()}, "/home/trim/wb2data/carrier2_%d_%d.pt" % (i, ls))
print("DONE_CARRIER2", flush=True)
