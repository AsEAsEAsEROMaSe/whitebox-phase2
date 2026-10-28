"""Ниточки для ПОХИБКИ НАРОДЖЕННЯ Rb(l) = (T − LLL) − (his − old) при оракульних входах (useall), 3% T на шар, незалежна від входів.
Граматика: стан шару l−1 (C, C∘C, C², H, κ21/κ22/κ31 істинні, вектори var, D3, κ4, c1) перенесений п'ятьма правилами з T_v = W diag(v),
v ∈ {1, c1, mu, var, D3/var, κ4/var²}; плюс стан шару l зі скалюваннями; плюс шматки onestep (LLL, A1g, A1ng, A2, A2c, A3, A3c).
Коеф. на дампах 0–3, R² на 4–7 з тими самими коефіцієнтами; жадібний спільний до 8; GBM-стеля на топ-60.
Передбачення ДО прогону: одиночний ≤0.3, спільний ≤0.5, GBM ≤0.6 ⟹ член не з цієї граматики (шанс ниточки >0.9: ~15%)."""
import os, sys, numpy as np, torch, time, itertools
sys.argv = [sys.argv[0]]
os.environ.update(D21USE="1", D21FEED="1", K4FEED="1", K31FEED="1")
import lean_oracle as lo, lean_k3_aug_gpu as lk
from sklearn.ensemble import HistGradientBoostingRegressor
TRAIN = [0, 1, 2, 3]; TEST = [4, 5, 6, 7]; LAYERS = list(range(4, 16)); NS = 100000; TOPK = 60; dev = torch.device("cuda")
REC = []
_lin, _wick = lk.K3Blocks.linear_, lk.K3Blocks.wick_
def lin(self, W): REC.append(("W", W.detach().clone())); _lin(self, W)
def wick(self, w1): REC.append(("w1", w1.detach().clone())); _wick(self, w1)
lk.K3Blocks.linear_, lk.K3Blocks.wick_ = lin, wick
class StateHook:
    def __init__(self, dh): self.dh = dh; self.st = {}
    def __call__(self, l, D3, D21, live):
        H = D21.detach().float().clone(); H.fill_diagonal_(0.0)
        self.st[l] = dict(mu=live["mu"].detach().float().clone(), C=live["C"].detach().float().clone(), var=live["var"].detach().float().clone(), H=H, D3h=D3.detach().float().clone())
        return self.dh(l, D3, D21, live)
def offd(M): M = M.clone(); M.fill_diagonal_(0.0); return M
def collect(i):
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; n = w.shape[1]; kp = np.load(lo.BANK.format(i)); pc = np.load("/home/trim/wb2data/pieces_useall_%d.npz" % i)
    statics = lk.build_static_coefs(n); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None; REC.clear()
    fit = lo.FitC(); lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, g_hook=fit)
    REC.clear(); dh = lo.D21Hook(kp); sh = StateHook(dh); lk.D21_HOOK = sh
    lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.FeedTrue(kp, dh), g_use_hook=lo.approx_pre,
                       shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    lk.D21_HOOK = None
    Ws = [r[1] for r in REC if r[0] == "W"]; w1s = [r[1] for r in REC if r[0] == "w1"]
    L = {}
    for l in range(1, 16):
        k22, k31, k4, mu21 = lo.true_pair(kp, l); a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64)
        t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
        st = sh.st[l]
        L[l] = dict(C=offd(st["C"]), mu=st["mu"], var=st["var"], H=st["H"], D3h=st["D3h"], Tt=offd(t(mu21)), K22=offd(t(k22)), K31=offd(t(k31)), K4=t(k4), D3t=t(E3 - 3*a*E2 + 2*a**3),
                    W=Ws[l-1] if l >= 2 else None, c1=w1s[l-2] if l >= 2 else None)
        if ("Rb_%d" % l) in pc.files:
            L[l].update({k: t(pc["%s_%d" % (k, l)]) for k in ("Rb", "LLL", "A1g", "A1ng", "A2", "A2c", "A3", "A3c")})
    return L
def candidates(L, l):
    cur, prv = L[l], L[l-1]; W = cur["W"]
    vecs_prev = {"1": None, "c1": cur["c1"], "mu": prv["mu"], "var": prv["var"], "d3/v": prv["D3t"] / prv["var"], "k4/v2": prv["K4"] / prv["var"]**2}
    base_prev = {"C": prv["C"], "CC": prv["C"] * prv["C"], "C2": prv["C"] @ prv["C"], "H": prv["H"], "Tt": prv["Tt"], "K22": prv["K22"], "K31": prv["K31"], "K31T": prv["K31"].T, "C*K31": prv["C"] * prv["K31"], "C*Tt": prv["C"] * prv["Tt"]}
    for vn, v in vecs_prev.items():
        T = W if v is None else W * v[None, :]; TT = T * T
        for bn, X in base_prev.items():
            yield "T%s[%s]" % (bn, vn), T @ X @ T.T
            yield "TT%s[%s]" % (bn, vn), TT @ X @ T.T
            yield "Th(T%s)[%s]" % (bn, vn), (T * (T @ X.T)) @ T.T
            yield "TT%sTT[%s]" % (bn, vn), TT @ X @ TT.T
        for dn, dv in (("var", prv["var"]), ("d3", prv["D3t"]), ("k4", prv["K4"]), ("d3h", prv["D3h"])):
            yield "TTdiag(%s)[%s]" % (dn, vn), (TT * dv[None, :]) @ T.T
            yield "TTdiag(%s)TT[%s]" % (dn, vn), (TT * dv[None, :]) @ TT.T
    cur_base = {"C": cur["C"], "H": cur["H"], "HT": cur["H"].T, "CC": cur["C"] * cur["C"], "Tt": cur["Tt"], "K22": cur["K22"], "K31": cur["K31"], "K31T": cur["K31"].T}
    if "LLL" in cur: cur_base.update({k: cur[k] for k in ("LLL", "A1g", "A1ng", "A2", "A2c", "A3", "A3c")})
    vecs = {"1": None, "mu": cur["mu"], "var": cur["var"], "d3/v": cur["D3h"] / cur["var"], "k4/v2": cur["K4"] / cur["var"]**2}
    for bn, M in cur_base.items():
        for (vi, a), (vj, b) in itertools.product(vecs.items(), vecs.items()):
            X = M
            if a is not None: X = X * a[:, None]
            if b is not None: X = X * b[None, :]
            yield ("%s" % bn) if (a is None and b is None) else "%s[%s,%s]" % (bn, vi, vj), X
def run():
    g = torch.Generator(device="cpu").manual_seed(11); n = 1024
    ii = torch.randint(0, n, (NS + 2000,), generator=g); jj = torch.randint(0, n, (NS + 2000,), generator=g); m = ii != jj; ii, jj = ii[m][:NS].to(dev), jj[m][:NS].to(dev)
    stats = {}; t0 = time.perf_counter()
    for i in TRAIN + TEST:
        L = collect(i); o = 0 if i in TRAIN else 3
        for l in LAYERS:
            if "Rb" not in L[l]: continue
            y = L[l]["Rb"][ii, jj]
            for name, M in candidates(L, l):
                x = M[ii, jj]; s = stats.setdefault(l, {}).setdefault(name, [0.0]*6)
                s[o] += float((x*y).sum()); s[o+1] += float((x*x).sum()); s[o+2] += float((y*y).sum())
        print("dump%d: %d кандидатів, t=%.0fs" % (i, len(stats[LAYERS[-1]]), time.perf_counter() - t0), flush=True); del L; torch.cuda.empty_cache()
    agg = {}
    for l in LAYERS:
        for name, s in stats.get(l, {}).items():
            a = s[0] / max(s[1], 1e-30); r2tr = 1 - (s[2] - 2*a*s[0] + a*a*s[1]) / max(s[2], 1e-30); r2te = 1 - (s[5] - 2*a*s[3] + a*a*s[4]) / max(s[5], 1e-30)
            agg.setdefault(name, []).append((r2tr, r2te))
    rows = sorted(((np.mean([v[1] for v in vals]), np.mean([v[0] for v in vals]), name) for name, vals in agg.items()), reverse=True)
    print("\n=== Rb: топ-30 одиночних (сер. по шарах; R²_test з train-коеф | train | по шарах test)")
    for r2te, r2tr, name in rows[:30]: print("   %-28s test %.4f | train %.4f | %s" % (name, r2te, r2tr, " ".join("%.2f" % v[1] for v in agg[name])))
    keep = [r[2] for r in rows[:TOPK]]
    Xtr, ytr, Xte, yte = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}, {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
    for i in TRAIN + TEST:
        L = collect(i)
        for l in LAYERS:
            if "Rb" not in L[l]: continue
            cols = {name: M[ii, jj].double() for name, M in candidates(L, l) if name in keep}
            X = torch.stack([cols[nm] for nm in keep], 1); y = L[l]["Rb"][ii, jj].double()
            (Xtr if i in TRAIN else Xte)[l].append(X); (ytr if i in TRAIN else yte)[l].append(y)
        del L; torch.cuda.empty_cache()
    LY = [l for l in LAYERS if Xtr[l]]
    print("\n=== Rb: жадібний спільний (коеф. TRAIN, R² TEST), середнє по шарах %s" % LY)
    chosen = []
    for step in range(8):
        best = None
        for c in range(TOPK):
            if c in chosen: continue
            cols = chosen + [c]; r2s = []
            for l in LY:
                A = torch.cat(Xtr[l])[:, cols]; b = torch.cat(ytr[l]); sol = torch.linalg.lstsq(A, b[:, None]).solution[:, 0]
                At = torch.cat(Xte[l])[:, cols]; bt = torch.cat(yte[l]); res = bt - At @ sol; r2s.append(1 - float((res*res).sum()/(bt*bt).sum()))
            r = float(np.mean(r2s))
            if best is None or r > best[0]: best = (r, c, r2s)
        chosen.append(best[1]); print("   +%-28s -> R²_test %.4f (%s)" % (keep[best[1]], best[0], " ".join("%.2f" % v for v in best[2])), flush=True)
    # GBM-стеля на топ-60 + шар
    Xa = np.concatenate([np.concatenate([torch.cat(Xtr[l]).float().cpu().numpy(), np.full((len(torch.cat(ytr[l])), 1), l, np.float32)], 1) for l in LY]); ya = np.concatenate([torch.cat(ytr[l]).float().cpu().numpy() for l in LY])
    Xb = np.concatenate([np.concatenate([torch.cat(Xte[l]).float().cpu().numpy(), np.full((len(torch.cat(yte[l])), 1), l, np.float32)], 1) for l in LY]); yb = np.concatenate([torch.cat(yte[l]).float().cpu().numpy() for l in LY])
    sub = np.random.default_rng(0).choice(len(Xa), min(len(Xa), 2000000), replace=False)
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True, random_state=0).fit(Xa[sub], ya[sub])
    p = gb.predict(Xb); r = yb - p
    print("\n=== Rb: GBM на топ-60 (iters %d): R²_test %.4f ε %.3f" % (gb.n_iter_, 1 - (r*r).sum()/(yb*yb).sum(), np.linalg.norm(r)/np.linalg.norm(yb)))
    for l in LY:
        mm = Xb[:, -1] == l; rr = yb[mm] - p[mm]; print("   l=%d R² %.3f" % (l, 1 - (rr*rr).sum()/(yb[mm]**2).sum()))
    print("DONE_THREADS_RB")
run()
