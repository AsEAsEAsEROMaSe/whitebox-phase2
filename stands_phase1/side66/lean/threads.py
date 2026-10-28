"""Автоматичний перебір «ниточок»: граматика n²-кандидатів зі стану його ланцюга (шар l і перенесене з l−1)
проти оракульних цілей: R = κ21_true − D21_his (позадіаг.) і S_old (зріз старого ярусу).
Одиночний R² на кожному кандидаті (коеф. підібраний), жадібний спільний до 8 членів на топ-60;
коефіцієнти на дампах TRAIN, R² на TEST з тими самими коефіцієнтами (контроль «чужа»).
Передбачення ДО прогону: найкращий одиночний R²(R) ≤ 0.5, спільний ≤ 0.6 на тесті; для S_old ≤ 0.9. Шанс ниточки >0.95: ~10–15%."""
import os, sys, numpy as np, torch, time, itertools
sys.argv = [sys.argv[0]] + (sys.argv[1:] if len(sys.argv) > 1 else ["0"])
import lean_oracle as lo, lean_k3_aug_gpu as lk
TRAIN = [0, 1, 2, 3]; TEST = [4, 5, 6, 7]; LAYERS = list(range(5, 16)); NS = 100000; AOLD = 4; TOPK = 60
dev = torch.device("cuda")
REC = []
_lin, _wick = lk.K3Blocks.linear_, lk.K3Blocks.wick_
def lin(self, W): REC.append(("W", W.detach().clone())); _lin(self, W)
def wick(self, w1): REC.append(("w1", w1.detach().clone())); _wick(self, w1)
lk.K3Blocks.linear_, lk.K3Blocks.wick_ = lin, wick
class Probe:
    def __init__(self): self.S = {}
    def __call__(self, l, K3, D3, D21):
        def sl(sub):
            if not sub: return torch.zeros_like(D3), torch.zeros_like(D21)
            a, b = K3.dslices(subset=sub); b = b.detach().float().clone(); b.fill_diagonal_(0.0); return a.detach().float().clone(), b
        self.S[l] = dict(young=sl([s for s in K3.sources if s.get("age", 0) < AOLD]), old=sl([s for s in K3.sources if s.get("age", 0) >= AOLD]),
                         his=(D3.detach().float().clone(), D21.detach().float().clone()))
class StateHook:
    def __init__(self): self.st = {}
    def __call__(self, l, D3, D21, live):
        self.st[l] = dict(mu=live["mu"].detach().float().clone(), C=live["C"].detach().float().clone(), var=live["var"].detach().float().clone()); return D3, D21, None
def offd(M): M = M.clone(); M.fill_diagonal_(0.0); return M
def collect(i):
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; n = w.shape[1]; kp = np.load(lo.BANK.format(i))
    statics = lk.build_static_coefs(n)
    lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None; REC.clear()
    fit = lo.FitC(); lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, g_hook=fit)
    REC.clear(); pr = Probe(); sh = StateHook(); lk.K3_PROBE = pr; lk.D21_HOOK = sh
    lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef),
                       g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    lk.K3_PROBE = None; lk.D21_HOOK = None
    Ws = [r[1] for r in REC if r[0] == "W"]; w1s = [r[1] for r in REC if r[0] == "w1"]
    L = {}
    for l in range(1, 16):
        a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64)
        _, _, _, mu21 = lo.true_pair(kp, l); d3t = E3 - 3*a*E2 + 2*a**3
        Tt = torch.as_tensor(mu21, dtype=torch.float32, device=dev); Tt = offd(Tt)
        st = sh.st[l]; D3h, H = pr.S[l]["his"]; H = offd(H)
        L[l] = dict(C=offd(st["C"]), mu=st["mu"], var=st["var"], H=H, D3h=D3h, Sy=pr.S[l]["young"][1], So=pr.S[l]["old"][1], D3y=pr.S[l]["young"][0], D3o=pr.S[l]["old"][0],
                    Tt=Tt, D3t=torch.as_tensor(d3t, dtype=torch.float32, device=dev), R=Tt - H,
                    T=(Ws[l-1] * w1s[l-2][None, :]).float() if l >= 2 else None)
    return L
def candidates(L, l):
    """yield (name, matrix) для шару l"""
    cur, prv = L[l], L[l-1]; T = cur["T"]; TT = T * T
    base = {"C": cur["C"], "H": cur["H"], "HT": cur["H"].T, "Sy": cur["Sy"], "SyT": cur["Sy"].T, "So": cur["So"], "SoT": cur["So"].T, "CC": cur["C"] * cur["C"], "C2": cur["C"] @ cur["C"]}
    for nm, X in (("Cp", prv["C"]), ("Hp", prv["H"]), ("Sop", prv["So"]), ("Rp", prv["R"]), ("Ttp", prv["Tt"])):
        base["T%sT" % nm] = T @ X @ T.T; base["TT%sT" % nm] = TT @ X @ T.T; base["T%sTT" % nm] = T @ X @ TT.T; base["Th(T%s)T" % nm] = (T * (T @ X.T)) @ T.T
    for nm, dvec in (("d3hp", prv["D3h"]), ("d3tp", prv["D3t"])):
        base["TTdiag(%s)T" % nm] = (TT * dvec[None, :]) @ T.T
    vecs = {"1": None, "mu": cur["mu"], "var": cur["var"], "d3/v": cur["D3h"] / cur["var"], "mu/s": cur["mu"] / cur["var"].sqrt()}
    names = list(base)
    for nm in names:
        M = base[nm]
        for (vi, a), (vj, b) in itertools.product(vecs.items(), vecs.items()):
            if a is None and b is None: yield nm, M; continue
            X = M
            if a is not None: X = X * a[:, None]
            if b is not None: X = X * b[None, :]
            yield "%s[%s,%s]" % (nm, vi, vj), X
    for p in range(len(names)):
        for q in range(len(names)):
            yield "%s@%s" % (names[p], names[q]), base[names[p]] @ base[names[q]]
            if q >= p: yield "%s*%s" % (names[p], names[q]), base[names[p]] * base[names[q]]
def run():
    g = torch.Generator(device="cpu").manual_seed(7)
    n = 1024; ii = torch.randint(0, n, (NS + 2000,), generator=g); jj = torch.randint(0, n, (NS + 2000,), generator=g); m = ii != jj; ii, jj = ii[m][:NS].to(dev), jj[m][:NS].to(dev)
    stats = {}   # (target, l) -> dict name -> [sxy_train, sxx_train, syy_train, sxy_test, sxx_test, syy_test]
    samples = {}  # (dump, target, l) -> dict name -> vector (only for names in keep set, second pass)
    t0 = time.perf_counter()
    for i in TRAIN + TEST:
        L = collect(i); split = "train" if i in TRAIN else "test"
        for l in LAYERS:
            ys = {"R": L[l]["R"][ii, jj], "So": L[l]["So"][ii, jj]}
            for name, M in candidates(L, l):
                x = M[ii, jj]
                for tg, y in ys.items():
                    s = stats.setdefault((tg, l), {}).setdefault(name, [0.0]*6); o = 0 if split == "train" else 3
                    s[o] += float((x*y).sum()); s[o+1] += float((x*x).sum()); s[o+2] += float((y*y).sum())
        print("dump%d зібрано, %d кандидатів, t=%.0fs" % (i, len(stats[("R", LAYERS[0])]), time.perf_counter() - t0), flush=True)
        del L; torch.cuda.empty_cache()
    # одиночний R²: коеф a = sxy/sxx з TRAIN; R²_test = 1 − (syy − 2a sxy + a² sxx)/syy на TEST
    keep = {}
    for tg in ("R", "So"):
        agg = {}
        for l in LAYERS:
            for name, s in stats[(tg, l)].items():
                a = s[0] / max(s[1], 1e-30); r2tr = 1 - (s[2] - 2*a*s[0] + a*a*s[1]) / max(s[2], 1e-30); r2te = 1 - (s[5] - 2*a*s[3] + a*a*s[4]) / max(s[5], 1e-30)
                agg.setdefault(name, []).append((r2tr, r2te))
        rows = sorted(((np.mean([v[1] for v in vals]), np.mean([v[0] for v in vals]), name) for name, vals in agg.items()), reverse=True)
        print("\n=== ЦІЛЬ %s: топ-25 одиночних (сер. по шарах %s; R²_test з train-коефіцієнтом | R²_train)" % (tg, LAYERS))
        for r2te, r2tr, name in rows[:25]: print("   %-34s test %.4f | train %.4f | по шарах test: %s" % (name, r2te, r2tr, " ".join("%.2f" % v[1] for v in agg[name])))
        keep[tg] = [r[2] for r in rows[:TOPK]]
    # жадібний спільний на топ-K: другий прохід, зберігаємо вибірки
    for tg in ("R", "So"):
        Xtr, ytr, Xte, yte = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}, {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
        for i in TRAIN + TEST:
            L = collect(i)
            for l in LAYERS:
                cols = {}
                for name, M in candidates(L, l):
                    if name in keep[tg]: cols[name] = M[ii, jj].double()
                X = torch.stack([cols[nm] for nm in keep[tg]], 1); y = (L[l]["R"] if tg == "R" else L[l]["So"])[ii, jj].double()
                (Xtr if i in TRAIN else Xte)[l].append(X); (ytr if i in TRAIN else yte)[l].append(y)
            del L; torch.cuda.empty_cache()
        print("\n=== ЦІЛЬ %s: жадібний спільний (коеф. з TRAIN, R² на TEST), середнє по шарах" % tg)
        chosen = []
        for step in range(8):
            best = None
            for c in range(TOPK):
                if c in chosen: continue
                cols = chosen + [c]; r2s = []
                for l in LAYERS:
                    A = torch.cat(Xtr[l])[:, cols]; b = torch.cat(ytr[l]); sol = torch.linalg.lstsq(A, b[:, None]).solution[:, 0]
                    At = torch.cat(Xte[l])[:, cols]; bt = torch.cat(yte[l]); res = bt - At @ sol; r2s.append(1 - float((res*res).sum()/(bt*bt).sum()))
                r = float(np.mean(r2s))
                if best is None or r > best[0]: best = (r, c, r2s)
            chosen.append(best[1]); print("   +%-34s -> R²_test %.4f  (по шарах %s)" % (keep[tg][best[1]], best[0], " ".join("%.2f" % v for v in best[2])), flush=True)
    print("DONE_THREADS")
run()
