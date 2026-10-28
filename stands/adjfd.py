"""Спряжений вал скінченною різницею (autograd не влазить у 8 ГБ): v = J_l X = y(D21_use_l + X) − y0.
Ціль: X = R_l (істинна нев'язка D21 на шарі l, позадіаг.) — видима частина. Ознаки: J_l c для реалізовних кандидатів
граматики threads (без оракульних Rp/Ttp), нормованих до ||R_l||. Пломба лінійності: J(0.5 R)·2 проти J(R).
Аналіз окремо (adjfd_an.py): видимий R² (коеф. з TRAIN дампів 0-3, тест 4-7) проти поелементного.
ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: найкращий одиночний видимий R²_test ≥ 0.5 з шансом 15%; поелементні ті самі кандидати ≤ 0.1."""
import os, sys, time, numpy as np, torch
DUMPS = [int(x) for x in os.environ.get("DUMPS", "0").split(",")]
LAYS = [int(x) for x in os.environ.get("LAYS", "6,9,12,15").split(",")]
OUT = "/home/trim/wb2data/adjfd"; os.makedirs(OUT, exist_ok=True)
src = open("/home/trim/wb2lean/threads.py").read(); cut = src.rstrip().rfind("\nrun()"); assert cut > 0
ns = {"__name__": "threads_lib", "__file__": "/home/trim/wb2lean/threads.py"}; exec(compile(src[:cut], "threads.py", "exec"), ns)
lo, lk = ns["lo"], ns["lk"]
NAMES = ["C", "H", "HT", "Sy", "SyT", "So", "SoT", "CC", "C2", "TCpT", "THpT", "TSopT", "TTCpT", "TCpTT", "Th(TCp)T", "Th(THp)T",
         "TTdiag(d3hp)T", "H[d3/v,1]", "H[mu,1]", "C[mu,1]", "THpT[mu/s,mu/s]", "H@C", "C@H"]
KW = dict(k4_aug=True, aug_no_k211=True, g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True,
          no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
class Pert:
    def __init__(self, l, X, X3=None): self.l = l; self.X = X; self.X3 = X3
    def __call__(self, l, D3, D21, live):
        if l == self.l and self.X3 is not None: return D3 + self.X3.to(D3.dtype), D21, None
        if l == self.l and self.X is not None: return D3, D21 + self.X.to(D21.dtype), None
        return D3, D21, None
def run_net(i):
    t0 = time.perf_counter(); L = ns["collect"](i)
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; statics = lk.build_static_coefs(w.shape[1])
    lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None
    fit = lo.FitC(); lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, g_hook=fit, k4_aug=True); coef = fit.coef
    def fwd(l, X, X3=None):
        lk.D21_HOOK = Pert(l, X, X3); y = lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, g_hook=lo.Replay(coef), **KW)[-1]
        lk.D21_HOOK = None; return y.double().cpu().numpy()
    y0 = fwd(-1, None); out = dict(y0=y0, gt=d["final_means"].astype(np.float64))
    print("dump%d base mse %.4e t=%.0fs" % (i, np.mean((y0-out["gt"])**2), time.perf_counter()-t0), flush=True)
    for l in LAYS:
        R = L[l]["R"]; nR = float(R.norm()); out["nR%d" % l] = nR
        out["vR%d" % l] = fwd(l, R) - y0; vh = fwd(l, 0.5*R) - y0
        r3 = L[l]["D3t"] - L[l]["D3h"]; out["v3R%d" % l] = fwd(l, None, r3) - y0
        lin = float(np.linalg.norm(2*vh - out["vR%d" % l]) / np.linalg.norm(out["vR%d" % l]))
        cands = {nm: M for nm, M in ns["candidates"](L, l) if nm in NAMES}
        miss = [nm for nm in NAMES if nm not in cands]
        for nm, M in cands.items():
            M = M.clone(); M.fill_diagonal_(0.0); s = nR / max(float(M.norm()), 1e-30)
            out["v%d|%s" % (l, nm)] = (fwd(l, M*s) - y0) / s
            # поелементний R² на тому ж кандидаті (для порівняння): зберігаємо скалярні добутки
            out["e%d|%s" % (l, nm)] = np.array([float((M*R).sum()), float((M*M).sum()), float((R*R).sum())])
        print("  l=%2d ||J r3||/||J R|| %.3f | ||J R||/||y0|| %.3e нелінійність %.4f кандидатів %d (нема: %s) t=%.0fs" % (l, np.linalg.norm(out["v3R%d" % l])/max(np.linalg.norm(out["vR%d" % l]),1e-300), np.linalg.norm(out["vR%d" % l])/np.linalg.norm(y0), lin, len(cands), miss, time.perf_counter()-t0), flush=True)
        out["lin%d" % l] = lin
        np.savez("%s/fd_%d.npz" % (OUT, i), **out)
    del L; torch.cuda.empty_cache()
if __name__ == "__main__":
    for i in DUMPS: run_net(i)
    print("DONE_FD", flush=True)
