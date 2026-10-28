"""Спряжений вал (adjoint) ланцюга 504aldo: що з нев'язки D21 бачить вихід.
Ідея юзера «турбіна»: вал точний, похибка = впорскування на кожному ступені; до виходу доходить лише проєкція на
чутливі напрями J_l (use-side D21_l, D3_l -> final mean). Міряємо autograd-ом: дельти-листки на use-side,
K випадкових проб u ⟹ G_lu = J_l^T u для всіх шарів одним зворотним проходом.
Пломби: (P1) лінійність: Σ_l <G_lu, R_l> проти u^T (y_orc - y_base) з повним вікном оракула (corr ~1, нахил ~1);
(P2) скінченна різниця на одному шарі. Метрики: видима енергія нев'язки ||J R||² (Хатчінсон), спектр J_l (Грам проб),
видимий R² кандидатів threads (коеф. з TRAIN) проти поелементного.
ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: J_l ефективного рангу ≤ 30 на глибоких шарах (шанс 50%); видимий R² найкращого одиночного
кандидата ≥ 0.5 (шанс 15%), поелементний був ≤ 0.14; якщо ні, ідея «передбачити лише видиму частину» мертва."""
import os, sys, time, json, numpy as np, torch
MODE = os.environ.get("MODE", "collect")
DUMPS = [int(x) for x in os.environ.get("DUMPS", "0").split(",")]
K = int(os.environ.get("K", "32")); OUT = "/home/trim/wb2data/adj"
sys.argv = [sys.argv[0], "0"]
import lean_oracle as lo, lean_k3_aug_gpu as lk
dev = torch.device("cuda"); os.makedirs(OUT, exist_ok=True)
KW = dict(k4_aug=True, aug_no_k211=True, g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True,
          no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
def offd(M): M = M.clone(); M.fill_diagonal_(0.0); return M
class DeltaHook:
    """use-side D21 += dO_l (позадіаг.), D3 += dD_l; зберігає свої D21/D3 і нев'язку R_l, r_l."""
    def __init__(self, kp, oracle=False): self.kp = kp; self.oracle = oracle; self.dO = {}; self.dD = {}; self.R = {}; self.r3 = {}
    def __call__(self, l, D3, D21, live):
        if l < 1: return D3, D21, None
        kp = self.kp; a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64)
        _, _, _, mu21 = lo.true_pair(kp, l); k3 = E3 - 3*a*E2 + 2*a**3
        T21 = offd(torch.as_tensor(mu21, dtype=D21.dtype, device=D21.device)); T3 = torch.as_tensor(k3, dtype=D3.dtype, device=D3.device)
        self.R[l] = (T21 - offd(D21.detach())).float(); self.r3[l] = (T3 - D3.detach()).float()
        if self.oracle: return T3, T21, None
        dO = torch.zeros_like(D21, requires_grad=True); dD = torch.zeros_like(D3, requires_grad=True)
        self.dO[l] = dO; self.dD[l] = dD
        return D3 + dD, D21 + offd(dO), None
def chain(w, statics, kp, hook):
    lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None
    fit = lo.FitC()
    with torch.no_grad(): lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, g_hook=fit, k4_aug=True)
    lk.D21_HOOK = hook
    pred = lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, g_hook=lo.Replay(fit.coef), **KW)
    lk.D21_HOOK = None
    return pred
def collect(i):
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(lo.BANK.format(i))
    statics = lk.build_static_coefs(w.shape[1]); t0 = time.perf_counter()
    with torch.no_grad():
        yo = chain(w, statics, kp, DeltaHook(kp, oracle=True))[-1].double().cpu().numpy()
    torch.set_grad_enabled(True)
    h = DeltaHook(kp); pred = chain(w, statics, kp, h); y = pred[-1]
    yb = y.detach().double().cpu().numpy()
    print("dump%d base mse %.4e orc mse %.4e forward t=%.0fs" % (i, np.mean((yb-gt)**2), np.mean((yo-gt)**2), time.perf_counter()-t0), flush=True)
    Ls = sorted(h.dO); n = y.shape[0]; g = torch.Generator(device="cpu").manual_seed(1000 + i)
    U = torch.randn(K, n, generator=g, dtype=torch.float64)
    GO = {l: np.zeros((K, n, n), dtype=np.float32) for l in Ls}; GD = {l: np.zeros((K, n), dtype=np.float32) for l in Ls}
    leaves = [h.dO[l] for l in Ls] + [h.dD[l] for l in Ls]
    for k in range(K):
        s = (y * U[k].to(y.device, y.dtype)).sum()
        gr = torch.autograd.grad(s, leaves, retain_graph=(k < K-1), allow_unused=True)
        for j, l in enumerate(Ls):
            go = gr[j]; gd = gr[len(Ls)+j]
            if go is not None: GO[l][k] = offd(go).float().cpu().numpy()
            if gd is not None: GD[l][k] = gd.float().cpu().numpy()
        if k == 0: print("  перший зворотний t=%.0fs, nan %d" % (time.perf_counter()-t0, int(sum(np.isnan(GO[l][0]).sum() for l in Ls))), flush=True)
    # P1: лінійність
    pred_lin = np.zeros(K);
    for l in Ls: pred_lin += np.einsum("kij,ij->k", GO[l], h.R[l].cpu().numpy()) + GD[l] @ h.r3[l].cpu().numpy()
    act = U.numpy() @ (yo - yb)
    cc = float(np.corrcoef(pred_lin, act)[0, 1]); sl = float((pred_lin*act).sum()/(pred_lin*pred_lin).sum())
    print("  P1 лінійність: corr %.4f нахил act/lin %.3f | ||dy|| %.3e" % (cc, sl, np.linalg.norm(yo-yb)), flush=True)
    stats = {}
    for l in Ls:
        G = GO[l].reshape(K, -1); Gm = G @ G.T; ev = np.linalg.eigvalsh(Gm.astype(np.float64))[::-1]
        R = h.R[l].cpu().numpy().ravel(); vis = G @ R
        # видима енергія ||J R||² ≈ mean_k <G_k,R>² (u ~ N(0,I)); «ізотропний» еталон: ||J||_F² ||R||²/n² = mean_k ||G_k||² ||R||² / n²
        e_vis = float(np.mean(vis**2)); e_iso = float(np.mean((G*G).sum(1)) * (R*R).sum() / R.size)
        pr = float(ev.sum()**2 / (ev**2).sum())
        stats[l] = dict(e_vis=e_vis, e_iso=e_iso, pr=pr, ev=[float(x) for x in ev[:8]], gnorm=float(np.sqrt(np.mean((G*G).sum(1)))))
        print("  l=%2d ||G|| %.3e PR(Грам проб) %.1f/%d | видима/ізотропна енергія R %.3f" % (l, stats[l]["gnorm"], pr, K, e_vis/max(e_iso,1e-300)), flush=True)
    np.savez_compressed("%s/adj_%d.npz" % (OUT, i), **{"GO%d" % l: (GO[l] / max(float(np.abs(GO[l]).max()), 1e-30)).astype(np.float16) for l in Ls}, **{"GS%d" % l: np.float64(max(float(np.abs(GO[l]).max()), 1e-30)) for l in Ls}, **{"GD%d" % l: GD[l] for l in Ls},
                        **{"R%d" % l: h.R[l].cpu().numpy() for l in Ls}, **{"r3%d" % l: h.r3[l].cpu().numpy() for l in Ls}, U=U.numpy(), yb=yb, yo=yo, gt=gt)
    json.dump(dict(p1_corr=cc, p1_slope=sl, stats=stats), open("%s/adj_%d.json" % (OUT, i), "w"))
    print("  збережено t=%.0fs" % (time.perf_counter()-t0), flush=True)
    torch.set_grad_enabled(False)
if __name__ == "__main__":
    for i in DUMPS: collect(i); torch.cuda.empty_cache()
    print("DONE_ADJ", flush=True)
