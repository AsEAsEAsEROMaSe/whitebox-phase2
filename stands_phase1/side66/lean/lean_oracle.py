"""Оракул κ₂₂/κ₃₁ на lean-ланцюгу 504aldo (float64 torch → GPU-патч).
Пломба: mode=rec на dense-ланцюгу — на l=1 регресія wk4_mat проти істинної κ₂₂(p) мусить дати R²≈1 і сталий нахил
(це визначає одиниці); wk4_row проти κ₄_iiii так само. Передбачення ДО прогону: regen-ланцюг ≈ 2.2e-8; +істинна κ₂₂/κ₃₁ у
use-side → 1.1…1.5e-8 (наш K4P: 0.0162→0.0118% = ×1.9 MSE); шанс ≥×1.5: 50%."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_k3_aug_gpu as lk
DEV = lk.DEV
DUMP = "/home/trim/wb2data/dumps_p2/mlp_{:04d}.npz"; BANK = "/home/trim/wb2data/kp_p2/mlp_{:05d}.npz"
dumps = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 3 else []; cfg = sys.argv[2] if len(sys.argv) > 3 else ""; mode = sys.argv[3] if len(sys.argv) > 3 else ""
dtype = torch.float64 if os.environ.get("F64", "0") == "1" else torch.float32
W22 = float(os.environ.get("W22", "1")); W31 = float(os.environ.get("W31", "1"))
def zd(M): M = M.clone(); M.fill_diagonal_(0.0); return M
def true_pair(kp, l):
    a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64); E4 = kp["pre_m4"][l].astype(np.float64)
    M11 = kp["pre_M11"][l].astype(np.float64); M21 = kp["pre_M21"][l].astype(np.float64); M31 = kp["pre_M31"][l].astype(np.float64); M22 = kp["pre_M22"][l].astype(np.float64)
    C11 = M11 - np.outer(a, a)
    # центральні: mu21[i,j]=E[(pi-ai)^2 (pj-aj)], mu31, mu22 з сирих
    mu21 = M21 - 2*a[:, None]*M11 - a[None, :]*E2[:, None] + 2*np.outer(a*a, a)
    mu31 = M31 - 3*a[:, None]*M21 + 3*(a*a)[:, None]*M11 - a[None, :]*E3[:, None] + 3*a[None, :]*(a[:, None]*E2[:, None]) - 3*np.outer(a**3, a)
    mu22 = (M22 - 2*a[None, :]*M21 - 2*a[:, None]*M21.T + (a*a)[None, :]*E2[:, None] + (a*a)[:, None]*E2[None, :]
            + 4*np.outer(a, a)*M11 - 3*np.outer(a*a, a*a))
    v = E2 - a*a
    k22 = mu22 - np.outer(v, v) - 2*C11*C11
    k31 = mu31 - 3*v[:, None]*C11
    k4 = E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v
    return k22, k31, k4, mu21
def r2(x, y):
    x = x.ravel(); y = y.ravel(); a = float((x*y).sum()/(x*x).sum()); return 1 - float(((y - a*x)**2).sum()/(y*y).sum()), a
class Hook:
    def __init__(self, kp): self.kp = kp; self.lines = []; self.coef3 = []
    def __call__(self, l, wk4_row, wk4_mat, wk431, live):
        if wk4_mat is None or l < 1: return wk4_row, wk4_mat, wk431
        k22, k31, k4, _ = true_pair(self.kp, l)
        offm = ~np.eye(k22.shape[0], dtype=bool)
        wm = wk4_mat.detach().cpu().double().numpy(); wr = wk4_row.detach().cpu().double().numpy()
        r_m, a_m = r2(wm[offm], k22[offm]); r_r, a_r = r2(wr, k4)
        line = "l=%2d wk4_mat~κ₂₂ R² %.3f нахил %.3f | wk4_row~κ₄ R² %.3f нахил %.3f" % (l, r_m, a_m, r_r, a_r)
        if wk431 is not None:
            w3 = wk431.detach().cpu().double().numpy(); r_3, a_3 = r2(w3[offm], k31[offm]); line += " | wk431~κ₃₁ R² %.3f нахил %.3f" % (r_3, a_3)
        if mode == "fit" and live["D21"] is not None:
            C = live["C"].detach().double(); mu = live["mu"].detach().double(); var = live["var"].detach().double()
            D21 = live["D21"].detach().double(); D3 = live["D3"].detach().double(); k4r = wk4_row.detach().double()
            dev = C.device; m = torch.as_tensor(offm, device=dev)
            Coff = C.clone(); Coff.fill_diagonal_(0.0)
            modes = {"C": Coff, "D21": D21, "D21T": D21.T, "D21*D3/v": D21*(D3/var)[:, None], "D21*mu": D21*mu[:, None],
                     "C*D3": Coff*D3[:, None], "C*mu": Coff*mu[:, None], "C*k4": Coff*k4r[:, None], "C*var": Coff*var[:, None],
                     "C*muj": Coff*mu[None, :], "CC": Coff*Coff, "D21*k4/v2": D21*(k4r/var**2)[:, None],
                     "D21T*D3j/vj": D21.T*(D3/var)[None, :], "D21*(D3/v)^2": D21*(D3/var)[:, None]**2, "D21*D3/v*muj": D21*(D3/var)[:, None]*mu[None, :],
                     "D21*mu/v": D21*(mu/var)[:, None], "D21*C": D21*Coff, "D21*D3/v*varj": D21*(D3/var)[:, None]*var[None, :],
                     "D21*D3^2/v^2": D21*(D3*D3/var**2)[:, None], "C*D3^2/v": Coff*(D3*D3/var)[:, None], "C*D3*mu": Coff*(D3*mu)[:, None]}
            y = torch.as_tensor(k31, device=dev)[m]
            X = torch.stack([v[m] for v in modes.values()], 1)
            ss = float((y*y).sum())
            def fit(cols):
                Xc = X[:, cols]; sol = torch.linalg.lstsq(Xc, y[:, None]).solution[:, 0]; res = y - Xc @ sol
                return 1 - float((res*res).sum())/ss, sol
            names = list(modes)
            singles = " ".join("%s %.3f" % (nm, fit([k])[0]) for k, nm in enumerate(names))
            rall, sol = fit(list(range(len(names))))
            line += " || memoryless κ₃₁: " + singles + " | ВСІ %.3f" % rall
            r3, sol3 = fit([names.index(k) for k in MEM3]); line += " | 3 моди %.3f coef %s" % (r3, ["%.4g" % float(v) for v in sol3])
            self.coef3.append([float(v) for v in sol3])
        if mode == "mem" and live["D21"] is not None and wk431 is not None:
            C = live["C"].detach(); var = live["var"].detach(); D21 = live["D21"].detach(); D3 = live["D3"].detach(); k4r = wk4_row.detach()
            Coff = C.clone(); Coff.fill_diagonal_(0.0)
            mm = {"D21*D3/v": D21*(D3/var)[:, None], "C*k4": Coff*k4r[:, None], "C": Coff}
            a = MEMTAB[l]; new31 = sum(float(a[k])*mm[nm] for k, nm in enumerate(MEM3)); new31.fill_diagonal_(0.0)
            w3 = new31.detach().cpu().double().numpy(); r_n, a_n = r2(w3[offm], k31[offm]); line += " | mem κ₃₁ R² %.3f нахил %.3f" % (r_n, a_n)
            wk431 = new31.to(wk431.dtype)*W31 + (1-W31)*wk431
        self.lines.append(line)
        if mode in ("k22", "both"):
            wk4_mat = torch.as_tensor(np.where(offm, k22/A22, 0.0), dtype=wk4_mat.dtype, device=wk4_mat.device)*W22 + (1-W22)*wk4_mat
        if mode in ("k31", "both") and wk431 is not None:
            wk431 = torch.as_tensor(np.where(offm, k31/A31, 0.0), dtype=wk431.dtype, device=wk431.device)*W31 + (1-W31)*wk431
        return wk4_row, wk4_mat, wk431
MEM3 = ["D21*D3/v", "C*k4", "C"]
import json
MEMTAB = None
if os.environ.get("MEMCOEF"):
    _t = [json.load(open(f)) for f in os.environ["MEMCOEF"].split(",")]; MEMTAB = {}
    for l in _t[0]: MEMTAB[int(l)] = np.mean([t[l] for t in _t], axis=0)
D21USE = int(os.environ.get("D21USE", "0")); D21FEED = int(os.environ.get("D21FEED", "0")); D21R = int(os.environ.get("D21R", "0")); D21W = float(os.environ.get("D21W", "1"))
D21DIAG = int(os.environ.get("D21DIAG", "1"))
D21OFF = int(os.environ.get("D21OFF", "1")); D21PROJ = int(os.environ.get("D21PROJ", "0")); D21PBASIS = os.environ.get("D21PBASIS", "res")   # 1: також істинне D3 (власна κ₃); 0: лише позадіагональ
D21LMIN = int(os.environ.get("D21LMIN", "0")); D21LMAX = int(os.environ.get("D21LMAX", "99"))
OLDSKIP = int(os.environ.get("OLDSKIP", "0")); OLDAGE = int(os.environ.get("OLDAGE", "4")); K3REF = {}
def k3_stash(l, K3, D3, D21): K3REF["K3"] = K3
class D21Hook:
    """Аудит і оракул (2,1)-зрізу κ₃ передактивацій. Істина: mu21[i,j]=E[(p_i-a_i)^2(p_j-a_j)] (центральний = кумулянт для 3-го
    порядку), D3 = E3-3aE2+2a^3. Передбачення ДО прогону: R² його D21 на l=1 ≈ 0.99, до l=15 падає до ~0.9 (ε~2..10%);
    істинне D21 у use-side на regen: 2.20e-8 → 1.0…1.6e-8 (шанс ≥×1.5: 40%); +feed додає мало; +κ₃₁ разом → ≤5e-9 (шанс 30%)."""
    def __init__(self, kp): self.kp = kp; self.lines = []
    def __call__(self, l, D3, D21, live):
        self.last_l = l
        if l < 1: return D3, D21, None
        if not (D21LMIN <= l <= D21LMAX): return D3, D21, None   # вікно шарів оракула (атрибуція)
        if OLDSKIP and (l % OLDSKIP == 1) and "K3" in K3REF:
            # читання старого ярусу лише на шарах l % OLDSKIP == 0: тут зріз ЛИШЕ молодих джерел (вік < OLDAGE), use і feed
            K3 = K3REF["K3"]; young = [src for src in K3.sources if src.get("age", 0) < OLDAGE]
            D3y, D21y = K3.dslices(subset=young) if young else (D3*0, D21*0)
            D21y = D21y.clone(); D21y.fill_diagonal_(0.0); self.lines.append("l=%2d OLDSKIP: молодих %d з %d, ‖D21y‖/‖D21‖ %.3f" % (l, len(young), len(K3.sources), float(D21y.norm()/D21.norm())))
            return D3y, D21y, D21y
        kp = self.kp; a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64)
        _, _, _, mu21 = true_pair(kp, l); k3 = E3 - 3*a*E2 + 2*a**3
        n = mu21.shape[0]; offm = ~np.eye(n, dtype=bool)
        his = D21.detach().cpu().double().numpy(); hd3 = D3.detach().cpu().double().numpy()
        r_o, a_o = r2(his[offm], mu21[offm]); r_d, a_d = r2(hd3, k3)
        eps = float(np.sqrt(((his[offm]-mu21[offm])**2).sum()/(mu21[offm]**2).sum()))
        E4 = kp["pre_m4"][l].astype(np.float64); v = E2 - a*a; m4 = E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4
        N = float(kp["n_samples"]); nb = float(np.sqrt((np.outer(m4, v)[offm]).sum()/N/(mu21[offm]**2).sum()))
        line = "l=%2d D21~κ₂₁ R² %.4f нахил %.3f ε %.3f (шум банку %.3f) | D3~κ₃ R² %.4f нахил %.3f" % (l, r_o, a_o, eps, nb, r_d, a_d)
        tru = np.where(offm, mu21, 0.0)
        if D21PROJ:
            # низькорангова поправка: його D21 + проєкція нев'язки R=істина−його на топ-r підпростір
            # res: власний SVD нев'язки (оракульний базис) · hisR: правий (j) топ-r підпростір ЙОГО D21 · hisL: лівий (i)
            Rm = np.where(offm, mu21 - his, 0.0)
            if D21PBASIS == "res":
                U, S, Vh = np.linalg.svd(Rm); corr = (U[:, :D21PROJ]*S[:D21PROJ]) @ Vh[:D21PROJ]
            else:
                U, S, Vh = np.linalg.svd(np.where(offm, his, 0.0))
                corr = Rm @ Vh[:D21PROJ].T @ Vh[:D21PROJ] if D21PBASIS == "hisR" else U[:, :D21PROJ] @ (U[:, :D21PROJ].T @ Rm)
            tru = np.where(offm, his + corr, 0.0)
            line += " | proj %s r=%d: частка ‖R‖² %.3f" % (D21PBASIS, D21PROJ, float((corr**2).sum()/(Rm**2).sum()))
        if D21R:
            U, S, Vh = np.linalg.svd(tru); tru = np.where(offm, (U[:, :D21R]*S[:D21R]) @ Vh[:D21R], 0.0)
            line += " | truth rank %d energy %.5f" % (D21R, float((S[:D21R]**2).sum()/(S**2).sum()))
        self.lines.append(line)
        T21 = (torch.as_tensor(tru, dtype=D21.dtype, device=D21.device)*D21W + (1-D21W)*D21) if D21OFF else D21
        T3 = torch.as_tensor(k3, dtype=D3.dtype, device=D3.device)*D21W + (1-D21W)*D3 if D21DIAG else D3
        D3u, D21u = (T3, T21) if D21USE else (D3, D21)
        return D3u, D21u, (T21 if D21FEED else None)
A22 = float(os.environ.get("A22", "1")); A31 = float(os.environ.get("A31", "1"))   # одиниці: κ_true/A = wk (з пломби rec)
class FitC:   # G_off ≈ λ_l C_off (пас 1) / replay (пас 2)
    def __init__(self): self.coef = []
    def __call__(self, l, G, live):
        n = G.shape[0]; m = ~torch.eye(n, dtype=torch.bool, device=G.device); y = zd(G)[m]; x = zd(live["C"])[m]
        a = float((x*y).sum()/(x*x).sum()); self.coef.append(a); return torch.diag(torch.diagonal(G)) + a*zd(live["C"])
class Replay:
    def __init__(self, coef): self.coef = coef
    def __call__(self, l, G, live):
        lk.REGEN_STATE["lam"] = float(self.coef[l]); lk.REGEN_STATE["g"] = torch.diagonal(G).clone()
        return torch.diag(torch.diagonal(G)) + self.coef[l]*zd(live["C"])
K31FEED = float(os.environ.get("K31FEED", "0")); K4FEED = float(os.environ.get("K4FEED", "0"))
class FeedTrue:
    """g_feed_hook: G_off ← κ₃₁_true (wk431 = 0.5·c·G_off, c=2 ⟹ G_off = wk431 ≈ κ₃₁, нахил ~1 на l=1), diag(G) ← κ₄_true/c.
    Замінює memoryless-регенерацію ЛИШЕ у народженні κ₃ (feed), use-side лишається його. Питання: чи нев'язка D21 (ε 3..7%,
    низькорангова, j-домінантна) походить з неточного κ₃₁ у feed. Передбачення: ε падає до ~шуму банку на l≤8 (шанс 40%)."""
    def __init__(self, kp, dh): self.kp = kp; self.dh = dh
    def __call__(self, G, C_pre_off):
        l = getattr(self.dh, "last_l", 0); Gp = approx_pre(G, C_pre_off)
        if l < 1: return Gp
        k22, k31, k4, _ = true_pair(self.kp, l); n = G.shape[0]; offm = ~np.eye(n, dtype=bool)
        dg = torch.diagonal(Gp).clone(); off = zd(Gp)
        if K4FEED: dg = torch.as_tensor(k4/2.0, dtype=G.dtype, device=G.device)*K4FEED + (1-K4FEED)*dg
        if K31FEED: off = torch.as_tensor(np.where(offm, k31/A31, 0.0), dtype=G.dtype, device=G.device)*K31FEED + (1-K31FEED)*off
        return torch.diag(dg) + off
def approx_pre(G, C_pre_off):
    lam = lk.REGEN_STATE.get("lam")
    return G if lam is None else torch.diag(torch.diagonal(G)) + lam*zd(C_pre_off)
if os.environ.get("ARM_DROP"):
    _a = os.environ["ARM_DROP"].split(":"); lk.ARM_DROP = (int(_a[0]), tuple(_a[1].split(",")))
if __name__ == "__main__":
    statics = None; mses = []
    for i in dumps:
        d = np.load(DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(BANK.format(i))
        if statics is None: statics = lk.build_static_coefs(w.shape[1])
        hook = Hook(kp); dh = D21Hook(kp); t0 = time.perf_counter(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = k3_stash if OLDSKIP else None
        if cfg == "dense":
            lk.WK_HOOK = hook; lk.D21_HOOK = dh
            pred = lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True)
        elif cfg == "regen":
            fit = FitC()
            lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, g_hook=fit)
            lk.WK_HOOK = hook; lk.D21_HOOK = dh
            pred = lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=Replay(fit.coef),
                                      g_feed_hook=(FeedTrue(kp, dh) if (K31FEED or K4FEED) else approx_pre), g_use_hook=approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True,
                                      leg_trunc={"rres": 16, "d21": 16}, fac_schedule=(eval(os.environ["FAC"]) if os.environ.get("FAC") else None))
        else: raise SystemExit("cfg?")
        pred = pred.detach().cpu().double().numpy(); mse = float(np.mean((pred[-1] - gt)**2)); mses.append(mse)
        print("[%s/%s] dump%d mse_final=%.4e t=%.1fs" % (cfg, mode, i, mse, time.perf_counter()-t0), flush=True)
        for ln in hook.lines: print("   ", ln, flush=True)
        for ln in dh.lines: print("   ", ln, flush=True)
        if mode == "fit" and hook.coef3: json.dump({str(l+1): c for l, c in enumerate(hook.coef3)}, open("memcoef_%d.json" % i, "w"))
    print("[%s/%s] MEAN mse_final=%.4e over %s" % (cfg, mode, float(np.mean(mses)), dumps), flush=True)
