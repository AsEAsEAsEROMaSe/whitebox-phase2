"""Структура нев'язки D21 його ланцюга проти істини (regen=V25, use-side без підміни).
Питання: нев'язка R = κ₂₁_true − D21_his (позадіагональ) — низькорангова (пропущений структурний член) чи плоска
(дрібна частина/шум)? Лежить у топ-r підпросторі істини чи ізотропна? Пояснюється memoryless-модами стану?
Передбачення ДО прогону: спектр R плоский (топ-64 < 25% енергії), у топ-256 істини < 40% ⟹ ізотропна; моди R² < 0.3."""
import os, sys, numpy as np, torch, time
sys.argv = [sys.argv[0]] + (sys.argv[1:] if len(sys.argv) > 1 else ["0"])
import lean_oracle as lo, lean_k3_aug_gpu as lk
dumps = [int(x) for x in sys.argv[1].split(",")]
class ResHook:
    def __init__(self, kp): self.kp = kp; self.lines = []
    def __call__(self, l, D3, D21, live):
        if l < 1: return D3, D21, None
        kp = self.kp; _, _, _, mu21 = lo.true_pair(kp, l)
        n = mu21.shape[0]; dev = D21.device
        T = torch.as_tensor(mu21, dtype=torch.float64, device=dev); T.fill_diagonal_(0.0)
        H = D21.detach().double().clone(); H.fill_diagonal_(0.0)
        R = T - H; m = ~torch.eye(n, dtype=torch.bool, device=dev)
        eT = float((T*T).sum()); eR = float((R*R).sum()); eH = float((H*H).sum())
        sR = torch.linalg.svdvals(R); sT = torch.linalg.svdvals(T)
        UT, ST, VhT = torch.linalg.svd(T)
        fr = lambda s, r: float((s[:r]**2).sum()/(s**2).sum())
        # частка енергії R у топ-r підпросторі істини (ліва і права проєкція)
        def inT(r):
            P = UT[:, :r] @ (UT[:, :r].T @ R) ; Q = (R @ VhT[:r].T) @ VhT[:r]
            return float((P*P).sum()/eR), float((Q*Q).sum()/eR)
        # антисиметрія/симетрія нев'язки, кореляція з транспонованою істиною
        sym = float(((R+R.T)/2).pow(2).sum()/eR)
        # регресія R на moди стану
        C = live["C"].detach().double(); mu = live["mu"].detach().double(); var = live["var"].detach().double(); D3d = D3.detach().double()
        Coff = C.clone(); Coff.fill_diagonal_(0.0)
        modes = {"C": Coff, "H": H, "HT": H.T, "T_T": T.T, "C*D3/v": Coff*(D3d/var)[:, None], "C*D3j/vj": Coff*(D3d/var)[None, :],
                 "C*mu": Coff*mu[:, None], "CC": Coff*Coff, "H*mu": H*mu[:, None], "C*var": Coff*var[:, None], "C*varj": Coff*var[None, :],
                 "CCT": Coff@Coff, "C*C*D3j": (Coff*Coff)*(D3d/var)[None, :], "H*C": H*Coff}
        y = R[m]; X = torch.stack([v[m] for v in modes.values()], 1); ss = float((y*y).sum())
        def fit(cols):
            Xc = X[:, cols]; sol = torch.linalg.lstsq(Xc, y[:, None]).solution[:, 0]; res = y - Xc @ sol
            return 1 - float((res*res).sum())/ss
        names = list(modes); singles = " ".join("%s %.3f" % (nm, fit([k])) for k, nm in enumerate(names)); rall = fit(list(range(len(names))))
        a1, b1 = inT(64); a2, b2 = inT(256); a3, b3 = inT(512)
        line = ("l=%2d ε %.3f ‖H‖/‖T‖ %.4f | спектр R топ16 %.3f топ64 %.3f топ256 %.3f (істина: %.3f %.3f %.3f) | R у топ-r істини L/R: 64 %.2f/%.2f 256 %.2f/%.2f 512 %.2f/%.2f | sym %.2f | моди: %s | ВСІ %.3f"
                % (l, np.sqrt(eR/eT), np.sqrt(eH/eT), fr(sR,16), fr(sR,64), fr(sR,256), fr(sT,16), fr(sT,64), fr(sT,256), a1,b1,a2,b2,a3,b3, sym, singles, rall))
        self.lines.append(line)
        if os.environ.get("SAVE"): self.saved = getattr(self, "saved", {}); self.saved["his_%d" % l] = H.float().cpu().numpy(); self.saved["d3_%d" % l] = D3.detach().float().cpu().numpy()
        return D3, D21, None
class OldProbe:
    """K3_PROBE: на шарі l зріз ЛИШЕ старих джерел (ті, що існували на попередньому шарі) = його лінійний перенос κ₃(p_{l-1});
    народження = повний зріз − старий."""
    def __init__(self): self.prev = 0; self.saved = {}
    def __call__(self, l, K3, D3, D21):
        cnt = len(K3.sources)
        if self.prev > 0 and l >= 1:
            D3o, D21o = K3.dslices(subset=K3.sources[:self.prev])
            D21o = D21o.clone(); D21o.fill_diagonal_(0.0)
            self.saved["old_%d" % l] = D21o.float().cpu().numpy(); self.saved["d3old_%d" % l] = D3o.float().cpu().numpy(); self.saved["nsrc_%d" % l] = np.array([self.prev, cnt])
        self.prev = cnt
statics = None
for i in dumps:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; kp = np.load(lo.BANK.format(i))
    if statics is None: statics = lk.build_static_coefs(w.shape[1])
    lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; t0 = time.perf_counter()
    fit = lo.FitC(); lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, g_hook=fit)
    rh = ResHook(kp); lk.D21_HOOK = rh; gt = d["final_means"].astype(np.float64)
    op = OldProbe(); lk.K3_PROBE = op if os.environ.get("SAVE") else None
    LT = {"rres": int(os.environ.get("RRES", "16")), "d21": int(os.environ.get("D21LEG", "16"))}
    if os.environ.get("CFG", "regen") == "dense":
        pred = lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True)
    else:
        pred = lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef),
                       g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc=LT)
    pred = pred.detach().cpu().double().numpy(); mse = float(np.mean((pred[-1] - gt)**2))
    print("[res %s LT=%s] dump%d mse_final=%.4e t=%.1fs" % (os.environ.get("CFG", "regen"), LT, i, mse, time.perf_counter()-t0), flush=True)
    for ln in rh.lines: print("   ", ln, flush=True)
    if os.environ.get("SAVE"): np.savez(os.environ["SAVE"] % i, **rh.saved, **op.saved)
