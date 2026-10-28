"""Атрибуція решти оракула D21 (×1.6) у ПЕРЕНОСІ (стенд 26.09): щошару l вкидаємо нев'язку зрізу
R21 = D21_true(l) − D21_ours(l), R3 = D3_true − D3_ours як нове CP-джерело (M⊗P⊗P, P = I) у стан κ₃ ПЕРЕД wick_,
далі воно несеться W і живить усі споживачі. D21Hook друкує ε(l) ДО вкидання ⟹ бачимо, чи нев'язка шару l+1
є переносом нев'язки шару l. Режими: base · zero (пломба: джерело з M=0 ⟹ MSE = base біт у біт) · inj<lmin>
(вкидання з шару lmin, use-side наш) · injuse<lmin> (вкидання + істинний D21 у use) · use (лише use, відоме 1.31e-8).
Передбачення ДО прогону: ε(l+1) при inj падає лише на 20–40% (решта нев'язки народжується з (1,1,1)-частини
щошару), MSE inj ≈ ×1.1–1.2, injuse ≈ use ×1.05 (шанс, що inj дає ≥×1.4: 25%)."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
modes = os.environ.get("MODES", "base,zero,inj1,inj4,injuse1,use").split(",")
LMAX = int(os.environ.get("INJLMAX", "99"))


class Inj:
    def __init__(self, kp, lmin, zero=False):
        self.kp, self.lmin, self.zero, self.scale, self.lines = kp, lmin, zero, None, []

    def __call__(self, l, K3, D3, D21):
        if l < self.lmin or l > LMAX or D21 is None:
            return
        kp = self.kp; a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64)
        k3 = E3 - 3*a*E2 + 2*a**3; mu21 = lo.true_pair(kp, l)[3]
        n = D21.shape[0]; dev = D21.device
        T21 = torch.as_tensor(mu21, dtype=torch.float64, device=dev); T21.fill_diagonal_(0.0)
        T3 = torch.as_tensor(k3, dtype=torch.float64, device=dev)
        R21 = T21 - D21.double(); R21.fill_diagonal_(0.0); R3 = T3 - D3.double()
        if self.zero:
            R21.zero_(); R3.zero_()
        eye = torch.eye(n, dtype=D21.dtype, device=dev)
        mk = lambda M: dict(X1=None, Y1=None, X2=None, Y2=None, X3=None, Y3=None, X4=None, Y4=None, M=M.to(D21.dtype), P=eye.clone(), age=0)
        if self.scale is None and not self.zero:
            M0 = R21.T.clone(); M0.fill_diagonal_(0.0); M0 = M0 + torch.diag(R3)
            d3s, d21s = K3.dslices(subset=[mk(M0)]); d3s = d3s.double(); d21s = d21s.double()
            s_off = float((d21s*R21).sum()/(d21s*d21s).sum()); s_d = float((d3s*R3).sum()/(d3s*d3s).sum())
            self.scale = (s_off, s_d)
            M = R21.T*s_off; M.fill_diagonal_(0.0); M = M + torch.diag(R3*s_d)
            d3s, d21s = K3.dslices(subset=[mk(M)]); d3s = d3s.double(); d21s = d21s.double()
            self.lines.append("l=%2d калібровка: s_off %.6f s_diag %.6f | пломба dslices(джерело) vs R: D21 %.2e D3 %.2e" % (
                l, s_off, s_d, float((d21s-R21).norm()/R21.norm()), float((d3s-R3).norm()/R3.norm())))
        s_off, s_d = self.scale if self.scale else (1.0, 1.0)
        M = R21.T*s_off; M.fill_diagonal_(0.0); M = M + torch.diag(R3*s_d)
        K3.sources.append(mk(M))


R = {}; EPS = {}
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(lo.BANK.format(i))
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for nm in modes:
        lk.REGEN_STATE.clear(); t0 = time.time()
        lo.D21USE = 1 if nm.startswith("injuse") or nm == "use" else 0
        dh = lo.D21Hook(kp); lk.D21_HOOK = dh
        inj = None
        if nm == "zero": inj = Inj(kp, 1, zero=True)
        elif nm.startswith("injuse"): inj = Inj(kp, int(nm[6:]))
        elif nm.startswith("inj"): inj = Inj(kp, int(nm[3:]))
        lk.K3_INJ = inj
        hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
        mse = float(np.mean((p[-1].detach().cpu().double().numpy() - gt)**2)); R.setdefault(nm, []).append(mse)
        eps = [float(ln.split("ε ")[1].split()[0]) for ln in dh.lines if "ε " in ln]; EPS.setdefault(nm, []).append(eps)
        print("dump %d %-8s %.4e  (%.0f с)  ε: %s" % (i, nm, mse, time.time() - t0, " ".join("%.3f" % e for e in eps)), flush=True)
        if inj is not None:
            for ln in inj.lines: print("   ", ln, flush=True)
        lk.K3_INJ = None; lk.D21_HOOK = None; torch.cuda.empty_cache()
for k in modes:
    print("MEAN %-8s %.4e  ε сер по дампах: %s" % (k, np.mean(R[k]), " ".join("%.3f" % e for e in np.mean(EPS[k], axis=0))), flush=True)
