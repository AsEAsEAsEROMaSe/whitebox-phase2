"""Пломба 30.09 (ідея «обрізання за метрикою споживача»): чи чутливість виходу до D21 на шарі L анізотропна НЕ за спектром D21.
Збурення δ однакової норми Фробеніуса (eps·‖D21‖_F) уздовж: top-k сингулярних напрямків D21, напрямків 500..500+k, випадкових.
Якщо ΔMSE(хвіст/випадкові) ≫ ΔMSE(топ) ⟹ енергетичне обрізання тримає не те; метрика має бути зваженою чутливістю.
Передбачення: хвіст/топ ≥ 3 (шанс 50%)."""
import os, sys, time, numpy as np, torch
L = int(os.environ.get("L", "12")); EPS = float(os.environ.get("EPS", "0.03")); K = 16
sys.argv = sys.argv[:1]; sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32
for i in [0, 1, 2]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["all_layer_means"].astype(np.float64)
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K_POST_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
    def run(mode, seed=0):
        cap = {}
        def hook(l, D3, D21, live):
            if l != L: return D3, D21, None
            U, S, Vh = torch.linalg.svd(D21)
            nrm = D21.norm(); g = torch.Generator(device=D21.device); g.manual_seed(seed)
            if mode == "base": return D3, D21, None
            if mode == "top": delta = (U[:, :K] * S[:K]) @ Vh[:K]
            elif mode == "tail": delta = (U[:, 500:500+K] * S[:K]) @ Vh[500:500+K]   # хвостові напрямки з топовою вагою
            elif mode == "mid": delta = (U[:, 128:128+K] * S[:K]) @ Vh[128:128+K]
            elif mode == "rand": delta = torch.randn(D21.shape, generator=g, device=D21.device, dtype=D21.dtype)
            delta = delta * (EPS * nrm / delta.norm())
            cap["S"] = S[[0, K, 128, 500]].cpu().numpy()
            return D3, D21 + delta, None
        lk.D21_HOOK = hook; lk.REGEN_STATE.clear()
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG).detach().cpu().double().numpy()
        lk.D21_HOOK = None
        return ((p - gt) ** 2).mean(1)[-1], cap.get("S")
    b, _ = run("base")
    out = {}
    for mode in ["top", "mid", "tail", "rand"]:
        t0 = time.time(); m, S = run(mode); out[mode] = m / b
        print("dump %d L %d eps %.2f %-4s mse×%.4f  S=%s (%.0f с)" % (i, L, EPS, mode, m / b, S, time.time() - t0), flush=True)
    print("dump %d SUMMARY top %.4f mid %.4f tail %.4f rand %.4f" % (i, out["top"], out["mid"], out["tail"], out["rand"]), flush=True)
    torch.cuda.empty_cache()
print("DONE_SENS21", flush=True)
