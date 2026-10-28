"""Стенд 30.09: пам'ять моря κ₄ (Sea4) у ланцюзі REG на дампах. base проти mem (WINDOW шарів пам'яті).
Передбачення ДО прогону: оракул D21 ×1.68 = стеля; носій пояснює ~65% дисперсії зрізу (2,1,1) ⟹ MSE ×1.15–1.4 (шанс 35%);
альтернатива ≤×1.03 як d4 (одношаговий k_max=4)."""
import os, sys, time, numpy as np, torch
_a = sys.argv; sys.argv = sys.argv[:1]
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import lean_oracle as lo; lk = lo.lk; import sea4, d4dense
sys.argv = _a
dt = torch.float32; torch.set_num_threads(2)
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
modes = os.environ.get("MODES", "base,mem").split(","); WINDOW = int(os.environ.get("WINDOW", "6"))
R = {}
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); n = w.shape[1]
    st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.D4 = None; lk.K3_PROBE = None; lk.K3_INJ = None; lk.K_POST_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for nm in modes:
        lk.REGEN_STATE.clear(); t0 = time.time()
        if nm.startswith("mem") or nm == "pairs":
            pfn = None
            if "pairs" in nm:
                pfn = sea4.PairFn(os.environ.get("PAIRFIT", "/home/trim/wb2data/pairfit_%s_%d.npz" % ("pairs_sea3" if "mem" in nm.replace("mem31", "mem") and nm != "pairs" else "pairs", i)), lk.DEV)
            S4 = sea4.Sea4(n, lk.DEV, window=WINDOW, pairfn=pfn, use_S=(nm != "pairs")); lk.D4 = S4
            d4dense.D4SCALE = float(os.environ.get("D4SCALE", "1"))
            def probe(l, K3, D3, D21): S4.stash(l, K3, D21)
            def d21hook(l, D3, D21, live):
                wk = lk.WickCache(live["mu"], live["var"]); S4.set_w(l, {k: wk(k, 1) for k in (1, 2, 3)}, lk.zero_diag_(live["C"].clone()), mu=live["mu"].clone())
                return D3, D21, None
            lk.K3_PROBE = probe; lk.D21_HOOK = d21hook
            if nm.startswith("mem31"):
                K31FAC = float(os.environ.get("K31FAC", "1"))
                def wkhook(l, wk4_row, wk4_mat, wk431, live):
                    if wk431 is None or S4.cur is None or S4.cur["l"] != l or l not in S4.LAY or "w" not in S4.LAY[l]:
                        return wk4_row, wk4_mat, wk431
                    S4.build_S(); K = S4.k31_mem()
                    if K is not None:
                        S4.log("  mem31 l=%d: ‖wk431‖ %.3e ‖K31_mem‖ %.3e cos %+.3f" % (l, wk431.norm(), K.norm(), float((wk431.float() * K).sum() / wk431.norm() / K.norm())))
                        wk431 = wk431 + K31FAC * K.to(wk431.dtype)
                    return wk4_row, wk4_mat, wk431
                lk.WK_HOOK = wkhook
            else:
                lk.WK_HOOK = None
        else:
            lk.D4 = None; lk.K3_PROBE = None; lk.D21_HOOK = None
        hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
        mse = float(np.mean((p[-1].detach().cpu().double().numpy() - gt) ** 2))
        R.setdefault(nm, []).append(mse)
        print("dump %d %-6s mse %.4e  (%.0f с)%s" % (i, nm, mse, time.time() - t0,
              "" if lk.D4 is None else "  ×base %.4f" % (R["base"][-1] / mse if "base" in R else float("nan"))), flush=True)
        lk.D4 = None; lk.K3_PROBE = None; lk.D21_HOOK = None; lk.WK_HOOK = None; torch.cuda.empty_cache()
print("DONE_SEAFEED", {k: ["%.4e" % v for v in vs] for k, vs in R.items()}, flush=True)
