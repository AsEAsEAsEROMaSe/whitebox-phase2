"""01.10: парні блоки ЛАНЦЮГА на шарі ls, рядки/стовпці S (32): mu, var, C, D21, D3, K4/K22/K31 (memoryless wk4) → npz."""
import os, sys, numpy as np, torch
_a = sys.argv; sys.argv = sys.argv[:1]
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import lean_oracle as lo; lk = lo.lk
sys.argv = _a; torch.set_num_threads(2); dt = torch.float32; dev = lk.DEV; m = 32
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
class Stop(Exception): pass
for arg in sys.argv[1:]:
    i, ls = [int(x) for x in arg.split(":")]
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; n = w.shape[1]
    rng = np.random.default_rng(11 + i); S = np.sort(rng.choice(n, m, replace=False)); St = torch.tensor(S, device=dev)
    st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K3_PROBE = None; lk.K3_PROBE_POST = None; lk.K_POST_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    got = {}
    sub = lambda M: M[St][:, St].double().cpu().numpy()
    def d21hook(l, D3, D21, live):
        if l == ls: got.update(mu=live["mu"][St].double().cpu().numpy(), var=live["var"][St].double().cpu().numpy(), C=sub(live["C"]), D3=D3[St].double().cpu().numpy(), D21=sub(D21))
        return D3, D21, None
    def wkhook(l, wk4_row, wk4_mat, wk431, live):
        if l == ls:
            got.update(K4=wk4_row[St].double().cpu().numpy(), K22=sub(wk4_mat), K31=sub(wk431) if wk431 is not None else np.zeros((m, m))); raise Stop()
        return wk4_row, wk4_mat, wk431
    lk.D21_HOOK = d21hook; lk.WK_HOOK = wkhook; lk.REGEN_STATE.clear()
    try:
        lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG)
    except Stop:
        pass
    lk.D21_HOOK = None; lk.WK_HOOK = None
    np.savez("/home/trim/wb2data/blocksS_%d_%d.npz" % (i, ls), S=S, **got); print("saved", i, ls, {k: v.shape for k, v in got.items()}, flush=True)
print("DONE_BLOCKS", flush=True)
