"""Похибка народження κ₃ на кроці при ОРАКУЛЬНИХ входах. Ланцюг regen з хуками: D21FEED (істинні D21/D3 у народження),
K4FEED/K31FEED (істинні κ₄/κ₃₁ у народження), D21USE (істинні у використання ⟹ майже точний стан). Зберігає his_l (власний зріз
ланцюга), old_l (зріз перенесених джерел) для onestep.py (розклад R на перенос і народження).
Передбачення ДО прогону: народження (birth−his_birth) 2.9–3.2% T у базі; при feed+k4+k31 ~2.5% (мало); при useall ~2% ⟹ формула кроку
губить ~2% T на шар незалежно від входів (шанс 60%); альтернатива: падає до <1% ⟹ уся похибка від входів (40%)."""
import os, sys, numpy as np, torch, time
TAG = os.environ["TAG"]; dumps = [int(x) for x in sys.argv[1].split(",")]; sys.argv = [sys.argv[0]]
import lean_oracle as lo, lean_k3_aug_gpu as lk
class OldProbe:
    def __init__(self): self.prev = 0; self.saved = {}
    def __call__(self, l, K3, D3, D21):
        cnt = len(K3.sources)
        if self.prev > 0 and l >= 1:
            D3o, D21o = K3.dslices(subset=K3.sources[:self.prev]); D21o = D21o.clone(); D21o.fill_diagonal_(0.0)
            self.saved["old_%d" % l] = D21o.float().cpu().numpy(); self.saved["d3old_%d" % l] = D3o.float().cpu().numpy(); self.saved["nsrc_%d" % l] = np.array([self.prev, cnt])
        self.prev = cnt
class SaveHook:
    def __init__(self, dh): self.dh = dh; self.saved = {}
    def __call__(self, l, D3, D21, live):
        H = D21.detach().clone(); H.fill_diagonal_(0.0); self.saved["his_%d" % l] = H.float().cpu().numpy(); self.saved["d3_%d" % l] = D3.detach().float().cpu().numpy()
        return self.dh(l, D3, D21, live)
statics = None
for i in dumps:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(lo.BANK.format(i))
    if statics is None: statics = lk.build_static_coefs(w.shape[1])
    lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None; t0 = time.perf_counter()
    fit = lo.FitC(); lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, g_hook=fit)
    dh = lo.D21Hook(kp); sh = SaveHook(dh); op = OldProbe(); lk.D21_HOOK = sh; lk.K3_PROBE = op
    pred = lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef),
                              g_feed_hook=(lo.FeedTrue(kp, dh) if (lo.K31FEED or lo.K4FEED) else lo.approx_pre), g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    lk.D21_HOOK = None; lk.K3_PROBE = None
    pred = pred.detach().cpu().double().numpy(); mse = float(np.mean((pred[-1] - gt)**2))
    print("[%s] dump%d mse_final=%.4e t=%.0fs" % (TAG, i, mse, time.perf_counter() - t0), flush=True)
    for ln in dh.lines:
        if ln.startswith(("l= 1 ", "l= 8 ", "l=15 ")): print("   ", ln[:200], flush=True)
    np.savez("/home/trim/wb2lean/his_%s_dump%d.npz" % (TAG, i), **sh.saved, **op.saved)
