"""17.10.2026, ідея юзера «антитетика для кумулянтів»: два ланцюги з ПРОТИЛЕЖНИМ обрізанням/зсувом, суміш з фітом ваги (LOO по дампах).
Пари дзеркал (кожна дає p_plus, p_minus на дампі; база reg = ланцюг з λ-законом):
  lam  : λ_l ← λ_l·(1±δ)        (сила регенерованого κ4 поза діагоналлю)
  k4   : діагональ κ4 g ← g·(1±δ) (A4)
  sign : W ← −W                   🔒 пломба: має дати БІТ У БІТ ту саму відповідь (ReLU-мережа без біасів: p = h@W, −W міняє знак p,
                                 ReLU(−p) ≠ ReLU(p), тож НЕ тотожно; це контроль, що «дзеркало» справді інше)
Суміш: pred = (1−a)·p_plus + a·p_minus, a фітиться LOO (7 дампів фіт, 1 оцінка). Також a=0.5 (чиста антитетика).
Передбачення ДО прогону: lam δ=0.3: суміш ×1.00…1.03 (λ уже на оптимумі, симетричне збурення гаситься в 2-му порядку);
k4 δ=0.5: ×1.00…1.05; sign: MSE ×0.5…2 (інша мережа), суміш ×1.1…1.3 як «дві мережі» (шанс 20%)."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dev = lk.DEV; dt = torch.float32
dumps = [int(x) for x in sys.argv[1].split(",")]; pair = sys.argv[2]; delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
st = None; P = {}; GT = {}
class ReplayS(lo.Replay):
    def __init__(self, coef, s): super().__init__(coef); self.s = s
    def __call__(self, l, G, live):
        lk.REGEN_STATE["lam"] = float(self.coef[l])*self.s; lk.REGEN_STATE["g"] = torch.diagonal(G).clone()
        return torch.diag(torch.diagonal(G)) + self.coef[l]*self.s*lo.zd(live["C"])
def chain(w, lam_s=1.0, g_s=1.0):
    lk.REGEN_STATE.clear(); lk.C_HOOK = None; lk.D21_HOOK = None; lk.WK_HOOK = None; lk.K3_PROBE = None
    fit = lo.FitC(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    if g_s != 1.0:
        def wk(l, wr, wm, w431, live): return (wr*g_s if wr is not None else wr), wm, w431
        lk.WK_HOOK = wk
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, aug_no_k211=True, g_hook=ReplayS(fit.coef, lam_s), g_feed_hook=lo.approx_pre,
                           g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    lk.WK_HOOK = None
    return p[-1].detach().cpu().double().numpy()
for i in dumps:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); GT[i] = gt
    if st is None: st = lk.build_static_coefs(w.shape[1])
    base = chain(w)
    if pair == "lam": pp, pm = chain(w, lam_s=1+delta), chain(w, lam_s=1-delta)
    elif pair == "k4": pp, pm = chain(w, g_s=1+delta), chain(w, g_s=1-delta)
    elif pair == "sign": pp, pm = base, chain(-w)
    else: raise SystemExit("pair?")
    P[i] = (base, pp, pm)
    m = lambda p: float(np.mean((p-gt)**2))
    print("d%d base %.4e · plus %.4e · minus %.4e · a=0.5 %.4e (×%.3f)" % (i, m(base), m(pp), m(pm), m(0.5*(pp+pm)), m(base)/m(0.5*(pp+pm))), flush=True)
# LOO фіт a
res = []
for i in dumps:
    num = den = 0.0
    for j in dumps:
        if j == i: continue
        b, pp, pm = P[j]; x = pm - pp; r = GT[j] - pp; num += float((x*r).sum()); den += float((x*x).sum())
    a = num/den; b, pp, pm = P[i]; mix = (1-a)*pp + a*pm
    mb, mm = float(np.mean((b-GT[i])**2)), float(np.mean((mix-GT[i])**2)); res.append((mb, mm, a))
    print("LOO d%d a=%.3f: base %.4e → mix %.4e (×%.3f)" % (i, a, mb, mm, mb/mm), flush=True)
mb = np.mean([r[0] for r in res]); mm = np.mean([r[1] for r in res])
print("MEAN pair=%s δ=%.2f: base %.4e → mix %.4e  ×%.4f  кращих %d/%d  a=%.3f±%.3f" % (pair, delta, mb, mm, mb/mm, sum(r[1] < r[0] for r in res), len(res), np.mean([r[2] for r in res]), np.std([r[2] for r in res])), flush=True)
