"""17.10.2026, ідея юзера «проєктор критичної ентропії»: усі джерела κ3 віку ≥1 щошару проєктуються на k напрямків стану h_l,
які найсильніше доходять до виходу (ліві сингулярні вектори лінеаризованого оператора J_l = W_{l+1} D_{l+1} ... W_15 D_15,
Φ з живих дампів reg-ланцюга, без істини). Контролі при тому самому k: базис за енергією ніг (штатний confine_old_) і випадковий.
Ланцюг dense (k4_aug=True), 8 дампів. Вихід: MSE по режимах.
Передбачення ДО прогону: crit k=256 ×0.9–1.0 від бази (джерела несуть поправку до середнього через нелінійні зрізи, а не лінійно),
crit кращий за енергетичний при тому самому k (шанс 60%); при k=128 усі режими гірші за базу ×1.3+."""
import os, sys, math, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dev = lk.DEV; dt = torch.float32
dumps = [int(x) for x in sys.argv[1].split(",")]; ks = [int(x) for x in sys.argv[2].split(",")]
st = None; res = {}
for i in dumps:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); NL, n = w.shape[0], w.shape[1]
    if st is None: st = lk.build_static_coefs(n)
    W = [torch.tensor(w[l], dtype=torch.float64, device=dev) for l in range(NL)]
    Ph = []
    for l in range(NL):
        z = np.load('/home/trim/wblogs/live/d%d_l%02d.npz' % (i, l)); mu = torch.tensor(z['mu'], dtype=torch.float64, device=dev)
        C = torch.tensor(z['C'], dtype=torch.float64, device=dev); t = mu / torch.diagonal(C).clamp_min(1e-30).sqrt()
        Ph.append(0.5 * (1 + torch.erf(t / math.sqrt(2))))
    U = {NL-1: torch.linalg.svd(torch.diag(Ph[NL-1]))[0].float()}; J = torch.diag(Ph[NL-1])
    for l in range(NL-2, -1, -1):
        J = W[l+1] @ J; U[l] = torch.linalg.svd(J, full_matrices=False)[0].float(); J = torch.diag(Ph[l]) @ J
    g = torch.Generator(device=dev).manual_seed(1000 + i)
    R = {l: torch.linalg.qr(torch.randn(n, n, generator=g, device=dev))[0] for l in range(NL)}
    CUR = [0]
    def probe(l, K3, D3, D21): CUR[0] = l
    def run(mode, k):
        lk.REGEN_STATE.clear(); lk.C_HOOK = None; lk.D21_HOOK = None; lk.WK_HOOK = None
        lk.K3_PROBE = probe
        if mode == "crit": lk.CONFINE_Q_FN = lambda G, r: U[CUR[0]][:, :r].to(G.dtype)
        elif mode == "rand": lk.CONFINE_Q_FN = lambda G, r: R[CUR[0]][:, :r].to(G.dtype)
        else: lk.CONFINE_Q_FN = None
        kw = dict(k4_aug=True)
        if mode != "base": kw["age_rank"] = (1, k)
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, **kw)[-1].detach().cpu().double().numpy()
        lk.CONFINE_Q_FN = None; lk.K3_PROBE = None
        return float(np.mean((p - gt) ** 2))
    b = run("base", 0); res.setdefault(("base", 0), []).append(b)
    line = "d%d base %.4e" % (i, b)
    for k in ks:
        for mode in ("crit", "energy", "rand"):
            m = run(mode, k); res.setdefault((mode, k), []).append(m); line += " | %s%d %.4e (×%.3f)" % (mode, k, m, b / m)
    print(line, flush=True)
B = np.mean(res[("base", 0)])
for key, v in sorted(res.items()): print("MEAN %-8s k=%4d  %.4e  ×%.4f від бази" % (key[0], key[1], np.mean(v), B / np.mean(v)), flush=True)
