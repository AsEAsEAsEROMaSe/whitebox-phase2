"""17.10.2026, ідея юзера «кинути назад точне середнє мережі й глянути на кілька мереж».
Для кожного дампу: похибка середнього ланцюга e_l = mu_chain_l - mu_true_l по шарах (істина з банку kp_p2, mean = E[h_l] після ReLU;
ланцюг mu з живих дампів reg = E[p_l] до ReLU, тому порівнюємо pre_mean). Дивимось:
(1) профіль норми похибки по шарах; (2) кореляція похибки між шарами в межах мережі після переносу назад (e_15 проти перенесеної
e_l через лінеаризований J_l); (3) чи повторюється щось між мережами: кореляція e по нейронах між мережами (має бути ≈0,
бо нейрони різні), і кореляція ознак нейрона (t, ряд W) з похибкою в межах мережі та стабільність цього між мережами."""
import sys, math, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo
dev = 'cuda'; dumps = [int(x) for x in sys.argv[1].split(",")]
feat_corr = []
for i in dumps:
    kp = np.load(lo.BANK.format(i)); d = np.load(lo.DUMP.format(i)); W = [torch.tensor(d['weights'][l], dtype=torch.float64, device=dev) for l in range(16)]
    mu_t = torch.tensor(kp['pre_mean'], dtype=torch.float64, device=dev)
    lv = [np.load('/home/trim/wblogs/live/d%d_l%02d.npz' % (i, l)) for l in range(16)]
    mu_c = torch.stack([torch.tensor(z['mu'], dtype=torch.float64, device=dev) for z in lv]); C = [torch.tensor(z['C'], dtype=torch.float64, device=dev) for z in lv]
    e = mu_c - mu_t
    s = torch.stack([torch.diagonal(C[l]).sqrt() for l in range(16)]); t = mu_c / s; Ph = 0.5 * (1 + torch.erf(t / math.sqrt(2)))
    rel = (e.norm(dim=1) / mu_t.norm(dim=1)).cpu().numpy()
    # перенос похибки вперед лінеаризацією: δp_{l+1} = (Φ_l ∘ δp_l) @ W_{l+1}; порівняти з реальною e_{l+1}
    fwd = []
    for l in range(15):
        pred = (Ph[l] * e[l]) @ W[l+1]; r = e[l+1]
        cc = float((pred * r).sum() / (pred.norm() * r.norm())); gain = float(r.norm() / pred.norm())
        fwd.append((cc, gain))
    # ознаки нейрона проти похибки на шарі 15: t, |рядок W| , сума ваг у нейрон
    l = 15; x = e[l]; f1 = t[l]; f2 = W[l].norm(dim=0); f3 = W[l].sum(0)
    cf = [float(torch.corrcoef(torch.stack([x, f]))[0, 1]) for f in (f1, f2, f3)]
    feat_corr.append(cf)
    print("d%d  відн. похибка μ по шарах (1,4,8,12,15): %s" % (i, " ".join("%.3f%%" % (100*rel[l]) for l in (1,4,8,12,15))))
    print("    перенос e_l→e_{l+1} лінеаризацією: cos(l=1,4,8,12,14) %s · ‖e_{l+1}‖/‖перенос‖ %s" % (
        " ".join("%.2f" % fwd[l][0] for l in (1,4,8,12,14)), " ".join("%.2f" % fwd[l][1] for l in (1,4,8,12,14))))
    print("    шар 15: corr(e, t) %.3f · corr(e, ‖w_j‖) %.3f · corr(e, Σw_j) %.3f" % tuple(cf), flush=True)
fc = np.array(feat_corr); print("МІЖ МЕРЕЖАМИ: corr(e,t) %.3f±%.3f · corr(e,‖w‖) %.3f±%.3f · corr(e,Σw) %.3f±%.3f" % (fc[:,0].mean(), fc[:,0].std(), fc[:,1].mean(), fc[:,1].std(), fc[:,2].mean(), fc[:,2].std()))
