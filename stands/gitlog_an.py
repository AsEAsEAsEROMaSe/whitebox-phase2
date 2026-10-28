"""Аналіз git-логу (gitlog.py): кумулятивні коміти за народженням, адитивність, профіль доставки джерела за віком,
колінеарність доставок (чи «решта ∝ перша»), поява коміту по шарах і частка лінійного переносу середнього, дослівний лог нейронів."""
import sys, numpy as np, torch
dump = int(sys.argv[1]); suf = sys.argv[2] if len(sys.argv) > 2 else ""
z = np.load("/home/trim/wb2data/gitlog_%d%s.npz" % (dump, suf)); names = list(z["names"]); P = z["preds"]; gt = z["gt"]
pred = {n: P[i] for i, n in enumerate(names)}; base = pred["base"][-1]; none = pred["none"][-1]; e = base - gt
nrm = np.linalg.norm; cos = lambda a, b: float(a @ b / (nrm(a) * nrm(b) + 1e-300))
print("dump %d  MSE base %.4e  none %.4e  ‖e‖ %.3e  ‖base−none‖ %.3e" % (dump, np.mean(e**2), np.mean((none-gt)**2), nrm(e), nrm(base-none)))
# 1. кумулятивний лог
print("\n== КОМІТИ ЗА НАРОДЖЕННЯМ (cum≤b − cum≤b−1), вихід ==")
print(" b   ‖Δ_b‖      /‖e‖   MSE після b   ×до b   cos(Δ_b, −e_{b−1})  cos(Δ_b, single_b)  ‖single_b‖/‖Δ_b‖")
prev = none; D = []; S = []
for b in range(15):
    cur = pred["cum<=%d" % b][-1]; d = cur - prev; D.append(d); ep = prev - gt; s = pred["single%d" % b][-1] - none; S.append(s)
    print("%2d  %.3e  %6.2f   %.4e  %6.2f      %+.3f              %+.3f            %.3f" % (b, nrm(d), nrm(d)/nrm(e), np.mean((cur-gt)**2), np.mean(ep**2)/np.mean((cur-gt)**2), cos(d, -ep), cos(d, s), nrm(s)/nrm(d)))
    prev = cur
D = np.array(D); S = np.array(S); tot = base - none; sumS = S.sum(0)
print("адитивність: ‖(base−none) − Σ single‖/‖base−none‖ = %.4f ; cos = %.5f" % (nrm(tot - sumS)/nrm(tot), cos(tot, sumS)))
sv = np.linalg.svd(D, compute_uv=False); print("матриця комітів 15×1024: eff.rank %.2f, s1/s15 %.1f, енергія топ-3 %.3f" % ((sv**2).sum()**2/(sv**4).sum(), sv[0]/sv[-1], (sv[:3]**2).sum()/(sv**2).sum()))
G = D @ D.T / np.outer(nrm(D, axis=1), nrm(D, axis=1)); off = G[~np.eye(15, dtype=bool)]
print("cos між комітами різних народжень: сер %+.3f, |cos| сер %.3f, макс %.3f" % (off.mean(), np.abs(off).mean(), np.abs(off).max()))
# 2. профіль доставки джерела за віком
print("\n== ПРОФІЛЬ ДОСТАВКИ ДЖЕРЕЛА b ЗА ВІКОМ a (only b≤a − only b≤a−1), вихід ==")
for b in [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 3, 6, 9, 12]:
    ages = [a for a in range(1, 16 - b) if "only%d<=%d" % (b, a) in pred]
    if not ages: continue
    prev = none; dl = []
    for a in ages:
        cur = pred["only%d<=%d" % (b, a)][-1]; dl.append(cur - prev); prev = cur
    dl = np.array(dl); totb = pred["single%d" % b][-1] - none; last = pred["only%d<=%d" % (b, ages[-1])][-1] - none
    print("b=%2d  ‖total‖ %.3e  пломба ‖only≤%d − single‖/‖total‖ %.2e" % (b, nrm(totb), ages[-1], nrm(last - totb)/nrm(totb)))
    en = (dl**2).sum(1); print("   a:      " + " ".join("%6d" % a for a in ages))
    print("   ‖d_a‖/‖tot‖ " + " ".join("%6.3f" % (nrm(d)/nrm(totb)) for d in dl))
    print("   cos(d_a,d_1) " + " ".join("%+6.2f" % cos(d, dl[0]) for d in dl))
    print("   cos(d_a,tot) " + " ".join("%+6.2f" % cos(d, totb) for d in dl))
    if len(dl) > 1:
        rest = dl[1:].sum(0); c = float(rest @ dl[0] / (dl[0] @ dl[0])); r2 = 1 - nrm(rest - c*dl[0])**2/nrm(rest)**2
        Gd = dl @ dl.T / np.outer(nrm(dl, axis=1), nrm(dl, axis=1)); offd = Gd[~np.eye(len(dl), dtype=bool)]
        svd_ = np.linalg.svd(dl, compute_uv=False)
        print("   решта(a≥2) ∝ d_1: c=%+.3f R²=%.3f ; ‖решта‖/‖d_1‖ %.3f ; cos між доставками сер %+.3f |.| %.3f ; eff.rank %.2f з %d" % (c, r2, nrm(rest)/nrm(dl[0]), offd.mean(), np.abs(offd).mean(), (svd_**2).sum()**2/(svd_**4).sum(), len(dl)))
        # найкращий лінійний прогноз решти з перших k доставок (МНК, in-sample стеля)
        for k in (1, 2, 3):
            if len(dl) > k:
                X = dl[:k].T; rest_k = dl[k:].sum(0); coef = np.linalg.lstsq(X, rest_k, rcond=None)[0]; print("   решта(a>%d) з перших %d: R² %.3f" % (k, k, 1 - nrm(rest_k - X @ coef)**2/nrm(rest_k)**2))
# 3. поява коміту по шарах і частка лінійного переносу середнього
w = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % dump)["weights"].astype(np.float64)
mu = z["mu"]; var = z["var"]; ml = list(z["mu_layers"]); Phi = lambda t: 0.5*(1+torch.erf(torch.as_tensor(t)/np.sqrt(2))).numpy()
print("\n== ПОЯВА КОМІТУ b ПО ШАРАХ l (‖Δ_b^(l)‖/‖m^(l)‖·1e3) і cos(Δ^(l+1), Φ⊙(Δ^(l)W)) ==")
for b in [0, 3, 6, 9, 12]:
    prevc = pred["none"] if b == 0 else pred["cum<=%d" % (b-1)]; curc = pred["cum<=%d" % b]; line = "b=%2d " % b; line2 = "      cosT "
    for l in range(b+1, 16):
        dlay = curc[l] - prevc[l]; line += " l%02d %5.2f" % (l, 1e3*nrm(dlay)/nrm(curc[l]))
        if l < 15 and (l+1) in ml:
            t = mu[ml.index(l+1)] / np.sqrt(var[ml.index(l+1)]); T = Phi(t) * (dlay @ w[l+1]); dn = curc[l+1] - prevc[l+1]
            line2 += "      %+5.2f" % cos(dn, T) + ("/%4.2f" % (nrm(T)/nrm(dn)))
    print(line); print(line2)
# 4. дослівний лог трьох нейронів з найбільшою |e| і трьох випадкових
print("\n== ДОСЛІВНИЙ ЛОГ (значення ×1e3): none → коміти b=0..14 → base | істина ==")
rng = np.random.default_rng(0); idx = list(np.argsort(-np.abs(e))[:3]) + list(rng.choice(1024, 3, replace=False))
for j in idx:
    print("нейрон %4d: none %+8.4f | " % (j, 1e3*none[j]) + " ".join("%+.3f" % (1e3*D[b, j]) for b in range(15)) + " | base %+8.4f істина %+8.4f  e %+.2e" % (1e3*base[j], 1e3*gt[j], e[j]))
