"""breaks2 (20.10.2026 white-box-89, юзер: «незалежно це і треба; трішки поренжити, нащупати криві і екстраполювати»).
Нейрони перемикаються незалежно ⟹ задача на переломі одновимірна для кожного нейрона. Для шарів LS:
(A) крива по t = μ/σ передактивації: справжні P(p<0), E[ReLU p]/σ, щільність ρ(0)·σ проти гауса з ТИМИ САМИМИ μ,σ;
    біни по t, медіана і розкид різниці між нейронами (чи це гладка універсальна крива, чи хмара);
(B) вікно EPS = 0.05/0.1/0.2/0.4: щільність біля перелому з вікна і екстраполяція EPS→0 (квадратична), реальна проти гауса.
Пломба: на шарі 0 (p гаусів точно) усі різниці = шум.
ПЕРЕДБАЧЕННЯ: різниця E[ReLU]/σ як функція t гладка (розкид між нейронами < 30% від середнього в біні) на шарах 12–14;
ρ(0) відхиляється від гауса на 5–15%. Шанс, що крива універсальна між мережами: 30%.
usage: breaks2.py <k> <out.json>"""
import json, os, sys, time, math
import numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]
LS = [int(x) for x in os.environ.get("LS", "0,10,12,14").split(",")]
N = int(float(os.environ.get("NS", str(2 ** 23)))); B = 32768
EPSS = (0.05, 0.1, 0.2, 0.4)
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(770000 + k)
W = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)).to(dev)
gg = torch.Generator(device=dev).manual_seed(66000 + k)
acc = {l: {"s1": torch.zeros(1024, dtype=torch.float64, device=dev), "s2": torch.zeros(1024, dtype=torch.float64, device=dev),
           "neg": torch.zeros(1024, dtype=torch.float64, device=dev), "relu": torch.zeros(1024, dtype=torch.float64, device=dev)}
       for l in LS}
t0 = time.time()
# pass 1: μ, σ, P(p<0), E[relu p]
n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        p = h @ W[l]
        if l in acc:
            a = acc[l]; pd = p.double()
            a["s1"] += pd.sum(0); a["s2"] += (pd * pd).sum(0)
            a["neg"] += (p < 0).sum(0, dtype=torch.float64); a["relu"] += torch.relu(pd).sum(0)
        h = torch.relu(p)
    n += B
stat = {}
for l in LS:
    a = acc[l]; mu = a["s1"] / n; sd = torch.sqrt(a["s2"] / n - mu * mu)
    stat[l] = (mu, sd, a["neg"] / n, a["relu"] / n)
# pass 2: window counts around 0 for the density, all EPS at once
win = {l: torch.zeros(len(EPSS), 1024, dtype=torch.float64, device=dev) for l in LS}
gg = torch.Generator(device=dev).manual_seed(67000 + k)
n2 = 0
while n2 < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        p = h @ W[l]
        if l in win:
            sd = stat[l][1].float()
            for i, e in enumerate(EPSS):
                win[l][i] += (p.abs() < e * sd).sum(0, dtype=torch.float64)
        h = torch.relu(p)
    n2 += B
nd = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64, device=dev), torch.tensor(1.0, dtype=torch.float64, device=dev))
res = {"k": k, "N": N, "layers": {}}
edges = [-1e9, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 1e9]
for l in LS:
    mu, sd, neg, relu = stat[l]
    t = mu / sd
    Pg = nd.cdf(-t); Rg = sd * (torch.exp(nd.log_prob(t)) + t * nd.cdf(t))
    dP = (neg - Pg); dR = (relu - Rg) / sd
    rho_g = torch.exp(nd.log_prob(t))                         # gauss density at 0 in sigma units
    rhos = torch.stack([win[l][i] / n2 / (2 * e) for i, e in enumerate(EPSS)])   # density estimate per window (sigma units)
    x = torch.tensor(EPSS, dtype=torch.float64, device=dev)
    X = torch.stack([torch.ones_like(x), x * x], 1)          # rho(eps) = rho0 + c eps^2 (symmetric window)
    coef = torch.linalg.lstsq(X, rhos).solution              # (2, 1024)
    rho0 = coef[0]
    noiseR = torch.sqrt(torch.clamp(relu * 0 + (sd * sd), min=0)) / math.sqrt(n) / sd   # rough: sd/sqrt(N) in sigma units
    L = {"median_abs_t": float(t.abs().median()), "noise_dR_sigma_units": float(noiseR.median())}
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (t >= lo) & (t < hi)
        if int(s.sum()) < 8:
            continue
        b = {"t": [lo, hi], "n": int(s.sum())}
        for name, v in (("dP", dP), ("dR", dR), ("drho0_rel", rho0 / rho_g - 1), ("drho_eps0.1_rel", rhos[1] / rho_g - 1)):
            vs = v[s]
            b[name] = [float(vs.median()), float(vs.mean()), float(vs.std())]
        bins.append(b)
    L["bins"] = bins
    res["layers"][str(l)] = L
    print("net %d layer %d median|t| %.2f noise_dR %.1e" % (k, l, L["median_abs_t"], L["noise_dR_sigma_units"]), flush=True)
    for b in bins:
        print("   t[%5.1f,%5.1f) n=%4d  dR med %+.2e mean %+.2e sd %.2e | dP med %+.2e sd %.2e | rho0 rel med %+.3f sd %.3f | rho.1 rel %+.3f"
              % (b["t"][0], b["t"][1], b["n"], *b["dR"], b["dP"][0], b["dP"][2], b["drho0_rel"][0], b["drho0_rel"][2], b["drho_eps0.1_rel"][0]), flush=True)
res["t"] = time.time() - t0
json.dump(res, open(OUT, "w"), indent=1)
