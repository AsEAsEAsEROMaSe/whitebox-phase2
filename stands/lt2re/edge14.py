"""edge14 (20.10.2026 white-box-89). locstep: похибка m на ш14 = сума майже незалежних локальних похибок кроків (~5.5e-5 від ‖m‖
на крок, у квадратурі), тобто лікувати треба формулу КРОКУ. Тут: для передактивацій шарів LS справжні μ,σ,γ3,γ4 по нейрону (МК),
і E[ReLU]/σ = ψ(t) − γ3 tφ/6 + γ4 (t²−1)φ/24 + γ3²(t⁴−6t²+3)φ/72 проти справжнього. Дві половини ⟹ шум.
ПЕРЕДБАЧЕННЯ: Еджворт зі справжніми γ знімає ≥70% гаусової похибки (1e-3σ) ⟹ похибка кроку ланцюга = похибка ЙОГО γ3/γ4. Шанс: 60%.
usage: edge14.py <k> <out.json>"""
import json, sys, math, os, time
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]; dev = "cuda"
LS = [int(x) for x in os.environ.get("LS", "12,14").split(",")]; N = int(float(os.environ.get("NS", str(2 ** 26)))); B = 32768
g = torch.Generator(device="cpu").manual_seed(770000 + k)
W = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)).to(dev)
gg = torch.Generator(device=dev).manual_seed(33000 + k)
# shift by layer-wise pre-activation mean from a pilot batch to keep float64 power sums well conditioned
S = {(l, h_): torch.zeros(5, 1024, dtype=torch.float64, device=dev) for l in LS for h_ in (0, 1)}
sh = {}
n = 0; t0 = time.time()
while n < N:
    half = 0 if n < N // 2 else 1
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        p = h @ W[l]
        if l in LS:
            if l not in sh: sh[l] = p.mean(0)
            x = (p - sh[l]).double(); a = S[(l, half)]
            a[0] += x.sum(0); a[1] += (x * x).sum(0); a[2] += (x ** 3).sum(0); a[3] += (x ** 4).sum(0); a[4] += torch.relu(p).double().sum(0)
        if l == max(LS): break
        h = torch.relu(p)
    n += B
nd = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64, device=dev), torch.tensor(1.0, dtype=torch.float64, device=dev))
res = {"k": k, "N": N, "layers": {}}
edges = [-1e9, -3, -2, -1, 0, 1, 2, 3, 1e9]
def stats(a, cnt, l):
    m1, m2, m3, m4 = a[0] / cnt, a[1] / cnt, a[2] / cnt, a[3] / cnt
    var = m2 - m1 ** 2; sd = var.sqrt()
    c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2
    mu = m1 + sh[l].double(); t = mu / sd; g3 = c3 / sd ** 3; g4 = c4 / sd ** 4
    ph = torch.exp(nd.log_prob(t)); Ph = nd.cdf(t); psi = ph + t * Ph
    R = a[4] / cnt / sd
    E1 = psi - g3 * t * ph / 6; E2 = E1 + g4 * (t * t - 1) * ph / 24; E3 = E2 + g3 * g3 * (t ** 4 - 6 * t * t + 3) * ph / 72
    return t, g3, g4, R, {"gauss": R - psi, "e3": R - E1, "e34": R - E2, "e34s": R - E3}
for l in LS:
    c = N // 2
    tA, g3A, g4A, RA, dA = stats(S[(l, 0)], c, l); tB, g3B, g4B, RB, dB = stats(S[(l, 1)], c, l)
    L = {"g3_mean": float(((g3A + g3B) / 2).mean()), "g3_sd": float(((g3A + g3B) / 2).std()), "g4_mean": float(((g4A + g4B) / 2).mean()),
         "g4_noise_sd": float(((g4A - g4B) / 2).std()), "g3_noise_sd": float(((g3A - g3B) / 2).std())}
    t = (tA + tB) / 2
    for key in dA:
        d = (dA[key] + dB[key]) / 2; nz = (dA[key] - dB[key]) / 2
        L[key] = {"rms": float(d.pow(2).mean().sqrt()), "noise_rms": float(nz.pow(2).mean().sqrt()),
                  "bins": [[lo, hi, int(((t >= lo) & (t < hi)).sum()), float(d[(t >= lo) & (t < hi)].mean()), float(d[(t >= lo) & (t < hi)].pow(2).mean().sqrt())]
                           for lo, hi in zip(edges[:-1], edges[1:]) if int(((t >= lo) & (t < hi)).sum()) > 4]}
    res["layers"][str(l)] = L
    print("net %d L%d g3 %.3f±%.3f (noise %.3f) g4 %.3f (noise %.3f) | rms resid σ-units: " % (k, l, L["g3_mean"], L["g3_sd"], L["g3_noise_sd"], L["g4_mean"], L["g4_noise_sd"])
          + " ".join("%s %.2e(n %.1e)" % (key, L[key]["rms"], L[key]["noise_rms"]) for key in dA), flush=True)
res["t"] = time.time() - t0
json.dump(res, open(OUT, "w"), indent=1)
