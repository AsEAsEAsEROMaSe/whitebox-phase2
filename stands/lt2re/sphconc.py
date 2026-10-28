"""sphconc (20.10.2026 white-box-89): гейт до FORM/SORM на сфері. f(z) = u·h14(z), u = напрямок похибки ланцюга (chres e на шарі 14),
контроль u_r випадковий. Мірило: частка |f − E f| у топ 1% / 0.1% семплів, ексцес f, а також частка в топ-1% за ‖z‖ (радіальна)
і за кутом (найбільші |f|/‖z‖). Гаус: топ 1% |g| несе 4.4%, топ 0.1% 0.62% від E|g|.
ПЕРЕДБАЧЕННЯ: f не зосереджена (топ 1% < 8%), так само як випадковий u ⟹ FORM мертвий. Шанс зосередженості: 5%.
usage: sphconc.py <k>   (k з chres_a.json)"""
import json, sys, math, os
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False)
k = int(sys.argv[1]); dev = "cuda"; N = int(float(os.environ.get("NS", str(2 ** 22)))); B = 32768
d = json.load(open("/home/trim/wblogs/chres_a.json"))[str(k)]["layers"]["14"]
u = torch.tensor(d["e"], dtype=torch.float32, device=dev); u = u / u.norm()
gr = torch.Generator(device="cpu").manual_seed(5 + k); ur = torch.randn(1024, generator=gr).to(dev); ur = ur / ur.norm()
g = torch.Generator(device="cpu").manual_seed(770000 + k)
W = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)).to(dev)
gg = torch.Generator(device=dev).manual_seed(44000 + k)
F, Fr, R = [], [], []
n = 0
while n < N:
    z = torch.randn(B, 1024, generator=gg, device=dev); h = z
    for l in range(15):
        h = torch.relu(h @ W[l])
    F.append((h @ u).cpu()); Fr.append((h @ ur).cpu()); R.append(z.norm(dim=1).cpu()); n += B
F, Fr, R = torch.cat(F).double(), torch.cat(Fr).double(), torch.cat(R).double()
def conc(f, key=None):
    a = (f - f.mean()).abs(); key = a if key is None else key
    o = torch.argsort(key, descending=True); tot = a.sum()
    return [float(a[o[:int(len(a) * q)]].sum() / tot) for q in (0.01, 0.001)], float(((f - f.mean()) ** 4).mean() / ((f - f.mean()) ** 2).mean() ** 2 - 3)
for name, f in (("u_err", F), ("u_rand", Fr)):
    (c1, c01), ex = conc(f)
    (r1, r01), _ = conc(f, R)
    (a1, a01), _ = conc(f / R * R.mean())
    print("net %d %-6s top1%% %.4f top0.1%% %.4f exkurt %+.3f | by‖z‖ top1%% %.4f | angular top1%% %.4f top0.1%% %.4f" % (k, name, c1, c01, ex, r1, a1, a01), flush=True)
