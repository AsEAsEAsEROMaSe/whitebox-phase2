"""21.10.2026 white-box-2e (= f9): accC3, звідки похибка C ланцюга. accC2 дав: гаусів крок від СПРАВЖНІХ передактивацій має 6% похибку
позадіагоналі і ламає ланцюг (goff ×0.066), тоді як справжня позадіагональ ×1.33, з c3c4 ×4.48/×6.41.
Тут: та сама МК (сід 55000+k), C ланцюга (Crec), gauss(chain pre) = Мелер від μ ланцюга і W^T C_{l-1} W. Похибки позадіагоналі проти істини.
Режими: gchain = гаусова частина ланцюга (без його негаусових поправок); mix = gauss(true pre) + (C_chain − gauss(chain pre)),
тобто справжній гаусів крок плюс негаусова поправка ланцюга. Пломба: off+c3c4 = accC2 (×4.484 / ×6.408).
ПЕРЕДБАЧЕННЯ: негаусова частина ланцюга корелює з істинною (corr > 0.7), gchain ламає як goff, mix ≥ ×1.2 (шанс 40%).
usage: accC2e.py <k>"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False); torch.set_num_threads(1)
k = int(sys.argv[1]); N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(55000 + k)
mt = torch.tensor(m_true, dtype=torch.float32, device=dev)
S = torch.zeros(16, 4, 1024, dtype=torch.float64, device=dev); sh = {}
G = torch.zeros(15, 1024, 1024, dtype=torch.float32, device=dev); Q = torch.zeros(15, 1024, 1024, dtype=torch.float32, device=dev); n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        p = h @ W[l]
        if l not in sh: sh[l] = p.mean(0)
        x = p - sh[l]; xd = x.double()
        S[l, 0] += xd.sum(0); S[l, 1] += (xd * xd).sum(0); S[l, 2] += (xd ** 3).sum(0); S[l, 3] += (xd ** 4).sum(0)
        if 1 <= l <= 14: Q[l] += x.T @ x
        h = torch.relu(p)
        if 1 <= l <= 14:
            xc = h - mt[l]; G[l] += xc.T @ xc
    n += B
a = (S / n).cpu().numpy(); m1, m2, m3, m4 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
var = m2 - m1 ** 2; c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2
Cs, Vs, Gs = {}, {}, {}
for l in range(1, 15):
    Cpost = G[l].double() / n; Cs[l] = Cpost
    Vs[l] = torch.diag(Cpost).cpu().numpy()
    mp = torch.as_tensor(m1[l], device=dev); Cpre = Q[l].double() / n - torch.outer(mp, mp)
    mu = mp + sh[l].double(); sd = torch.sqrt(torch.diag(Cpre)); t = mu / sd
    phi = torch.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi); Phi = 0.5 * (1 + torch.erf(t / math.sqrt(2)))
    rho = Cpre / torch.outer(sd, sd); rho.fill_diagonal_(0.0)
    He = [torch.ones_like(t), t.clone()]
    for j in range(2, 12): He.append(t * He[-1] - (j - 1) * He[-2])
    acc = torch.outer(sd * Phi, sd * Phi) * rho; rk = rho.clone(); fac = 1.0
    for kk in range(2, 13):
        rk = rk * rho; fac *= kk
        ak = sd * ((-1) ** kk) * phi * He[kk - 2]
        acc += torch.outer(ak, ak) * rk / fac
    Gs[l] = acc
del S, G, Q; torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
o = {"K": -1, "mode": "", "truth": T0, "rec": [], "Crec": True}; mod.ORAC = o
p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
base = float(np.mean((p[-1] - m_true[15]) ** 2))
pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in o["pre"]}
Cp = {li: cp.asnumpy(cp.asarray(v)).astype(np.float64) for li, v in o.get("Cpost", {}).items()}
mod.ORAC = None; o.clear(); cp.get_default_memory_pool().free_all_blocks()
print("net %d base %.4e, Cpost layers %s" % (k, base, sorted(Cp)), flush=True)
def gauss(mu, Cpre):
    sd = torch.sqrt(torch.diag(Cpre)); t = mu / sd
    phi = torch.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi); Phi = 0.5 * (1 + torch.erf(t / math.sqrt(2)))
    rho = Cpre / torch.outer(sd, sd); rho.fill_diagonal_(0.0)
    He = [torch.ones_like(t), t.clone()]
    for j in range(2, 12): He.append(t * He[-1] - (j - 1) * He[-2])
    acc = torch.outer(sd * Phi, sd * Phi) * rho; rk = rho.clone(); fac = 1.0
    for kk in range(2, 13):
        rk = rk * rho; fac *= kk
        ak = sd * ((-1) ** kk) * phi * He[kk - 2]
        acc += torch.outer(ak, ak) * rk / fac
    return acc
Wd = W.double().cpu(); off = 1 - torch.eye(1024, dtype=torch.float64)
del W; torch.cuda.empty_cache()
Cs = {l: v.cpu() for l, v in Cs.items()}; Gs = {l: v.cpu() for l, v in Gs.items()}; torch.cuda.empty_cache()
def rel(A, B): return float(torch.sqrt(torch.mean(((A - B) * off) ** 2)) / torch.sqrt(torch.mean((B * off) ** 2)))
Gc, Mix = {}, {}
for l in range(1, 15):
    Cc = torch.as_tensor(Cp[l])
    Cpre_c = Wd[l].T @ torch.as_tensor(Cp[l - 1]) @ Wd[l]
    gc = gauss(torch.as_tensor(pre[l][0]), Cpre_c)
    Gc[l] = gc; Mix[l] = Gs[l] + (Cc - gc) * off
    if l in (2, 4, 8, 10, 12, 14):
        print("net %d l%d err vs true: chain %.4f | gauss(true pre) %.4f | gauss(chain pre) %.4f | gauss(chain pre) vs chain %.4f | mix %.4f | NG true %.4f vs NG chain corr %.3f" % (
            k, l, rel(Cc, Cs[l]), rel(Gs[l], Cs[l]), rel(gc, Cs[l]), rel(gc, Cc), rel(Mix[l], Cs[l]), rel(Gs[l], Cs[l]),
            float(torch.sum((Cs[l] - Gs[l]) * (Cc - gc) * off) / torch.sqrt(torch.sum(((Cs[l] - Gs[l]) * off) ** 2) * torch.sum(((Cc - gc) * off) ** 2)))), flush=True)
del Wd
Cg = {l: cp.asarray(Cs[l].float().cpu().numpy()) for l in Cs}; Gg = {l: cp.asarray(Gs[l].float().cpu().numpy()) for l in Gs}
Mg = {l: cp.asarray(Mix[l].float().cpu().numpy()) for l in Mix}; GCg = {l: cp.asarray(Gc[l].float().cpu().numpy()) for l in Gc}
del Cs, Gs, Mix, Gc; torch.cuda.empty_cache()
c3t = [cp.asarray(c3[l].astype(np.float32)) for l in range(16)]; c4t = [cp.asarray(c4[l].astype(np.float32)) for l in range(16)]
Tg = [[cp.asarray(m_true[l].astype(np.float32)), cp.asarray(Vs.get(l, np.zeros(1024)).astype(np.float32)), cp.zeros(1024, dtype=cp.float32), cp.zeros(1024, dtype=cp.float32)] for l in range(16)]
def run(**x):
    d = dict(K=-1, mode="", truth=Tg, rec=[], C=Cg, c3t=c3t, c4t=c4t); d.update(x); mod.ORAC = d
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None; cp.get_default_memory_pool().free_all_blocks()
    return float(np.mean((p[-1] - m_true[15]) ** 2))
A15 = tuple(range(1, 16)); A14 = tuple(range(1, 15)); KK = dict(D3set=A15, G4set=A15)
res = {"k": k, "base": run()}
for nm, x in (("off+c3c4", dict(Kset=A14, mode="c", **KK)),
              ("gchain", dict(Kset=A14, mode="c", C=GCg)),
              ("mix", dict(Kset=A14, mode="c", C=Mg)), ("mix+c3c4", dict(Kset=A14, mode="c", C=Mg, **KK))):
    res[nm] = run(**x)
    print("net %d %s ×%.3f" % (k, nm, res["base"] / res[nm]), flush=True)
json.dump(res, open("/home/trim/wblogs/accC2e_n%d.json" % k, "w"), indent=1)
