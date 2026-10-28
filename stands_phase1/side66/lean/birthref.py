"""Похибка народження κ₃(h) референсною kprop (kprop_ds.relu_kprop) на ПОВНИХ істинних кумулянтах p підмножини S (32 нейрони).
Режими входу: до κ₃ · до κ₄ · до κ₅ (решта 0); k_max = 3 і 4. Ціль: істинна κ₃(h) на S³ (слайси (1,1,1) і (2,1)).
🔒 шум: половини вибірки. Передбачення ДО: при точних входах до κ₄ похибка (1,1,1)+(2,1) ≤1% (формула майже точна, і тоді
похибка народження ланцюга = неточні ВХОДИ/представлення), шанс 50%; альтернатива ~3% = обрізання ряду (тоді κ₅/k_max=4 вирішує)."""
import sys, math, itertools, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference"]
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower, DSTensor
dev = "cuda"; i = int(sys.argv[1]); ls = int(sys.argv[2]); m = 32; N = 2**22; CH = 2**15
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); W = torch.tensor(d["weights"], device=dev); L, n, _ = W.shape
kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i)
rng = np.random.default_rng(11 + i); S = torch.tensor(np.sort(rng.choice(n, m, replace=False)), device=dev)
a = torch.tensor(kp["pre_mean"][ls], device=dev, dtype=torch.float32)[S]
g = torch.Generator(device=dev); g.manual_seed(7 + i)
M = {h: {k: 0 for k in range(1, 6)} for h in (0, 1)}; H = {h: [0, 0, 0] for h in (0, 1)}; cnt = {0: 0, 1: 0}
for c in range(N // CH):
    hh = c % 2; z = torch.randn(CH, n, device=dev, generator=g); p = z @ W[0]
    for l in range(ls): p = torch.relu(p) @ W[l+1]
    q = (p[:, S] - a).double(); r = torch.relu(p[:, S]).double()
    M[hh][1] = M[hh][1] + q.sum(0); M[hh][2] = M[hh][2] + q.T @ q
    for s0 in range(0, CH, 2048):
        qs = q[s0:s0+2048]; b = qs.shape[0]
        q2 = (qs[:, :, None]*qs[:, None, :]).reshape(b, -1); q3 = (q2[:, :, None]*qs[:, None, :]).reshape(b, -1)
        M[hh][3] = M[hh][3] + q2.T @ qs; M[hh][4] = M[hh][4] + q2.T @ q2; M[hh][5] = M[hh][5] + q2.T @ q3
    H[hh][0] = H[hh][0] + r.sum(0); r2_ = (r[:, :, None]*r[:, None, :]).reshape(CH, -1); H[hh][1] = H[hh][1] + r.T @ r; H[hh][2] = H[hh][2] + r2_.T @ r
    cnt[hh] += CH
def cums(Ms, Nh):
    m1 = Ms[1]/Nh; mu = {}
    # центрування на справжнє середнє (a з банку ≈ середнє, дотягуємо зсувом): переходимо до центральних моментів через біноміальний зсув — простіше
    raw = {k: Ms[k]/Nh for k in range(1, 6)}
    e = m1
    m2 = raw[2].reshape(m, m) - torch.outer(e, e)
    R3 = raw[3].reshape(m, m, m); R4 = raw[4].reshape(m, m, m, m); R5 = raw[5].reshape(m, m, m, m, m)
    R2 = raw[2].reshape(m, m)
    def sym_sum(T, perms): return sum(T.permute(*p) for p in perms)/1.0
    # центральні моменти з сирих (зсув на e)
    E = e
    C3 = R3 - (R2[:, :, None]*E[None, None, :] + R2[:, None, :]*E[None, :, None] + R2[None, :, :]*E[:, None, None]) + 2*E[:, None, None]*E[None, :, None]*E[None, None, :]
    # для 4-го і 5-го: центруємо семпли точніше неможливо постфактум без зсувів вищих порядків ⟹ розмір зсуву e малий (a≈mean), тому
    # використовуємо наближення першого порядку по e для R4,R5 (похибка O(e²), e~1e-4 σ)
    C4 = R4 - sum(R3.permute(*pp)[..., None].movedim(-1, k) * 0 for k, pp in [])  # placeholder
    return e, m2, C3, R4, R5
Nh = cnt[0]
def build(h):
    Ms = M[h]; e = Ms[1]/Nh
    R2 = (Ms[2]/Nh).reshape(m, m); R3 = (Ms[3]/Nh).reshape(m, m, m); R4 = (Ms[4]/Nh).reshape(m, m, m, m); R5 = (Ms[5]/Nh).reshape(m, m, m, m, m)
    return e, R2, R3, R4, R5
def moments_to_cumulants(R2, R3, R4, R5):
    # q центровані на a≈mean ⟹ вважаємо E[q]=e≈0 малим і ігноруємо (перевірка: max|e|/σ друкується)
    k2 = R2; k3 = R3
    idx4 = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]
    k4 = R4 - sum(torch.einsum("ab,cd->abcd", R2, R2).permute(*np.argsort(p)) for p in idx4)
    # κ5 = μ5 − Σ_{10} μ3⊗μ2
    k5 = R5.clone()
    for pair in itertools.combinations(range(5), 2):
        rest = [x for x in range(5) if x not in pair]; order = list(pair) + rest
        T = torch.einsum("ab,cde->abcde", R2, R3); k5 = k5 - T.permute(*np.argsort(order))
    return k2, k3, k4, k5
out = {}
for h in (0, 1, "all"):
    if h == "all":
        Ms = {k: M[0][k] + M[1][k] for k in range(1, 6)}; Nn = 2*Nh; Hs = [H[0][j] + H[1][j] for j in range(3)]
    else: Ms = M[h]; Nn = Nh; Hs = H[h]
    e = Ms[1]/Nn; R2 = (Ms[2]/Nn).reshape(m, m); R3 = (Ms[3]/Nn).reshape(m, m, m); R4 = (Ms[4]/Nn).reshape(m, m, m, m); R5 = (Ms[5]/Nn).reshape(m, m, m, m, m)
    k2, k3, k4, k5 = moments_to_cumulants(R2, R3, R4, R5)
    mean = (a.double() + e)
    # істинна κ₃(h)
    hm = Hs[0]/Nn; Hh2 = (Hs[1]/Nn); Hh3 = (Hs[2]/Nn).reshape(m, m, m)
    Ch = Hh2 - torch.outer(hm, hm)
    K3h = Hh3 - (Hh2[:, :, None]*hm[None, None, :] + Hh2[:, None, :]*hm[None, :, None] + Hh2[None, :, :]*hm[:, None, None]) + 2*hm[:, None, None]*hm[None, :, None]*hm[None, None, :]
    out[h] = dict(k=(mean, k2, k3, k4, k5), K3h=K3h, e=e)
print("max|E[q]|/σ %.2e" % float((out["all"]["e"].abs()/torch.diagonal(out["all"]["k"][1]).sqrt()).max()))
iu = torch.tensor([[x, y, z] for x in range(m) for y in range(m) for z in range(m) if x < y < z], device=dev)
p21 = torch.tensor([[x, x, y] for x in range(m) for y in range(m) if x != y], device=dev)
def sel(T, ix): return T[ix[:, 0], ix[:, 1], ix[:, 2]]
T = out["all"]["K3h"]; TA = out[0]["K3h"]; TB = out[1]["K3h"]
nz = lambda x: float(x.norm())
print("шум половин: (1,1,1) %.4f · (2,1) %.4f (відн. ‖A−B‖/2‖T‖)" % (nz(sel(TA, iu)-sel(TB, iu))/2/nz(sel(T, iu)), nz(sel(TA, p21)-sel(TB, p21))/2/nz(sel(T, p21))))
T = T.cpu(); iu = iu.cpu(); p21 = p21.cpu()
import os
if os.environ.get("XH", "0") == "1":   # незалежні половини: входи з A, істина з B (і навпаки), похибка мінус шум ціліі
    runs = [(0, 1), (1, 0)]
else: runs = [("all", "all")]
for kmax in (3, 4):
    for name, top in [("до κ₃", 3), ("до κ₄", 4), ("до κ₅", 5)]:
        e1s, e2s = [], []
        for hin, hout in runs:
            mean, k2, k3, k4, k5 = [x.double().cpu() for x in out[hin]["k"]]; Tt = out[hout]["K3h"].cpu()
            tw = {1: mean, 2: k2, 3: k3, 4: k4, 5: k5}; tw = {k: v for k, v in tw.items() if k <= top}
            Kin = DSTower.from_tower(tw); Ko = relu_kprop(Kin, k_max=kmax); P3 = Ko[3].to_tensor()
            e1s.append(float(((sel(P3, iu)-sel(Tt, iu))**2).sum()/(sel(T, iu)**2).sum())); e2s.append(float(((sel(P3, p21)-sel(Tt, p21))**2).sum()/(sel(T, p21)**2).sum()))
        e1 = np.mean(e1s); e2 = np.mean(e2s)
        if runs[0][0] != "all":   # мінус шум двох незалежних половин (дисперсія різниці = 2·шум половини)
            n1 = float(((sel(TA.cpu(), iu)-sel(TB.cpu(), iu))**2).sum()/(sel(T, iu)**2).sum()); n2 = float(((sel(TA.cpu(), p21)-sel(TB.cpu(), p21))**2).sum()/(sel(T, p21)**2).sum())
            print("k_max=%d %-6s (1,1,1) сира %.4f знешумлена %.4f · (2,1) сира %.4f знешумлена %.4f" % (kmax, name, e1**0.5, max(e1-n1, 0)**0.5, e2**0.5, max(e2-n2, 0)**0.5), flush=True)
        else: print("k_max=%d %-6s (1,1,1) %.4f (2,1) %.4f" % (kmax, name, e1**0.5, e2**0.5), flush=True)
print("DONE_BIRTHREF")
