"""birthref з N=2²⁶, без κ₅; незалежні половини; атрибуція нових діаграм k_max=4 за класом (кортеж порядків блоків).
Передбачення ДО: більшу частину виграшу (1,1,1) дає клас (2,3) = C·κ₃ (шанс 45%)."""
import sys, os, math, itertools, collections, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference"]
import mlp_kprop.kprop_ds as kds
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
dev = "cuda"; i = int(sys.argv[1]); ls = int(sys.argv[2]); m = 32; N = 2**int(os.environ.get("LOGN", "26")); CH = 2**15
fn = "/home/trim/wb2data/brc_%d_%d.npz" % (i, ls)
if not os.path.exists(fn):
    d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); W = torch.tensor(d["weights"], device=dev); L, n, _ = W.shape
    kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i)
    rng = np.random.default_rng(11 + i); S = torch.tensor(np.sort(rng.choice(n, m, replace=False)), device=dev)
    a = torch.tensor(kp["pre_mean"][ls], device=dev, dtype=torch.float32)[S]
    g = torch.Generator(device=dev); g.manual_seed(1000 + i)
    Z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
    acc = {h: dict(q1=Z(m), q2=Z(m, m), q3=Z(m*m, m), q4=Z(m*m, m*m), h1=Z(m), h2=Z(m, m), h3=Z(m*m, m)) for h in (0, 1)}
    for c in range(N // CH):
        hh = c % 2; z = torch.randn(CH, n, device=dev, generator=g); p = z @ W[0]
        for l in range(ls): p = torch.relu(p) @ W[l+1]
        q = (p[:, S] - a).double(); r = torch.relu(p[:, S]).double(); A = acc[hh]
        q2 = (q[:, :, None]*q[:, None, :]).reshape(CH, -1); r2 = (r[:, :, None]*r[:, None, :]).reshape(CH, -1)
        A["q1"] += q.sum(0); A["q2"] += q.T @ q; A["q3"] += q2.T @ q; A["q4"] += q2.T @ q2
        A["h1"] += r.sum(0); A["h2"] += r.T @ r; A["h3"] += r2.T @ r
    np.savez(fn, a=a.double().cpu().numpy(), Nh=N//2, **{"%d_%s" % (h, k): v.cpu().numpy() for h in (0, 1) for k, v in acc[h].items()})
D = np.load(fn); Nh = float(D["Nh"]); a = torch.tensor(D["a"])
def get(h):
    if h == "all": g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k]))/(2*Nh)
    else: g_ = lambda k: torch.tensor(D["%d_%s" % (h, k)])/Nh
    e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m)
    # точне центрування на вибіркове середнє
    E_ = e
    C2 = R2 - torch.outer(E_, E_)
    C3 = R3 - (R2[:, :, None]*E_[None, None, :] + R2[:, None, :]*E_[None, :, None] + R2[None, :, :]*E_[:, None, None]) + 2*E_[:, None, None]*E_[None, :, None]*E_[None, None, :]
    # 4-й: зсув e ~1e-3 σ ⟹ поправки O(e) до μ4 ~1e-3 відносно — нехтуємо
    k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
    hm = g_("h1"); H2 = g_("h2"); H3 = g_("h3").reshape(m, m, m)
    K3h = H3 - (H2[:, :, None]*hm[None, None, :] + H2[:, None, :]*hm[None, :, None] + H2[None, :, :]*hm[:, None, None]) + 2*hm[:, None, None]*hm[None, :, None]*hm[None, None, :]
    return (a + e, C2, C3, k4), K3h
iu = torch.tensor([[x, y, z] for x in range(m) for y in range(m) for z in range(m) if x < y < z]); p21 = torch.tensor([[x, x, y] for x in range(m) for y in range(m) if x != y])
sel = lambda T, ix: T[ix[:, 0], ix[:, 1], ix[:, 2]]
kin = {h: get(h) for h in (0, 1, "all")}; T = kin["all"][1]; TA = kin[0][1]; TB = kin[1][1]
den1 = float((sel(T, iu)**2).sum()); den2 = float((sel(T, p21)**2).sum())
n1 = float(((sel(TA, iu)-sel(TB, iu))**2).sum())/den1; n2 = float(((sel(TA, p21)-sel(TB, p21))**2).sum())/den2
print("dump%d шар %d N=2^%d шум (1,1,1) %.4f (2,1) %.4f" % (i, ls, int(math.log2(N)), (n1/4)**0.5, (n2/4)**0.5), flush=True)
T3 = {k: v for k, v in GATI(3).items()}; T4 = GATI(4)
def cls(vp): return tuple(sorted(sum(b) for b in vp))
newcls = collections.Counter(cls(vp) for ip, vd in T4.items() for vp in vd if vp not in T3.get(ip, {}))
def run(kmax, keep=None, top=4):
    if keep is not None:
        def patched(k_max, d_max=None, use_mean_var=False):
            out = {}
            for ip, vd in T4.items():
                out[ip] = {vp: c for vp, c in vd.items() if vp in T3.get(ip, {}) or cls(vp) in keep}
            return out
        kds.get_all_terms_iso = patched
    else: kds.get_all_terms_iso = GATI
    e1s, e2s = [], []
    for hin, hout in [(0, 1), (1, 0)]:
        (mean, k2, k3, k4), _ = kin[hin]; Tt = kin[hout][1]
        tw = {1: mean, 2: k2, 3: k3, 4: k4} if top == 4 else {1: mean, 2: k2, 3: k3}
        P3 = relu_kprop(DSTower.from_tower(tw), k_max=kmax)[3].to_tensor()
        e1s.append(float(((sel(P3, iu)-sel(Tt, iu))**2).sum())/den1); e2s.append(float(((sel(P3, p21)-sel(Tt, p21))**2).sum())/den2)
    e1 = np.mean(e1s) - n1; e2 = np.mean(e2s) - n2
    return e1, e2
def fmt(e): return "%+.4f" % (np.sign(e)*abs(e)**0.5)
b1, b2 = run(3); f1, f2 = run(4)
print("  k_max=3 (1,1,1) %s (2,1) %s | k_max=4 (1,1,1) %s (2,1) %s   [знешумлено; від'ємне = нижче шуму]" % (fmt(b1), fmt(b2), fmt(f1), fmt(f2)), flush=True)
if os.environ.get("TERMS", "0") == "1":
    TERMS = {"шлях (0,1,1)²(1,0,1)": [((1,1,1), ((0,1,1),(0,1,1),(1,0,1)))], "трикутник": [((1,1,1), ((0,1,1),(1,0,1),(1,1,0)))],
             "C·κ₃ (1,1,1)": [((1,1,1), ((0,1,1),(1,1,1)))], "C³ у (2,1)": [((2,1), ((1,1),(1,1),(1,1)))]}
    TERMS["шлях+C³(2,1)"] = TERMS["шлях (0,1,1)²(1,0,1)"] + TERMS["C³ у (2,1)"]
    TERMS["усі чотири"] = sum(TERMS[k] for k in ["шлях (0,1,1)²(1,0,1)", "трикутник", "C·κ₃ (1,1,1)", "C³ у (2,1)"]) if False else [t for k in ["шлях (0,1,1)²(1,0,1)", "трикутник", "C·κ₃ (1,1,1)", "C³ у (2,1)"] for t in TERMS[k]]
    for nm, lst in TERMS.items():
        def patched(k_max, d_max=None, use_mean_var=False, lst=lst):
            out = {}
            for ip, vd in T4.items():
                out[ip] = {vp: c for vp, c in vd.items() if vp in T3.get(ip, {}) or (ip, vp) in lst}
            return out
        kds.get_all_terms_iso = patched
        e1s, e2s = [], []
        for hin, hout in [(0, 1), (1, 0)]:
            (mean, k2, k3, k4), _ = kin[hin]; Tt = kin[hout][1]
            P3 = relu_kprop(DSTower.from_tower({1: mean, 2: k2, 3: k3, 4: k4}), k_max=4)[3].to_tensor()
            e1s.append(float(((sel(P3, iu)-sel(Tt, iu))**2).sum())/den1); e2s.append(float(((sel(P3, p21)-sel(Tt, p21))**2).sum())/den2)
        e1 = np.mean(e1s) - n1; e2 = np.mean(e2s) - n2
        print("   %-24s (1,1,1) похибка %s частка %5.2f | (2,1) %s частка %5.2f" % (nm, fmt(e1), (b1-e1)/(b1-f1), fmt(e2), (b2-e2)/(b2-f2)), flush=True)
    raise SystemExit
print("  атрибуція (k3 + ОДИН клас нових діаграм): частка знятої дисперсії похибки (1,1,1) / (2,1)")
for c, cnt in newcls.most_common():
    e1, e2 = run(4, keep={c})
    print("   клас %-14s діаграм %4d  (1,1,1) %5.2f  (2,1) %5.2f" % (c, cnt, (b1-e1)/(b1-f1) if b1 != f1 else 0, (b2-e2)/(b2-f2) if b2 != f2 else 0), flush=True)
print("DONE_BR2")
