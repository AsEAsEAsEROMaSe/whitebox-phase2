"""chpre_f0r (21.10.2026 f0, завдання 5f): стеля c4 через низький ранг стану. U_r = топ-r власні вектори істинної Cov(h_l) (вхід шару l, МК 2^21),
y = U_rᵀ(h − m), c4_r(p_j) = κ4((y @ U_rᵀW)_j) прямою МК (те саме, що r⁴-тензор κ4(y) згорнутий з (U_rᵀW_j)^⊗4).
Варіант «+d»: плюс діагональ решти Σ_a W_aj⁴ κ4(x⊥_a). Контроль: випадковий 32-вимірний підпростір.
Оракул: справжній c3 (D3set 1..15) + G4set 1..15 з c4_r, проти both (повний c4) і d3 на мережі 0.
ПЕРЕДБАЧЕННЯ: r=16 бере < 50% лог-виграшу g4 поверх d3 (abcd n⁴ не з низького рангу: ранг стану ~3 міряли на 256 у фазі 1, тут 1024 і середнє не прибране). Шанс r≤16 ≥ 70%: 20%.
Рестарт 21.10.2026 13:50 (SCALED=1): прогін 1 дав ×0.10 для всіх r (частка −860%), бо амплітуда c4_r занижена у 2…800 разів, а заміна g4 майже нулем гірша за власний g4 ланцюга. Форма жива: corr(c4_r, c4) на ш15 0.73 (r4) … 0.87 (r64), випадковий 32: −0.01.
Тепер c4_r × оракульний нахил по шару + контроль both_zero (G4set з нулем).
ПЕРЕДБАЧЕННЯ: both_r64s бере 30…60% лог-виграшу both над d3; r16s < 30%. Шанс r16s ≥ 50%: 15%.
usage: chpre_f0r.py <k> <out.json>"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]
N1 = int(float(os.environ.get("NS1", str(2 ** 21)))); N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192
RS = (4, 8, 16, 32, 64); LS = tuple(range(1, 16))
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_orc", "/home/trim/wb2lean/lt2re/estimator_sb2_orc_f0.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
dev = "cuda"; W = w.to(dev)
mt = {l: torch.tensor(m_true[l - 1], dtype=torch.float32, device=dev) for l in LS}
# прохід 1: Cov(h_l) входу шару l
gg = torch.Generator(device=dev).manual_seed(55000 + k)
Cs = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in LS}; S1 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in LS}
n = 0
while n < N1:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        if l in Cs:
            x = h - mt[l]; Cs[l] += (x.T @ x).double(); S1[l] += x.double().sum(0)
        if l < 15: h = torch.relu(h @ W[l])
    n += B
U = {}; info = {}
gr = torch.Generator(device="cpu").manual_seed(99000 + k)
for l in LS:
    mu = S1[l] / n; C = Cs[l] / n - torch.outer(mu, mu)
    ev, V = torch.linalg.eigh(C); ev = ev.flip(0); V = V.flip(1)
    U[l] = V[:, :max(RS)].float().contiguous()
    tr = float(ev.sum()); info[l] = {"pr": float(ev.sum() ** 2 / (ev ** 2).sum()), "evfrac": {r: float(ev[:r].sum() / tr) for r in RS}}
    Q, _ = torch.linalg.qr(torch.randn(1024, 32, generator=gr, dtype=torch.float64)); U[(l, "rand")] = Q.float().to(dev)
    print("net %d L%d PR %.2f evfrac %s" % (k, l, info[l]["pr"], json.dumps(info[l]["evfrac"])), flush=True)
del Cs, S1; torch.cuda.empty_cache()
# прохід 2: c4 повний, c4 проєкцій і κ4 діагоналі решти
VAR = ["full"] + ["r%d" % r for r in RS] + ["rand32"]
Z = lambda: torch.zeros(4, 1024, dtype=torch.float64, device=dev)
P = {l: {v: Z() for v in VAR} for l in LS}; D = {l: {v: Z() for v in VAR[1:]} for l in LS}
gg = torch.Generator(device=dev).manual_seed(66000 + k); n = 0
def pw(acc, q):
    q = q.double(); q2 = q * q; acc[0] += q.sum(0); acc[1] += q2.sum(0); acc[2] += (q2 * q).sum(0); acc[3] += (q2 * q2).sum(0)
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        if l in P:
            x = h - mt[l]; pw(P[l]["full"], x @ W[l]); y64 = x @ U[l]
            for r in RS:
                y = y64[:, :r]; Ur = U[l][:, :r]; pw(P[l]["r%d" % r], y @ (Ur.T @ W[l])); pw(D[l]["r%d" % r], x - y @ Ur.T)
            Uq = U[(l, "rand")]; y = x @ Uq; pw(P[l]["rand32"], y @ (Uq.T @ W[l])); pw(D[l]["rand32"], x - y @ Uq.T)
        if l < 15: h = torch.relu(h @ W[l])
    n += B
def cum(a):
    m1, m2, m3, m4 = (a / n).cpu().numpy()
    v = m2 - m1 ** 2
    return m3 - 3 * m1 * m2 + 2 * m1 ** 3, m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * v ** 2
c3 = np.zeros((16, 1024)); C4 = {v: np.zeros((16, 1024)) for v in VAR}; C4d = {v: np.zeros((16, 1024)) for v in VAR[1:]}
stat = {}
for l in LS:
    c3[l], C4["full"][l] = cum(P[l]["full"]); W4 = (w[l].numpy().astype(np.float64)) ** 4
    st = {}
    for v in VAR[1:]:
        C4[v][l] = cum(P[l][v])[1]; C4d[v][l] = C4[v][l] + cum(D[l][v])[1] @ W4
        st[v] = [float(np.corrcoef(C4[v][l], C4["full"][l])[0, 1]), float(np.corrcoef(C4d[v][l], C4["full"][l])[0, 1]),
                 float(C4[v][l] @ C4["full"][l] / (C4[v][l] @ C4[v][l]))]
    stat[l] = st
    print("net %d L%d corr(c4_r,c4) [lr, lr+d, slope_lr] %s" % (k, l, json.dumps(st)), flush=True)
del W, P, D, U; torch.cuda.empty_cache()
# шар 0: вхід гаусів ⟹ c3=c4=0 точно
Wg = [cp.asarray(x.numpy()) for x in w]
c3t = [cp.asarray(c3[l].astype(np.float32)) for l in range(16)]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
A = tuple(range(1, 16))
def run(c4arr=None, **kw):
    c4t = [cp.asarray((c4arr if c4arr is not None else C4["full"])[l].astype(np.float32)) for l in range(16)]
    mod.ORAC = dict(K=-1, mode="", truth=T0, rec=[], c3t=c3t, c4t=c4t, **kw)
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
res = {"k": k, "info": info, "stat": stat, "base": run()}
jobs = [("d3", None, dict(D3set=A)), ("both", None, dict(D3set=A, G4set=A))]
np.savez("/home/trim/wblogs/chpref0r_c4_n%d.npz" % k, c3=c3, **{"c4_" + v: C4[v] for v in VAR}, **{"c4d_" + v: C4d[v] for v in VAR[1:]})
SC = os.environ.get("SCALED") == "1"   # 21.10.2026 рестарт: c4_r на оракульному понашаровому нахилі (форма є, амплітуда занижена ×2…×800)
for v in VAR[1:]:
    if SC:
        Cs_ = C4[v].copy()
        for l in LS: Cs_[l] *= stat[l][v][2]
        jobs += [("both_" + v + "s", Cs_, dict(D3set=A, G4set=A))]
    else:
        jobs += [("both_" + v, C4[v], dict(D3set=A, G4set=A)), ("both_" + v + "d", C4d[v], dict(D3set=A, G4set=A))]
if SC: jobs += [("both_zero", np.zeros_like(C4["full"]), dict(D3set=A, G4set=A))]
for nm, arr, kw in jobs:
    res[nm] = run(arr, **kw)
    sh = ""
    if "d3" in res and "both" in res and nm.startswith("both_"):
        sh = "  частка %.0f%%" % (100 * math.log(res["d3"] / res[nm]) / math.log(res["d3"] / res["both"]))
    print("net %d %s ×%.4f%s" % (k, nm, res["base"] / res[nm], sh), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
