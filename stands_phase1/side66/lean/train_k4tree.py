"""Бекпроп як пошук (26.09 ніч): навчений модуль «ефект κ₄-моря на народження κ₃» у ланцюзі v34.
На шарах L0..14 у стан κ₃(p) вкидається хаб-джерело T_abc = Σ_типів g_b^k · X_k[b,a]·Y_k[b,c] (хаб b):
  тип 1: D21[b,a]·D21[b,c]/var_b (κ₃κ₃/κ₂) · тип 2: C[b,a]·C[b,c] · тип 3: D21[b,a]·C[b,c]/√var_b.
Ворота g^k_b = MLP_θ(t_b, log var_b, γ₃_b, l/16), останній шар нуль ⟹ старт = база. Ціль: фінальна MSE/mean(gt²),
градієнт через шари ≥L0 (пам'ять 5.3 ГБ), базиси обрізань сталі (градієнт зміщений ~×1.3, вирок лише за forward).
Дані: офіційний mini-100 (істина 1e9), train 0..79, test 80..99. Передбачення: test MSE ×0.85 (шанс 25–30%)."""
import os, sys, time, glob, json, numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; dev = "cuda"
L0 = int(os.environ.get("L0", "11")); LR = float(os.environ.get("LR", "3e-3")); EPOCHS = int(os.environ.get("EPOCHS", "10")); TAG = os.environ.get("TAG", "a")
NTR = int(os.environ.get("NTR", "80")); OUT = "/home/trim/wb2data/k4tree_%s.pt" % TAG
fs = sorted(glob.glob('/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a/data/mini-*.parquet'))
index = []
for f in fs:
    t = pq.read_table(f, columns=["mlp_name"])
    for r in range(t.num_rows): index.append((f, r, t.column("mlp_name")[r].as_py()))
def load(k):
    f, r, nm = index[k]; row = pq.read_table(f).slice(r, 1).to_pylist()[0]
    return nm, np.array(row["weights"], dtype=np.float32), np.array(row["final_means"], dtype=np.float64)
train = list(range(NTR)); test = list(range(80, 100))
class Gate(torch.nn.Module):
    def __init__(self, k=3, h=16):
        super().__init__(); self.a = torch.nn.Linear(4, h); self.b = torch.nn.Linear(h, h); self.c = torch.nn.Linear(h, k)
        torch.nn.init.zeros_(self.c.weight); torch.nn.init.zeros_(self.c.bias)
    def forward(self, x): return self.c(torch.tanh(self.b(torch.tanh(self.a(x)))))
gate = Gate().to(dev); opt = torch.optim.Adam(gate.parameters(), lr=LR)
state = {"ep": 0, "coef": {}, "base": {}}
if os.path.exists(OUT):
    ck = torch.load(OUT); gate.load_state_dict(ck["gate"]); opt.load_state_dict(ck["opt"]); state = ck["state"]; print("відновлено з епохи %d" % state["ep"], flush=True)
live = {}
def d21hook(l, D3, D21, lv):
    live["mu"] = lv["mu"]; live["var"] = lv["var"]
    if l == L0: torch.set_grad_enabled(True)
    return D3, D21, None
def inj(l, K3, D3, D21):
    if l < L0 or l > 14 or D21 is None or not ENABLED[0]: return
    mu, var = live["mu"], live["var"]; sd = var.clamp(min=1e-12).sqrt()
    x = torch.stack([mu / sd, torch.log(var.clamp(min=1e-12)), D3 / sd ** 3, torch.full_like(mu, l / 16.0)], 1).float()
    g = gate(x)                                      # (n, 3), хаб b
    Coff = live_C[0].clone(); Coff.fill_diagonal_(0.0); D21t = D21.T
    X = D21t * (g[:, 0] / sd)[None, :] + Coff * g[:, 1][None, :] + D21t * (g[:, 2] / sd)[None, :]
    # Y окремо на тип: тип1 D21ᵀ/√var, тип2 C, тип3 C ⟹ два хаб-джерела (пари ніг)
    X1 = D21t * (g[:, 0] / sd)[None, :]; Y1 = D21t / sd[None, :]
    X2 = Coff * g[:, 1][None, :] + D21t * (g[:, 2] / sd)[None, :]; Y2 = Coff
    K3.append_hub(X1.to(D21.dtype), Y1.to(D21.dtype), X2.to(D21.dtype), Y2.to(D21.dtype))
live_C = [None]
def chook(l, K, mu, C):
    live_C[0] = C
ENABLED = [True]
def run(w, gt, grad, enabled, coef):
    torch.set_grad_enabled(False); ENABLED[0] = enabled
    lk.D21_HOOK = d21hook if grad else (lambda l, D3, D21, lv: (live.update(mu=lv["mu"], var=lv["var"]) or (D3, D21, None)))
    lk.K3_INJ = inj; lk.K_POST_HOOK = chook; lk.REGEN_STATE.clear()
    p = lk.lean_k3_predict(w, dtype=dt, statics=ST, g_hook=lo.Replay(coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG)
    loss = ((p[-1].double() - gt) ** 2).mean() / (gt ** 2).mean(); torch.set_grad_enabled(False); return loss
ST = None
def prep(k):
    global ST
    nm, w, gtn = load(k); gt = torch.tensor(gtn, device=dev)
    if ST is None: ST = lk.build_static_coefs(w.shape[1])
    if nm not in state["coef"]:
        lk.D21_HOOK = None; lk.K3_INJ = None; lk.K_POST_HOOK = None; fit = lo.FitC(); lk.REGEN_STATE.clear()
        with torch.no_grad(): lk.lean_k3_predict(w, dtype=dt, statics=ST, k4_aug=True, g_hook=fit)
        state["coef"][nm] = fit.coef
    if nm not in state["base"]:
        with torch.no_grad(): state["base"][nm] = float(run(w, gt, False, False, state["coef"][nm]))
    return nm, w, gt
def K_POST_HOOK_fix():
    pass
for ep in range(state["ep"], EPOCHS):
    t0 = time.time(); rs = []
    rng = np.random.default_rng(ep); order = rng.permutation(train)
    for k in order:
        nm, w, gt = prep(int(k)); opt.zero_grad()
        loss = run(w, gt, True, True, state["coef"][nm])
        with torch.enable_grad(): loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0); opt.step()
        r = float(loss) / state["base"][nm]; rs.append(r)
        print("ep %d %-20s ratio %.4f (base %.3e)  %.0f с" % (ep, nm, r, state["base"][nm], time.time() - t0), flush=True)
        torch.cuda.empty_cache()
    te = []
    for k in test:
        nm, w, gt = prep(k)
        with torch.no_grad(): te.append(float(run(w, gt, False, True, state["coef"][nm])) / state["base"][nm])
    print("EPOCH %d: train ratio сер %.4f (геом %.4f) · TEST ratio сер %.4f геом %.4f min %.3f max %.3f · %.0f с" % (
        ep, np.mean(rs), np.exp(np.mean(np.log(rs))), np.mean(te), np.exp(np.mean(np.log(te))), min(te), max(te), time.time() - t0), flush=True)
    state["ep"] = ep + 1; torch.save(dict(gate=gate.state_dict(), opt=opt.state_dict(), state=state), OUT)
print("DONE_TRAIN")
