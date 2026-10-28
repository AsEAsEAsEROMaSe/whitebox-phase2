"""Потоковий тренер (27.09, ідея юзера «генерувати онлайн і одразу юзати»): підхоплює пари (state_gNNNNN, truth_NNNNN)
з wb2data/gen у міру появи (gen_truth на eaa8 + dumpstate на 26c6), тримає буфер останніх BUF мереж на GPU, вчить
модель: понейронний енкодер φ(x_j) (147 ознак → 64) + один крок повідомлень по нормованих C15, Cpost15, D21, D21ᵀ
(m_j = Σ_k M_jk ψ(h_k) і Σ_k M_jk² ψ(h_k)) + декодер → r̂_j; ціль r = gt15 − pred15, loss = mean((r̂−r)²)/mean(r²) на мережу.
Кожні EVAL кроків: R²_test і MSE_після/до (геом) на офіційних mini-100 (істина 1e9) + train-число. Чекпоінт wb2data/stream.pt.
Передбачення: R²_test на mini-100 при 1000 мережах ≤0.05 (шанс 65%); якщо ≥0.15 (MSE ×1.18) ⟹ гнати далі і думати про порт."""
import os, re, sys, glob, time, numpy as np, torch, torch.nn as nn
sys.argv = sys.argv[:1]; sys.path.insert(0, "/home/trim/wb2lean"); from pilot_lc import feats
dev = "cuda"; torch.manual_seed(0); GEN = "/home/trim/wb2data/gen"; BUF = int(os.environ.get("BUF", "400")); EVAL = int(os.environ.get("EVAL", "100")); BS = 4; LR = float(os.environ.get("LR", "3e-4")); SPN = int(os.environ.get("SPN", "3"))
CK = "/home/trim/wb2data/stream.pt"; LOG = lambda *a: print(*a, flush=True)
def mats(d):
    pre = d["pre15"]; sd = np.sqrt(np.clip(pre[:, 1], 1e-12, None)).astype(np.float32); nrm = np.outer(sd, sd)
    out = []
    for k in ("C15", "Cpost15", "D21_15"):
        M = d[k].astype(np.float32); np.fill_diagonal(M, 0.0); Mn = M / nrm; out.append(Mn)
    out.append(out[-1].T.copy())
    return np.stack(out, 0)   # (4, n, n)
def pack(f, truth=None):
    d = np.load(f, allow_pickle=True); X = feats(d).astype(np.float32); gt = d["gt"] if truth is None else truth
    r = (gt[-1] - d["pred"][-1]).astype(np.float32)
    return dict(X=torch.tensor(X, device=dev), M=torch.tensor(mats(d), device=dev).half(), r=torch.tensor(r, device=dev))
# тест = mini-100
test = [pack(f) for f in sorted(glob.glob(GEN + "/state_*.npz")) if not re.match(r"state_g\d{5}\.npz$", os.path.basename(f))]
LOG("test мереж", len(test), "ознак", test[0]["X"].shape[1])
xmu = torch.cat([t["X"] for t in test]).mean(0); xsd = torch.cat([t["X"] for t in test]).std(0) + 1e-6; rsd = torch.cat([t["r"] for t in test]).std()
class Net(nn.Module):
    def __init__(self, nf, h=64):
        super().__init__(); self.enc = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.msg = nn.Linear(h, h); self.dec = nn.Sequential(nn.Linear(h + 8 * h, h), nn.GELU(), nn.Linear(h, 1)); self.g = nn.Parameter(torch.zeros(1))
    def forward(self, X, M):
        h = self.enc((X - xmu) / xsd); p = self.msg(h); Mf = M.float(); ms = []
        for i in range(4): ms += [Mf[i] @ p, (Mf[i] * Mf[i]) @ p]
        return self.dec(torch.cat([h] + ms, 1))[:, 0] * rsd * self.g.exp() * 0.1
net = Net(test[0]["X"].shape[1]).to(dev); opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5); step = 0; seen = set(); buf = []
if os.path.exists(CK):
    ck = torch.load(CK); net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); step = ck["step"]; seen = set(ck["seen"]); LOG("відновлено: крок", step, "бачено", len(seen))
def evaluate(ds):
    net.eval(); ss_res = ss_tot = 0.0; lr_ = []
    with torch.no_grad():
        for t in ds:
            rh = net(t["X"], t["M"]); ss_res += float(((t["r"] - rh) ** 2).sum()); ss_tot += float((t["r"] ** 2).sum()); lr_.append(float(((t["r"] - rh) ** 2).mean() / (t["r"] ** 2).mean()))
    net.train(); return 1 - ss_res / ss_tot, float(np.exp(np.mean(np.log(lr_))))
idle = 0
while True:
    new = []
    for f in sorted(glob.glob(GEN + "/state_g[0-9][0-9][0-9][0-9][0-9].npz")):
        i = int(os.path.basename(f)[7:12]); tf = GEN + "/truth_%05d.npz" % i
        if i in seen or not os.path.exists(tf): continue
        try: new.append(pack(f, np.load(tf)["alm"])); seen.add(i)
        except Exception as e: LOG("skip", i, repr(e))
    if new: buf += new; buf = buf[-BUF:]; idle = 0
    if len(buf) < 8:
        idle += 1; time.sleep(30); continue
    if not new and idle > 0: pass
    # кроки: на кожну нову мережу ~8 кроків (буфер перемішаний), або чекаємо
    nsteps = SPN * len(new) if new else 0   # лише за новими даними: справді потік, без перевчання буфера
    for _ in range(nsteps):
        idx = np.random.randint(len(buf), size=BS); loss = 0.0
        for i in idx:
            t = buf[i]; rh = net(t["X"], t["M"]); loss = loss + ((rh - t["r"]) ** 2).mean() / (t["r"] ** 2).mean()
        loss = loss / BS; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
        if step % EVAL == 0:
            r2t, rt = evaluate(test); r2b, rb = evaluate(buf[-64:])
            LOG("step %6d бачено %4d буфер %3d | TEST mini-100 R² %+.4f ratio %.4f | train(64) R² %+.4f ratio %.4f | loss %.4f" % (step, len(seen), len(buf), r2t, rt, r2b, rb, float(loss)))
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, seen=sorted(seen)), CK + ".tmp"); os.replace(CK + ".tmp", CK)
    if not new: time.sleep(20)
