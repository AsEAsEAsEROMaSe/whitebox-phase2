"""«Міжфазова смуга» (ідея юзера): чи народжується похибка κ₃ (Rb, при майже точних входах, pieces_useall) у нейронах-джерелах
БІЛЯ ПОРОГУ ReLU. Модель «понейронне народження»: Rb ≈ Σ_a q_a F_a, F_a[i,j] = W[a,i]² W[a,j] (позадіаг.), W = weights[l]
(рядок a = нейрон шару l−1). МНК по q (1024 невідомих) на половині пар, R² на другій половині (контроль перенавчання).
🔒 контроль: той самий фіт на ПЕРЕМІШАНИХ рядках W (чужі джерела) мусить дати ≈0. Концентрація: частка енергії внесків
q_a²‖F_a‖² у 15% нейронів з найменшим |t_a| (t_a = a/σ шару l−1 з банку); еталон рівномірного = 15%.
ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: R²_test понейронної моделі ≥ 0.3 (шанс 40%); якщо так, ≥50% енергії в 15% біля порогу (шанс 30%)."""
import sys, numpy as np, torch
dev = "cuda"; DUMPS = [int(x) for x in sys.argv[1].split(",")]; LAYS = list(range(3, 16))
def solve(Wl, Rb, mtr, mte):
    n = Wl.shape[0]; W2 = Wl*Wl
    # F_a[i,j] = W2[a,i] W[a,j]; G_ab = Σ_{ij∈tr} F_a F_b ; b_a = Σ_{ij∈tr} F_a Rb
    # через маску: G = Σ_ij m_ij W2[a,i]W[a,j]W2[b,i]W[b,j] = ((W2 ⊙ ...)) — рахуємо блоками по i
    G = torch.zeros(n, n, dtype=torch.float64, device=dev); bvec = torch.zeros(n, dtype=torch.float64, device=dev)
    for i0 in range(0, n, 64):
        i1 = min(n, i0+64)
        F = W2[:, i0:i1, None] * Wl[:, None, :]           # (a, i-блок, j)
        F = F * mtr[None, i0:i1, :]
        Ff = F.reshape(n, -1)
        G += (Ff @ Ff.T).double(); bvec += (Ff @ Rb[i0:i1].reshape(-1)).double()
    q = torch.linalg.solve(G + 1e-12*G.diagonal().mean()*torch.eye(n, device=dev, dtype=torch.float64), bvec)
    pred = (W2.T.double() * q[None, :]) @ Wl.double()     # Σ_a q_a W2[a,i] W[a,j]
    def r2(m): y = Rb.double()[m]; p = pred[m]; return 1 - float(((y-p)**2).sum()/(y*y).sum())
    return q, r2(mtr > 0.5), r2(mte), pred
for i in DUMPS:
    d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); Wall = d["weights"]
    kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i); P = np.load("/home/trim/wb2data/pieces_useall_%d.npz" % i)
    g = torch.Generator(device=dev); g.manual_seed(5 + i)
    for l in LAYS:
        if "Rb_%d" % l not in P.files: continue
        Wl = torch.tensor(Wall[l], device=dev, dtype=torch.float32); n = Wl.shape[0]
        Rb = torch.tensor(P["Rb_%d" % l], device=dev, dtype=torch.float32); Rb.fill_diagonal_(0.0)
        off = ~torch.eye(n, dtype=torch.bool, device=dev); half = torch.rand(n, n, generator=g, device=dev) < 0.5
        mtr = (off & half).float(); mte = off & ~half
        q, r2tr, r2te, pred = solve(Wl, Rb, mtr, mte)
        perm = torch.randperm(n, generator=g, device=dev); _, _, r2sh, _ = solve(Wl[perm][:, torch.randperm(n, generator=g, device=dev)], Rb, mtr, mte)
        a = kp["pre_mean"][l-1].astype(np.float64); v = kp["pre_m2"][l-1].astype(np.float64) - a*a; t = a/np.sqrt(v)
        Fn2 = ((Wl*Wl)**2).sum(1) * (Wl*Wl).sum(1)          # ≈ ‖F_a‖²
        en = (q.cpu().numpy()**2) * Fn2.double().cpu().numpy()
        order = np.argsort(np.abs(t)); k15 = int(0.15*n)
        frac = float(en[order[:k15]].sum()/en.sum())
        cq = float(np.corrcoef(np.abs(q.cpu().numpy()), np.abs(t))[0, 1])
        print("dump%d l=%2d  понейронне народження R² train %.3f test %.3f | 🔒чужі джерела %.3f | енергія в 15%% біля порогу %.3f (еталон 0.150) | corr(|q|,|t|) %+.3f | мед|t| %.2f"
              % (i, l, r2tr, r2te, r2sh, frac, cq, float(np.median(np.abs(t)))), flush=True)
print("DONE_TBAND")
