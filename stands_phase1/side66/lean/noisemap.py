"""Ідея юзера: «де шум, там сидить щось, що важко міряти». Карта вибіркового розкиду оцінки T[i,j]=E[pc_i² pc_j]:
V[i,j] = E[pc_i⁴ pc_j²] − T² (6-ті моменти). Чи похибка народження Rb (pieces_useall, шари 12..15, де шум N=2²⁰ ≲10%)
живе там, де V велике, ПОНАД масштаб |T|? Метрики: corr(|Rb|, √V); частковий corr після МНК-зняття log|T| і log(σ_iσ_j...);
🔒 перемішана карта (перестановка пар) ≈ 0. Критерій ДО прогону: частковий corr(log|Rb|, log√V | log|T|, log σ²_iσ_j) ≥ 0.3,
шанс 20%."""
import sys, numpy as np, torch
dev = "cuda"; i = int(sys.argv[1]); N = 2**20; CH = 2**15; LAYS = [12, 13, 14, 15]
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); W = torch.tensor(d["weights"], device=dev)
kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i); P = np.load("/home/trim/wb2data/pieces_useall_%d.npz" % i)
L, n, _ = W.shape
a = {l: torch.tensor(kp["pre_mean"][l], device=dev, dtype=torch.float32) for l in LAYS}
S3 = {l: torch.zeros(n, n, dtype=torch.float64, device=dev) for l in LAYS}; S6 = {l: torch.zeros(n, n, dtype=torch.float64, device=dev) for l in LAYS}
g = torch.Generator(device=dev); g.manual_seed(77)
for c in range(N // CH):
    z = torch.randn(CH, n, device=dev, generator=g); p = z @ W[0]
    for l in range(L):
        if l in LAYS:
            pc = p - a[l]; q = pc*pc
            S3[l] += (q.T @ pc).double(); S6[l] += ((q*q).T @ q).double()
        if l < L-1: p = torch.relu(p) @ W[l+1]
off = ~torch.eye(n, dtype=torch.bool, device=dev)
def lg(x): return torch.log(x.abs() + 1e-30)
for l in LAYS:
    T = S3[l]/N; V = S6[l]/N - T*T; Rb = torch.tensor(P["Rb_%d" % l], device=dev, dtype=torch.float64)
    var = torch.tensor(kp["pre_m2"][l] - kp["pre_mean"][l]**2, device=dev, dtype=torch.float64)
    y = lg(Rb)[off]; x = 0.5*lg(V)[off]; tt = lg(T)[off]; sc = (lg(var)[:, None] + 0.5*lg(var)[None, :]).expand(n, n)[off]
    def corr(u, v): u = u - u.mean(); v = v - v.mean(); return float((u*v).sum()/(u.norm()*v.norm()))
    def resid(u, cols):
        X = torch.stack([torch.ones_like(u)] + cols, 1); sol = torch.linalg.lstsq(X, u[:, None]).solution[:, 0]; return u - X @ sol
    pc_ = corr(resid(y, [tt, sc]), resid(x, [tt, sc]))
    perm = torch.randperm(x.numel(), device=dev)
    print("dump%d l=%2d corr(log|Rb|, log√V) %.3f | corr(log|Rb|, log|T|) %.3f | corr(log√V, log|T|) %.3f | ЧАСТКОВИЙ (|T|, σ знято) %.3f | 🔒перемішана %.3f | corr(|Rb|,√V) лін %.3f"
          % (i, l, corr(y, x), corr(y, tt), corr(x, tt), pc_, corr(resid(y, [tt, sc]), resid(x[perm], [tt, sc])), corr(Rb.abs()[off], V.clamp_min(0).sqrt()[off])), flush=True)
print("DONE_NOISEMAP")
