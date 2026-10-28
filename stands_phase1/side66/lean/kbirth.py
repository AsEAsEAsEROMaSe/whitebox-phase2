"""κ₅-хаб у народженні κ₃ (вивід 23.09): E[p_a p_b r_c] ∋ c3_c·κ5(a,b,c,c,c)/σ_c³, деревне κ5(a,b,c,c,c) ≈ κ3(a,b,c)·κ4_c/σ_c²
⟹ внесок = перенос ІСТИННОГО κ3 з воротами g_c замість c1_c на ОДНІЙ нозі: F(g) = E[x_i² z_j] + 2E[x_i z_i x_j],
x = (c1⊙pc)@W, z = (g⊙pc)@W (pc центр. передактивації шару l−1, W = weights[l]). Ціль: Rb (pieces_useall, шари 12..15).
Ворота: g1 = c3·κ4/σ⁵ (дерево) · g2 = c3/σ (без κ4) · g3 = c1·κ4/σ⁴ (γ₄ на c1) · g4 = c3·γ3/σ² · 🔒 g1 перемішані.
Критерій ДО прогону: R²(Rb | F(g1)) ≥ 0.3 і перемішані ≈ 0; шанс 25%."""
import sys, numpy as np, torch
dev = "cuda"; i = int(sys.argv[1]); N = 2**21; CH = 2**15; LAYS = [12, 13, 14, 15]
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); W = torch.tensor(d["weights"], device=dev); L, n, _ = W.shape
P = np.load("/home/trim/wb2data/pieces_useall_%d.npz" % i)
SRC = [l-1 for l in LAYS]
g = torch.Generator(device=dev); g.manual_seed(4242 + i)
def fwd():
    z = torch.randn(CH, n, device=dev, generator=g); p = z @ W[0]
    for l in range(L):
        yield l, p
        if l < L-1: p = torch.relu(p) @ W[l+1]
# pass 1: моменти для SRC
S = {l: torch.zeros(8, n, dtype=torch.float64, device=dev) for l in SRC}
for c in range(N // CH):
    for l, p in fwd():
        if l in S:
            pd = p.double(); h = torch.relu(pd); q = torch.ones_like(pd)
            for k in range(5): S[l][k] += q.sum(0); q = q*pd
            S[l][5] += h.sum(0); S[l][6] += (h*pd).sum(0)
        if l >= max(SRC): break
st = {}
for l in SRC:
    m = [S[l][k]/N for k in range(5)]; a = m[1]; v = m[2]-a*a; s = v.sqrt()
    mu3 = m[3]-3*a*m[2]+2*a**3; mu4 = m[4]-4*a*m[3]+6*a*a*m[2]-3*a**4; k4 = mu4-3*v*v
    c0 = S[l][5]/N; c1 = (S[l][6]/N - c0*a)/v
    st[l] = dict(a=a, v=v, s=s, k3=mu3, k4=k4, c1=c1, H3=torch.zeros(n, dtype=torch.float64, device=dev))
g.manual_seed(4242 + i)
for c in range(N // CH):
    for l, p in fwd():
        if l in st:
            zz = (p.double()-st[l]["a"])/st[l]["s"]; st[l]["H3"] += (torch.relu(p.double())*(zz**3-3*zz)).sum(0)
        if l >= max(SRC): break
gates = {}
for l in SRC:
    x = st[l]; c3 = x["H3"]/N/6.0; s = x["s"]
    g1 = c3*x["k4"]/s**5; gates[l] = {"g1 c3·κ4/σ⁵": g1, "g2 c3/σ": c3/s, "g3 c1·κ4/σ⁴": x["c1"]*x["k4"]/s**4, "g4 c3·γ3/σ²": c3*(x["k3"]/s**3)/s**2,
                                      "🔒g1 перемішані": g1[torch.randperm(n, generator=torch.Generator(device=dev).manual_seed(7), device=dev)]}
# pass 2: кандидати F(g)
F = {(l, k): torch.zeros(n, n, dtype=torch.float64, device=dev) for l in SRC for k in gates[l]}
g.manual_seed(999 + i)
for c in range(N // CH):
    for l, p in fwd():
        if l in st:
            pc = p - st[l]["a"].float(); Wn = W[l+1]; xx = (st[l]["c1"].float()*pc) @ Wn
            for k, gv in gates[l].items():
                zv = (gv.float()*pc) @ Wn
                F[(l, k)] += ((xx*xx).T @ zv + 2*(xx*zv).T @ xx).double()
        if l >= max(SRC): break
off = ~torch.eye(n, dtype=torch.bool, device=dev)
for l in SRC:
    lo = l+1; Rb = torch.tensor(P["Rb_%d" % lo], device=dev, dtype=torch.float64)[off]
    cols = []; line = "dump%d l=%2d ‖Rb‖ " % (i, lo) + "%.3e" % float(Rb.norm()) + " | R²(коеф):"
    for k in gates[l]:
        f = (F[(l, k)]/N)[off]; a_ = float((f*Rb).sum()/(f*f).sum()); r2 = 1 - float(((Rb-a_*f)**2).sum()/(Rb*Rb).sum())
        line += " %s %.3f(%.2g)" % (k, r2, a_)
        if not k.startswith("🔒"): cols.append(f)
    X = torch.stack(cols, 1); sol = torch.linalg.lstsq(X, Rb[:, None]).solution[:, 0]; r2a = 1 - float(((Rb - X@sol)**2).sum()/(Rb*Rb).sum())
    print(line + " | усі 4: %.3f" % r2a, flush=True)
print("DONE_KBIRTH")
