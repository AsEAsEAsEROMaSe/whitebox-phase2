"""Одношаговий розклад народження κ₂₁ (стенд С122, 19.09.2026). h = c0 + c1·(p−a) + r (r = нелінійний залишок ReLU по нейрону),
p' = h@W. Тоді центральний (2,1)-момент p' = LLL + A1 + A2 + A3 за числом залишків r у трьох множниках:
  x = (c1·(p−a))@W (лінійний перенос κ₃(p)), y = r@W (центрований), T[i,j] = E[(x+y)_i² (x+y)_j]
  LLL = E[x_i² x_j] · A1 = E[2x_i y_i x_j + x_i² y_j] (хаб: один залишок) · A2 = E[y_i² x_j + 2x_i y_i y_j] · A3 = E[y_i² y_j]
  same-нейронні частини A2/A3 (обидва/три r на одному нейроні a): через M2[a,k]=E[r_a² x_k], E[r_a³]; cross = total − same.
Питання: які частки T несуть A2_cross і A3_cross (K=3-побудова 504aldo їх не народжує), і чи нев'язка його D21 (his_d21_dump%d.npz)
пояснюється ними. Пломби: T(семпли) ≈ mu21(банк) R²>0.99 на всіх шарах; сума частин = T до 1e-6. Передбачення: A2_cross+A3_cross
~3–5% ‖T‖ і R² нев'язки на них ≥0.5 (шанс 35%); альтернатива: нев'язка сидить в A1 (неточний хаб/зв'язок) — шанс 45%."""
import os, sys, time, numpy as np, torch
dev = "cuda"; i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N = int(os.environ.get("N", str(2**20))); CH = 2**15; seed = int(os.environ.get("SEED", "1"))
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); W = torch.tensor(d["weights"], device=dev)   # (16, n, n)
kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i); HIS = os.environ.get("HIS", "/home/trim/wb2lean/his_d21_dump%d.npz"); his = np.load(HIS % i) if os.path.exists(HIS % i) else None
L, n, _ = W.shape; layers = list(range(0, L-1))
def fwd(gen):
    z = torch.randn(CH, n, device=dev, generator=gen); p = z @ W[0]
    for l in range(L):
        yield l, p
        if l < L-1: p = torch.relu(p) @ W[l+1]
# pass 1: per-neuron a, E[h], E[hp], E[p²]
g = torch.Generator(device=dev); g.manual_seed(seed)
S = {k: torch.zeros(L, n, dtype=torch.float64, device=dev) for k in ("p", "p2", "h", "hp")}
for c in range(N // CH):
    for l, p in fwd(g):
        h = torch.relu(p); pd = p.double(); hd = h.double()
        S["p"][l] += pd.sum(0); S["p2"][l] += (pd*pd).sum(0); S["h"][l] += hd.sum(0); S["hp"][l] += (hd*pd).sum(0)
a = S["p"]/N; var = S["p2"]/N - a*a; c0 = S["h"]/N; c1 = (S["hp"]/N - c0*a)/var
from math import sqrt
c1g = 0.5*(1+torch.erf(a/var.sqrt()/sqrt(2)))   # гаусів c1 = Φ(t): це w1 його ланцюга
for l in range(L): print("l=%2d c1_true/Φ(t) − 1: RMS %.4f mean %+.4f | corr(δ, γ₃) %.2f" % (l, float(((c1[l]/c1g[l]-1)**2).mean().sqrt()), float((c1[l]/c1g[l]-1).mean()), float(torch.corrcoef(torch.stack([c1[l]/c1g[l]-1, torch.tensor(kp["pre_m3"][l], device=dev, dtype=torch.float64)]))[0,1])), flush=True)
print("пломба a: max|a−банк|/sd = %.2e" % float(((a - torch.tensor(kp["pre_mean"], device=dev, dtype=torch.float64)).abs()/var.sqrt()).max()), flush=True)
# pass 2: pieces
def acc(u, v, w): return (u*v).T.double() @ w.double() if False else ((u*v).T @ w).double()
P = {k: torch.zeros(L, n, n, dtype=torch.float64, device=dev) for k in ("T", "LLL", "A1", "A2", "A3", "M2", "LLLg", "A1g")}
H3 = torch.zeros(L, n, dtype=torch.float64, device=dev); R3 = torch.zeros(L, n, dtype=torch.float64, device=dev); G2 = torch.zeros(L, n, dtype=torch.float64, device=dev); CS = torch.zeros(L, n, n, dtype=torch.float64, device=dev)
g = torch.Generator(device=dev); g.manual_seed(seed); t0 = time.perf_counter()
for c in range(N // CH):
    for l, p in fwd(g):
        if l == L-1: break
        pc = p - a[l].float(); h = torch.relu(p); r = h - c0[l].float() - c1[l].float()*pc
        x = (c1[l].float()*pc) @ W[l+1]; y = r @ W[l+1]; s = x + y
        xg = (c1g[l].float()*pc) @ W[l+1]; yg = s - xg
        P["LLLg"][l+1] += acc(xg, xg, xg); P["A1g"][l+1] += 2*acc(xg, yg, xg) + acc(xg, xg, yg)
        P["T"][l+1] += acc(s, s, s); P["LLL"][l+1] += acc(x, x, x)
        P["A1"][l+1] += 2*acc(x, y, x) + acc(x, x, y); P["A2"][l+1] += acc(y, y, x) + 2*acc(x, y, y); P["A3"][l+1] += acc(y, y, y)
        P["M2"][l+1] += ((r*r).T @ x).double(); R3[l+1] += (r*r*r).double().sum(0); G2[l] += (pc*pc*r).double().sum(0); CS[l] += (pc.T @ pc).double(); zz = pc / var[l].sqrt().float(); H3[l] += (h*(zz*zz*zz - 3*zz)).double().sum(0)
for k in P: P[k] /= N
R3 /= N; G2 /= N; CS /= N; H3 /= N
print("pass2 %.0fs" % (time.perf_counter()-t0), flush=True)
def r2(x, y):
    a_ = float((x*y).sum()/(x*x).sum()); return 1 - float(((y - a_*x)**2).sum()/(y*y).sum()), a_
offm = ~torch.eye(n, dtype=torch.bool, device=dev)
def nrm(M): return float(M[offm].pow(2).sum().sqrt())
for l in range(1, L):
    Wl = W[l].double(); T = P["T"][l]; M2 = P["M2"][l]
    A2s = (Wl*Wl).T @ M2 + 2*((Wl*M2).T @ Wl)          # y_i² x_j same + 2 x_i y_i y_j same
    A3s = (Wl*Wl*R3[l][:, None]).T @ Wl
    A2c = P["A2"][l] - A2s; A3c = P["A3"][l] - A3s
    # банк
    kpa = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); M11 = kp["pre_M11"][l].astype(np.float64); M21 = kp["pre_M21"][l].astype(np.float64)
    mu21 = torch.tensor(M21 - 2*kpa[:, None]*M11 - kpa[None, :]*E2[:, None] + 2*np.outer(kpa*kpa, kpa), device=dev)
    rb, ab = r2(T[offm], mu21[offm]); sumchk = float((P["LLL"][l]+P["A1"][l]+P["A2"][l]+P["A3"][l]-T)[offm].abs().max()/T[offm].abs().max())
    nT = nrm(T); line = "l=%2d пломба T~банк R² %.4f нахил %.3f сума %.1e | ‖·‖/‖T‖: LLL %.3f A1 %.3f A2 %.3f (same %.3f cross %.3f) A3 %.3f (same %.3f cross %.3f)" % (
        l, rb, ab, sumchk, nrm(P["LLL"][l])/nT, nrm(P["A1"][l])/nT, nrm(P["A2"][l])/nT, nrm(A2s)/nT, nrm(A2c)/nT, nrm(P["A3"][l])/nT, nrm(A3s)/nT, nrm(A3c)/nT)
    if his is not None:
        H = torch.tensor(his["his_%d" % l], device=dev, dtype=torch.float64); Rh = mu21 - H; Rh[~offm] = 0
        dL = P["LLLg"][l] - P["LLL"][l]; dA = P["A1g"][l] - P["A1"][l]
        # гаусів хаб: E[p_a p_b r_c] = β_ac β_bc E[p_c² r_c], β_ac = C_ac/C_cc  ⟹ A1g2 = (Z∘Z)ᵀ(g∘W) + 2 (Z∘W)ᵀ(g∘Z), Z = diag(1/var) C diag(c1) W
        lp = l-1
        kpa0 = kp["pre_mean"][lp].astype(np.float64); E20 = kp["pre_m2"][lp].astype(np.float64); M110 = kp["pre_M11"][lp].astype(np.float64); M210 = kp["pre_M21"][lp].astype(np.float64); E30 = kp["pre_m3"][lp].astype(np.float64)
        mu21p_full = torch.tensor(M210 - 2*kpa0[:, None]*M110 - kpa0[None, :]*E20[:, None] + 2*np.outer(kpa0*kpa0, kpa0), device=dev); mu21p_full[~offm] = 0
        D3t_prev = torch.tensor(E30 - 3*kpa0*E20 + 2*kpa0**3, device=dev)
        Cl = CS[lp]; Z = (Cl / Cl.diagonal()[:, None]) @ (c1[lp][:, None] * Wl); gb = G2[lp][:, None]
        A1gauss = (Z*Z).T @ (gb*Wl) + 2*(Z*Wl).T @ (gb*Z); A1ng = P["A1"][l] - A1gauss
        line += " | ‖A1−A1gauss‖/‖T‖ %.3f (A1gauss~A1 R² %.3f)" % (nrm(A1ng)/nT, r2(A1gauss[offm], P["A1"][l][offm])[0])
        # перенос нев'язки попереднього шару як slice-supported тензора: K = c1²⊗c1·R_l, κ = c1³·ΔD3
        if his is not None and l >= 2:
            kpa0 = kp["pre_mean"][lp].astype(np.float64); E20 = kp["pre_m2"][lp].astype(np.float64); M110 = kp["pre_M11"][lp].astype(np.float64); M210 = kp["pre_M21"][lp].astype(np.float64); E30 = kp["pre_m3"][lp].astype(np.float64)
            mu21p = torch.tensor(M210 - 2*kpa0[:, None]*M110 - kpa0[None, :]*E20[:, None] + 2*np.outer(kpa0*kpa0, kpa0), device=dev)
            Rp = mu21p - torch.tensor(his["his_%d" % lp], device=dev, dtype=torch.float64); Rp[~offm] = 0
            dD3 = torch.tensor(E30 - 3*kpa0*E20 + 2*kpa0**3, device=dev) - torch.tensor(his["d3_%d" % lp], device=dev, dtype=torch.float64)
            K = (c1[lp]**2)[:, None] * c1[lp][None, :] * Rp; kap = c1[lp]**3 * dD3
            Rpred = (Wl*Wl).T @ (K @ Wl) + 2*(Wl*(K @ Wl)).T @ Wl + (Wl*Wl*kap[:, None]).T @ Wl
            rr, aa = r2(Rpred[offm], Rh[offm]); inj = Rh - aa*Rpred
            line += " | перенос R_{l-1}→R_l: R² %.3f нахил %.3f ‖pred‖/‖R‖ %.3f; ін'єкція/‖T‖ %.3f, її R² на A1ng %.3f A2c %.3f" % (rr, aa, nrm(Rpred)/nrm(Rh), nrm(inj)/nT, r2(A1ng[offm], inj[offm])[0], r2(A2c[offm], inj[offm])[0])
        line += " | ‖LLLg−LLL‖/‖T‖ %.3f ‖A1g−A1‖/‖T‖ %.3f" % (nrm(dL)/nT, nrm(dA)/nT)
        # slice-only перенос ІСТИНИ: тензор, що живе лише на зрізах (T_aac = D21[a,c], T_aaa = D3) — скільки наступного зрізу дає
        # позазрізовий вміст κ₃(p)? Lsl = (W∘W)ᵀ K W + 2 (W∘(K W))ᵀ W + (W∘W∘κ)ᵀ W, K = c1²⊗c1·D21_true, κ = c1³·D3_true
        Kt = (c1[lp]**2)[:, None] * c1[lp][None, :] * mu21p_full; kap_t = c1[lp]**3 * D3t_prev
        Lsl = (Wl*Wl).T @ (Kt @ Wl) + 2*(Wl*(Kt @ Wl)).T @ Wl + (Wl*Wl*kap_t[:, None]).T @ Wl
        rs, as_ = r2(Lsl[offm], P["LLL"][l][offm]); dsl = P["LLL"][l] - as_*Lsl
        line += " | SLICE-ONLY перенос істини: R² %.4f нахил %.3f ‖LLL−a·Lsl‖/‖T‖ %.3f (без a: %.3f)" % (rs, as_, nrm(dsl)/nT, nrm(P["LLL"][l]-Lsl)/nT)
        # ЗАМИКАННЯ позазрізового внеску O = LLL − Lsl через n²-об'єкти шару l−1 (чи можлива парна κ₃-рекурсія без ніг)
        O = P["LLL"][l] - Lsl; O[~offm] = 0
        Ct = c1[lp][:, None] * CS[lp] * c1[lp][None, :]; Cn = Wl.T @ Ct @ Wl; Kn = Wl.T @ Kt @ Wl; KWl = Kt @ Wl; KtW = Kt.T @ Wl; CW = Ct @ Wl
        cl = {"B3": (Wl*KtW).T @ Wl, "B4": (Wl*Wl).T @ KtW, "B5": (Wl*CW).T @ KWl, "B6": (Wl*KWl).T @ CW, "B7": (Wl*CW).T @ KtW,
              "Cn*Kn": Cn*Kn, "Cn*KnT": Cn*Kn.T, "dKn*Cn": Kn.diagonal()[:, None]*Cn, "Cn*Cn*d": (Cn*Cn)*(kap_t @ Wl)[None, :], "Cn*dKnj": Cn*Kn.diagonal()[None, :],
              "Lsl": Lsl, "CnCn": Cn @ Cn}
        yo = O[offm]; Xo = torch.stack([v[offm] for v in cl.values()], 1); sso = float((yo*yo).sum())
        def fito(cols):
            Xc = Xo[:, cols]; sol = torch.linalg.lstsq(Xc, yo[:, None]).solution[:, 0]; return 1 - float(((yo - Xc@sol)**2).sum()/sso), sol
        nm = list(cl); sing = " ".join("%s %.3f" % (k, fito([j])[0]) for j, k in enumerate(nm)); rall, sol = fito(list(range(len(nm))))
        line += " | ЗАМИКАННЯ O (%.3f T): " % (nrm(O)/nT) + sing + " | ВСІ %.3f" % rall
        parts = {"LLL": P["LLL"][l], "A1": P["A1"][l], "A2c": A2c, "A3c": A3c, "A2c+A3c": A2c+A3c, "dLLL": dL, "A1ng": A1ng}
        if ("old_%d" % l) in his.files:
            Ho = torch.tensor(his["old_%d" % l], device=dev, dtype=torch.float64); Hb = H - Ho
            Rold = P["LLL"][l] - Ho; Rold[~offm] = 0; Rb = (T - P["LLL"][l]) - Hb; Rb[~offm] = 0
            line += " | РОЗКЛАД R: перенос (LLL−old) %.3f, народження (birth−his_birth) %.3f [‖·‖/‖T‖]; R² R на них %.3f / %.3f; ‖old‖/‖LLL‖ %.3f ‖his_birth‖/‖birth‖ %.3f nsrc %s" % (
                nrm(Rold)/nT, nrm(Rb)/nT, r2(Rold[offm], Rh[offm])[0], r2(Rb[offm], Rh[offm])[0], nrm(Ho)/nrm(P["LLL"][l]), nrm(Hb)/nrm(T-P["LLL"][l]), his["nsrc_%d" % l].tolist())
            parts["Rold"] = Rold; parts["Rbirth"] = Rb
            # кандидат на пропущений член народження: c̃3_c·γ3_c·Cov(lin_a,lin_b) (з E[p̃_a p̃_b He3(p̃_c)] = κ5 + ρ_ab γ3_c + 3ρ_ac κ̃21[c,b] + 3ρ_bc κ̃21[c,a])
            g3 = torch.tensor(kp["pre_m3"][lp].astype(np.float64) - 3*kpa0*E20 + 2*kpa0**3, device=dev) / var[lp]**1.5
            q3 = (H3[lp]/6.0) * g3; CL = c1[lp][:, None] * CS[lp] * c1[lp][None, :]; Mm = Wl.T @ CL @ Wl; qW = q3 @ Wl
            T1 = Mm.diagonal()[:, None] * qW[None, :] + 2*qW[:, None] * Mm; T1[~offm] = 0
            cands = {"A1ng": A1ng, "A2c": A2c, "A3c": A3c, "T1(D3·C)": T1, "A1gauss": A1gauss}
            line += " | НАРОДЖЕННЯ Rb (%.3f T) R²(coef) на: " % (nrm(Rb)/nT) + " ".join("%s %.3f(%.2f)" % ((k,)+r2(v[offm], Rb[offm])) for k, v in cands.items())
            yb = Rb[offm]; Xb = torch.stack([cands[k][offm] for k in ("A1ng", "A2c", "A3c", "T1(D3·C)")], 1); solb = torch.linalg.lstsq(Xb, yb[:, None]).solution[:, 0]
            line += " | МНК: R² %.3f coef %s | ‖T1‖/‖T‖ %.3f, T1~R R² %.3f" % (1 - float(((yb - Xb@solb)**2).sum()/(yb*yb).sum()), ["%.2f" % float(v) for v in solb], nrm(T1)/nT, r2(T1[offm], Rh[offm])[0])
        if os.environ.get("SAVEP"):
            SVP = globals().setdefault("SVP", {})
            for k in ("LLL", "A1", "A2", "A3", "LLLg", "A1g"): SVP["%s_%d" % (k, l)] = P[k][l].float().cpu().numpy()
            for k in ("A2c", "A3c", "A1ng"): SVP["%s_%d" % (k, l)] = parts[k].float().cpu().numpy()
            if "Rbirth" in parts: SVP["Rb_%d" % l] = parts["Rbirth"].float().cpu().numpy(); SVP["Rold_%d" % l] = parts["Rold"].float().cpu().numpy()
        line += " | нев'язка його (ε %.3f) R² на: " % (nrm(Rh)/nT) + " ".join("%s %.3f(%.2f)" % ((k,)+r2(v[offm], Rh[offm])) for k, v in parts.items())
        y = Rh[offm]; X = torch.stack([parts[k][offm] for k in ("LLL", "A1", "A2c", "A3c")], 1); sol = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
        line += " | МНК 4: R² %.3f coef %s" % (1 - float(((y - X@sol)**2).sum()/(y*y).sum()), ["%.2f" % float(v) for v in sol])
    print(line, flush=True)
if os.environ.get("SAVEP"): np.savez(os.environ["SAVEP"] % i, **globals()["SVP"]); print("saved pieces", os.environ["SAVEP"] % i, flush=True)
