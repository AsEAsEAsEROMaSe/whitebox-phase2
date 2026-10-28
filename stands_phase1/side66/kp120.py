"""kp120 (18.09.2026): носій зрізки (3,1) κ₄ передактивацій (κ₃₁) — хаб-рука-модель на ІСТИННОМУ стані (банк keenanpepper N=1e8).
Механізм: κ₄(h)_aaab (три однакові індекси + один інший) ≈ q(a)·c₁(b)·ρ_ab до першого порядку в ρ (Stein: E[g(z_a)He₁(z_b)] = ρ E[g']).
  q(a) = 3E[h̃_a² F'] − 3Var(h_a)c₁(a) = 3σ³(1−Φ)[Φ(1+t²) + tφ − 2(φ+tΦ)²],  c₁(b) = σ_b Φ(t_b),  t = μ/σ передактивації, h̃ центроване.
Перенос: хаб-нога U^{g→l} = (U∘Φ)@W (як kp119d), рука V^{g→l} = ((c₁∘ρ)@W) далі так само; власний ексцес κ₄(h)_aaaa окремо.
κ₃₁(p_l)_ij = Σ_g Σ_a { κ₄aaaa(a) U_ai³ U_aj + q(a)[U_ai³ V_aj + 3 U_ai² V_ai U_aj] }.
🔒 пломби: (1) МК на 2-D гаусовій парі: κ(h_a,h_a,h_a,h_b) проти q(a)c₁(b)ρ при ρ=0.1 (перший порядок), кілька t;
           (2) шар l=1 (p гаусів точно на l=0 ⟹ κ₄(h₀) = власне + пара) мусить дати R² ≥ 0.9.
Передбачення ДО прогону: R²(усі джерела) на l=15 ~0.8–0.9 (як хаб κ₂₁ 0.95); молоді ≤4 джерела ~0.6; шанс молодих ≥0.9: 25%.
Вихід: R² по шарах і за віком; лог /home/trim/wblogs/kp120.log"""
import os, sys, time, math, numpy as np, torch
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
DEV = os.environ.get("DEV", "cuda"); DT = torch.float64
NETS = [int(x) for x in os.environ.get("NETS", "0").split(",")]
DUMP = "/home/trim/wb2data/dumps_p2/mlp_{:04d}.npz"; BANK = "/home/trim/wb2data/kp_p2/mlp_{:05d}.npz"
SQ2PI = math.sqrt(2*math.pi)
def phi(t): return torch.exp(-0.5*t*t)/SQ2PI
def Phi(t): return 0.5*(1 + torch.erf(t/math.sqrt(2)))
def coefs(mu, sd):
    t = mu/sd; P = Phi(t); p = phi(t); m1 = p + t*P
    c1 = sd*P
    var = sd*sd*(P*(1+t*t) + t*p - m1*m1)
    q = 3*sd**3*(1-P)*(P*(1+t*t) + t*p - 2*m1*m1)
    return c1, var, q, t
def true_k31(kp, l):
    a = torch.as_tensor(kp["pre_mean"][l], dtype=DT, device=DEV); E2 = torch.as_tensor(kp["pre_m2"][l], dtype=DT, device=DEV); E3 = torch.as_tensor(kp["pre_m3"][l], dtype=DT, device=DEV)
    M11 = torch.as_tensor(kp["pre_M11"][l], dtype=DT, device=DEV); M21 = torch.as_tensor(kp["pre_M21"][l], dtype=DT, device=DEV); M31 = torch.as_tensor(kp["pre_M31"][l], dtype=DT, device=DEV)
    C11 = M11 - torch.outer(a, a)
    mu31 = M31 - 3*a[:, None]*M21 + 3*(a*a)[:, None]*M11 - a[None, :]*E3[:, None] + 3*a[None, :]*(a[:, None]*E2[:, None]) - 3*torch.outer(a**3, a)
    v = E2 - a*a
    return mu31 - 3*v[:, None]*C11, C11, a, torch.sqrt(v)
def r2(M, T):
    m = ~torch.eye(M.shape[0], dtype=torch.bool, device=DEV); x = M[m]; y = T[m]
    a = float((x*y).sum()/(x*x).sum()); return 1 - float(((y - a*x)**2).sum()/(y*y).sum()), a
# --- пломба 1: МК 2-D
g = torch.Generator(device="cpu").manual_seed(7); N = 8_000_000; rho = float(os.environ.get("RHO", "0.02"))
for ta, tb in ((0.5, -0.3), (1.5, 1.0), (-0.5, 0.8)):
    z = torch.randn(N, 2, generator=g, dtype=torch.float64); zb = rho*z[:, 0] + math.sqrt(1-rho*rho)*z[:, 1]; za = z[:, 0]
    sa, sb = 1.3, 0.8; ha = torch.relu(sa*(za+ta)); hb = torch.relu(sb*(zb+tb)); ha = ha - ha.mean(); hb = hb - hb.mean()
    k = float((ha**3*hb).mean() - 3*(ha*ha).mean()*(ha*hb).mean())
    c1b, _, qa, _ = coefs(torch.tensor([sb*tb, sa*ta], dtype=DT), torch.tensor([sb, sa], dtype=DT))
    c1b = coefs(torch.tensor([sb*tb], dtype=DT), torch.tensor([sb], dtype=DT))[0]; qa = coefs(torch.tensor([sa*ta], dtype=DT), torch.tensor([sa], dtype=DT))[2]
    lg("🔒 МК t_a=%.1f t_b=%.1f: κ(aaab)=%.5e  модель q c₁ ρ=%.5e  (відн. %.3f; O(ρ³)≈1%%)" % (ta, tb, k, float(qa*c1b)*rho, k/(float(qa*c1b)*rho)))
for i in NETS:
    d = np.load(DUMP.format(i)); kp = np.load(BANK.format(i)); nm = str(d["mlp_name"])
    W = [torch.as_tensor(d["weights"][l], dtype=DT, device=DEV).T.contiguous() for l in range(16)]   # h @ W[l] : (in a, out i)
    hm = torch.as_tensor(kp["mean"], dtype=DT, device=DEV); hm2 = torch.as_tensor(kp["post_m2"], dtype=DT, device=DEV); hm3 = torch.as_tensor(kp["post_m3"], dtype=DT, device=DEV); hm4 = torch.as_tensor(kp["post_m4"], dtype=DT, device=DEV)
    hv = hm2 - hm*hm; k4h = hm4 - 4*hm*hm3 + 6*hm*hm*hm2 - 3*hm**4 - 3*hv*hv
    # перевірка орієнтації W: p_1 = W[1]ᵀ? ми хочемо p_i(l) = Σ_a h_a(l-1) W[l][a,i]; звірити mean(p_1) з банком
    mu0 = torch.as_tensor(kp["pre_mean"][0], dtype=DT, device=DEV); sd0 = torch.sqrt(torch.as_tensor(kp["pre_m2"][0], dtype=DT, device=DEV) - mu0*mu0)
    m_h0 = hm[0]; p1 = m_h0 @ W[1]; err = float((p1 - torch.as_tensor(kp["pre_mean"][1], dtype=DT, device=DEV)).norm()/torch.as_tensor(kp["pre_mean"][1], dtype=DT, device=DEV).norm())
    if err > 0.05:
        W = [w.T.contiguous() for w in W]; p1 = m_h0 @ W[1]; err2 = float((p1 - torch.as_tensor(kp["pre_mean"][1], dtype=DT, device=DEV)).norm()/torch.as_tensor(kp["pre_mean"][1], dtype=DT, device=DEV).norm())
        lg("орієнтацію W перевернуто: %.3f → %.3f" % (err, err2))
    else: lg("%s (%d): орієнтація W ок (%.4f)" % (nm, i, err))
    chains = {}   # g -> (U, V, q, k4d)
    for l in range(1, 16):
        # стан шару l-1 (передактивації) з банку: t, σ, ρ
        mu = torch.as_tensor(kp["pre_mean"][l-1], dtype=DT, device=DEV); sd = torch.sqrt(torch.clamp(torch.as_tensor(kp["pre_m2"][l-1], dtype=DT, device=DEV) - mu*mu, min=1e-12))
        c1, var, q, t = coefs(mu, sd); Ph = Phi(t)
        C = torch.as_tensor(kp["pre_M11"][l-1], dtype=DT, device=DEV) - torch.outer(mu, mu); rho = C/torch.outer(sd, sd); rho.fill_diagonal_(0.0)
        for gk in list(chains):
            U, V, qq, kd = chains[gk]; chains[gk] = ((U*Ph[None, :]) @ W[l], (V*Ph[None, :]) @ W[l], qq, kd)   # U[a,i]: рядок a = нейрон народження
        chains[l-1] = (W[l].clone(), (c1[None, :]*rho) @ W[l], q, k4h[l-1])
        K = torch.zeros(1024, 1024, dtype=DT, device=DEV); Kt, C11, a_l, sd_l = true_k31(kp, l)
        parts = []
        for gk in sorted(chains, reverse=True):   # від наймолодшого
            U, V, qq, kd = chains[gk]; U2 = U*U; U3 = U2*U
            Kg = U3.T @ (kd[:, None]*U) + U3.T @ (qq[:, None]*V) + 3*(U2*V).T @ (qq[:, None]*U)
            K += Kg; parts.append((l-1-gk, r2(K, Kt)[0]))
        rr, sc = r2(K, Kt)
        Kd = sum((chains[gk][0]**3).T @ (chains[gk][3][:, None]*chains[gk][0]) for gk in chains); rd, _ = r2(Kd, Kt)
        if l in (1, 2, 4, 8, 12, 15):
            lg("  l=%2d κ₃₁: R² усі %.3f масштаб %.2f | лише власний ексцес %.3f | за віком (age:R²): %s" % (l, rr, sc, rd, " ".join("%d:%.2f" % p for p in parts[:8])))
    lg("%s готово" % nm)
print("DONE_KP120")
