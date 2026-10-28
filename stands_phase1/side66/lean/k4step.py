"""Одношаговий хаб κ₄ (стенд 26.09): κ₄(h_l) з діаграм k_max=4 на n²-блоках (+ κ₃-CP лише у конвертації pK→K),
перенесена W_{l+1} на 32 нейрони S ⟹ тривісний блок κ₄(p_{l+1})[a,a,b,c] проти істини brc (q4).
Режими: seal (n=32, W=I, істинні входи, проти референсу relu_kprop k_max=4 з тим самим списком діаграм) і chain
(входи ланцюга на шарі ls−1, перенос W_ls, проти істини на шарі ls).
Передбачення ДО прогону: R² одношагового хаба (усі зрізи (2,1,1)+(2,2)+(3,1)+(4)+море) проти істинного тривісного блоку
0.5–0.8 (шанс ≥0.8: 25%); (2,1,1)-зріз несе більшість (як зріз κ₃ несе 80% D21)."""
import os, sys, time, itertools, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import k4eng
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
from mlp_kprop.partitions import vec_part_coef
m = 32
EXT = "abcd"


def kind(v):
    nz = [x for x in v if x > 0]
    if len(nz) == 1: return "vec%d" % nz[0]
    if len(nz) == 2: return "pair%s" % (tuple(sorted(nz)),)
    if len(nz) == 3: return "k3" if tuple(sorted(nz)) == (1, 1, 1) else "tri"
    return "k4" if tuple(sorted(nz)) == (1, 1, 1, 1) else "quad"


OK_KINDS = ("vec2", "vec3", "vec4", "pair(1, 1)", "pair(1, 2)", "pair(2, 2)", "pair(1, 3)")


def computable(ip, T4):
    return [(vp, float(c) * float(vec_part_coef(vp, divide_fac=True))) for vp, c in T4[ip].items() if all(kind(v) in OK_KINDS for v in vp)]


def block_matrix(u, v, ku, kv, B):
    """M[u, v] для блоку з порядками (ku, kv)."""
    if (ku, kv) == (1, 1): return B["C"]
    if (ku, kv) == (2, 1): return B["D21"]
    if (ku, kv) == (1, 2): return B["D21"].T
    if (ku, kv) == (2, 2): return B["K22"]
    if (ku, kv) == (3, 1): return B["K31"]
    if (ku, kv) == (1, 3): return B["K31"].T
    raise ValueError((ku, kv))


def build(vp, ip, B, wick, assign, WS):
    """Diagram для vp на зрізі ip; assign: vertex -> tuple імен зовнішніх осей."""
    nv = len(ip); d = k4eng.Diagram(nv)
    korder = [0] * nv
    for v in vp:
        nz = [i for i in range(nv) if v[i] > 0]
        for i in nz: korder[i] += v[i]
        if len(nz) == 1:
            u = nz[0]; d.add_vec(u, {2: B["var"], 3: B["D3"], 4: B["K4"]}[v[u]])
        else:
            u, w = nz; d.add_block(u, w, block_matrix(u, w, v[u], v[w], B))
    for u in range(nv):
        d.add_vec(u, wick(korder[u], ip[u]))
        for nm in assign[u]: d.add_ext(u, nm, WS)
    return d


def assignments(ip):
    """(assign dict, вага) для зрізу ip: (2,1,1) → 6 пар × 2 порядки /2; (1,1,1,1) → 24 /24; (2,2) → 3; (3,1) → 4; (4,) → 1."""
    out = []
    if ip == (1, 1, 1, 1):
        for p in itertools.permutations(EXT): out.append(({i: (p[i],) for i in range(4)}, 1.0 / 24))
    elif ip == (2, 1, 1):
        for pair in itertools.combinations(EXT, 2):
            rest = [c for c in EXT if c not in pair]
            for o in (rest, rest[::-1]): out.append(({0: pair, 1: (o[0],), 2: (o[1],)}, 0.5))
    elif ip == (2, 2):
        for pair in (("a", "b"), ("a", "c"), ("a", "d")):
            rest = tuple(c for c in EXT if c not in pair); out.append(({0: pair, 1: rest}, 1.0))
    elif ip == (3, 1):
        for s in EXT: out.append(({0: tuple(c for c in EXT if c != s), 1: (s,)}, 1.0))
    elif ip == (4,):
        out.append(({0: tuple(EXT)}, 1.0))
    return out


def slice_sum(ip, diagrams, B, wick, WS, n, log=None):
    """Σ_діаграм coef · Σ_assign вага · masked_value → тензор (32,)*4."""
    acc = torch.zeros((m,) * 4, dtype=WS.dtype, device=WS.device); t0 = time.time()
    for j, (vp, coef) in enumerate(diagrams):
        for assign, wt in assignments(ip):
            d = build(vp, ip, B, wick, assign, WS)
            val = k4eng.masked_value(d, n)
            if val is not None: acc += (coef * wt) * val
        if log and (j % 50 == 49): log("   %s діаграм %d/%d (%.0f с)" % (ip, j + 1, len(diagrams), time.time() - t0))
    return acc


def n2_slices(Bpost, WS, n):
    """Перенесені (2,2), (3,1), (4) зрізи пост-κ₄ (уже кумулянти)."""
    acc = torch.zeros((m,) * 4, dtype=WS.dtype, device=WS.device)
    ones = lambda: torch.ones(n, dtype=WS.dtype, device=WS.device)
    for ip, key in (((2, 2), "K22"), ((3, 1), "K31")):
        for assign, wt in assignments(ip):
            d = k4eng.Diagram(2); d.add_block(0, 1, Bpost[key])
            for u in (0, 1):
                for nm in assign[u]: d.add_ext(u, nm, WS)
            names, T = k4eng.contract(d, n); acc += wt * k4eng.to_abcd(names, T)
    d = k4eng.Diagram(1); d.add_vec(0, Bpost["K4"])
    for nm in EXT: d.add_ext(0, nm, WS)
    names, T = k4eng.contract(d, n); acc += k4eng.to_abcd(names, T)
    return acc


def conversion_211(mu_h, C_h, D21h, D3h, triples, WS, n):
    """K(2,1,1) = pK − 2 C_h[i,k]C_h[i,l] − 2 m_h[i] κ₃h[i,k,l], з маскою різних i,k,l. triples: список (L0, L1, L2) CP κ₃h
    (несиметризовано; симетризуємо по 3 виборах ноги для вершини 0 і 2 порядках решти)."""
    acc = torch.zeros((m,) * 4, dtype=WS.dtype, device=WS.device); ones = torch.ones(n, dtype=WS.dtype, device=WS.device)
    Coff = C_h.clone(); Coff.fill_diagonal_(0.0)
    for assign, wt in assignments((2, 1, 1)):
        # −2 C C (зірка у вершині 0)
        d = k4eng.Diagram(3); d.add_block(0, 1, Coff); d.add_block(0, 2, Coff)
        for u in range(3):
            for nm in assign[u]: d.add_ext(u, nm, WS)
        acc += (-2.0 * wt) * k4eng.masked_value(d, n)
        # −2 m κ₃: повний CP член мінус збіги (n²-зрізи κ₃h)
        p, q = assign[0]; s, t = assign[1][0], assign[2][0]
        A = torch.einsum("pi,qi,i->pqi", WS, WS, mu_h)
        full = torch.zeros((m,) * 4, dtype=WS.dtype, device=WS.device)
        for (L0, L1, L2) in triples:
            for legs in ((L0, L1, L2), (L1, L0, L2), (L2, L0, L1)):   # яка нога у вершині 0 (1/3)
                a0 = A.reshape(-1, n) @ legs[0]                          # (pq, r)
                b1 = WS @ legs[1]; c2 = WS @ legs[2]
                Tm = torch.einsum("er,sr,tr->est", a0, b1, c2).reshape(m, m, m, m)
                full += Tm / 3.0
        # перестановка осей (p,q,s,t) -> a,b,c,d
        names = (p, q, s, t); full = k4eng.to_abcd(names, full)
        # збіги: i=k → κ₃h[i,i,l] = D21h[i,l]; i=l → D21h[i,k]; k=l → κ₃h[i,k,k] = D21h[k,i]; i=k=l → D3h
        def two(vec0, M, names0, names1):
            d2 = k4eng.Diagram(2); d2.add_vec(0, vec0); d2.add_block(0, 1, M)
            for nm in names0: d2.add_ext(0, nm, WS)
            for nm in names1: d2.add_ext(1, nm, WS)
            nm_, T_ = k4eng.contract(d2, n); return k4eng.to_abcd(nm_, T_)
        D21o = D21h.clone(); D21o.fill_diagonal_(0.0)
        ik = two(mu_h, D21o, (p, q, s), (t,))
        il = two(mu_h, D21o, (p, q, t), (s,))
        kl = two(mu_h, D21o.T, (p, q), (s, t))
        d1 = k4eng.Diagram(1); d1.add_vec(0, mu_h * D3h)
        for nm in EXT: d1.add_ext(0, nm, WS)
        nm_, T_ = k4eng.contract(d1, n); ikl = k4eng.to_abcd(nm_, T_)
        acc += (-2.0 * wt) * (full - ik - il - kl - ikl)   # D21o без діагоналі ⟹ потрійний збіг віднімається один раз
    return acc


def dist_mask(dev):
    ar = torch.arange(m, device=dev); A, B_, C_ = torch.meshgrid(ar, ar, ar, indexing="ij")
    return (A != B_) & (A != C_) & (B_ != C_)


def r2(x, y):
    a = float((x * y).sum() / (x * x).sum()); return 1 - float(((y - a * x) ** 2).sum() / (y * y).sum()), a


def tri(T4):
    ar = torch.arange(m, device=T4.device); return T4[ar, ar]            # (a, b, c) = T4[a,a,b,c]


# ---------------- пломба: n = 32, W = I, істинні входи, проти референсу ----------------
def seal(fn):
    import mlp_kprop.kprop_ds as kds
    from mlp_kprop.kprop_ds import relu_kprop
    from mlp_kprop.diagslice import DSTower
    from mlp_kprop.wick import relu_wick_coef
    torch.set_default_dtype(torch.float64)
    D = np.load(fn); Nh = float(D["Nh"]); a = torch.tensor(D["a"])
    g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
    e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m)
    C2 = R2 - torch.outer(e, e)
    C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
    k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
    mean = a + e; var = torch.diagonal(C2)
    T3 = GATI(3); T4 = GATI(4)
    diag211 = computable((2, 1, 1), T4); diag1111 = computable((1, 1, 1, 1), T4)
    keep = {(2, 1, 1): set(vp for vp, c in diag211), (1, 1, 1, 1): set(vp for vp, c in diag1111)}
    def patched(k_max, d_max=None, use_mean_var=False):
        out = {}
        for ip, vd in T4.items():
            out[ip] = {vp: c for vp, c in vd.items() if ip not in keep or vp in keep[ip]}
        return out
    kds.get_all_terms_iso = patched
    Kb = relu_kprop(DSTower.from_tower({1: mean, 2: C2, 3: C3, 4: k4}), k_max=4)
    ref4 = Kb[4].to_tensor(); ref3 = Kb[3].to_tensor(); mu_h = Kb[1].to_tensor(); C_h = Kb[2].to_tensor()
    kds.get_all_terms_iso = GATI
    Coff = C2.clone(); Coff.fill_diagonal_(0.0); D21 = torch.einsum("aac->ac", C3).clone(); D21.fill_diagonal_(0.0)
    B = dict(var=var, D3=torch.einsum("aaa->a", C3).clone(), K4=torch.einsum("aaaa->a", k4).clone(), C=Coff, D21=D21,
             K22=torch.einsum("aabb->ab", k4).clone(), K31=torch.einsum("aaab->ab", k4).clone())
    B["K22"].fill_diagonal_(0.0); B["K31"].fill_diagonal_(0.0)
    wick = lambda k, p=1: relu_wick_coef(mean, var, k, p)
    WS = torch.eye(m, dtype=torch.float64); n = m
    print("пломба %s: діаграм (2,1,1) %d, (1,1,1,1) %d" % (os.path.basename(fn), len(diag211), len(diag1111)), flush=True)
    t0 = time.time(); S211 = slice_sum((2, 1, 1), diag211, B, wick, WS, n, log=print)
    D21h = torch.einsum("aac->ac", ref3).clone(); D3h = torch.einsum("aaa->a", ref3).clone()
    # κ₃h як CP рангу m²: X[i,(k,l)] = κ[i,k,l], P = δ_k, Y = δ_l (несиметризований, але κ симетрична ⟹ вибір ноги неістотний)
    X = ref3.reshape(m, m * m); P = torch.eye(m).repeat_interleave(m, dim=1); Y = torch.eye(m).repeat(1, m)
    S211 = S211 + conversion_211(mu_h, C_h, D21h, D3h, [(X, P, Y)], WS, n)
    S1111 = slice_sum((1, 1, 1, 1), diag1111, B, wick, WS, n, log=print)
    ar = torch.arange(m); A4, B4, C4, D4 = torch.meshgrid(ar, ar, ar, ar, indexing="ij")
    nd = 1 + (B4 != A4).int() + ((C4 != A4) & (C4 != B4)).int() + ((D4 != A4) & (D4 != B4) & (D4 != C4)).int()
    m211 = (nd == 3); m1111 = (nd == 4)
    for nm, mine, msk in (("(2,1,1)", S211, m211), ("(1,1,1,1)", S1111, m1111)):
        x = mine[msk]; y = ref4[msk]; r, sl = r2(x, y)
        print("   %s: нахил %.6f  R² %.6f  віднос. %.2e  (%.0f с)" % (nm, sl, r, float((x - y).norm() / y.norm()), time.time() - t0), flush=True)
    # маска: мої зрізи мусять бути нулями поза своїм класом збігів
    print("   нулі поза класом: S211 на nd≠3 %.2e · S1111 на nd≠4 %.2e" % (float(S211[~m211].norm() / S211[m211].norm()), float(S1111[~m1111].norm() / S1111[m1111].norm())), flush=True)
    # перенос: випадкова W (32×32) проти прямої контракції референсного тензора (лише клас nd==3 / nd==4)
    g = torch.Generator().manual_seed(5); Wr = torch.randn(m, m, generator=g, dtype=torch.float64) / m ** 0.5
    ref211 = ref4 * m211; ref1111 = ref4 * m1111
    direct211 = torch.einsum("ai,bj,ck,dl,ijkl->abcd", Wr, Wr, Wr, Wr, ref211); direct1111 = torch.einsum("ai,bj,ck,dl,ijkl->abcd", Wr, Wr, Wr, Wr, ref1111)
    t0 = time.time()
    pk = slice_sum((2, 1, 1), diag211, B, wick, Wr, n); cv = conversion_211(mu_h, C_h, D21h, D3h, [(X, P, Y)], Wr, n); mine211 = pk + cv
    mine1111 = slice_sum((1, 1, 1, 1), diag1111, B, wick, Wr, n)
    for nm, a_, b_ in (("(2,1,1) pK", pk, None), ("(2,1,1) конв.", cv, None), ("(2,1,1) перенос", mine211, direct211), ("(1,1,1,1) перенос", mine1111, direct1111)):
        if b_ is None:
            print("   %s: ‖·‖/‖direct211‖ %.3f" % (nm, float(a_.norm() / direct211.norm())), flush=True); continue
        r, sl = r2(a_.flatten(), b_.flatten())
        print("   %s: нахил %.6f R² %.6f віднос. %.2e (%.0f с)" % (nm, sl, r, float((a_ - b_).norm() / b_.norm()), time.time() - t0), flush=True)
    return


if __name__ == "__main__" and sys.argv[1] == "seal":
    seal(sys.argv[2] if len(sys.argv) > 2 else "/home/trim/wb2data/brc_0_12.npz")


# ---------------- режим ланцюга ----------------
def chain(i, ls, truek4=False):
    import lean_oracle as lo; lk = lo.lk
    dt = torch.float32; dev = lk.DEV
    REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; n = w.shape[1]; kp = np.load(lo.BANK.format(i))
    rng = np.random.default_rng(11 + i); S = torch.tensor(np.sort(rng.choice(n, m, replace=False)), device=dev)
    Db = np.load("/home/trim/wb2data/brc_%d_%d.npz" % (i, ls)); Nh = float(Db["Nh"])
    g_ = lambda k: (torch.tensor(Db["0_" + k]) + torch.tensor(Db["1_" + k])) / (2 * Nh); gh = lambda h, k: torch.tensor(Db["%d_%s" % (h, k)]) / Nh
    def k4_of(g):
        e = g("q1"); R2 = g("q2"); R4 = g("q4").reshape(m, m, m, m); C2 = R2 - torch.outer(e, e)
        return R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
    k4_t = k4_of(g_).to(dev); k4_a = k4_of(lambda k: gh(0, k)).to(dev); k4_b = k4_of(lambda k: gh(1, k)).to(dev)
    st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K3_PROBE = None; lk.K3_PROBE_POST = None; lk.K_POST_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    got = {}; lp = ls - 1
    def d21hook(l, D3, D21, live):
        if l == lp: got.update(mu=live["mu"].clone(), C=live["C"].clone(), var=live["var"].clone(), D3=D3.clone(), D21=D21.clone())
        return D3, D21, None
    def wkhook(l, wk4_row, wk4_mat, wk431, live):
        if l == lp: got.update(K4=wk4_row.clone(), K22=wk4_mat.clone(), K31=(wk431.clone() if wk431 is not None else None))
        return wk4_row, wk4_mat, wk431
    def kpost(l, K, mu, C):
        if l == lp: got.update(mu_h=mu.clone(), C_h=C.clone(), K22h=K[(2, 2)].clone(), K31h=K[(3, 1)].clone(), K4h=K[(4,)].clone())
    def k3post(l, K3):
        if l == lp:
            trip = []
            for s in K3.sources:
                P = s["P"]; Q = s.get("Q"); assert Q is None
                Pd = torch.eye(n, dtype=dt, device=dev) if P is None else P.clone()
                for xk, yk in K3.PAIRS:
                    if s.get(xk) is not None: trip.append((s[xk].clone(), Pd, s[yk].clone()))
                if s["M"] is not None: trip.append((s["M"].clone(), Pd, Pd))
            D3h, D21h = K3.dslices(); got.update(trip=trip, D3h=D3h.clone(), D21h=D21h.clone())
    lk.D21_HOOK = d21hook; lk.WK_HOOK = wkhook; lk.K_POST_HOOK = kpost; lk.K3_PROBE_POST = k3post; lk.REGEN_STATE.clear()
    lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG)
    lk.D21_HOOK = None; lk.WK_HOOK = None; lk.K_POST_HOOK = None; lk.K3_PROBE_POST = None
    W = torch.tensor(w[ls].T.copy(), dtype=dt, device=dev); WS = W[S].contiguous()
    wick = lk.WickCache(got["mu"], got["var"])
    Coff = got["C"].clone(); Coff.fill_diagonal_(0.0); D21 = got["D21"].clone(); D21.fill_diagonal_(0.0)
    K22 = got["K22"].clone(); K22.fill_diagonal_(0.0); K31 = got["K31"].clone() if got["K31"] is not None else torch.zeros_like(K22); K31.fill_diagonal_(0.0)
    if truek4:
        k22t, k31t, k4t, mu21 = lo.true_pair(kp, lp); a = kp["pre_mean"][lp].astype(np.float64); E2 = kp["pre_m2"][lp].astype(np.float64); E3 = kp["pre_m3"][lp].astype(np.float64)
        tt = lambda x: torch.tensor(x, dtype=dt, device=dev)
        K22 = tt(k22t); K22.fill_diagonal_(0.0); K31 = tt(k31t); K31.fill_diagonal_(0.0); got["K4"] = tt(k4t)
        D21 = tt(mu21); D21.fill_diagonal_(0.0); got["D3"] = tt(E3 - 3*a*E2 + 2*a**3)
    B = dict(var=got["var"], D3=got["D3"], K4=got["K4"], C=Coff, D21=D21, K22=K22, K31=K31)
    T4 = GATI(4); diag211 = computable((2, 1, 1), T4); diag1111 = computable((1, 1, 1, 1), T4)
    tag = "dump %d шар %d→%d%s" % (i, lp, ls, " істинні n²-κ₄ pre" if truek4 else "")
    print(tag + ": діаграм (2,1,1) %d, (1,1,1,1) %d, триплетів κ₃h %d" % (len(diag211), len(diag1111), len(got["trip"])), flush=True)
    t0 = time.time()
    with torch.no_grad():
        S211 = slice_sum((2, 1, 1), diag211, B, wick, WS, n, log=print)
        conv = conversion_211(got["mu_h"], got["C_h"], got["D21h"], got["D3h"], got["trip"], WS, n)
        S1111 = slice_sum((1, 1, 1, 1), diag1111, B, wick, WS, n, log=print)
        K22h = got["K22h"].clone(); K22h.fill_diagonal_(0.0); K31h = got["K31h"].clone(); K31h.fill_diagonal_(0.0)
        Sn2 = n2_slices(dict(K22=K22h, K31=K31h, K4=got["K4h"]), WS, n)
    msk = dist_mask(dev); y = tri(k4_t)[msk]; noise = float((tri(k4_a) - tri(k4_b))[msk].norm() / 2 / y.norm())
    print("   ‖істинний tri‖ %.3e шум %.3f  (%.0f с)" % (float(y.norm()), noise, time.time() - t0), flush=True)
    parts = {"(2,1,1) pK": S211, "(2,1,1) конв.": conv, "(2,1,1) K": S211 + conv, "n² зрізи": Sn2, "море": S1111,
             "(2,1,1)+n²": S211 + conv + Sn2, "усе": S211 + conv + Sn2 + S1111}
    for nm, T in parts.items():
        x = tri(T)[msk]; r, sl = r2(x, y); rel = float((x - y).norm() / y.norm())
        print("   %-14s R² %.3f нахил %.3f ‖x‖/‖y‖ %.3f віднос. %.3f" % (nm, r, sl, float(x.norm() / y.norm()), rel), flush=True)
    # спільний МНК по 4 компонентах (діагностика)
    Xm = torch.stack([tri(S211 + conv)[msk], tri(Sn2)[msk], tri(S1111)[msk]], 1).double()
    coef = torch.linalg.lstsq(Xm, y.double()[:, None]).solution[:, 0]; pred = Xm @ coef
    print("   МНК 3 компоненти: R² %.3f coef %s" % (1 - float(((y.double() - pred) ** 2).sum() / (y.double() ** 2).sum()), " ".join("%.2f" % c for c in coef)), flush=True)
    torch.save(dict(S211=S211.cpu(), conv=conv.cpu(), Sn2=Sn2.cpu(), S1111=S1111.cpu(), k4_t=k4_t.cpu(), S=S.cpu()), "/home/trim/wb2data/k4step_%d_%d%s.pt" % (i, ls, "_true" if truek4 else ""))


if __name__ == "__main__" and sys.argv[1] == "chain":
    for arg in sys.argv[2].split(","):
        i, ls = [int(x) for x in arg.split(":")]; chain(i, ls, truek4=os.environ.get("TRUEK4", "0") == "1")
