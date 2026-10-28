"""Проба носія 30.09: море κ₄(p_ls) на 32 нейронах від sea3-діаграм (κ₃-хаб + ребро C або D21), народжених на КОЖНОМУ
шарі s<ls з хабів ланцюга і перенесених лінійно (чиста пам'ять, без масок збігів, без ребра K22), проти ІСТИННОГО κ₄(p_ls)
(brc). Двохабова форма: κ₄(p)_{ijkl} = Σ_J A_iJ B_jJ Σ_d G_ld H_kd Y_dJ; A,B = ноги джерела s' на шарі ls (уже з воротами),
Y = нога-з'єднання на шарі s (до воріт), G = T_{ls←s} diag(w_c), H = T_{ls←s} diag(w_e) E_s. Передбачення: cos ≥0.5 (шанс 40%);
розклад за шаром народження s (пам'ять глибока чи молода)."""
import os, sys, time, itertools, numpy as np, torch
sys.argv = sys.argv[:1] + sys.argv[1:]; ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
import lean_oracle as lo; lk = lo.lk
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
from mlp_kprop.partitions import vec_part_coef
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; DEV = lk.DEV; m = 32
i = int(ARGS[0]); ls = int(ARGS[1])
T4 = GATI(4)
# діаграми sea3 без K22: (vp, vpc, cnt); структура: hyper = вектор з 3 ненулями, edge = інший
DIAGS = []
for vp in T4[(1, 1, 1, 1)]:
    if any(sum(1 for x in v if x > 0) == 3 and sum(v) == 3 for v in vp) and not any(sum(v) >= 4 and sum(1 for x in v if x > 0) >= 3 for v in vp):
        h = [v for v in vp if sum(1 for x in v if x > 0) == 3][0]; ed = [v for v in vp if v != h][0]
        if max(ed) >= 2 and sum(ed) == 4: continue    # K22-ребро пропускаємо
        DIAGS.append((vp, vec_part_coef(vp, divide_fac=True), T4[(1, 1, 1, 1)][vp], h, ed))
print("діаграми:", [(d[0], d[1], d[2]) for d in DIAGS], flush=True)
# істина
D = np.load("/home/trim/wb2data/brc_%d_%d.npz" % (i, ls)); Nh = float(D["Nh"]); a = torch.tensor(D["a"], dtype=torch.float64)
g_ = lambda k: (torch.tensor(D["0_" + k]) + torch.tensor(D["1_" + k])) / (2 * Nh)
e = g_("q1"); R2 = g_("q2"); R3 = g_("q3").reshape(m, m, m); R4 = g_("q4").reshape(m, m, m, m)
C2 = R2 - torch.outer(e, e)
k4 = R4 - sum(torch.einsum("ab,cd->abcd", C2, C2).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
k4h = {h: None for h in (0, 1)}   # половини для шуму
for hh in (0, 1):
    gh = lambda k: torch.tensor(D["%d_%s" % (hh, k)]) / Nh
    eh = gh("q1"); R2h = gh("q2"); R4h = gh("q4").reshape(m, m, m, m); C2h = R2h - torch.outer(eh, eh)
    k4h[hh] = R4h - sum(torch.einsum("ab,cd->abcd", C2h, C2h).permute(*np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
ar = torch.arange(m); A_, B_, C_, D_ = torch.meshgrid(ar, ar, ar, ar, indexing="ij")
sea = (A_ != B_) & (A_ != C_) & (A_ != D_) & (B_ != C_) & (B_ != D_) & (C_ != D_)
d = np.load(lo.DUMP.format(i)); w = d["weights"]; n = w.shape[1]
rng = np.random.default_rng(11 + i); S = torch.tensor(np.sort(rng.choice(n, m, replace=False)), device=DEV)
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K_POST_HOOK = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
LAY = {}      # s -> dict(legs=[(src_idx, [(L1,L2,L3),...])], D21, C, w)
class Stop(Exception): pass
RES = {}
def triples(src, cast):
    P = src["P"]; assert src.get("Q") is None
    Pd = cast(P) if P is not None else None
    out = []
    for xk, yk in (("X1", "Y1"), ("X3", "Y3")):
        if src.get(xk) is not None: out.append((cast(src[xk]), Pd, cast(src[yk])))
    if src.get("M") is not None: out.append((cast(src["M"]), Pd, Pd))
    return out
def probe(l, K3, D3, D21):
    if l < ls:
        if l == 0: return
        LAY[l] = dict(D21=D21.half().clone(), legs=[(si, triples(s, lambda t: None if t is None else t.half().clone())) for si, s in enumerate(K3.sources)])
        return
    RES["K3"] = K3; RES["D21"] = D21
def d21hook(l, D3, D21, live):
    if 1 <= l < ls:
        wk = lk.WickCache(live["mu"], live["var"]); LAY[l]["C"] = lk.zero_diag_(live["C"].clone()).half(); LAY[l]["w"] = {k: wk(k, 1).clone() for k in (1, 2, 3)}
    if l == ls:
        RES["mu"] = live["mu"].clone(); raise Stop()
    return D3, D21, None
lk.K3_PROBE = probe; lk.D21_HOOK = d21hook; lk.REGEN_STATE.clear(); t0 = time.time()
try:
    lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
except Stop:
    pass
lk.K3_PROBE = None; lk.D21_HOOK = None
K3 = RES["K3"]; mu = RES["mu"]
print("ланцюг до шару %d: %.0f с; джерел %d; пломба середнього: ‖mu[S]−(a+e)‖/‖a+e‖ = %.2e" % (
    ls, time.time() - t0, len(K3.sources), float((mu[S].double().cpu() - (a + e)).norm() / (a + e).norm())), flush=True)
# джерело, народжене на шарі s, має індекс s у K3.sources? перевірити за віком: age = ls - s
age = {si: s.get("age", 0) for si, s in enumerate(K3.sources)}
born = {si: ls - ag for si, ag in age.items()}
print("вік джерел на шарі ls:", [(si, born[si]) for si in born][:20], flush=True)
Ssub = lambda M: M[S].float()
def T_of(s):
    """T_{ls←s} = P джерела, народженого на шарі s (P на шарі ls після linear_)."""
    for si, s2 in enumerate(K3.sources):
        if born[si] == s and s2["P"] is not None: return s2["P"].float()
    raise KeyError(s)
acc = {k: torch.zeros(m, m, m, m, dtype=torch.float64, device=DEV) for k in range(len(DIAGS))}   # лабельовані, до симетризації
per_s = {}
t0 = time.time()
for s in sorted(LAY):
    T = T_of(s); wS = LAY[s]["w"]; Cs = LAY[s]["C"].float(); D21s = LAY[s]["D21"].float()
    acc_s = torch.zeros(m, m, m, m, dtype=torch.float64, device=DEV)
    for k, (vp, vpc, cnt, h, ed) in enumerate(DIAGS):
        hv = [q for q in range(4) if h[q] > 0]; ev = [q for q in range(4) if ed[q] > 0]
        vc = [q for q in hv if q in ev][0]; ve = [q for q in ev if q != vc][0]; hub1 = [q for q in hv if q != vc]
        deg = [h[q] + ed[q] for q in range(4)]
        # ребро E[c_e, c_c] орієнтоване (ve, vc): блок (k_e, k_c)
        ke, kc = ed[ve], ed[vc]
        E = Cs if (ke, kc) == (1, 1) else (D21s if (ke, kc) == (2, 1) else D21s.T)   # D21[a,c]=κ_{aac}: (2,1): κ_{ee c}=D21[e,c]; (1,2): κ_{e cc}=D21[c,e]
        G_S = Ssub(T * wS[deg[vc]][None, :])                 # (m, n): T diag(w_c)
        H_S = Ssub((T * wS[deg[ve]][None, :]) @ E)           # (m, n): T diag(w_e) E
        # цільові осі: hub1[0]->i, hub1[1]->j, ve->k, vc->l у лабельованому термі; збираємо в порядку (hub1[0],hub1[1],ve,vc)
        part = torch.zeros(m, m, m, m, dtype=torch.float64, device=DEV)
        for si, trips in LAY[s]["legs"]:
            src_now = K3.sources[si]
            now = triples(src_now, lambda t: None if t is None else t.float())
            for (Ls, Ln) in zip(trips, now):
                for c in range(3):
                    Y = Ls[c].float(); A = Ln[(c + 1) % 3]; B = Ln[(c + 2) % 3]
                    Z = torch.einsum("ld,kd,dJ->lkJ", G_S, H_S, Y)          # (m, m, J)
                    part += torch.einsum("iJ,jJ,lkJ->ijkl", Ssub(A), Ssub(B), Z).double() / 3.0
        # переставити осі (hub1[0], hub1[1], ve, vc) -> (0,1,2,3) позиції
        order = [hub1[0], hub1[1], ve, vc]; inv = [order.index(q) for q in range(4)]
        lab = part.permute(*inv)
        acc[k] += lab; acc_s += lab * 0   # (симетризація нижче)
        per_s.setdefault(s, {})[k] = lab.clone()
    torch.cuda.synchronize()
print("носій зібрано: %.0f с" % (time.time() - t0), flush=True)
def symz(k, lab):
    vp, vpc, cnt, h, ed = DIAGS[k]
    tot = torch.zeros_like(lab)
    for p in itertools.permutations(range(4)): tot += lab.permute(*p)
    return vpc * (cnt / 24.0) * tot
tru = k4.to(DEV)[sea.to(DEV)]
noise = float(((k4h[0] - k4h[1]) / 2).to(DEV)[sea.to(DEV)].norm() / tru.norm())
print("істина: ‖море‖ %.3e, шум половин %.3f" % (tru.norm(), noise), flush=True)
def rep(name, Tt):
    x = Tt[sea.to(DEV)]
    print("  %-28s ‖x‖/‖tru‖ %.3f  cos %+.3f  ‖tru−x‖²/‖tru‖² %.3f  найкращий скаляр c*=%.2f -> %.3f" % (
        name, x.norm() / tru.norm(), (x * tru).sum() / x.norm() / tru.norm(), ((tru - x) ** 2).sum() / (tru ** 2).sum(),
        (x * tru).sum() / (x * x).sum(), 1 - ((x * tru).sum() ** 2) / ((x * x).sum() * (tru * tru).sum())), flush=True)
full = sum(symz(k, acc[k]) for k in acc)
rep("усі s, усі діаграми", full)
rep("усі s, лише ребро C", symz(0, acc[0]) if DIAGS[0][3] is not None else full)
for s in sorted(per_s):
    rep("народжено на s=%d" % s, sum(symz(k, per_s[s][k]) for k in per_s[s]))
cum = None
for s in sorted(per_s, reverse=True):
    cum = sum(symz(k, per_s[s][k]) for k in per_s[s]) if cum is None else cum + sum(symz(k, per_s[s][k]) for k in per_s[s])
    rep("пам'ять s>=%d" % s, cum)

# ---- зрізи зі збігами індексів (те, що споживає ланцюг): [a,a,b,c], [a,a,b,b], [a,a,a,b] ----
K4T = k4.to(DEV); FULL = full
def slice_rep(name, pat):
    if pat == "211":
        a3, b3, c3 = torch.meshgrid(ar, ar, ar, indexing="ij"); msk = (a3 != b3) & (a3 != c3) & (b3 != c3); get = lambda T: torch.stack([T[x, x] for x in range(m)])[msk]
    elif pat == "22":
        msk = (A_ != B_); get = lambda T: torch.stack([T[x, x, y, y] for x in range(m) for y in range(m) if x != y])
    else:
        msk = (A_ != B_); get = lambda T: torch.stack([T[x, x, x, y] for x in range(m) for y in range(m) if x != y])
    t = get(K4T.double()); x = get(FULL.double())
    print("  зріз %-4s %-14s ‖x‖/‖tru‖ %.3f  cos %+.3f  ‖tru−x‖²/‖tru‖² %.3f" % (pat, name, x.norm() / t.norm(), (x * t).sum() / x.norm() / t.norm(), ((t - x) ** 2).sum() / (t ** 2).sum()), flush=True)
for pat in ("211", "22", "31"):
    slice_rep("носій", pat)
np.savez("/home/trim/wb2data/seacarry_%d_%d.npz" % (i, ls), carrier=full.cpu().numpy(), truth=k4.numpy(), S=S.cpu().numpy())

# ---- внесок кожного типу злиття осей у зріз [a,a,b,c] (лабельовані осі: 0,1 = hub-1; 2 = ve; 3 = vc) ----
a3, b3, c3 = torch.meshgrid(ar, ar, ar, indexing="ij"); msk3 = ((a3 != b3) & (a3 != c3) & (b3 != c3)).to(DEV)
def sym3(T):
    return sum(T.permute(*p) for p in itertools.permutations(range(3))) / 6.0
tot211 = None; parts211 = {}
for k in acc:
    vp, vpc, cnt, h, ed = DIAGS[k]
    for (p, q) in itertools.combinations(range(4), 2):
        mg = torch.diagonal(acc[k], dim1=p, dim2=q)          # (m, m, m_merged) — злита вісь остання
        t3 = sym3(mg) * vpc * cnt / 6.0
        parts211[(p, q)] = t3 if (p, q) not in parts211 else parts211[(p, q)] + t3
        tot211 = t3 if tot211 is None else tot211 + t3
tt = tot211[msk3]
chk = torch.stack([FULL.double()[x, x] for x in range(m)])   # [a, b, c] = FULL[a,a,b,c]
print("  зріз 211 із злиттів: ‖tot‖ %.3e; пломба проти симетризованого FULL: %.2e" % (tt.norm(), (tot211 - chk)[msk3].norm() / tt.norm()), flush=True)
for pq, t3 in parts211.items():
    x = t3[msk3]
    print("     злиття осей %s  ‖x‖/‖tot‖ %.3f  cos з tot %+.3f" % (pq, x.norm() / tt.norm(), (x * tt).sum() / x.norm() / tt.norm()), flush=True)

print("DONE_SEACARRY", i, ls, flush=True)
