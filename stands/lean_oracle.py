"""Оракул κ₂₂/κ₃₁ на lean-ланцюгу 504aldo (float64 torch → GPU-патч).
Пломба: mode=rec на dense-ланцюгу — на l=1 регресія wk4_mat проти істинної κ₂₂(p) мусить дати R²≈1 і сталий нахил
(це визначає одиниці); wk4_row проти κ₄_iiii так само. Передбачення ДО прогону: regen-ланцюг ≈ 2.2e-8; +істинна κ₂₂/κ₃₁ у
use-side → 1.1…1.5e-8 (наш K4P: 0.0162→0.0118% = ×1.9 MSE); шанс ≥×1.5: 50%."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_k3_aug_gpu as lk
DEV = lk.DEV
DUMP = "/home/trim/wb2data/dumps_p2/mlp_{:04d}.npz"; BANK = "/home/trim/wb2data/kp_p2/mlp_{:05d}.npz"
dumps = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 3 else []; cfg = sys.argv[2] if len(sys.argv) > 3 else ""; mode = sys.argv[3] if len(sys.argv) > 3 else ""
dtype = torch.float64 if os.environ.get("F64", "0") == "1" else torch.float32
W22 = float(os.environ.get("W22", "1")); W31 = float(os.environ.get("W31", "1"))
def zd(M): M = M.clone(); M.fill_diagonal_(0.0); return M
def true_pair(kp, l):
    a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64); E4 = kp["pre_m4"][l].astype(np.float64)
    M11 = kp["pre_M11"][l].astype(np.float64); M21 = kp["pre_M21"][l].astype(np.float64); M31 = kp["pre_M31"][l].astype(np.float64); M22 = kp["pre_M22"][l].astype(np.float64)
    C11 = M11 - np.outer(a, a)
    # центральні: mu21[i,j]=E[(pi-ai)^2 (pj-aj)], mu31, mu22 з сирих
    mu21 = M21 - 2*a[:, None]*M11 - a[None, :]*E2[:, None] + 2*np.outer(a*a, a)
    mu31 = M31 - 3*a[:, None]*M21 + 3*(a*a)[:, None]*M11 - a[None, :]*E3[:, None] + 3*a[None, :]*(a[:, None]*E2[:, None]) - 3*np.outer(a**3, a)
    mu22 = (M22 - 2*a[None, :]*M21 - 2*a[:, None]*M21.T + (a*a)[None, :]*E2[:, None] + (a*a)[:, None]*E2[None, :]
            + 4*np.outer(a, a)*M11 - 3*np.outer(a*a, a*a))
    v = E2 - a*a
    k22 = mu22 - np.outer(v, v) - 2*C11*C11
    k31 = mu31 - 3*v[:, None]*C11
    k4 = E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v
    return k22, k31, k4, mu21
def r2(x, y):
    x = x.ravel(); y = y.ravel(); a = float((x*y).sum()/(x*x).sum()); return 1 - float(((y - a*x)**2).sum()/(y*y).sum()), a
class Hook:
    def __init__(self, kp): self.kp = kp; self.lines = []; self.coef3 = []
    def __call__(self, l, wk4_row, wk4_mat, wk431, live):
        if os.environ.get("DUMPLIVE"):   # 16.10: власні mu, C ланцюга по шарах для k4pred з входами ланцюга
            os.makedirs("/home/trim/wblogs/live", exist_ok=True)
            np.savez("/home/trim/wblogs/live/d%d_l%02d.npz" % (CUR_DUMP[0], l), mu=live["mu"].detach().double().cpu().numpy(), C=live["C"].detach().float().cpu().numpy())
        if mode == "k4bl":   # 15.10: історія власних (mu, C) ланцюга для «лінії назад»
            BL_HIST[l] = (live["mu"].detach().double().clone(), live["C"].detach().double().clone())
        if wk4_mat is None or l < 1: return wk4_row, wk4_mat, wk431
        k22, k31, k4, _ = true_pair(self.kp, l)
        offm = ~np.eye(k22.shape[0], dtype=bool)
        wm = wk4_mat.detach().cpu().double().numpy(); wr = wk4_row.detach().cpu().double().numpy()
        r_m, a_m = r2(wm[offm], k22[offm]); r_r, a_r = r2(wr, k4)
        line = "l=%2d wk4_mat~κ₂₂ R² %.3f нахил %.3f | wk4_row~κ₄ R² %.3f нахил %.3f" % (l, r_m, a_m, r_r, a_r)
        if wk431 is not None:
            w3 = wk431.detach().cpu().double().numpy(); r_3, a_3 = r2(w3[offm], k31[offm]); line += " | wk431~κ₃₁ R² %.3f нахил %.3f" % (r_3, a_3)
        if EXPARTS and "4" in EXPARTS and l in EX_EPS and wk431 is not None:   # 16.10: зняти κ4-обличчя моди масштабу 4ε·Pair3(C⊗C)
            _e = EX_EPS[l]; _C = live["C"].detach().double(); _v = torch.diagonal(_C)
            wk4_row = (wk4_row.double() - 12.0*_e*_v*_v).to(wk4_row.dtype)
            wk4_mat = (wk4_mat.double() - (4.0*_e*(_v[:, None]*_v[None, :] + 2.0*_C*_C)).fill_diagonal_(0.0)).to(wk4_mat.dtype)
            wk431 = (wk431.double() - (12.0*_e*_v[:, None]*_C).fill_diagonal_(0.0)).to(wk431.dtype)
        if mode in ("k31fit", "k31law") and live.get("D21") is not None and wk431 is not None:   # 16.10: закон κ31 kai (форум 18246): b1·v_i C_ij + b2·κ3_i T_iij / v_i
            _C = live["C"].detach().double(); _v = live["var"].detach().double(); _D21 = live["D21"].detach().double(); _D3 = live["D3"].detach().double()
            _om = ~torch.eye(_C.shape[0], dtype=torch.bool, device=_C.device)
            _F = [(_v[:, None]*_C), (_D3/_v)[:, None]*_D21, (_D3/_v)[:, None]*_D21.T, wk431.detach().double()]
            if mode == "k31fit":
                _X = torch.stack([f[_om] for f in _F], 1); _y = torch.as_tensor(k31, dtype=torch.float64, device=_C.device)[_om]
                K31STATS.setdefault(CUR_DUMP[0], {})[l] = dict(XtX=(_X.T @ _X).cpu().tolist(), Xty=(_X.T @ _y).cpu().tolist(), yy=float(_y @ _y))
            else:
                _cols = [int(c) for c in os.environ.get("K31COLS", "0,1").split(",")]
                _A = np.zeros((len(_cols), len(_cols))); _bb = np.zeros(len(_cols))
                for _d, _st in K31LOAD.items():
                    if _d == CUR_DUMP[0] or l not in _st: continue
                    _A += np.array(_st[l]["XtX"])[np.ix_(_cols, _cols)]; _bb += np.array(_st[l]["Xty"])[_cols]
                _coef = np.linalg.solve(_A, _bb); _new = sum(float(c)*_F[k] for c, k in zip(_coef, _cols)); _new = _new.masked_fill(~_om, 0.0)
                line += " | k31law coef %s" % (["%.4g" % c for c in _coef])
                wk431 = _new.to(wk431.dtype)
        if mode == "fit" and live["D21"] is not None:
            C = live["C"].detach().double(); mu = live["mu"].detach().double(); var = live["var"].detach().double()
            D21 = live["D21"].detach().double(); D3 = live["D3"].detach().double(); k4r = wk4_row.detach().double()
            dev = C.device; m = torch.as_tensor(offm, device=dev)
            Coff = C.clone(); Coff.fill_diagonal_(0.0)
            modes = {"C": Coff, "D21": D21, "D21T": D21.T, "D21*D3/v": D21*(D3/var)[:, None], "D21*mu": D21*mu[:, None],
                     "C*D3": Coff*D3[:, None], "C*mu": Coff*mu[:, None], "C*k4": Coff*k4r[:, None], "C*var": Coff*var[:, None],
                     "C*muj": Coff*mu[None, :], "CC": Coff*Coff, "D21*k4/v2": D21*(k4r/var**2)[:, None],
                     "D21T*D3j/vj": D21.T*(D3/var)[None, :], "D21*(D3/v)^2": D21*(D3/var)[:, None]**2, "D21*D3/v*muj": D21*(D3/var)[:, None]*mu[None, :],
                     "D21*mu/v": D21*(mu/var)[:, None], "D21*C": D21*Coff, "D21*D3/v*varj": D21*(D3/var)[:, None]*var[None, :],
                     "D21*D3^2/v^2": D21*(D3*D3/var**2)[:, None], "C*D3^2/v": Coff*(D3*D3/var)[:, None], "C*D3*mu": Coff*(D3*mu)[:, None]}
            y = torch.as_tensor(k31, device=dev)[m]
            X = torch.stack([v[m] for v in modes.values()], 1)
            ss = float((y*y).sum())
            def fit(cols):
                Xc = X[:, cols]; sol = torch.linalg.lstsq(Xc, y[:, None]).solution[:, 0]; res = y - Xc @ sol
                return 1 - float((res*res).sum())/ss, sol
            names = list(modes)
            singles = " ".join("%s %.3f" % (nm, fit([k])[0]) for k, nm in enumerate(names))
            rall, sol = fit(list(range(len(names))))
            line += " || memoryless κ₃₁: " + singles + " | ВСІ %.3f" % rall
            r3, sol3 = fit([names.index(k) for k in MEM3]); line += " | 3 моди %.3f coef %s" % (r3, ["%.4g" % float(v) for v in sol3])
            self.coef3.append([float(v) for v in sol3])
        if mode == "mem" and live["D21"] is not None and wk431 is not None:
            C = live["C"].detach(); var = live["var"].detach(); D21 = live["D21"].detach(); D3 = live["D3"].detach(); k4r = wk4_row.detach()
            Coff = C.clone(); Coff.fill_diagonal_(0.0)
            mm = {"D21*D3/v": D21*(D3/var)[:, None], "C*k4": Coff*k4r[:, None], "C": Coff}
            a = MEMTAB[l]; new31 = sum(float(a[k])*mm[nm] for k, nm in enumerate(MEM3)); new31.fill_diagonal_(0.0)
            w3 = new31.detach().cpu().double().numpy(); r_n, a_n = r2(w3[offm], k31[offm]); line += " | mem κ₃₁ R² %.3f нахил %.3f" % (r_n, a_n)
            wk431 = new31.to(wk431.dtype)*W31 + (1-W31)*wk431
        self.lines.append(line)
        if not (D21LMIN <= l <= D21LMAX): return wk4_row, wk4_mat, wk431   # 15.10: вікно шарів і для κ₂₂/κ₃₁/κ₄
        _RM, _RV = rmask(l, k22.shape[0]) if RFRAC < 1 else (None, None)
        if mode in ("k22", "both", "all"):
            _new = torch.as_tensor(np.where(offm, k22/A22, 0.0), dtype=wk4_mat.dtype, device=wk4_mat.device)*W22 + (1-W22)*wk4_mat
            wk4_mat = _new if _RM is None else rmix(_new, wk4_mat, _RM)
        if mode in ("k31", "both", "k31k4", "all") and wk431 is not None:
            _new = torch.as_tensor(np.where(offm, k31/A31, 0.0), dtype=wk431.dtype, device=wk431.device)*W31 + (1-W31)*wk431
            wk431 = _new if _RM is None else rmix(_new, wk431, _RM)
        if mode in ("k4s", "k31k4s"):        # 13.10: ЛИШЕ масштаб: his κ₄ × оракульний нахил a_r (по шару і дампу)
            wk4_row = wk4_row * a_r
        if mode in ("k4t", "k31k4t") and K4TAB is not None:   # таблиця нахилів по шарах (LOO між дампами)
            wk4_row = wk4_row * float(K4TAB[min(l, len(K4TAB)-1)])
        if mode in ("k31", "k31k4s", "k31k4t") and wk431 is not None and mode != "k31":
            wk431 = torch.as_tensor(np.where(offm, k31/A31, 0.0), dtype=wk431.dtype, device=wk431.device)*W31 + (1-W31)*wk431
        if mode == "k4bl" and l >= 2 and all(j in BL_HIST for j in range(l)):   # 15.10: κ4(p_l) лінією назад з власних C,t ланцюга
            _pred = backline_k4(l)
            if os.environ.get("BLDBG"):
                _kt = torch.as_tensor(k4, device=_pred.device); _wr = wk4_row.detach().double()
                _mu, _C = BL_HIST[l]; _s = torch.diagonal(_C).sqrt(); _t = _mu/_s; _b = (_t*_t-1)*torch.exp(-_t*_t/2)/(24*_s**3)
                _r2 = lambda p_: float(1-((_b*(_kt-p_))**2).sum()/((_b*(_kt-_kt.mean()))**2).sum())
                _cc = lambda a_, b_: float(torch.corrcoef(torch.stack([a_, b_]))[0, 1])
                print("BL l=%2d  mean pred/true %.3f his/true %.3f | corr pred %.3f his %.3f | wR² pred %.3f his %.3f" % (l, float(_pred.mean()/_kt.mean()), float(_wr.mean()/_kt.mean()), _cc(_pred, _kt), _cc(_wr, _kt), _r2(_pred), _r2(_wr)) + " | mix .2 %.3f .35 %.3f .5 %.3f" % (_r2(0.2*_pred+0.8*_wr), _r2(0.35*_pred+0.65*_wr), _r2(0.5*_pred+0.5*_wr)), flush=True)
            wk4_row = (_pred/A4).to(wk4_row.dtype)*W4 + (1-W4)*wk4_row
        if mode in ("k4b", "k4lev") and live.get("D3") is not None:   # 15.10: розмита мапа γ4 = f(t, γ3) з ВЛАСНИХ mu,var,D3 ланцюга (коеф. LOO з банку)
            _mu = live["mu"].detach().double(); _v = live["var"].detach().double(); _t = _mu/_v.sqrt(); _g3 = live["D3"].detach().double()/_v**1.5
            _c = torch.tensor(K4BLUR["%d_%d" % (CUR_DUMP[0], l)], dtype=torch.float64, device=_mu.device)
            _F = torch.stack([torch.ones_like(_t), _t, _t*_t, _g3, _g3*_g3, _g3*_t], 1)
            _g4 = (_F @ _c) if mode == "k4b" else torch.full_like(_t, float(_c[0] + _c[1]*_t.mean()*0))
            if mode == "k4lev": _g4 = torch.full_like(_t, float((_F @ _c).mean()))
            wk4_row = ((_g4*_v*_v)/A4).to(wk4_row.dtype)*W4 + (1-W4)*wk4_row
        if mode in ("k4sel", "k4selp") and live.get("D3") is not None:   # 16.10: мапа (мікс 0.5) скрізь + точний κ4 на частці FRAC нейронів, відбір SEL
            _mu = live["mu"].detach().double(); _v = live["var"].detach().double(); _s = _v.sqrt(); _t = _mu/_s; _g3 = live["D3"].detach().double()/_v**1.5
            _c = torch.tensor(K4BLUR["%d_%d" % (CUR_DUMP[0], l)], dtype=torch.float64, device=_mu.device)
            _F = torch.stack([torch.ones_like(_t), _t, _t*_t, _g3, _g3*_g3, _g3*_t], 1)
            _his = wk4_row.detach().double(); _map = (_F @ _c)*_v*_v/A4; _mix = 0.5*_map + 0.5*_his
            _kt = torch.as_tensor(k4, dtype=torch.float64, device=_mu.device)/A4
            if mode == "k4selp": _kt = torch.as_tensor(np.load("/home/trim/wblogs/k4pred_%sd%d.npy" % (os.environ.get("K4PSUF", ""), CUR_DUMP[0]))[l], dtype=torch.float64, device=_mu.device)/A4   # 16.10: предиктор (гаус+κ3) замість істинного
            _b = ((_t*_t-1)*torch.exp(-_t*_t/2)/(24*_s**3)).abs()
            _sel = os.environ.get("SEL", "rand"); _fr = float(os.environ.get("FRAC", "0.1")); _k = int(round(_fr*_t.numel()))
            if _sel == "orac": _sc = _b*(_kt - _mix).abs()
            elif _sel == "dis": _sc = _b*(_map - _his).abs()
            elif _sel == "sens": _sc = _b*_v*_v
            else: _sc = torch.rand(_t.shape, generator=torch.Generator().manual_seed(1000*CUR_DUMP[0]+l), dtype=torch.float64).to(_mu.device)
            if os.environ.get("K4DBG"):
                _u = _v*_v/A4; _ix0 = torch.topk(_b*_v*_v, max(1, int(0.25*_t.numel()))).indices
                _rm = lambda x_: float(((x_[_ix0]/_u[_ix0])**2).mean().sqrt())
                print("K4DBG l=%2d γ4true rms %.4f mean %.4f | err his %.4f map %.4f mix %.4f (γ4 units, sens25)" % (l, _rm(_kt), float((_kt[_ix0]/_u[_ix0]).mean()), _rm(_his-_kt), _rm(_map-_kt), _rm(_mix-_kt)), flush=True)
            _out = _mix.clone()
            _ns = float(os.environ.get("K4NOISE", "0"))   # шум джерела в одиницях γ4 (МК з N семплів: √(24/N))
            if _ns > 0: _kt = _kt + _ns*_v*_v/A4*torch.randn(_t.shape, generator=torch.Generator().manual_seed(7+1000*CUR_DUMP[0]+l), dtype=torch.float64).to(_mu.device)
            if _k > 0: _ix = torch.topk(_sc, _k).indices; _out[_ix] = _kt[_ix]
            wk4_row = _out.to(wk4_row.dtype)
        if mode in ("k4", "k31k4", "all"):   # 13.10: істинна діагональ κ₄ (понейронна), одиниці A4 = нахил wk4_row~κ₄ з audit
            _new = torch.as_tensor(k4/A4, dtype=wk4_row.dtype, device=wk4_row.device)*W4 + (1-W4)*wk4_row
            wk4_row = _new if _RV is None else rmix(_new, wk4_row, _RV)
        return wk4_row, wk4_mat, wk431
MEM3 = ["D21*D3/v", "C*k4", "C"]
RFRAC = float(os.environ.get("RFRAC", "1")); MCN = int(os.environ.get("MCN", "0"))   # 16.10: випадкова частка точних пар / МК-оцінка замість точних
def rmask(l, n):
    g = np.random.default_rng(1000*CUR_DUMP[0] + l); return g.random((n, n)) < RFRAC, g.random(n) < RFRAC
def rmix(new, old, m):   # де маска: нове (точне), інакше своє ланцюга
    return torch.where(torch.as_tensor(m, device=old.device), new.to(old.dtype), old)
def mc_kp(w, i):
    Wt = torch.tensor(w, dtype=torch.float32, device=DEV); NL, n = Wt.shape[0], Wt.shape[1]
    A = {k: torch.zeros((NL, n, n), dtype=torch.float64, device=DEV) for k in ("pre_M11", "pre_M21", "pre_M31", "pre_M22")}
    V = {k: torch.zeros((NL, n), dtype=torch.float64, device=DEV) for k in ("pre_mean", "pre_m2", "pre_m3", "pre_m4")}
    g = torch.Generator(device=DEV).manual_seed(777 + i); done = 0
    while done < MCN:
        b = min(8192, MCN - done); h = torch.randn(b, n, generator=g, device=DEV)
        for l in range(NL):
            p_ = h @ Wt[l]; p2 = p_*p_
            V["pre_mean"][l] += p_.sum(0).double(); V["pre_m2"][l] += p2.sum(0).double(); V["pre_m3"][l] += (p2*p_).sum(0).double(); V["pre_m4"][l] += (p2*p2).sum(0).double()
            A["pre_M11"][l] += (p_.T @ p_).double(); A["pre_M21"][l] += (p2.T @ p_).double(); A["pre_M31"][l] += ((p2*p_).T @ p_).double(); A["pre_M22"][l] += (p2.T @ p2).double()
            h = torch.relu(p_)
        done += b
    out = {k: (v/MCN).cpu().numpy() for k, v in list(A.items()) + list(V.items())}; del A, V; torch.cuda.empty_cache(); out["n_samples"] = np.array(float(MCN)); return out
import json
K4BLUR = json.load(open("/home/trim/wb2lean/dig/k4blur_coef.json")) if os.path.exists("/home/trim/wb2lean/dig/k4blur_coef.json") else {}
CUR_DUMP = [0]
K31STATS = {}   # 16.10: k31fit
K31LOAD = {}
if os.path.exists("/home/trim/wblogs/k31law_stats.json"):
    K31LOAD = {int(d): {int(l): v for l, v in st.items()} for d, st in json.load(open("/home/trim/wblogs/k31law_stats.json")).items()}
BL_HIST = {}; BL_W = [None]
BL_COEF = json.load(open("/home/trim/wb2lean/dig/backline_coef.json")) if os.path.exists("/home/trim/wb2lean/dig/backline_coef.json") else {}
def _bl_pre(m, C):
    from math import sqrt, pi
    C = C.clone(); s = torch.diagonal(C).clamp_min(1e-30).sqrt(); t = m/s
    Ph = 0.5*(1+torch.erf(t/sqrt(2))); ph = torch.exp(-t*t/2)/sqrt(2*pi); Co = C.clone(); Co.fill_diagonal_(0)
    x = torch.linspace(-14, 14, 2001, dtype=torch.float64, device=C.device); wx = torch.exp(-x*x/2); wx = wx/wx.sum()
    r = torch.relu(m[:, None]+s[:, None]*x[None, :]); e1 = (r*wx).sum(1); rc = r-e1[:, None]
    k4own = (rc**4*wx).sum(1)-3*((rc**2*wx).sum(1))**2
    return dict(Ph=Ph, f2=ph/s, f3=-t*ph/s**2, Co=Co, k4own=k4own)
def backline_k4(L):
    W = BL_W[0]; c = BL_COEF[str(CUR_DUMP[0])]; G = W[L]; F = torch.zeros(4, W.shape[1], dtype=torch.float64, device=W.device)
    for l in range(L-1, -1, -1):
        P = _bl_pre(*BL_HIST[l]); V = P["Co"]@(P["Ph"][:, None]*G)
        F[0] += ((G**4)*P["k4own"][:, None]).sum(0); F[1] += 4*(G*P["f3"][:, None]*V**3).sum(0)
        U = G*P["f2"][:, None]*V; F[2] += 12*(U*(P["Co"]@U)).sum(0); G2 = G*G*P["f2"][:, None]; F[3] += 3*(G2*((P["Co"]**2)@G2)).sum(0)
        if l > 0: G = W[l]@(P["Ph"][:, None]*G)
    return c[0]*F[0]+c[1]*F[1]+c[2]*F[2]+c[3]*F[3]
K4TAB = [float(x) for x in os.environ["K4TAB"].split(",")] if os.environ.get("K4TAB") else None
import json
MEMTAB = None
if os.environ.get("MEMCOEF"):
    _t = [json.load(open(f)) for f in os.environ["MEMCOEF"].split(",")]; MEMTAB = {}
    for l in _t[0]: MEMTAB[int(l)] = np.mean([t[l] for t in _t], axis=0)
D21USE = int(os.environ.get("D21USE", "0")); D21FEED = int(os.environ.get("D21FEED", "0")); D21R = int(os.environ.get("D21R", "0")); D21W = float(os.environ.get("D21W", "1"))
D21DIAG = int(os.environ.get("D21DIAG", "1"))
D21OFF = int(os.environ.get("D21OFF", "1")); D21PROJ = int(os.environ.get("D21PROJ", "0")); D21PBASIS = os.environ.get("D21PBASIS", "res")   # 1: також істинне D3 (власна κ₃); 0: лише позадіагональ
D21LMIN = int(os.environ.get("D21LMIN", "0")); D21LMAX = int(os.environ.get("D21LMAX", "99"))
OLDSKIP = int(os.environ.get("OLDSKIP", "0")); OLDAGE = int(os.environ.get("OLDAGE", "4")); K3REF = {}
def k3_stash(l, K3, D3, D21): K3REF["K3"] = K3
EXPARTS = os.environ.get("EXTRACT", "")   # 16.10: "c","3","4","f" (C з ε·C теж), "F" (feed)
EX_EPS = {}
class ExtractD21:
    """16.10 (форум 18248, il.miscusi): мода глобального масштабу s=1+δ, Varδ=ε: δC = ε(C + m mᵀ), δκ3 = 2ε(m_a C_bc + m_b C_ac + m_c C_ab),
    δκ4 = 4ε(C_ab C_cd + C_ac C_bd + C_ad C_bc) (закон повної кумулянтності). ε з власного зрізу D21 ланцюга: F_ij = 2(2 m_i C_ij + m_j C_ii)."""
    def __init__(self, inner): self.inner = inner
    def __getattr__(self, k): return getattr(self.inner, k)
    def __call__(self, l, D3, D21, live):
        D3u, D21u, feed = self.inner(l, D3, D21, live)
        if l < 1 or D21 is None: return D3u, D21u, feed
        m = live["mu"].detach().double(); C = live["C"].detach().double(); v = torch.diagonal(C)
        F = 2.0*(2.0*m[:, None]*C + m[None, :]*v[:, None]); F.fill_diagonal_(0.0)
        e = float((D21.double()*F).sum()/(F*F).sum())
        e = float(os.environ["EXEPS"]) if os.environ.get("EXEPS") else e*float(os.environ.get("EXSCALE", "1"))   # 16.10: скан сили / стала ε
        EX_EPS[l] = e
        if "3" in EXPARTS:
            D3u = (D3u.double() - 6.0*e*m*v).to(D3u.dtype); D21u = (D21u.double() - e*F).to(D21u.dtype)
            if "F" in EXPARTS: feed = D21u
        return D3u, D21u, feed
def extract_c(l, mu, C_off, var):
    if l < 1 or l not in EX_EPS or "c" not in EXPARTS: return C_off, var
    e = EX_EPS[l]; m = mu.double(); full = "f" in EXPARTS
    Cn = C_off.double() - e*torch.outer(m, m) - (e*C_off.double() if full else 0.0); Cn.fill_diagonal_(0.0)
    vn = var.double() - e*m*m - (e*var.double() if full else 0.0)
    return Cn.to(C_off.dtype), vn.to(var.dtype)
class D21Hook:
    """Аудит і оракул (2,1)-зрізу κ₃ передактивацій. Істина: mu21[i,j]=E[(p_i-a_i)^2(p_j-a_j)] (центральний = кумулянт для 3-го
    порядку), D3 = E3-3aE2+2a^3. Передбачення ДО прогону: R² його D21 на l=1 ≈ 0.99, до l=15 падає до ~0.9 (ε~2..10%);
    істинне D21 у use-side на regen: 2.20e-8 → 1.0…1.6e-8 (шанс ≥×1.5: 40%); +feed додає мало; +κ₃₁ разом → ≤5e-9 (шанс 30%)."""
    def __init__(self, kp): self.kp = kp; self.lines = []
    def __call__(self, l, D3, D21, live):
        self.last_l = l
        if l < 1: return D3, D21, None
        if not (D21LMIN <= l <= D21LMAX): return D3, D21, None   # вікно шарів оракула (атрибуція)
        if OLDSKIP and (l % OLDSKIP == 1) and "K3" in K3REF:
            # читання старого ярусу лише на шарах l % OLDSKIP == 0: тут зріз ЛИШЕ молодих джерел (вік < OLDAGE), use і feed
            K3 = K3REF["K3"]; young = [src for src in K3.sources if src.get("age", 0) < OLDAGE]
            D3y, D21y = K3.dslices(subset=young) if young else (D3*0, D21*0)
            D21y = D21y.clone(); D21y.fill_diagonal_(0.0); self.lines.append("l=%2d OLDSKIP: молодих %d з %d, ‖D21y‖/‖D21‖ %.3f" % (l, len(young), len(K3.sources), float(D21y.norm()/D21.norm())))
            return D3y, D21y, D21y
        kp = self.kp; a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l].astype(np.float64); E3 = kp["pre_m3"][l].astype(np.float64)
        _, _, _, mu21 = true_pair(kp, l); k3 = E3 - 3*a*E2 + 2*a**3
        n = mu21.shape[0]; offm = ~np.eye(n, dtype=bool)
        his = D21.detach().cpu().double().numpy(); hd3 = D3.detach().cpu().double().numpy()
        r_o, a_o = r2(his[offm], mu21[offm]); r_d, a_d = r2(hd3, k3)
        eps = float(np.sqrt(((his[offm]-mu21[offm])**2).sum()/(mu21[offm]**2).sum()))
        E4 = kp["pre_m4"][l].astype(np.float64); v = E2 - a*a; m4 = E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4
        N = float(kp["n_samples"]); nb = float(np.sqrt((np.outer(m4, v)[offm]).sum()/N/(mu21[offm]**2).sum()))
        line = "l=%2d D21~κ₂₁ R² %.4f нахил %.3f ε %.3f (шум банку %.3f) | D3~κ₃ R² %.4f нахил %.3f" % (l, r_o, a_o, eps, nb, r_d, a_d)
        tru = np.where(offm, mu21, 0.0)
        if D21PROJ:
            # низькорангова поправка: його D21 + проєкція нев'язки R=істина−його на топ-r підпростір
            # res: власний SVD нев'язки (оракульний базис) · hisR: правий (j) топ-r підпростір ЙОГО D21 · hisL: лівий (i)
            Rm = np.where(offm, mu21 - his, 0.0)
            if D21PBASIS == "res":
                U, S, Vh = np.linalg.svd(Rm); corr = (U[:, :D21PROJ]*S[:D21PROJ]) @ Vh[:D21PROJ]
            else:
                U, S, Vh = np.linalg.svd(np.where(offm, his, 0.0))
                corr = Rm @ Vh[:D21PROJ].T @ Vh[:D21PROJ] if D21PBASIS == "hisR" else U[:, :D21PROJ] @ (U[:, :D21PROJ].T @ Rm)
            tru = np.where(offm, his + corr, 0.0)
            line += " | proj %s r=%d: частка ‖R‖² %.3f" % (D21PBASIS, D21PROJ, float((corr**2).sum()/(Rm**2).sum()))
        if D21R:
            U, S, Vh = np.linalg.svd(tru); tru = np.where(offm, (U[:, :D21R]*S[:D21R]) @ Vh[:D21R], 0.0)
            line += " | truth rank %d energy %.5f" % (D21R, float((S[:D21R]**2).sum()/(S**2).sum()))
        self.lines.append(line)
        T21 = (torch.as_tensor(tru, dtype=D21.dtype, device=D21.device)*D21W + (1-D21W)*D21) if D21OFF else D21
        T3 = torch.as_tensor(k3, dtype=D3.dtype, device=D3.device)*D21W + (1-D21W)*D3 if D21DIAG else D3
        if RFRAC < 1:   # 16.10: точні лише на випадковій частці (ті самі маски, що в Hook)
            _RM, _RV = rmask(l, D21.shape[0]); T21 = rmix(T21, D21, _RM); T3 = rmix(T3, D3, _RV)
        D3u, D21u = (T3, T21) if D21USE else (D3, D21)
        return D3u, D21u, (T21 if D21FEED else None)
A22 = float(os.environ.get("A22", "1")); A31 = float(os.environ.get("A31", "1")); A4 = float(os.environ.get("A4", "1")); W4 = float(os.environ.get("W4", "1"))   # одиниці: κ_true/A = wk (з пломби rec)
class FitC:   # G_off ≈ λ_l C_off (пас 1) / replay (пас 2)
    def __init__(self): self.coef = []
    def __call__(self, l, G, live):
        n = G.shape[0]; m = ~torch.eye(n, dtype=torch.bool, device=G.device); y = zd(G)[m]; x = zd(live["C"])[m]
        a = float((x*y).sum()/(x*x).sum()); self.coef.append(a); return torch.diag(torch.diagonal(G)) + a*zd(live["C"])
class Replay:
    def __init__(self, coef): self.coef = coef
    def __call__(self, l, G, live):
        lk.REGEN_STATE["lam"] = float(self.coef[l]); lk.REGEN_STATE["g"] = torch.diagonal(G).clone()
        return torch.diag(torch.diagonal(G)) + self.coef[l]*zd(live["C"])
K31FEED = float(os.environ.get("K31FEED", "0")); K4FEED = float(os.environ.get("K4FEED", "0"))
class FeedTrue:
    """g_feed_hook: G_off ← κ₃₁_true (wk431 = 0.5·c·G_off, c=2 ⟹ G_off = wk431 ≈ κ₃₁, нахил ~1 на l=1), diag(G) ← κ₄_true/c.
    Замінює memoryless-регенерацію ЛИШЕ у народженні κ₃ (feed), use-side лишається його. Питання: чи нев'язка D21 (ε 3..7%,
    низькорангова, j-домінантна) походить з неточного κ₃₁ у feed. Передбачення: ε падає до ~шуму банку на l≤8 (шанс 40%)."""
    def __init__(self, kp, dh): self.kp = kp; self.dh = dh
    def __call__(self, G, C_pre_off):
        l = getattr(self.dh, "last_l", 0); Gp = approx_pre(G, C_pre_off)
        if l < 1: return Gp
        k22, k31, k4, _ = true_pair(self.kp, l); n = G.shape[0]; offm = ~np.eye(n, dtype=bool)
        dg = torch.diagonal(Gp).clone(); off = zd(Gp)
        if K4FEED: dg = torch.as_tensor(k4/2.0, dtype=G.dtype, device=G.device)*K4FEED + (1-K4FEED)*dg
        if K31FEED: off = torch.as_tensor(np.where(offm, k31/A31, 0.0), dtype=G.dtype, device=G.device)*K31FEED + (1-K31FEED)*off
        return torch.diag(dg) + off
def approx_pre(G, C_pre_off):
    lam = lk.REGEN_STATE.get("lam")
    return G if lam is None else torch.diag(torch.diagonal(G)) + lam*zd(C_pre_off)
if os.environ.get("ARM_DROP"):
    _a = os.environ["ARM_DROP"].split(":"); lk.ARM_DROP = (int(_a[0]), tuple(_a[1].split(",")))
if __name__ == "__main__":
    statics = None; mses = []
    for i in dumps:
        d = np.load(DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = mc_kp(w, i) if MCN else dict(np.load(BANK.format(i)))
        if statics is None: statics = lk.build_static_coefs(w.shape[1])
        CUR_DUMP[0] = i; BL_HIST.clear(); BL_W[0] = torch.tensor(w, dtype=torch.float64, device=DEV); hook = Hook(kp); dh = D21Hook(kp); dh = ExtractD21(dh) if EXPARTS else dh; EX_EPS.clear(); lk.C_HOOK = extract_c if EXPARTS else None; t0 = time.perf_counter(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = k3_stash if OLDSKIP else None
        if cfg == "dense":
            lk.WK_HOOK = hook; lk.D21_HOOK = dh
            pred = lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True)
        elif cfg == "regen":
            fit = FitC()
            lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, g_hook=fit)
            lk.WK_HOOK = hook; lk.D21_HOOK = dh
            pred = lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=Replay(fit.coef),
                                      g_feed_hook=(FeedTrue(kp, dh) if (K31FEED or K4FEED) else approx_pre), g_use_hook=approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True,
                                      leg_trunc={"rres": 16, "d21": 16}, fac_schedule=(eval(os.environ["FAC"]) if os.environ.get("FAC") else None))
        else: raise SystemExit("cfg?")
        pred = pred.detach().cpu().double().numpy(); mse = float(np.mean((pred[-1] - gt)**2)); mses.append(mse)
        print("[%s/%s] dump%d mse_final=%.4e t=%.1fs" % (cfg, mode, i, mse, time.perf_counter()-t0), flush=True)
        if EXPARTS: print("   EPS " + " ".join("%d:%.4f" % (k, v) for k, v in sorted(EX_EPS.items())), flush=True)
        for ln in hook.lines: print("   ", ln, flush=True)
        for ln in dh.lines: print("   ", ln, flush=True)
        if mode == "fit" and hook.coef3: json.dump({str(l+1): c for l, c in enumerate(hook.coef3)}, open("memcoef_%d.json" % i, "w"))
    print("[%s/%s] MEAN mse_final=%.4e over %s" % (cfg, mode, float(np.mean(mses)), dumps), flush=True)
    if mode == "k31fit": json.dump({str(d): {str(l): v for l, v in st.items()} for d, st in K31STATS.items()}, open("/home/trim/wblogs/k31law_stats.json", "w"))
