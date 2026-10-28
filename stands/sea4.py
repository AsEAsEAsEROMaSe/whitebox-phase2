"""Sea4 (стенд 30.09): пам'ять моря κ₄ у хабовій граматиці. κ₄(h_s) sea3-діаграма (κ₃-хаб + ребро C) народжується на
кожному шарі s і переноситься лінійно (ноги = ноги джерел ланцюга на шарі l, T_{l←s} = P джерела, народженого на s).
На шарі l будуємо S[x;y,z] = κ₄(p_l)_{xxyz} (усі 6 злиттів осей двохабової форми, щільно n³ fp16 у E), звідки
Δ_new = 0.5 Σ_dbl w1 w1 w2 · S (поправка народження κ₃ у (1,1,1) від зрізу (2,1,1)); далі Dense4: перенос, D21/D3, cap."""
import time, torch, math, numpy as np
import d4dense, d4terms


class Sea4(d4dense.Dense4):
    def __init__(self, n, dev, window=6, slab=32, log=print, coef=0.5, pairfn=None, use_S=True):
        super().__init__(n, dev, slab=slab, log=log)
        self.pairfn = pairfn; self.use_S = use_S; self.mu = None
        self.window = window; self.LAY = {}; self.cur = None; self.coef = coef   # coef = vpc·cnt/24 для ((0,0,1,1),(1,1,0,1))
        self.t.update(mem=0.0); self.stats = []

    # ---- стеш на K3_PROBE (після linear_, до wick_) ----
    def stash(self, l, K3, D21):
        legs = []
        for si, s in enumerate(K3.sources):
            if s["P"] is None or s.get("Q") is not None or s.get("X1") is None:
                continue
            legs.append((si, s["X1"].half().clone(), s["P"].half().clone(), s["Y1"].half().clone()))
        self.LAY[l] = dict(legs=legs)
        for k in [k for k in self.LAY if k < l - self.window]:
            del self.LAY[k]
        self.cur = dict(l=l, K3=K3)

    def set_w(self, l, w, C_off, mu=None):
        self.mu = mu
        if l in self.LAY:
            self.LAY[l]["w"] = {k: v.float().clone() for k, v in w.items()}
            self.LAY[l]["C"] = C_off.half().clone()

    def densify(self, sources, D21_ref=None):
        pass

    def seal_d21(self):
        return d4dense.torch.zeros(self.n, device=self.dev), d4dense.torch.zeros((self.n, self.n), device=self.dev)

    # ---- S = κ₄(p_l)_{x;y,z} з пам'яті ----
    def build_S(self):
        t0 = time.time(); n = self.n; dev = self.dev; l = self.cur["l"]; K3 = self.cur["K3"]; E = self.E
        born = {si: l - s.get("age", 0) for si, s in enumerate(K3.sources)}
        Tof = {}
        for si, s in enumerate(K3.sources):
            if s["P"] is not None and born[si] not in Tof:
                Tof[born[si]] = s["P"].float()
        layers = [s for s in sorted(self.LAY) if s < l and "w" in self.LAY[s] and s in Tof]
        self.S_l = l; self.S_ncomb = 0
        if not layers:
            E.zero_(); self.SCE = 1.0; return 0
        cur = {si: (s["X1"].float(), s["P"].float(), s["Y1"].float()) for si, s in enumerate(K3.sources) if s["P"] is not None and s.get("X1") is not None}
        GH = {}
        for s in layers:
            T = Tof[s]; w = self.LAY[s]["w"]; Cs = self.LAY[s]["C"].float()
            G = T * w[2][None, :]                       # T diag(w2): вершина-з'єднання (deg 2)
            H = (T * w[1][None, :]) @ Cs                # T diag(w1) C: кінець ребра (deg 1)
            GH[s] = (G, H)
        # згруповані CP-частини: Q_tot[s] = Σ_{si,c} (A⊙B) Yᵀ (злиття hub-1); Zc_tot[(si,c)] = Σ_s (G⊙H) Y (злиття hub-2)
        combos = {}
        for s in layers:
            G, H = GH[s]; GHm = G * H
            Qt = torch.zeros((n, n), device=dev)
            for (si, X1s, Ps, Y1s) in self.LAY[s]["legs"]:
                if si not in cur: continue
                st = (X1s, Ps, Y1s); ct = cur[si]
                for c in range(3):
                    Y = st[c].float(); A = ct[(c + 1) % 3]; B = ct[(c + 2) % 3]
                    Qt += (A * B) @ Y.T
                    key = (si, c)
                    if key not in combos:
                        combos[key] = dict(A=A, B=B, Zc=torch.zeros((n, n), device=dev), Ys={})
                    combos[key]["Zc"] += GHm @ Y
                    combos[key]["Ys"][s] = st[c]
            GH[s] = (G, H, Qt)
        ncomb = len(combos)
        SCE = None; E.zero_(); mx = 0.0
        for sl in self.slabs():
            acc = torch.zeros((sl.stop - sl.start, n, n), device=dev)
            for s in layers:
                G, H, Qt = GH[s]
                acc += torch.einsum("xd,kd,ld->xkl", Qt[sl], H, G)                 # злиття (0,1): M[x;k,l]
            for key, cb in combos.items():
                A, B, Zc = cb["A"], cb["B"], cb["Zc"]
                acc += torch.einsum("xJ,iJ,jJ->xij", Zc[sl], A, B)                  # злиття (2,3): M[x;i,j]
                # перехресні злиття: R_tot = Σ_s einsum(H_s[sl], Y_s, G_s) тощо
                R02 = None; R03 = None; R12 = None; R13 = None
                for s, Yh in cb["Ys"].items():
                    G, H, _ = GH[s]; Y = Yh.float()
                    r = torch.einsum("xd,dJ,ld->xJl", H[sl], Y, G); R02 = r if R02 is None else R02 + r   # i=k=x: A_xJ H_xd Y_dJ B_jJ G_ld
                    r = torch.einsum("xd,dJ,kd->xJk", G[sl], Y, H); R03 = r if R03 is None else R03 + r   # i=l=x: A_xJ G_xd Y_dJ B_jJ H_kd
                acc += torch.einsum("xJ,jJ,xJl->xjl", A[sl], B, R02)                # M[x; j, l]
                acc += torch.einsum("xJ,jJ,xJk->xjk", A[sl], B, R03)                # M[x; j, k]
                acc += torch.einsum("xJ,iJ,xJl->xil", B[sl], A, R02)                # (1,2): j=k=x: B_xJ H_xd Y_dJ A_iJ G_ld  -> M[x; i, l]
                acc += torch.einsum("xJ,iJ,xJk->xik", B[sl], A, R03)                # (1,3): j=l=x  -> M[x; i, k]
                del R02, R03
            acc = acc + acc.transpose(1, 2)                                          # симетризація (y,z)
            acc = acc * (2.0 * self.coef / 3.0)
            m_ = float(acc.abs().max()); mx = max(mx, m_)
            if SCE is None:
                SCE = 100.0 / max(m_, 1e-30)
            E[sl] = (acc * SCE).half()
        self.SCE = SCE
        if mx * SCE > 6.0e4:
            self.log("  sea4: переповнення fp16 у S (max %.3e × %.3e) — множник /8" % (mx, SCE))
            E *= (1.0 / 8.0); self.SCE = SCE / 8.0
        torch.cuda.synchronize(); self.t["mem"] += time.time() - t0
        self.S_ncomb = ncomb
        return ncomb

    def k31_mem(self):
        """κ₄(p_l)_{aaab} з пам'яті: S[a; a, b] = E[a, a, b]; нулі на діагоналі."""
        if getattr(self, "S_ncomb", 0) == 0:
            return None
        K = torch.diagonal(self.E, dim1=0, dim2=1).float().T / self.SCE      # [a, b]
        K = K.clone(); K.fill_diagonal_(0.0)
        return K

    def mem_slab(self, sl, wick):
        """Δ_new[a_sl,b,c] = 0.5 Σ_dbl w w w2 · κ₄(p)_{dbl} з S, нулі на збігах."""
        n = self.n; E = self.E; s_ = sl.stop - sl.start
        w1 = wick(1, 1).float(); w2 = wick(2, 1).float()
        K0 = E[sl].float() / self.SCE                                    # κ₄_{a a b c}
        Kb = E[:, sl, :].float() / self.SCE                               # [x, a, z] = S[x; a, z]
        K1 = Kb.permute(1, 0, 2)                                          # κ₄_{a b b c} = S[b; a, c]
        K2 = Kb.permute(1, 2, 0)                                          # κ₄_{a b c c} = S[c; a, b]
        dn = 0.5 * (w2[sl][:, None, None] * w1[None, :, None] * w1[None, None, :] * K0
                    + w1[sl][:, None, None] * w2[None, :, None] * w1[None, None, :] * K1
                    + w1[sl][:, None, None] * w1[None, :, None] * w2[None, None, :] * K2)
        dn = dn * d4terms.distinct_mask(sl, n, self.dev).to(dn.dtype)
        if self.pairfn is not None:
            if self.pairfn.with_sea:
                dn = pair_delta_slab(self.pairfn, sl, n, self.blocks, self.mu, wick, self.dev, S_E=self.E if self.use_S else None, SCE=self.SCE)
            else:
                dn = (dn if self.use_S else 0.0) + pair_delta_slab(self.pairfn, sl, n, self.blocks, self.mu, wick, self.dev)
        return dn

    def birth(self, blocks, wick, w1, pk1=None, T3=None, T4=None, with_k4=False):
        t0 = time.time(); n = self.n; dev = self.dev
        if self.cur is None:
            return
        self.blocks = blocks
        ncomb = (self.build_S() if getattr(self, "S_l", None) != self.cur["l"] else self.S_ncomb) if self.use_S else 0
        if self.pairfn is not None and self.mu is not None:
            ncomb = max(ncomb, 1)
        w1 = w1.float(); newD = self.D is None
        cap = torch.zeros((n, n), dtype=torch.float32, device=dev) if pk1 is not None else None
        mx = 0.0; nrm2 = 0.0
        if ncomb == 0:
            if not newD:   # лише перенос старої Δ через ворота
                for sl in self.slabs():
                    old = self.D[sl].float() * (w1[sl][:, None, None] * w1[None, :, None] * w1[None, None, :])
                    self.D[sl] = old.half()
                    if cap is not None:
                        cap[sl] = 0.0
                self.birth_cap = None
                self._rescale()
            return
        if newD:
            for sl in self.slabs():
                mx = max(mx, float(self.mem_slab(sl, wick).abs().max()))
            self.SC = 500.0 / mx if mx > 0 else 1.0
            self.D = torch.zeros((n, n, n), dtype=torch.float16, device=dev)
        for sl in self.slabs():
            dn = self.mem_slab(sl, wick) * d4dense.D4SCALE
            nrm2 += float((dn * dn).sum())
            if cap is not None:
                cap[sl] = dn @ pk1.float()
            mx = max(mx, float(dn.abs().max()))
            if newD:
                self.D[sl] = (dn * self.SC).half()
            else:
                old = self.D[sl].float() * (w1[sl][:, None, None] * w1[None, :, None] * w1[None, None, :])
                self.D[sl] = (old + dn * self.SC).half()
        self._rescale()
        self.birth_cap = cap
        self.stats.append((self.cur["l"], ncomb, math.sqrt(nrm2)))
        torch.cuda.synchronize(); self.t["birth"] += time.time() - t0
        self.log("  sea4 l=%d: комбо %d, ‖Δ_new‖ %.3e, часи mem %.0f birth %.0f trans %.0f с" % (
            self.cur["l"], ncomb, math.sqrt(nrm2), self.t["mem"], self.t["birth"], self.t["trans"]))


# ---- 01.10: парна функція зрізу κ₄(p)_{hhxy} з блоків ланцюга (рідж ст.2, коефіцієнти з pairfit_*.npz) ----
class PairFn:
    def __init__(self, path, dev):
        Z = np.load(path)
        self.mean = torch.tensor(Z["mean"], dtype=torch.float32, device=dev); self.scale = torch.tensor(Z["scale"], dtype=torch.float32, device=dev)
        self.powers = Z["powers"]; self.coef = torch.tensor(Z["coef"], dtype=torch.float32, device=dev); self.b0 = float(Z["intercept"])
        self.with_sea = self.coef.shape[0] > self.powers.shape[0]

    def predict(self, F, sea=None):
        """F: список 21 тензора (broadcastable до (s,n,n)) стандартизованих ознак; sea: стандартизований носій або None."""
        Fs = [(f - self.mean[k]) / self.scale[k] for k, f in enumerate(F)]
        out = torch.full_like(Fs[3] + Fs[5] + Fs[0], self.b0)
        for t, pw in enumerate(self.powers):
            c = float(self.coef[t])
            if c == 0.0: continue
            term = None
            for k, p in enumerate(pw):
                if p == 0: continue
                f = Fs[k] if p == 1 else Fs[k] ** int(p)
                term = f if term is None else term * f
            out = out + c * term
        if self.with_sea and sea is not None:
            out = out + float(self.coef[-1]) * sea
        return out


def pair_delta_slab(pf, sl, n, blocks, mu, wick, dev, sub=8, S_E=None, SCE=1.0):
    """Δ_pairs[a_sl,b,c] = 0.5 Σ_dbl w w w2 · f(hub; x, y) (у справжніх одиницях), нулі на збігах. sub: підслаб по a."""
    var = blocks["var"].float(); s = torch.sqrt(var); C = blocks["C"].float() / (s[:, None] * s[None, :])
    D21 = blocks["D21"].float() / (s[:, None] ** 2 * s[None, :]); D3 = blocks["D3"].float() / s ** 3; mus = mu.float() / s
    w1 = wick(1, 1).float(); w2 = wick(2, 1).float()
    out = torch.zeros((sl.stop - sl.start, n, n), device=dev)
    for a0 in range(sl.start, sl.stop, sub):
        ss = slice(a0, min(a0 + sub, sl.stop)); ns = ss.stop - ss.start
        def M(Mat, r1, r2, pos):
            rows = ss if pos[r1] == 0 else slice(None); cols = ss if pos[r2] == 0 else slice(None)
            T = Mat[rows][:, cols]
            if pos[r1] > pos[r2]: T = T.T; r1, r2 = r2, r1
            shp = [1, 1, 1]; shp[pos[r1]] = T.shape[0]; shp[pos[r2]] = T.shape[1]
            return T.reshape(shp)
        def V(vec, r, pos):
            v = vec[ss] if pos[r] == 0 else vec
            shp = [1, 1, 1]; shp[pos[r]] = v.shape[0]; return v.reshape(shp)
        acc = torch.zeros((ns, n, n), device=dev)
        for pos, wts in ((dict(h=0, x=1, y=2), (w2, w1, w1)), (dict(h=1, x=0, y=2), (w1, w2, w1)), (dict(h=2, x=0, y=1), (w1, w1, w2))):
            z = torch.zeros((), device=dev)
            F = [V(mus, "h", pos), V(mus, "x", pos) + V(mus, "y", pos), V(mus, "x", pos) * V(mus, "y", pos),
                 M(C, "h", "x", pos) + M(C, "h", "y", pos), M(C, "h", "x", pos) * M(C, "h", "y", pos), M(C, "x", "y", pos),
                 M(D21, "h", "x", pos) + M(D21, "h", "y", pos), M(D21, "x", "h", pos) + M(D21, "y", "h", pos), M(D21, "x", "y", pos) + M(D21, "y", "x", pos),
                 M(D21, "h", "x", pos) * M(D21, "h", "y", pos),
                 z, z, z, z, z,
                 V(D3, "h", pos), V(D3, "x", pos) + V(D3, "y", pos), z, z,
                 M(C, "h", "x", pos) * M(C, "x", "y", pos) + M(C, "h", "y", pos) * M(C, "x", "y", pos),
                 M(C, "h", "x", pos) * M(C, "h", "y", pos) * M(C, "x", "y", pos)]
            sea = None
            if pf.with_sea and S_E is not None:
                # стандартизований носій S[hub; x, y] на цьому підслабі у розкладці (a,b,c)
                if pos["h"] == 0: Sv = S_E[ss].float()
                elif pos["h"] == 1: Sv = S_E[:, ss, :].float().permute(1, 0, 2)
                else: Sv = S_E[:, ss, :].float().permute(1, 2, 0)
                sea = Sv / SCE / (V(s, "h", pos) ** 2 * V(s, "x", pos) * V(s, "y", pos))
            pred = pf.predict(F, sea) * (V(s, "h", pos) ** 2 * V(s, "x", pos) * V(s, "y", pos))
            # ваги wick за позиціями a,b,c: (dbl=a: w2 w1 w1) тощо
            wa, wb, wc = (wts[pos["h"]] if False else None, None, None)
            wv = [None, None, None]; wv[pos["h"]] = w2; wv[pos["x"]] = w1; wv[pos["y"]] = w1
            acc = acc + 0.5 * wv[0][ss][:, None, None] * wv[1][None, :, None] * wv[2][None, None, :] * pred
        out[ss.start - sl.start: ss.stop - sl.start] = acc * d4terms.distinct_mask(ss, n, dev).to(acc.dtype)
    return out
