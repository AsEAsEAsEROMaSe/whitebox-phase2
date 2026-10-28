"""Щільний тензор поправки Δ (n³, fp16 на GPU зі шкалою) до κ₃ від діаграм k_max=4 у зрізі (1,1,1) (стенд 26.09).
Δ народжується щошару з κ₃(p) = Sym(CP-джерела lean) + Δ_pre через d4terms, несеться модовими добутками W,
і живить D21 (Δ_aac), D3 (Δ_aaa) та cap-контракції k211 (Δ°×₃u, без збігів індексів)."""
import torch, time, os
D4SCALE = float(os.environ.get('D4SCALE', '1'))
import d4terms


class Dense4:
    def __init__(self, n, dev, slab=64, jch=256, log=print):
        self.n, self.dev, self.s, self.jch, self.log = n, dev, slab, jch, log
        self.D = None          # (n,n,n) fp16 = Δ·SC
        self.E = torch.zeros((n, n, n), dtype=torch.float16, device=dev)   # scratch: щільний CP / буфер транспорту
        self.SC = 1.0; self.SCE = 1.0
        self.t = dict(dens=0.0, birth=0.0, trans=0.0)
        self.birth_cap = None
        self.diagrams = None

    def slabs(self):
        for a0 in range(0, self.n, self.s):
            yield slice(a0, min(a0 + self.s, self.n))

    # ---- транспорт: Δ_pre(l+1) = W ×₁×₂×₃ Δ_post ----
    def transport(self, W):
        if self.D is None:
            return
        t0 = time.time(); n = self.n; W = W.float()
        Dv = self.D.view(n, n * n); Ev = self.E.view(n, n * n)
        cw = 1 << 17
        for c0 in range(0, n * n, cw):
            cc = slice(c0, min(c0 + cw, n * n))
            Ev[:, cc] = (W @ Dv[:, cc].float()).half()
        for sl in self.slabs():
            slab = self.E[sl].float()
            slab = torch.matmul(W[None], slab)
            slab = torch.matmul(slab, W.T)
            self.D[sl] = slab.half()
        self._rescale()
        torch.cuda.synchronize(); self.t["trans"] += time.time() - t0

    def _absmax(self, T):
        m = 0.0
        for sl in self.slabs():
            m = max(m, float(T[sl].abs().max()))
        return m

    def _rescale(self):
        m = self._absmax(self.D)
        if m > 2.0e4:
            self.D *= 1.0 / 16.0; self.SC /= 16.0
        elif 0 < m < 30.0:
            self.D *= 16.0; self.SC *= 16.0

    # ---- зрізи ----
    def add_dslices(self, D3, D21):
        if self.D is None:
            return D3, D21
        n = self.n
        aac = torch.diagonal(self.D, dim1=0, dim2=1).T.float() / self.SC     # (n,n): [a,c] = Δ[a,a,c]
        aaa = torch.diagonal(aac, 0).clone()
        aac = aac.clone(); aac.fill_diagonal_(0.0)
        return (D3 + aaa.to(D3.dtype), D21 + aac.to(D21.dtype))

    def cap(self, u, s=None):
        """diag(s) (Δ° ×₃ u) diag(s), Δ° без збігів індексів; (n,n) або None."""
        if self.D is None:
            return None
        n = self.n; out = torch.zeros((n, n), dtype=torch.float32, device=self.dev); u = u.float()
        for sl in self.slabs():
            slab = self.D[sl].float() * d4terms.distinct_mask(sl, n, self.dev)
            out[sl] = slab @ u
        out = out / self.SC
        if s is not None:
            out = s.float()[:, None] * out * s.float()[None, :]
        return out

    # ---- щільний CP: E = Σ_джерел Σ_трійок X⊗P⊗Y (несиметризований), шкала SCE ----
    def densify(self, sources, D21_ref=None):
        t0 = time.time(); n = self.n; E = self.E; E.zero_()
        trip = []
        for src in sources:
            P, Q = src["P"], src.get("Q")
            assert P is not None
            def leg(k):
                v = src.get(k)
                if v is None:
                    return None
                return (Q @ v).float() if Q is not None else v.float()
            Pd = leg("P")
            for xk, yk in (("X1", "Y1"), ("X2", "Y2"), ("X3", "Y3"), ("X4", "Y4")):
                X = leg(xk)
                if X is not None:
                    trip.append((X, Pd, leg(yk)))
            M = leg("M")
            if M is not None:
                trip.append((M, Pd, Pd))
        self.ntrip = len(trip)
        if not trip:
            self.SCE = 1.0
            self.t["dens"] += time.time() - t0
            return
        # шкала: max|T| ~ кілька max|D21|
        ref = float(D21_ref.abs().max()) if D21_ref is not None else 1.0
        self.SCE = 500.0 / max(ref, 1e-30)
        Ev = E.view(n, n * n)
        for j0 in range(0, n, self.jch):
            jc = slice(j0, min(j0 + self.jch, n)); jl = jc.stop - jc.start
            for (L1, L2, L3) in trip:
                KR = (L2[:, jc].T[:, :, None] * L3[:, jc].T[:, None, :]).reshape(jl, n * n)   # (jl, n²): [j, b*n+c]
                for sl in self.slabs():
                    part = (L1[sl][:, jc] @ KR) * self.SCE
                    Ev[sl] += part.half()
        m = self._absmax(E)
        if not (m < 6.0e4):
            self.log("  densify: переповнення fp16 (max %.3e), шкала /64 і повтор" % m)
            self.SCE /= 64.0
            E.zero_()
            for j0 in range(0, n, self.jch):
                jc = slice(j0, min(j0 + self.jch, n)); jl = jc.stop - jc.start
                for (L1, L2, L3) in trip:
                    KR = (L2[:, jc].T[:, :, None] * L3[:, jc].T[:, None, :]).reshape(jl, n * n)
                    for sl in self.slabs():
                        Ev[sl] += ((L1[sl][:, jc] @ KR) * self.SCE).half()
        torch.cuda.synchronize(); self.t["dens"] += time.time() - t0

    def sym_slab(self, sl):
        """Sym(E)[sl] у fp32 у справжніх одиницях."""
        T1 = self.E[sl].float()
        S = T1 + T1.transpose(1, 2)
        T2 = self.E[:, sl, :].float().permute(1, 0, 2)
        S = S + T2 + T2.transpose(1, 2)
        T3 = self.E[:, :, sl].float().permute(2, 0, 1)
        S = S + T3 + T3.transpose(1, 2)
        return S * (1.0 / (6.0 * self.SCE))

    def seal_d21(self):
        """D21 і D3 з Sym(E) (лише CP-частина) для пломби проти K3.dslices()."""
        n = self.n
        D21 = torch.zeros((n, n), dtype=torch.float32, device=self.dev); D3 = torch.zeros(n, dtype=torch.float32, device=self.dev)
        for sl in self.slabs():
            S = self.sym_slab(sl)
            ia = torch.arange(sl.stop - sl.start, device=self.dev)
            aac = S[ia, sl.start + ia, :]                      # [a_local, c] = S[a,a,c]
            D21[sl] = aac
            D3[sl] = aac[ia, sl.start + ia]
        D21.fill_diagonal_(0.0)
        return D3, D21

    # ---- народження: Δ_post = wick(Δ_pre) + Δ_new(κ₃ = Sym(E) + Δ_pre, блоки) ----
    def birth(self, blocks, wick, w1, pk1=None, T3=None, T4=None, with_k4=False):
        t0 = time.time(); n = self.n; dev = self.dev
        if self.diagrams is None:
            self.diagrams = d4terms.select_diagrams(T3, T4, with_k4=with_k4)
            self.log("  d4: діаграм %d" % len(self.diagrams))
        bl = {k: v.float() for k, v in blocks.items()}
        comp = d4terms.compile_diagrams(self.diagrams, bl, lambda k, p=1: wick(k, p).float(), n)
        w1 = w1.float()
        newD = self.D is None
        if newD:
            self.D = torch.zeros((n, n, n), dtype=torch.float16, device=dev); self.SC = 1.0
        cap = torch.zeros((n, n), dtype=torch.float32, device=dev) if pk1 is not None else None
        mx = 0.0
        if newD:   # перший прохід лише для шкали (на шарі 0 CP-джерел нема, дешево)
            for sl in self.slabs():
                k3 = self.sym_slab(sl) if self.ntrip else None
                dn = d4terms.delta_slab_fast(comp, bl, k3, sl, n, dev)
                if dn is not None:
                    mx = max(mx, float(dn.abs().max()))
            self.SC = 500.0 / mx if mx > 0 else 1.0
        for sl in self.slabs():
            k3 = self.sym_slab(sl) if self.ntrip else torch.zeros((sl.stop - sl.start, n, n), dtype=torch.float32, device=dev)
            if not newD:
                k3 = k3 + self.D[sl].float() / self.SC
            dn = d4terms.delta_slab_fast(comp, bl, k3, sl, n, dev)
            if dn is None:
                dn = torch.zeros_like(k3)
            dn = dn * D4SCALE
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
        torch.cuda.synchronize(); self.t["birth"] += time.time() - t0
