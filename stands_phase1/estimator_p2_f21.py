"""С118 — ДЕТЕРМІНОВАНА РЕКУРСІЯ ТРЕТЬОГО ПОРЯДКУ (хабова форма κ₃), БЕЗ МОНТЕ-КАРЛО.

Стан на передактивації p_l: mp (n), Σ_p (n×n, fp64) і κ₃ у хабовій формі: джерело, народжене на
шарі k з гаусового p_k (ланцюгові доданки Мелера 3-го порядку, (a,b) ≤ 2):
   T_ijk = Σ_ab (a+b)!·sym3[ c_{a+b,i} · (ρ^a diag c_a)_ij · (ρ^b diag c_b)_ik ]
переноситься ТОЧНО матрицями (p→h: ×Φ(t); h→p: ×W), живе MAXAGE шарів (fp32, Штрассен).
Крок m: mpΦ + sφ − (γ₃/6)stφ + (γ₄c/24)s(t²−1)φ, γ₄c — універсальна пошарова крива (24 мережі).
Крок Σ: Мелер n≤NM + парний Еджворт з κ_aab + точна 1-D діагональ.
[стенд rec118h, 8 мереж: 0.1271% → 0.0492% (×6.7 у E²) точно; MAXAGE/обрізання — ~×5]
"""
from __future__ import annotations
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

# 🆕С120/09.09 f21: БЕЗ згортки (fold убиває F21: rec118h FOLD=4 0.0525 проти FOLD=99 0.0398), усі джерела живуть хабово
_F21B = 1.3          # вага f21b-джерела (скан rec118h: 0.7→0.0404 · 1.0→0.0392 · 1.3→0.0389 · 1.6→0.0395)
_F21PRH = 0.7        # хаби F21: |3c₃| > _F21PRH·max (0.5→0.0392/0.692B · 0.7→0.0392/0.594B · 0.85→0.0428)
_F21AGE = 99         # вік життя F21-джерел (8 ⟹ −10% їх ціни без втрат за rec118h)
_AMAX = 1
_ALPHA = -0.00050    # 🆕С118/03.09: спільний масштаб виходу, fit на 24 dev24 (A −0.00050 · B −0.00049), a* sd 1.3e-4
_NM = 6              # порядок Мелера у кроці Σ
_PRH = 0.05          # хаби: лишати i з φ(t_i) > _PRH·max φ (rec118h-конфіг F21: 0.05/0.03)
_PRC = 0.03          # стовпці при переносі: лишати j з Φ(t_j) > _PRC
_G4C = 1.0
_STRASSEN_L = 3      # L=4 давав 100 с backend на мережу (дрібні блоки) і TimeExhausted 120 с
_G4CURVE = [0.0, 0.0098, 0.0163, 0.0217, 0.0267, 0.0313, 0.0353, 0.0392,
            0.0427, 0.0461, 0.0492, 0.0525, 0.0553, 0.0584, 0.0609, 0.0635]
_INV_SQRT_2PI = 0.3989422804014327
_INV_SQRT_2 = 0.7071067811865476
_FACT = [1, 1, 2, 6, 24, 120, 720, 5040, 40320]


def _mm(a, b, level=_STRASSEN_L):
    """Штрассен–Віноград, рівні як осі батчу (С115). При level=0 тотожно fnp.matmul."""
    if level <= 0:
        return fnp.matmul(a, b)
    A = a[None]
    B = b[None]
    done = 0
    for _ in range(level):
        if A.shape[1] % 2 or A.shape[2] % 2 or B.shape[2] % 2:
            break
        m = A.shape[1] // 2
        k = A.shape[2] // 2
        n = B.shape[2] // 2
        A11 = A[:, :m, :k]; A12 = A[:, :m, k:]
        A21 = A[:, m:, :k]; A22 = A[:, m:, k:]
        B11 = B[:, :k, :n]; B12 = B[:, :k, n:]
        B21 = B[:, k:, :n]; B22 = B[:, k:, n:]
        A = fnp.concatenate([A11 + A22, A21 + A22, A11, A22,
                             A11 + A12, A21 - A11, A12 - A22], axis=0)
        B = fnp.concatenate([B11 + B22, B11, B12 - B22, B21 - B11,
                             B22, B11 + B12, B21 + B22], axis=0)
        done += 1
    M = fnp.matmul(A, B)
    for _ in range(done):
        r = M.shape[0] // 7
        M1 = M[0*r:1*r]; M2 = M[1*r:2*r]; M3 = M[2*r:3*r]; M4 = M[3*r:4*r]
        M5 = M[4*r:5*r]; M6 = M[5*r:6*r]; M7 = M[6*r:7*r]
        top = fnp.concatenate([M1 + M4 - M5 + M7, M3 + M5], axis=2)
        bot = fnp.concatenate([M2 + M4, M1 - M2 + M3 + M6], axis=2)
        M = fnp.concatenate([top, bot], axis=1)
    return M[0]


def _phi(x):
    return fnp.exp(-0.5 * x * x) * _INV_SQRT_2PI


def _Phi(x):
    ax = fnp.abs(x) * _INV_SQRT_2
    tt = 1.0 / (1.0 + 0.3275911 * ax)
    poly = tt * (0.254829592 + tt * (-0.284496736 + tt * (1.421413741
                 + tt * (-1.453152027 + tt * 1.061405429))))
    er = 1.0 - poly * fnp.exp(-ax * ax)
    return 0.5 * (1.0 + fnp.sign(x) * er)


def _He(n, t):
    a, b = fnp.ones_like(t), t
    if n == 0:
        return a
    for k in range(1, n):
        a, b = b, t * b - k * a
    return b


def _hcoef(s, t, N, ph, Ph):
    c = [s * (ph + t * Ph), s * Ph]
    for n in range(2, N + 1):
        c.append(s * ((-1) ** n) * _He(n - 2, t) * ph / _FACT[n])
    return c


def _mask_top(v, thr, n_round=16):
    """булева маска v > thr, кількість округлена вниз до кратного n_round (беремо найбільші)"""
    cnt = int(fnp.sum((v > thr).astype(fnp.float32)))
    cnt = max(n_round, (cnt // n_round) * n_round)
    if cnt >= v.shape[0]:
        return None
    return v >= fnp.sort(v)[v.shape[0] - cnt]


class _Source:
    """κ₃(h_{k+1}) з гаусового p_k у хабовій формі; матриці fp32 у поточному просторі."""

    def __init__(self, s, t, rho, c, ph):
        I = _mask_top(ph, _PRH * float(fnp.max(ph)))
        rho32 = rho.astype(fnp.float32)
        c32 = [x.astype(fnp.float32) for x in c]
        rp = {1: rho32, 2: rho32 * rho32}
        self.arm = {}
        for a in range(1, _AMAX + 1):
            A = rp[a] * c32[a][None, :]
            self.arm[a] = A[I] if I is not None else A
        self.hubc = {q: (c32[q][I] if I is not None else c32[q]) for q in range(2, 2 * _AMAX + 1)}
        self.I = I
        self.hub = None
        self.age = 0
        self.act = None
        self.ph_scale = None

    def scale_cols(self, Ph32):
        act = _mask_top(Ph32, _PRC)
        for a in self.arm:
            A = self.arm[a] * Ph32[None, :]
            self.arm[a] = A[:, act] if act is not None else A
        if self.hub is not None:
            for q in self.hub:
                H = self.hub[q] * Ph32[None, :]
                self.hub[q] = H[:, act] if act is not None else H
        else:
            self.ph_scale = Ph32
        self.act = act

    def push(self, X32):
        Xa = X32[self.act] if self.act is not None else X32
        for a in self.arm:
            self.arm[a] = _mm(self.arm[a], Xa)
        if self.hub is None:
            # перший перенос: diag(c_q)[I] · diag(Φ) · X = (c_q Φ)[I] ⊙ X[I]
            XI = X32[self.I] if self.I is not None else X32
            if self.ph_scale is None:
                self.hub = {q: self.hubc[q][:, None] * XI for q in self.hubc}
            else:
                sc = self.ph_scale
                if self.act is not None:
                    sc = sc * self.act.astype(fnp.float32)
                scI = sc[self.I] if self.I is not None else sc
                self.hub = {q: (self.hubc[q] * scI)[:, None] * XI for q in self.hubc}
        else:
            for q in self.hub:
                self.hub[q] = _mm(self.hub[q], Xa)
        self.age += 1
        self.act = None

    def fold(self, V32):
        """ядро цього джерела у базисі V (n×r): Σ_ab f·sym3(Σ_i (HmV)_i ⊗ (AmV)_i ⊗ (BmV)_i)"""
        r = V32.shape[1]
        Y = {a: fnp.matmul(self.arm[a], V32).T for a in self.arm}
        C = None
        for a in range(1, _AMAX + 1):
            for b in range(1, _AMAX + 1):
                f = float(_FACT[a + b])
                X = fnp.matmul(self.hub[a + b], V32).T
                KR = (Y[a][:, None, :] * Y[b][None, :, :]).reshape(r * r, -1)
                H = fnp.matmul(X, KR.T).reshape(r, r, r)
                H = f * (H + fnp.transpose(H, (1, 0, 2)) + fnp.transpose(H, (1, 2, 0)))
                C = H if C is None else C + H
        return C.astype(fnp.float64)

    def kab(self):
        K = None
        for a in range(1, _AMAX + 1):
            for b in range(1, _AMAX + 1):
                f = float(_FACT[a + b])
                Hm = self.hub[a + b]; Am = self.arm[a]; Bm = self.arm[b]
                term = 2.0 * _mm((Hm * Am).T, Bm) + _mm((Am * Bm).T, Hm)
                K = term * f if K is None else K + term * f
        return K


class _SourceF21:
    """f21b: негаусів зворотний зв'язок κ₂₁(p_l) у породження κ₃(h_{l+1}): E[He₃(x_i)He₁He₁] ∋ 3κ₂₁_ij ρ_ik.
    Хаб i (вага w=_F21B·3c₃), руки A=c₁∘κ₂₁[i,:] (свій ланцюг) і B=c₁∘ρ_i (= arm[1] батьківського g-джерела того ж шару);
    хаб H = (w/c₂)∘hub[2] батька (масштаб рядків комутує з переносом) ⟹ ОДИН новий ланцюг замість трьох."""

    def __init__(self, parent, c, k_aab):
        w = (_F21B * 3.0) * c[3]
        aw = fnp.abs(w)
        mf = _mask_top(aw, _F21PRH * float(fnp.max(aw)))
        I = parent.I
        if mf is None:
            mf = fnp.ones(w.shape[0], dtype=bool)
        if I is not None:
            mf = mf & I
            sel = mf[I]
        else:
            sel = mf
        self.parent = parent
        self.sel = sel
        self.hw = (w / c[2])[mf].astype(fnp.float32)
        A = (k_aab * c[1][None, :]).astype(fnp.float32)
        self.A = A[mf]
        self.age = 0
        self.act = None

    def scale_cols(self, Ph32):
        act = _mask_top(Ph32, _PRC)
        A = self.A * Ph32[None, :]
        self.A = A[:, act] if act is not None else A
        self.act = act

    def push(self, X32):
        Xa = X32[self.act] if self.act is not None else X32
        self.A = _mm(self.A, Xa)
        self.age += 1
        self.act = None

    def kab(self):
        H = self.hw[:, None] * self.parent.hub[2][self.sel]
        B = self.parent.arm[1][self.sel]
        A = self.A
        return 2.0 * (_mm((H * A).T, B) + _mm((H * B).T, A) + _mm((A * B).T, H))


def _core_contract(C, V):
    r = V.shape[1]
    G2 = (V[:, :, None] * V[:, None, :]).reshape(V.shape[0], r * r)
    return fnp.matmul(fnp.matmul(G2, C.reshape(r * r, r)), V.T)


def _core_transport(C, Mx):
    C2 = fnp.einsum("abc,da->dbc", C, Mx)
    C2 = fnp.einsum("dbc,eb->dec", C2, Mx)
    return fnp.einsum("dec,fc->def", C2, Mx)


def _orth_rows(K, Om):
    """ортонормований базис (n×r) простору рядків K: Y = Kᵀ(KΩ), V = Y (YᵀY)^{-1/2}"""
    Y = fnp.matmul(K.T, fnp.matmul(K, Om))
    G = fnp.matmul(Y.T, Y)
    ev, U = fnp.linalg.eigh(G)
    inv = 1.0 / fnp.sqrt(fnp.maximum(ev, 1e-300))
    return fnp.matmul(Y, fnp.matmul(U * inv[None, :], U.T))


class Estimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width = mlp.width
        depth = len(mlp.weights)
        Ws = [fnp.asarray(w, dtype=fnp.float64) for w in mlp.weights]
        n = width
        mp = fnp.zeros(n, dtype=fnp.float64)
        Sp = flops.as_symmetric(fnp.matmul(Ws[0].T, Ws[0]), symmetry=(0, 1))
        sources = []
        out = []
        Knext = None
        for l in range(depth):
            s = fnp.sqrt(fnp.maximum(fnp.diagonal(Sp), 1e-30))
            t = mp / s
            inv = 1.0 / s
            rho = Sp * fnp.outer(inv, inv)
            ph = _phi(t); Ph = _Phi(t)
            c = _hcoef(s, t, max(_NM, 4), ph, Ph)
            # κ₃(p_l) з усіх живих джерел
            Kab = Knext
            if Kab is not None:
                k_aab = Kab * fnp.outer(inv * inv, inv)
                g3 = fnp.diagonal(k_aab)
            else:
                k_aab = None
                g3 = fnp.zeros(n, dtype=fnp.float64)
            g4 = _G4C * _G4CURVE[min(l, 15)]
            # крок m
            m = mp * Ph + s * ph - (g3 / 6.0) * s * t * ph + (g4 / 24.0) * s * (t * t - 1.0) * ph
            out.append(m)
            if l + 1 >= depth:
                break
            # крок Σ: Мелер + парний Еджворт + точна діагональ
            Cov = None
            rp = None
            for k in range(1, _NM + 1):
                rp = rho if rp is None else rp * rho
                term = float(_FACT[k]) * fnp.outer(c[k], c[k]) * rp
                Cov = term if Cov is None else Cov + term
            if k_aab is not None:
                Cov = Cov + fnp.outer(c[2], c[1]) * k_aab + fnp.outer(c[1], c[2]) * k_aab.T \
                    + rho * (6.0 * fnp.outer(c[3], c[2]) * k_aab + 6.0 * fnp.outer(c[2], c[3]) * k_aab.T
                             + 4.0 * fnp.outer(c[4] * g3, c[1]) + 4.0 * fnp.outer(c[1], c[4] * g3))
            e2 = (mp * mp + s * s) * Ph + mp * s * ph + s * s * ph * (g3 / 3.0 - g4 * t / 12.0)
            Cov = Cov + fnp.diag(e2 - m * m - fnp.diagonal(Cov))
            Cov = flops.as_symmetric(Cov, symmetry=(0, 1))
            Wn = Ws[l + 1]
            # перенос джерел p_l → h_{l+1} → p_{l+1}; нове джерело з гаусового p_l
            Ph32 = Ph.astype(fnp.float32)
            W32 = Wn.astype(fnp.float32)
            for src in sources:
                src.scale_cols(Ph32)
            # нове джерело НЕ масштабується Φ: його c_a уже містять ReLU шару l (як у rec118h)
            g = _Source(s, t, rho, c, ph)
            sources.append(g)
            if k_aab is not None:
                sources.append(_SourceF21(g, c, k_aab))
            for src in sources:
                src.push(W32)
            mp = fnp.matmul(m, Wn)
            # 🆕k3b: крок Sp у fp32 (einsum був 25% флопів у fp64); ρ потребує ~1e-7 відн., похибка методу 6e-4
            Sp = fnp.einsum("ij,ia,jb->ab", flops.as_symmetric(Cov.astype(fnp.float32), symmetry=(0, 1)), W32, W32).astype(fnp.float64)
            Sp = flops.as_symmetric(0.5 * (Sp + Sp.T), symmetry=(0, 1))
            # κ_aab(p_{l+1}) усіх джерел + перенесене ядро; старі джерела згортаються у ядро
            Knext = None
            for src in sources:
                Kk = src.kab().astype(fnp.float64)
                Knext = Kk if Knext is None else Knext + Kk
            sources = [src for src in sources if not (isinstance(src, _SourceF21) and src.age >= _F21AGE)]
        out[-1] = out[-1] * (1.0 + _ALPHA)
        return fnp.stack(out, axis=0)
