import os
# AUGMENTED lean K3 (k_max=3, kind=AUGMENT): adds the r=1 (matrix-core) kappa4 channel
# on top of lean_k3.py -- the "true (2,1,1) diagram terms" (F48 lever 1, top-10 rung 2).
# K4 is tracked as Sym(G x metric) with G an (n, n) matrix core instead of the SIMPLE
# scalar radial core. New content, all in <= n^3 ops (never materializing n^3 tensors):
#   USE:  WK4 dslices from G: (4,)=m*diag(G); (2,2)=(m/6)(Gaa+Gbb); (3,1)=(m/2)G;
#         (2,1,1)=(m/6) 1_a x G_bc; (1,1,1,1)=0.   [verified: verify_aug_formulas.py]
#   PROD: pK slices (3,1) dense n^2 + (2,1,1) evaluated CAPPED (summed over the 2-block
#         axis with a!=b, a!=c exclusion) via generic einsum builder; extended pK->K;
#         harmonic projection to G: cA*(diag K4 + diag rowsum K22 + K211cap)
#         + cS*Sym(K31) + cI*(sum K4 + sum K22)*I, cA=6/(n+4), cS=12/(n+4),
#         cI=-3/((n+2)(n+4)); plus factored contributions pK211->K211 (cA) and
#         pK111->K211 (-2*cA) from hub-leg contractions.
#   FEED: K4 -> pK(1,1,1) hub sources (reference factor_k3.py augment terms 1 & 3;
#         term 2 vanishes for the diagonal avg metric).
#   TRANSPORT: G <- W G W^T (metric stays metric_c).
# Base file: copy of lean_k3.py (F44 port anchor); default flags reproduce it exactly.
#
# Lean reimplementation of K3-simple factored kprop (torch), structured 1:1 with the
# planned flopscope port (F44 design):
#   - shared-P hub blocks: per source layer {X1,Y1,X2,Y2,M,P}; B1=Sym(X1 x P x Y1),
#     B2=Sym(X2 x P x Y2), B3=Sym(M x P x P); all identity-born legs share one P.
#   - combined hub matmuls in the (2,1) dslice (right-operand P^T terms merged).
#   - all combinatorial coefficients precomputed as tables (imported from the reference
#     repo's pure-python functions here; the port embeds the dumped numbers).
#   - only n^2 / n^3 array ops in the layer loop -> maps op-for-op onto fnp.
# Verification: parity vs the reference mlp_kprop on random small shapes and real dumps.
# Run with the kprop venv:
#   cd C:\kaggle_local\whest\mlp_cumulant_propagation
#   uv run python C:\kaggle_local\whest\scratch\lean_k3.py [--small | n_mlps]
import glob
import math
import sys
import time

import numpy as np
import torch
import os as _os_
DEV = _os_.environ.get('LEAN_DEV', 'cpu')
WK_HOOK = None
K3_PROBE = None
K_POST_HOOK = None      # стенд 26.09: (l, K, mu, C) після збірки пост-стану
K3DAMP = float(os.environ.get("K3DAMP", "0")); K3DAMP_AGE = int(os.environ.get("K3DAMP_AGE", "1"))
K3_PROBE_POST = None    # стенд 26.09: (l, K3) після народження (append_ds)
CONFINE_Q_FN = None
ARM_DROP = None        # (age_min, keys): дешевший старий ярус = менше пар ніг, не менший ранг
D21_HOOK = None        # (l, D3, D21, live) -> (D3u, D21u, D21feed|None)
D21_HOOK_FEED = None
C_HOOK = None          # 16.10: (l, mu, C_offdiag, var) -> (C_offdiag, var), екстракція моди масштабу (форум 18248)
BIRTH_SCALE = float(os.environ.get('BIRTH_SCALE', '1')); BIRTH_SCALE_LMIN = int(os.environ.get('BIRTH_SCALE_LMIN', '0'))
DEAD_TH = float(os.environ['DEAD_TH']) if os.environ.get('DEAD_TH') else None   # стенд 14.10: виключення мертвих (t<-DEAD_TH) з κ3/G/D4 (і C при DEAD_C=1)
DEAD_C = os.environ.get('DEAD_C', '0') == '1'
DEAD_LIN = float(os.environ['DEAD_LIN']) if os.environ.get('DEAD_LIN') else None   # також лінійні t>DEAD_LIN (контроль: мусить шкодити)
DEAD_STATS = []
T_CUR = {}
MU_HOOK = None
if os.environ.get('HUBDEAD_TH'):   # стенд 14.10: хаб-стовпці мертвих при народженні (t_b < -HUBDEAD_TH) прибрати
    _hd = float(os.environ['HUBDEAD_TH'])
    _hl = float(os.environ.get('HUBLIN_TH', '99'))   # стенд 15.10: і лінійні (t_b > HUBLIN_TH)
    def _hubdead(l):
        return (T_CUR['t'] >= -_hd) & (T_CUR['t'] <= _hl)
DEAD_PARTS = os.environ.get('DEAD_PARTS', 'k3,g,d4').split(',')
HUB_MASK = None   # стенд 12.10: (l) -> (n,) маска стовпців хабу новонародженого (X-ноги і M) або None
if os.environ.get('HUBDEAD_TH'):
    HUB_MASK = _hubdead
K3_INJ = None           # стенд 26.09: (l, K3, D3, D21) -> None, вкидання джерела у стан κ₃ перед wick_
TERM_SEEN = None
HYB_K = None
PATH4 = float(_os_.environ.get("PATH4", "0"))
GPARTS = None         # dict або None (стенд 25.09)
D4 = None             # d4dense.Dense4 або None (стенд 26.09: щільна поправка k_max=4 у (1,1,1))
D4_SEAL = {}
SKIP_BIRTH = set(int(x) for x in _os_.environ.get("SKIP_BIRTH", "").split(",") if x != "")
HYB_SET = None
TERM_SCALE = None     # {група: множник} (стенд ручок 23.09); None = без змін
def _svd(M):
    M = M.detach()   # стенд бекпропу 26.09: базис сталий, обрізання = проєкція (чисельно те саме)
    try:
        return torch.linalg.svd(M, full_matrices=False)
    except Exception:
        if not bool(torch.isfinite(M).all()): raise FloatingPointError('non-finite')
        U, S, Vh = torch.linalg.svd(M.double().cpu(), full_matrices=False); U, S, Vh = U.to(M.device), S.to(M.device), Vh.to(M.device)
        return U.to(M.dtype), S.to(M.dtype), Vh.to(M.dtype)
def term_group(int_part, factors):
    n3 = sum(1 for vs, v in factors if vs == 3); n4 = sum(1 for vs, v in factors if vs == 4)
    return (tuple(int_part), n3, n4)

# ---- reference-repo combinatorics (pure python; port embeds the resulting numbers) ----
from mlp_kprop.kprop_harmonic import get_all_terms_iso
from mlp_kprop.partitions import (
    vec_part_coef, check_vec_partition, vector_partitions, int_partition_coef,
)
from mlp_kprop.cumulants import _pK_to_K_coef
from mlp_kprop.harmonic import proj_coef, _multigraph_coef
from mlp_kprop.wick import _relu_wick_poly

torch.set_grad_enabled(False)
torch.set_num_threads(22)

DUMP_DIR = r"C:\kaggle_local\whest\scratch\mlp_dumps_p2"

# ---------------------------------------------------------------------------
# Coefficient tables (computed once; static rationals independent of n except
# where noted). In the flopscope port these are embedded constants.
# ---------------------------------------------------------------------------

def build_term_table():
    """The 94 SIMPLE-mode nonlin terms -> evaluable records.

    Each record: (int_part, factors, coef, wick_kp) where
      factors: tuple of (vsum, vec) slice requests
      coef   = count * vec_part_coef(vec_part, divide_fac=True)
      wick_kp: per output axis (k, p) with k = sum of vec legs on that axis, p = int_part[axis]
    Terms whose factors hit an identically-zero WK4 slice ((3,1)-type) are dropped.
    """
    terms_iso = get_all_terms_iso(k_max=3, d_max=4)
    table = []
    for int_part, vec_part_dict in terms_iso.items():
        if len(int_part) > 3 or int_part in [(3, 1), (2, 1, 1)] or int_part == (1, 1, 1):
            continue
        for vec_part, count in vec_part_dict.items():
            if (int_part, set(vec_part)) == ((2, 1, 1), {(1, 1, 1)}):
                continue
            d = len(int_part)
            # WK4 (HTensor r=2, diagonal metric) has only (4,) and (2,2) nonzero.
            dead = False
            for v in vec_part:
                if sum(v) == 4:
                    nz = tuple(sorted((x for x in v if x > 0), reverse=True))
                    if nz not in [(4,), (2, 2)]:
                        dead = True
            if dead:
                continue
            k_axes = check_vec_partition(vec_part, d)
            coef = count * vec_part_coef(vec_part, divide_fac=True)
            factors = tuple((sum(v), v) for v in vec_part)
            wick_kp = tuple((int(k_axes[ax]), int(int_part[ax])) for ax in range(d))
            table.append((tuple(int_part), factors, float(coef), wick_kp))
    return table


def build_term_tables_aug():
    """AUGMENT-mode term tables. Returns (dense, capped211):
    dense    - int_parts of len <= 2 (incl. the new (3,1)); evaluated on the n^2 grid.
    capped211- int_part (2,1,1) terms; evaluated directly as the 2-block-axis-capped
               (n, n) matrix (the only reduction the r=1 projection ever needs).
    (1,1,1) stays manual (factored hubs); ((2,1,1), {(1,1,1)}) is the WK3-leg
    contraction handled by k211_from_sources. With the r=1 K4 channel every vsum-4
    factor pattern within <= 3 axes is alive, so no dead-filter."""
    terms_iso = get_all_terms_iso(k_max=3, d_max=4)
    dense, capped = [], []
    for int_part, vec_part_dict in terms_iso.items():
        if len(int_part) > 3 or int_part == (1, 1, 1):
            continue
        for vec_part, count in vec_part_dict.items():
            if (int_part, set(vec_part)) == ((2, 1, 1), {(1, 1, 1)}):
                continue
            d = len(int_part)
            k_axes = check_vec_partition(vec_part, d)
            coef = count * vec_part_coef(vec_part, divide_fac=True)
            factors = tuple((sum(v), v) for v in vec_part)
            wick_kp = tuple((int(k_axes[ax]), int(int_part[ax])) for ax in range(d))
            rec = (tuple(int_part), factors, float(coef), wick_kp)
            if int_part == (2, 1, 1):
                capped.append(rec)
            else:
                dense.append(rec)
    return dense, capped


def build_pk_to_k_table(aug=False):
    """pK -> K conversion per output slice: list of (blocks, coef)."""
    out_parts = [(1,), (2,), (1, 1), (2, 1), (2, 2), (3,), (4,)]
    if aug:
        out_parts = out_parts + [(3, 1)]
    table = {}
    for part in out_parts:
        entries = []
        for vpart in vector_partitions(tuple(part)):
            coef = vec_part_coef(vpart, divide_fac=False) * _pK_to_K_coef(vpart)
            if coef == 0:
                continue
            entries.append((tuple(vpart), float(coef)))
        table[tuple(part)] = entries
    return table


def build_pk_to_k_211_table():
    """pK -> K for the CAPPED (2,1,1) slice: (vpart, coef) entries whose product is
    evaluated capped over the 2-block axis. vparts containing a (1,1,1) block (the
    factored pK(1,1,1)) are excluded -- that is the pK111->K211 hub contraction."""
    entries = []
    for vpart in vector_partitions((2, 1, 1)):
        if any(tuple(v) == (1, 1, 1) for v in vpart):
            continue
        coef = vec_part_coef(vpart, divide_fac=False) * _pK_to_K_coef(vpart)
        if coef == 0:
            continue
        entries.append((tuple(vpart), float(coef)))
    return entries


def build_static_coefs(n: int):
    """n-dependent scalars for K4 projection and WK4 slices."""
    # K4 = P2 * (mg4L * sum(pK4slice_(4,)) * ipc4 + mg22L * sum(pK4slice_(2,2)) * ipc22)
    P2 = float(proj_coef(n, 4, 2)[2])
    mg4L = _multigraph_coef([((0, 0), 2)], [4])
    mg22L = _multigraph_coef([((0, 0), 1), ((1, 1), 1)], [2, 2])
    ipc4 = int_partition_coef((4,))       # = 1
    ipc22 = int_partition_coef((2, 2))    # = 3
    # WK4 slices: (4,) = mg4 * sigma * c^2 * ones_vec ; (2,2) = mg22 * sigma * c^2 * (ones - I)
    mg4 = _multigraph_coef([((0, 0), 2)], [4], lap_coef=False)
    mg22 = _multigraph_coef([((0, 0), 1), ((1, 1), 1)], [2, 2], lap_coef=False)
    # r=1 (matrix-core) harmonic projection coefficients, verified numerically in
    # verify_aug_formulas.py against DS_harmonic_proj(r_out=1, geq=True):
    #   (4,)/(2,2) slices: cA on diag/diag-rowsum, cI on total-sum * I
    #   (3,1): cS on Sym(S);  (2,1,1): cA on the capped sum
    # factored contributions (reference factor_k3.py): pK211->K211 cA, pK111->K211 -2cA
    cA = 6.0 / (n + 4)
    cS = 12.0 / (n + 4)
    cI = -3.0 / ((n + 2) * (n + 4))
    return dict(P2=P2, k4_c4=float(mg4L * ipc4), k4_c22=float(mg22L * ipc22),
                wk4_c4=float(mg4), wk4_c22=float(mg22),
                cA=cA, cS=cS, cI=cI)


def build_wick_polys():
    """Coefficient lists for the k<p relu wick cases; (p, k) pairs with k < p <= 4."""
    polys = {}
    for p in range(1, 5):
        for k in range(0, p):
            P1, P2 = _relu_wick_poly(p, k)
            polys[(p, k)] = (list(map(float, P1.coef)), list(map(float, P2.coef)))
    return polys


TERM_TABLE = build_term_table()
TERM_TABLE_AUG, TERM_TABLE_211 = build_term_tables_aug()
if _os_.environ.get("EXTRA_K4TERMS"):   # стенд 23.09: вибрані діаграми k_max=4 у (2,1)/(2,)/(1,1) слайсах
    _T3 = get_all_terms_iso(k_max=3, d_max=4); _T4 = get_all_terms_iso(k_max=4)
    _want = eval(_os_.environ["EXTRA_K4TERMS"])   # список (int_part, vec_part)
    _sc = float(_os_.environ.get("EXTRA_K4SCALE", "1"))
    for _ip, _vp in _want:
        _c = _T4[_ip][_vp]; _d = len(_ip); _ka = check_vec_partition(_vp, _d)
        TERM_TABLE_AUG.append((tuple(_ip), tuple((sum(v), v) for v in _vp), float(_sc * _c * vec_part_coef(_vp, divide_fac=True)),
                               tuple((int(_ka[ax]), int(_ip[ax])) for ax in range(_d))))
PK2K_TABLE = build_pk_to_k_table()
PK2K_TABLE_AUG = build_pk_to_k_table(aug=True)
PK2K211_TABLE = build_pk_to_k_211_table()
WICK_POLYS = build_wick_polys()

# probabilists' Hermite polynomial coefficient lists He_0..He_6 (ascending powers)
HE_COEFS = {0: [1.0], 1: [0.0, 1.0], 2: [-1.0, 0.0, 1.0], 3: [0.0, -3.0, 0.0, 1.0],
            4: [3.0, 0.0, -6.0, 0.0, 1.0], 5: [0.0, 15.0, 0.0, -10.0, 0.0, 1.0],
            6: [-15.0, 0.0, 45.0, 0.0, -15.0, 0.0, 1.0]}

SQRT_2PI = math.sqrt(2.0 * math.pi)


def horner(coefs, x):
    y = torch.zeros_like(x)
    for c in reversed(coefs):
        y = y * x + c
    return y


def norm_pdf(x):
    return torch.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x):
    return torch.special.ndtr(x)


class WickCache:
    """w(k, p) = E[d^k relu(Z)^p] for Z ~ N(mean, var), vectorized over neurons."""

    def __init__(self, mean, var):
        var = torch.clamp(var, min=1e-10)
        self.mean = mean
        self.sigma = torch.sqrt(var)
        self.alpha = mean / self.sigma
        self.phi = norm_pdf(self.alpha)
        self.Phi = norm_cdf(self.alpha)
        self._cache = {}

    def __call__(self, k: int, p: int = 1):
        key = (k, p)
        if key in self._cache:
            return self._cache[key]
        if k < p:
            c1, c2 = WICK_POLYS[(p, k)]
            val = self.sigma ** (p - k) * (
                horner(c1, self.alpha) * self.phi + horner(c2, self.alpha) * self.Phi
            )
        elif p > 1:
            val = math.factorial(p) * self(k - p + 1, 1)
        elif k == 0:
            val = self.sigma * self.phi + self.mean * self.Phi
        elif k == 1:
            val = self.Phi
        else:
            val = ((-1) ** (k - 2)) * self.sigma ** (-(k - 1)) * \
                horner(HE_COEFS[k - 2], self.alpha) * self.phi
        self._cache[key] = val
        return val


def zero_diag_(A):
    A.fill_diagonal_(0.0)
    return A


class K3Blocks:
    """Factored K3 as shared-P hub blocks, one entry per source layer:
    B1 = Sym sum_j X1[:,j] (x) P[:,j] (x) Y1[:,j]   (the 3 from fac2=3I folded into X1)
    B2 = Sym sum_j X2[:,j] (x) P[:,j] (x) Y2[:,j]
    B3 = Sym sum_j M[:,j] (x) P[:,j] (x) P[:,j]
    P is None while the source is at birth (P = I)."""

    def __init__(self):
        self.sources = []  # list of dicts {X1, Y1, X2, Y2, M, P}

    KEYS = ("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "M")
    PAIRS = (("X1", "Y1"), ("X2", "Y2"), ("X3", "Y3"), ("X4", "Y4"))

    def linear_(self, W):
        for s in self.sources:
            s["age"] = s.get("age", 0) + 1
            if ARM_DROP is not None and s["age"] >= ARM_DROP[0]:
                if ARM_DROP[1] == ("ALL",):
                    s["_dead"] = True
                    continue
                for key in ARM_DROP[1]:
                    s[key] = None
            if s.get("Q") is not None:
                s["Q"] = W @ s["Q"]
                continue
            for key in self.KEYS:
                if s.get(key) is not None:
                    s[key] = W @ s[key]
            s["P"] = W.clone() if s["P"] is None else W @ s["P"]
        if ARM_DROP is not None and ARM_DROP[1] == ("ALL",):
            self.sources = [s for s in self.sources if not s.get("_dead")]

    def wick_(self, w1):
        col = w1[:, None]
        for s in self.sources:
            if s.get("Q") is not None:
                s["Q"] = col * s["Q"]
                continue
            for key in self.KEYS + ("P",):
                if s.get(key) is not None:
                    s[key] = col * s[key]

    @staticmethod
    def _top_basis(Gm, rank, eig, sketch=None):
        """Top-rank basis of PSD Gm. eig='exact': eigh. eig='randq' (q passes):
        subspace iteration with a fixed sketch (what the flopscope port will do;
        orthonormalization via QR here, Cholesky-QR in the port)."""
        if eig == "exact":
            _, evecs = torch.linalg.eigh(Gm.detach().double())
            return evecs[:, -rank:].to(Gm.dtype).contiguous()
        q = int(eig[4:])
        Y = sketch[:, :rank] if sketch is not None else None
        if Y is None:
            g = torch.Generator().manual_seed(1234)
            Y = torch.randn(Gm.shape[0], rank, generator=g, dtype=Gm.dtype).to(Gm.device)
        for _ in range(q):
            Y = Gm @ Y
            Y, _ = torch.linalg.qr(Y)
        return Y.contiguous()

    def factorize_schedule_(self, schedule, basis="P", eig="exact"):
        """Low-rank confinement of old sources (F52 structural lever). schedule:
        tuple of (min_age, rank); a source of age a (linear steps survived) gets
        target rank min(r for (ma, r) in schedule if a >= ma). Every leg of a
        source equals T @ birth_leg with T = P the shared propagator, so all legs
        live in colspace(P); we project them onto P's top-(rank) left singular
        space: store Q (n x rank) and per-leg factors Q^T leg (rank x n). After
        this, transport touches ONLY Q (one matmul/source/layer) and dslices/
        contractions become n^2 R sandwiches (exact for the factored rep)."""
        for s in self.sources:
            if s["P"] is None:
                continue
            targets = [r for (ma, r) in schedule if s.get("age", 0) >= ma]
            if not targets:
                continue
            rank = min(targets)
            if s.get("Q") is None:
                n = s["P"].shape[0]
                if rank > n:  # == n allowed: lossless, exercises the factored path
                    continue
                if basis in ("gram", "gram4"):
                    # top eigenspace of the LEG Gram: weights directions by the
                    # actual represented content, not just the propagator.
                    # gram4: build from X1/Y1/M/P only (halves the build cost)
                    keys = (("X1", "Y1", "M", "P") if basis == "gram4"
                            else self.KEYS + ("P",))
                    Gm = torch.zeros((n, n), dtype=s["P"].dtype, device=s["P"].device)
                    for key in keys:
                        if s.get(key) is not None:
                            Gm = Gm + s[key] @ s[key].T
                    Q = self._top_basis(Gm, rank, eig)
                else:
                    U, _, _ = torch.linalg.svd(s["P"].detach(), full_matrices=False)
                    Q = U[:, :rank].contiguous()
                for key in self.KEYS + ("P",):
                    if s.get(key) is not None:
                        s[key] = Q.T @ s[key]
                s["Q"] = Q
            elif s["Q"].shape[1] > rank:
                # re-truncate thin: Q is no longer orthonormal (it has been
                # transported), so orthonormalize via QR and reduce the small core
                Qt, Rq = torch.linalg.qr(s["Q"].detach())
                if basis in ("gram", "gram4"):
                    keys = (("X1", "Y1", "M", "P") if basis == "gram4"
                            else self.KEYS + ("P",))
                    core = torch.zeros((Rq.shape[0],) * 2, dtype=Rq.dtype, device=Rq.device)
                    for key in keys:
                        if s.get(key) is not None:
                            F = Rq @ s[key]
                            core = core + F @ F.T
                    U = self._top_basis(core, rank, eig)
                else:
                    U, _, _ = torch.linalg.svd((Rq @ s["P"]).detach(), full_matrices=False)
                    U = U[:, :rank]
                T = (U.T @ Rq).contiguous()
                for key in self.KEYS + ("P",):
                    if s.get(key) is not None:
                        s[key] = T @ s[key]
                s["Q"] = Qt @ U

    def dslices(self, subset=None):
        """Returns (D3 vector, D21 matrix with zeroed diagonal): the (3,) and (2,1)
        dslices of the represented (symmetrized) tensor. D21[a,c] = T_aac.
        subset: optional source list to restrict to (for truncated contractions)."""
        n = None
        D3 = None
        D21 = None
        for s in (self.sources if subset is None else subset):
            M, P = s["M"], s["P"]
            Q = s.get("Q")
            pairs = [(s.get(xk), s.get(yk)) for xk, yk in self.PAIRS]
            if n is None:
                ref = Q if Q is not None else next(
                    m for m in [M] + [p[0] for p in pairs] if m is not None)
                n = ref.shape[0]
                D3 = torch.zeros(n, dtype=ref.dtype, device=ref.device)
                D21 = torch.zeros((n, n), dtype=ref.dtype, device=ref.device)
            if Q is not None:
                # factored source: legs are R x n factors, dense leg = Q @ factor.
                # Same formulas as the dense P-branch below, with every trailing
                # n^3 matmul re-associated through the factor: A @ leg^T =
                # (A @ factor^T) @ Q^T -> all n^2 R. Every sandwich shares the
                # SAME final @ Q^T, so it is accumulated once per source.
                Pm = Q @ P
                Mm = Q @ M if M is not None else None
                acc_P = 2.0 * (Mm * Pm) if M is not None else None
                inner = None  # (n, R) accumulator, deferred @ Q^T
                for (Fx, Fy), _ in zip(pairs, self.PAIRS):
                    if Fx is None:
                        continue
                    Xm = Q @ Fx
                    Ym = Q @ Fy
                    t = (Xm * Pm) @ Fy.T + (Pm * Ym) @ Fx.T
                    inner = t if inner is None else inner + t
                    xy = Xm * Ym
                    acc_P = xy if acc_P is None else acc_P + xy
                    D3 += (Xm * Pm * Ym).sum(dim=1)
                if M is not None:
                    t = (Pm * Pm) @ M.T
                    inner = t if inner is None else inner + t
                    D3 += (Mm * Pm * Pm).sum(dim=1)
                if acc_P is not None:
                    t = acc_P @ P.T
                    inner = t if inner is None else inner + t
                if inner is not None:
                    D21 += (1.0 / 3.0) * (inner @ Q.T)
                continue
            if P is None:  # birth: P = I, everything collapses to n^2 ops
                for Xb, Yb in pairs:
                    if Xb is None:
                        continue
                    xd, yd = Xb.diagonal(), Yb.diagonal()
                    D21 += (1.0 / 3.0) * (xd[:, None] * Yb.T + (Xb * Yb) + yd[:, None] * Xb.T)
                    D3 += xd * yd
                if M is not None:
                    md = M.diagonal()
                    D21 += (1.0 / 3.0) * (2.0 * torch.diag(md) + M.T)
                    D3 += md
            else:
                acc_P = 2.0 * (M * P) if M is not None else None  # right-operand-P^T merge
                for Xb, Yb in pairs:
                    if Xb is None:
                        continue
                    D21 += (1.0 / 3.0) * ((Xb * P) @ Yb.T + (P * Yb) @ Xb.T)
                    xy = Xb * Yb
                    acc_P = xy if acc_P is None else acc_P + xy
                    D3 += (Xb * P * Yb).sum(dim=1)
                if M is not None:
                    D21 += (1.0 / 3.0) * ((P * P) @ M.T)
                    D3 += (M * P * P).sum(dim=1)
                if acc_P is not None:
                    D21 += (1.0 / 3.0) * (acc_P @ P.T)
        if D21 is None:
            return None, None
        return D3, zero_diag_(D21)

    def gram(self, n, dtype, device):
        """Accumulated leg Gram Sum_s Sum_legs M M^T (present matrices only; the birth
        P=I isotropic leg is excluded so Q tracks the structured content)."""
        G = torch.zeros((n, n), dtype=dtype, device=device)
        for s in self.sources:
            for key in self.KEYS + ("P",):
                Mx = s.get(key)
                if Mx is not None:
                    G = G + Mx @ Mx.T
        return G

    def project_(self, Q):
        """Confine every tensor leg to span(Q) (Q: n x R orthonormal) -> symmetric
        Tucker-rank-R approximation of the accumulated K[3]. Birth P=I is materialized
        to its projection Q Q^T so future steps treat it generally."""
        QQt = None
        for s in self.sources:
            for key in self.KEYS + ("P",):
                Mx = s.get(key)
                if Mx is not None:
                    s[key] = Q @ (Q.T @ Mx)
            if s["P"] is None:
                if QQt is None:
                    QQt = Q @ Q.T
                s["P"] = QQt

    def confine_old_(self, age_min, rank, eig="exact"):
        """F66 option C on the aug chain: confine every leg of sources with age >= age_min
        (age = number of transports survived) to ONE shared rank-`rank` basis Q built from
        the accumulated Gram of those sources' legs (symmetric Tucker truncation of the OLD
        content only; young sources stay dense). Compounds layer to layer. Sources that
        are Q-factored (fac_schedule) are left alone."""
        old = [s for s in self.sources
               if s["P"] is not None and s.get("age", 0) >= age_min and s.get("Q") is None]
        if not old:
            return
        n = old[0]["P"].shape[0]
        if rank >= n:
            return
        G = torch.zeros((n, n), dtype=old[0]["P"].dtype, device=old[0]["P"].device)
        for s in old:
            for key in self.KEYS + ("P",):
                Mx = s.get(key)
                if Mx is not None:
                    G = G + Mx @ Mx.T
        Q = self._top_basis(G, rank, eig) if CONFINE_Q_FN is None else CONFINE_Q_FN(G, rank)
        for s in old:
            for key in self.KEYS + ("P",):
                if s.get(key) is not None:
                    s[key] = Q @ (Q.T @ s[key])

    def append_hub(self, X1, Y1, X2, Y2, X3=None, Y3=None, X4=None, Y4=None):
        self.sources.append(dict(X1=X1, Y1=Y1, X2=X2, Y2=Y2, X3=X3, Y3=Y3, X4=X4, Y4=Y4,
                                 M=None, P=None, age=0))

    def append_ds(self, M):
        # merge into the birth source of this layer if it exists (shares P = I)
        if self.sources and self.sources[-1]["P"] is None and self.sources[-1]["M"] is None:
            self.sources[-1]["M"] = M
        else:
            self.sources.append(dict(X1=None, Y1=None, X2=None, Y2=None, X3=None,
                                     Y3=None, M=M, P=None, age=0))


def get_wk_slice(vsum, vec, mu, C_offdiag, C_diag, D3, D21, wk4_row, wk4_mat,
                 wk431=None):
    """Diagonal slice of WK[vsum] for a vec_part vector `vec`, expanded for broadcasting
    over the output axes. Returns None if the tower level is absent this layer."""
    d = len(vec)
    if vsum == 2:
        if d == 1:  # vec (2,)
            return C_diag
        # d == 2: (1,1), (2,0), (0,2)
        if vec == (1, 1):
            return C_offdiag
        if vec == (2, 0):
            return C_diag[:, None]
        return C_diag[None, :]  # (0, 2)
    if vsum == 3:
        if D3 is None:
            return None
        if d == 1:  # (3,)
            return D3
        if vec == (2, 1):
            return D21
        if vec == (1, 2):
            return D21.T
        if vec == (3, 0):
            return D3[:, None]
        return D3[None, :]  # (0, 3)
    if vsum == 4:
        if wk4_row is None:
            return None
        if d == 1:  # (4,)
            return wk4_row
        if vec == (2, 2):
            return wk4_mat
        if vec == (3, 1):  # r=1 channel only
            return None if wk431 is None else wk431
        if vec == (1, 3):
            return None if wk431 is None else wk431.T
        if vec == (4, 0):
            return wk4_row[:, None]
        return wk4_row[None, :]  # (0, 4)
    raise AssertionError((vsum, vec))


# ---------------------------------------------------------------------------
# AUGMENT: capped (2,1,1) evaluation + hub-leg contractions
# ---------------------------------------------------------------------------

AXES211 = "abc"  # axis 0 = the 2-block (capped), axes 1/2 = the single legs


def fetch_operand_prod(vsum, vec, arrs):
    """(tensor, idx) for a WK-slice factor of a (2,1,1) term. arrs holds the runtime
    slice arrays; returns None if the tower level is absent this layer."""
    nzax = [i for i, x in enumerate(vec) if x > 0]
    pat = tuple(sorted((x for x in vec if x > 0), reverse=True))
    if vsum == 2:
        if pat == (2,):
            return arrs["C_diag"], AXES211[nzax[0]]
        return arrs["C_off"], AXES211[nzax[0]] + AXES211[nzax[1]]
    if vsum == 3:
        if arrs["D3"] is None:
            return None
        if pat == (3,):
            return arrs["D3"], AXES211[nzax[0]]
        return arrs["D21"], AXES211[vec.index(2)] + AXES211[vec.index(1)]
    if vsum == 4:
        if arrs["wk4_row"] is None:
            return None
        if pat == (4,):
            return arrs["wk4_row"], AXES211[nzax[0]]
        if pat == (2, 2):
            return arrs["wk4_mat"], AXES211[nzax[0]] + AXES211[nzax[1]]
        if pat == (3, 1):
            return arrs["wk431"], AXES211[vec.index(3)] + AXES211[vec.index(1)]
        # (2,1,1): (m/6) * G on the two single-leg axes, constant along the 2-block
        ones_ax = [i for i, x in enumerate(vec) if x == 1]
        return arrs["wk4_211"], AXES211[ones_ax[0]] + AXES211[ones_ax[1]]
    raise AssertionError((vsum, vec))


def fetch_operand_pk(vec, pk):
    """(tensor, idx) for a pK-slice block of a capped (2,1,1) pK->K product."""
    nz = tuple(sorted((x for x in vec if x > 0), reverse=True))
    nzax = [i for i, x in enumerate(vec) if x > 0]
    if nz == (1,):
        return pk[(1,)], AXES211[nzax[0]]
    if nz == (2,):
        return pk[(2,)], AXES211[nzax[0]]
    if nz == (3,):
        return pk[(3,)], AXES211[nzax[0]]
    if nz == (1, 1):
        return pk[(1, 1)], AXES211[nzax[0]] + AXES211[nzax[1]]
    if nz == (2, 1):
        return pk[(2, 1)], AXES211[vec.index(2)] + AXES211[vec.index(1)]
    raise AssertionError(vec)


def eval_capped(operands, wicks, n, dtype):
    """sum over a (a != b, a != c) of prod(operands) * prod(per-axis wick vecs),
    output (b, c). operands: list of (tensor, idx); wicks: {axis_char: vec or None}.
    Never materializes an n^3 tensor (einsum contracts over 'a')."""

    def ein(sub):
        args, subs = [], []
        for t, idx in operands:
            new = "".join(sub.get(ch, ch) for ch in idx)
            if len(new) == 2 and new[0] == new[1]:
                t = torch.diagonal(t)  # e.g. 'ab' under a->b becomes diag, idx 'b'
                new = new[0]
            args.append(t)
            subs.append(new)
        for ax, w in wicks.items():
            if w is None:
                continue
            args.append(w)
            subs.append(sub.get(ax, ax))
        expr = ",".join(subs) + "->bc"
        return torch.einsum(expr, *args)

    full = ein({})
    return full - ein({"a": "b"}) - ein({"a": "c"})


def eval_pk211_capped(table, arrs, wick, n, dtype):
    """Capped (2,1,1) pK production: sum of all generic (2,1,1) terms, each directly
    as its 2-block-axis-capped (n, n) matrix. Symmetrized over (b, c), zero diag."""
    acc = torch.zeros((n, n), dtype=dtype, device=DEV)
    for int_part, factors, coef, wick_kp in table:
        ops = []
        ok = True
        for vsum, vec in factors:
            o = fetch_operand_prod(vsum, vec, arrs)
            if o is None:
                ok = False
                break
            ops.append(o)
        if not ok:
            continue
        wicks = {AXES211[ax]: wick(k, p) for ax, (k, p) in enumerate(wick_kp)}
        acc = acc + coef * eval_capped(ops, wicks, n, dtype)
    return zero_diag_(0.5 * (acc + acc.T))


def eval_k211_capped(pk, pk211_cap, n, dtype):
    """Capped pK -> K conversion for the (2,1,1) slice."""
    acc = torch.zeros((n, n), dtype=dtype, device=DEV)
    for vpart, coef in PK2K211_TABLE:
        if vpart == ((2, 1, 1),):
            acc = acc + coef * pk211_cap
            continue
        ops = [fetch_operand_pk(v, pk) for v in vpart]
        acc = acc + coef * eval_capped(ops, {}, n, dtype)
    return zero_diag_(0.5 * (acc + acc.T))


def _rot_term(cap_c, B, C, s, n, dtype):
    """(S B) diag(cap_c) (S C)^T with S = diag(s) (s=None -> identity scaling) and
    B/C = None meaning the identity matrix (birth P legs)."""
    if B is None and C is None:
        v = cap_c if s is None else s * s * cap_c
        return torch.diag(v)
    if B is None:
        left = cap_c[:, None] if s is None else (s * cap_c)[:, None]
        right = C.T if s is None else C.T * s[None, :]
        return left * right
    if C is None:
        left = B if s is None else s[:, None] * B
        right = cap_c[None, :] if s is None else (cap_c * s)[None, :]
        return left * right
    Bs = B if s is None else s[:, None] * B
    Cs = C if s is None else s[:, None] * C
    return (Bs * cap_c[None, :]) @ Cs.T


def k211_from_sources_naive(sources, u, s, n, dtype, extra_triples=(), rep=None):
    """Original triple-loop hub-leg contraction (kept as parity reference for the
    merged-matmul restructure below)."""
    acc = torch.zeros((n, n), dtype=dtype, device=DEV)

    def cap(Mx):
        return u.clone() if Mx is None else Mx.T @ u

    def add_triple(A, B, C, sign=1.0):
        nonlocal acc
        for X, Y, Z in ((A, B, C), (B, C, A), (C, A, B)):
            acc = acc + sign * _rot_term(cap(X), Y, Z, s, n, dtype)

    for src in sources:
        P = src["P"]
        for xk, yk in (("X1", "Y1"), ("X2", "Y2"), ("X3", "Y3"), ("X4", "Y4")):
            if src.get(xk) is not None:
                add_triple(src[xk], P, src[yk])
        if src["M"] is not None:
            add_triple(src["M"], P, P)
    for A, B, C in extra_triples:
        add_triple(A, B, C)
    if rep is not None:
        rep3, rep21 = rep
        A_rep = torch.diag(rep3) + 3.0 * rep21.T
        add_triple(A_rep, None, None, sign=-1.0)
    return (0.5 / 3.0) * (acc + acc.T)


def k211_from_sources(sources, u, s, n, dtype, extra_triples=(), rep=None,
                      no_mid=False):
    """Hub-leg contraction, MERGED-MATMUL restructure (exact == naive version):
    - the s scaling factors out of every rotation term as diag(s) core diag(s),
      applied once at the end (commutes with the final symmetrization);
    - per source, all rotation terms with P as LEFT operand collapse into one
      matmul P @ R_acc, all right-P terms into L_acc @ P^T; only the middle
      rotations Y_k diag(P^T u) X_k^T stay one matmul per hub pair.
    Full source: 5 matmuls instead of 12. Birth sources (P = I): unchanged 3.
    u: cap weight vector; s: elementwise weight on uncapped legs (None = 1).
    rep: (rep3, rep21) repeated-content correction, subtracted via the factored
    form (-A_rep, I, I) with A_rep = diag(rep3) + 3*rep21^T.
    Returns Sym(sum of rotations) / 3 (caller applies the projection coef)."""
    acc = torch.zeros((n, n), dtype=dtype, device=DEV)

    for src in sources:
        P = src["P"]
        M = src["M"]
        Q = src.get("Q")
        if Q is not None:
            # factored source: every rotation term is Q (R x R core) Q^T; caps go
            # through qu = Q^T u. Sum the cores, reconstruct ONCE per source.
            qu = Q.T @ u
            cP = P.T @ qu
            G = torch.zeros((P.shape[0], P.shape[0]), dtype=dtype, device=DEV)
            Rf = None  # (n, R): sum of diag(c) factor^T pieces left-multiplied by P
            Lf = None  # (R, n): sum of factor diag(c) pieces right-multiplied by P^T
            for xk, yk in (("X1", "Y1"), ("X2", "Y2"), ("X3", "Y3"), ("X4", "Y4")):
                Fx, Fy = src.get(xk), src.get(yk)
                if Fx is None:
                    continue
                cX = Fx.T @ qu
                cY = Fy.T @ qu
                r = cX[:, None] * Fy.T
                Rf = r if Rf is None else Rf + r
                lt = Fx * cY[None, :]
                Lf = lt if Lf is None else Lf + lt
                G = G + (Fy * cP[None, :]) @ Fx.T  # middle rotation core
            if M is not None:
                cM = M.T @ qu
                r = cM[:, None] * P.T + cP[:, None] * M.T
                Rf = r if Rf is None else Rf + r
                lt = M * cP[None, :]
                Lf = lt if Lf is None else Lf + lt
            if Rf is not None:
                G = G + P @ Rf
            if Lf is not None:
                G = G + Lf @ P.T
            acc = acc + Q @ (G @ Q.T)
            continue
        if P is not None:
            cP = P.T @ u
            R_acc = None  # sum of diag(c) Z^T factors multiplied by P from the left
            L_acc = None  # sum of Z diag(c) factors multiplied by P^T from the right
            for xk, yk in (("X1", "Y1"), ("X2", "Y2"), ("X3", "Y3"), ("X4", "Y4")):
                X, Y = src.get(xk), src.get(yk)
                if X is None:
                    continue
                cX = X.T @ u
                cY = Y.T @ u
                r = cX[:, None] * Y.T
                R_acc = r if R_acc is None else R_acc + r
                lt = X * cY[None, :]
                L_acc = lt if L_acc is None else L_acc + lt
                if not no_mid:
                    acc = acc + (Y * cP[None, :]) @ X.T  # middle rotation
            if M is not None:
                cM = M.T @ u
                r = cM[:, None] * P.T + cP[:, None] * M.T
                R_acc = r if R_acc is None else R_acc + r
                lt = M * cP[None, :]
                L_acc = lt if L_acc is None else L_acc + lt
            # SYM-MERGE (exact): only Sym(acc) is used downstream, so the
            # right-P term L_acc @ P^T may be replaced by its transpose
            # P @ L_acc^T and merged with P @ R_acc -> ONE matmul per source.
            if R_acc is not None or L_acc is not None:
                Rt = R_acc if R_acc is not None else torch.zeros_like(P)
                if L_acc is not None:
                    Rt = Rt + L_acc.T
                acc = acc + P @ Rt
        else:
            # birth source: P = I; rot1/rot3 are n^2, rot2 is one matmul per pair
            for xk, yk in (("X1", "Y1"), ("X2", "Y2"), ("X3", "Y3"), ("X4", "Y4")):
                X, Y = src.get(xk), src.get(yk)
                if X is None:
                    continue
                cX = X.T @ u
                cY = Y.T @ u
                acc = acc + cX[:, None] * Y.T
                acc = acc + (Y * u[None, :]) @ X.T
                acc = acc + X * cY[None, :]
            if M is not None:
                cM = M.T @ u
                acc = acc + torch.diag(cM) + u[:, None] * M.T + M * u[None, :]

    def add_triple_small(A, B, C, sign=1.0):
        # extra_triples / rep entries always carry a None operand -> all n^2
        nonlocal acc
        for X, Y, Z in ((A, B, C), (B, C, A), (C, A, B)):
            cx = u if X is None else X.T @ u
            if Y is None and Z is None:
                acc = acc + sign * torch.diag(cx)
            elif Y is None:
                acc = acc + sign * (cx[:, None] * Z.T)
            elif Z is None:
                acc = acc + sign * (Y * cx[None, :])
            else:
                acc = acc + sign * ((Y * cx[None, :]) @ Z.T)

    for A, B, C in extra_triples:
        add_triple_small(A, B, C)
    if rep is not None:
        rep3, rep21 = rep
        A_rep = torch.diag(rep3) + 3.0 * rep21.T
        add_triple_small(A_rep, None, None, sign=-1.0)
    out = (0.5 / 3.0) * (acc + acc.T)
    if s is not None:
        out = out * (s[:, None] * s[None, :])
    return out


CORR_FEATURES = ("1", "mu", "var", "sigma", "alpha", "|alpha|", "phi", "Phi",
                 "pred", "D3", "D21n", "K4", "sigma*phi")
N_CORR_FEAT = len(CORR_FEATURES)


def build_corr_feats(mu_pre, var_pre, pred, D3, D21n, K4, dtype):
    """Per-neuron feature matrix (n x F) for the online mean correction. All quantities
    are already computed in the K3 loop; the correction delta = feats @ beta is
    flopscope-cheap (one dot product per neuron per layer)."""
    n = mu_pre.shape[0]
    sigma = torch.sqrt(torch.clamp(var_pre, min=1e-10))
    alpha = mu_pre / sigma
    phi = torch.exp(-0.5 * alpha * alpha) / SQRT_2PI
    Phi = norm_cdf(alpha)
    ones = torch.ones(n, dtype=dtype, device=DEV)
    k4v = torch.full((n,), float(K4), dtype=dtype, device=DEV) if not torch.is_tensor(K4) else K4
    cols = [ones, mu_pre, var_pre, sigma, alpha, alpha.abs(), phi, Phi,
            pred, D3, D21n, k4v, sigma * phi]
    return torch.stack(cols, dim=1)  # (n, F)


def _rank_at(spec, layer):
    """leg_trunc rank spec: int, or list indexed by birth layer (last entry
    repeats for later layers)."""
    if isinstance(spec, (list, tuple)):
        return int(spec[min(layer, len(spec) - 1)])
    return int(spec)


REGEN_STATE = {}


def lean_k3_predict(weights: np.ndarray, dtype=torch.float64, statics=None,
                    shadow_B2=False, shadow_B1=False, no_bprime=False,
                    no_k3_feedback=False, no_wk22_fac=False, return_diag=False,
                    force_mean=None, svd_d21=False, d21_rank=None,
                    tucker_rank=None, mean_corr=None, collect_feats=False,
                    no_m_chain=False, no_k4=False, k4_vec=False, k4_prop=None,
                    k4_aug=False, debug_state=None, aug_no_a=False, aug_no_b=False,
                    aug_no_feed=False, aug_no_31=False, aug_no_211=False,
                    aug_recent_k=None, fac_schedule=None, fac_basis="P",
                    fac_eig="exact", t211_drop=None, leg_trunc=None, spectra=None,
                    aug_no_mid=False, aug_no_k211=False, g_hook=None,
                    g_lin_diag=False, g_use_hook=None, g_feed_hook=None,
                    age_rank=None, age_eig="exact"):
    """weights: (L, n, n) in the x@w convention (columns = output neurons).
    aug_no_k211: drop the ENTIRE capped (2,1,1) kappa4 channel (pk211 production,
      hub contractions a/b, birth part, and the K211_cap term of the G projection):
      G is then built from the n^2 slices K(4,), K(2,2), K(3,1) only.
    g_hook: callable(layer_index, G, live_dict) -> G, applied right after the
      kappa4 matrix core G is produced (memoryless regeneration probes).

    leg_trunc (F63 lens on the aug legs): dict with optional ranks
      d21  - rank of the D21 used inside the hub legs (X1,Y1,X2,Y2); off-diag only
             (the diagonal is a P-column-scaled leg, exact for free)
      g    - rank of the off-diagonal part of G inside X3 (diag(G) kept exact)
      rres - rank of Rres = S21 - S_sep inside the M leg (S_sep rides on the A leg)
    Truncation happens at BIRTH only; transport of a rank-r matrix is exact, so this
    is the faithful compounding test (F46-UPDATE lesson), not an output proxy.
    spectra: optional list; per layer a dict of singular values gets appended.

    Ablation flags (DS-shadow-at-birth: a dropped sub-block's dslices are NOT
    subtracted from K_ds3, so its diagonal content migrates into the exact M-chain
    and only its fully-off-diagonal content is lost):
      shadow_B2: drop the B2 hub sub-block (X2, Y2)
      shadow_B1: drop the B1 hub sub-block (X1, Y1)
      no_bprime: drop the WK21 (b-bar-prime) term inside Y1
    """
    L, n = weights.shape[0], weights.shape[1]
    if statics is None:
        statics = build_static_coefs(n)
    table211 = (TERM_TABLE_211 if t211_drop is None else
                [rec for i, rec in enumerate(TERM_TABLE_211) if i not in t211_drop])
    Ws = [torch.from_numpy(w.astype(np.float64).T).contiguous().to(dtype).to(DEV) for w in weights]
    metric_c = 2.0  # E[W W^T] = w_var * I (He var 2/n summed over n inputs)

    mu = torch.zeros(n, dtype=dtype, device=DEV)
    C = torch.eye(n, dtype=dtype, device=DEV)
    K3 = None          # K3Blocks after first nonlin
    K4_sigma = None    # scalar core of the r=2 radial K4
    K4_vecstate = None  # augmented-lite v0: per-neuron diagonal kappa4 content
    K4_matstate = None  # augmented-lite v1: full (2,2)-slice kappa4 matrix
    G = None           # AUGMENT: (n, n) matrix core of the r=1 K4 = Sym(G x metric)
    rows = []
    feats_collect = []   # (L, n, F) online-correction features, if collect_feats
    predpre_collect = [] # (L, n) post-nonlin mean BEFORE the correction delta
    diag_mu_in = []    # pre-nonlin per-neuron mean (mu) each layer
    diag_var_in = []   # pre-nonlin per-neuron variance (diag C) each layer
    diag_d3 = []       # K3 3rd-dslice diagonal (n,) each layer (non-Gaussianity signal)
    diag_d21n = []     # per-neuron abs-row-sum of the (2,1) dslice each layer
    diag_k4 = []       # incoming K4 scalar core each layer
    diag_svd = []      # singular values of the (2,1) dslice matrix each layer

    _dead_prev = None
    for l in range(L):
        if HYB_K is not None and l == HYB_K:   # стенд 23.09: від шару HYB_K ланцюг стає dense
            _hs = HYB_SET or {"k211", "B2", "wk22", "31", "trunc", "hooks"}
            if "k211" in _hs: aug_no_k211 = False
            if "B2" in _hs: shadow_B2 = False
            if "wk22" in _hs: no_wk22_fac = False
            if "31" in _hs: aug_no_31 = False
            if "trunc" in _hs: leg_trunc = None
            if "hooks" in _hs: g_hook = None; g_use_hook = None; g_feed_hook = None
        W = Ws[l]
        # ---- linear step ----
        Wk = W   # стенд 14.10: Wk = W без стовпців насичених нейронів попереднього шару (лише дорогі частини)
        if DEAD_TH is not None and _dead_prev is not None:
            Wk = W * (~_dead_prev).to(W.dtype)[None, :]
            DEAD_STATS.append((l, float(_dead_prev.float().mean())))
        mu = W @ mu
        if DEAD_C and Wk is not W:
            _vd = torch.diagonal(C) * _dead_prev.to(C.dtype)
            C = Wk @ C @ Wk.T + torch.diag((W * W) @ _vd)
        else:
            C = W @ C @ W.T
        C = 0.5 * (C + C.T)
        if K3 is not None:
            K3.linear_(Wk if 'k3' in DEAD_PARTS else W)
            if D4 is not None:
                D4.transport(Wk if 'd4' in DEAD_PARTS else W)
        # K4: scalar core unchanged; metric set to metric_c (avg metric)
        if G is not None:
            if g_lin_diag and REGEN_STATE.get("lam") is not None:
                # regenerated G = diag(g) + lam*C_off_post transports to
                # lam*C_pre_off + W diag(g - lam*var_post) W^T; keep only the
                # DIAGONAL of the second term ((W*W) action, n^2) -> G_pre =
                # lam*C_pre_off + diag((W*W) g). Probe of the free-port approximation.
                G = REGEN_STATE["lam"] * zero_diag_(C.clone())                     + torch.diag((W * W) @ REGEN_STATE["g"])
            else:
                _Wg = Wk if 'g' in DEAD_PARTS else W
                G = _Wg @ G @ _Wg.T
                G = 0.5 * (G + G.T)

        # ---- WK slices ----
        var = torch.clamp(torch.diagonal(C), min=1e-10)
        mu_pre = mu.clone()          # pre-nonlin mean (feature for online correction)
        var_pre = var                # pre-nonlin variance
        T_CUR['t'] = mu / var.sqrt()  # стенд 14.10: t поточного шару для HUBDEAD
        if DEAD_TH is not None:
            _tt = mu / var.sqrt()
            _dead_prev = _tt < -DEAD_TH
            if DEAD_LIN is not None:
                _dead_prev = _dead_prev | (_tt > DEAD_LIN)
        k4_in = K4_sigma if K4_sigma is not None else 0.0
        if return_diag:
            diag_mu_in.append(mu.clone())
            diag_var_in.append(var.clone())
        C_offdiag = zero_diag_(C.clone())
        if K3 is not None:
            D3, D21 = K3.dslices()
            if K3_PROBE is not None:
                K3_PROBE(l, K3, D3, D21)
            if D4 is not None:
                D4_SEAL["D21_cp"] = D21.clone(); D4_SEAL["D3_cp"] = D3.clone()
                D3, D21 = D4.add_dslices(D3, D21)
            if d21_rank is not None and d21_rank < n:
                U, S, Vh = _svd(D21)
                D21 = zero_diag_(U[:, :d21_rank] @ (U[:, :d21_rank].T @ D21))
        else:
            D3, D21 = None, None
        D3u, D21u = D3, D21
        D21_HOOK_FEED = None
        if D21_HOOK is not None and D21 is not None:
            D3u, D21u, D21_HOOK_FEED = D21_HOOK(l, D3, D21, dict(mu=mu, C=C, var=var))
        if return_diag:
            diag_d3.append(D3.clone() if D3 is not None else torch.zeros(n, dtype=dtype, device=DEV))
            diag_d21n.append(D21.abs().sum(dim=1) if D21 is not None
                             else torch.zeros(n, dtype=dtype, device=DEV))
            diag_k4.append(float(K4_sigma) if K4_sigma is not None else 0.0)
            if svd_d21:
                diag_svd.append(torch.linalg.svdvals(D21).clone() if D21 is not None
                                else torch.zeros(n, dtype=dtype, device=DEV))
        if C_HOOK is not None:
            C_offdiag, var = C_HOOK(l, mu, C_offdiag, var)
        wk431, wk4_211 = None, None
        if k4_aug:
            if G is not None:
                # dslices of Sym(G x metric_c*I)  [verify_aug_formulas.py piece 1]
                G_use = G if g_use_hook is None else g_use_hook(G, C_offdiag)
                dG = torch.diagonal(G_use)
                wk4_row = metric_c * dG
                wk4_mat = zero_diag_((metric_c / 6.0) * (dG[:, None] + dG[None, :]))
                wk431 = zero_diag_(0.5 * metric_c * G_use.clone())
                wk4_211 = (metric_c / 6.0) * G_use
            else:
                wk4_row, wk4_mat = None, None
        elif K4_sigma is not None and not no_k4:
            if k4_prop == "mat":
                # v1: exact (W*W)-sandwich transport of the full (2,2) kappa4 matrix
                WW = W * W
                T = WW @ K4_matstate @ WW.T / metric_c ** 2
                wk4_row = statics["wk4_c4"] * metric_c ** 2 * torch.diagonal(T)
                wk4_mat = zero_diag_(statics["wk4_c22"] * metric_c ** 2 * T.clone())
            elif k4_vec:
                gv = K4_vecstate
                if k4_prop == "ww":
                    # exact one-layer propagation of the diagonal kappa4 content,
                    # replacing one avg-metric cup by the true (W*W) row action
                    gv = (W * W) @ gv / metric_c
                wk4_row = statics["wk4_c4"] * metric_c ** 2 * gv
                wk4_mat = zero_diag_(statics["wk4_c22"] * metric_c ** 2
                                     * 0.5 * (gv[:, None] + gv[None, :]))
            else:
                g4 = statics["wk4_c4"] * K4_sigma * metric_c ** 2
                g22 = statics["wk4_c22"] * K4_sigma * metric_c ** 2
                wk4_row = torch.full((n,), g4, dtype=dtype, device=DEV)
                wk4_mat = zero_diag_(torch.full((n, n), g22, dtype=dtype, device=DEV))
        else:
            wk4_row, wk4_mat = None, None

        if WK_HOOK is not None:
            wk4_row, wk4_mat, wk431 = WK_HOOK(l, wk4_row, wk4_mat, wk431,
                                              dict(mu=mu, C=C, var=var, D3=D3, D21=D21, G=G))
        # ---- nonlin: pK slices from the term table ----
        wick = WickCache(mu, var)
        pk = {}
        for int_part, factors, coef, wick_kp in (TERM_TABLE_AUG if k4_aug
                                                 else TERM_TABLE):
            d = len(int_part)
            shape = (n,) * d
            prod = None
            ok = True
            for vsum, vec in factors:
                f = get_wk_slice(vsum, vec, mu, C_offdiag, var, D3u, D21u, wk4_row,
                                 wk4_mat, wk431)
                if f is None:
                    ok = False
                    break
                prod = f if prod is None else prod * f
            if not ok:
                continue
            if prod is None:  # empty vec_part -> all-ones
                prod = torch.ones(shape, dtype=dtype, device=DEV)
            elif prod.shape != shape:
                prod = prod.expand(shape).clone() if d > 1 else prod
            term = coef * prod
            if TERM_SEEN is not None:
                TERM_SEEN.add(term_group(int_part, factors))
            if TERM_SCALE is not None:
                term = term * TERM_SCALE.get(term_group(int_part, factors), 1.0)
            for ax, (k, p) in enumerate(wick_kp):
                wv = wick(k, p)
                view = [1] * d
                view[ax] = -1
                term = term * wv.reshape(view)
            key = tuple(int_part)
            pk[key] = term if key not in pk else pk[key] + term
        # symmetrize + zero repeated
        for key in [(1, 1), (2, 2)]:
            if key in pk:
                pk[key] = 0.5 * (pk[key] + pk[key].T)
        for key in [(1, 1), (2, 1), (2, 2), (3, 1)]:
            if key in pk:
                zero_diag_(pk[key])

        # ---- AUGMENT: capped (2,1,1) pK production + pK211->K211 hub contraction ----
        # F52 combined contraction: every rotation term is LINEAR in the cap
        # vector u, and post-wick legs = diag(w1) @ pre-wick legs, so
        #   cA*contrib_a - 2cA*contrib_b(old sources)
        #     = cA * diag(w1) core(u_a - 2*w1*u_b) diag(w1)
        # -> ONE heavy contraction over the old sources per layer instead of two.
        # The birth source (exists only in contrib_b) and the n^2 rep/extra
        # corrections are handled separately below.
        pk211_cap, contrib_a = None, None
        contrib_comb, rep_a_term = None, None
        if aug_no_k211:
            aug_no_a = aug_no_b = aug_no_211 = True
        fast_comb = (k4_aug and not aug_no_a and not aug_no_b
                     and aug_recent_k is None)
        if k4_aug:
            arrs211 = dict(C_diag=var, C_off=C_offdiag, D3=D3u, D21=D21u,
                           wk4_row=wk4_row, wk4_mat=wk4_mat, wk431=wk431,
                           wk4_211=wk4_211)
            pk211_cap = (torch.zeros((n, n), dtype=dtype, device=DEV) if aug_no_211 else
                         eval_pk211_capped(table211, arrs211, wick, n, dtype))
            if fast_comb and K3 is not None and K3.sources:
                w1v = wick(1)
                u_eff = wick(1, 2) - 2.0 * (w1v * pk[(1,)])
                contrib_comb = k211_from_sources(K3.sources, u=u_eff, s=w1v,
                                                 n=n, dtype=dtype,
                                                 no_mid=aug_no_mid)
                if D4 is not None and D4.D is not None:
                    contrib_comb = contrib_comb + D4.cap(u_eff, w1v).to(dtype)
                rep_a_term = k211_from_sources([], u=wick(1, 2), s=w1v, n=n,
                                               dtype=dtype, rep=(D3, D21))
            elif not fast_comb and not aug_no_a and K3 is not None and K3.sources:
                # the ((2,1,1), {(1,1,1)}) term: WK3 legs capped with E[d relu^2],
                # uncapped legs weighted by E[d relu] (pre-wick sources)
                if aug_recent_k is None:
                    srcs, rep_a = K3.sources, (D3, D21)
                else:
                    srcs = K3.sources[-aug_recent_k:]
                    rd3, rd21 = K3.dslices(subset=srcs)
                    rep_a = (rd3, rd21)
                contrib_a = k211_from_sources(srcs, u=wick(1, 2), s=wick(1),
                                              n=n, dtype=dtype, rep=rep_a)

        # ---- manual (1,1,1) factored part: wick old blocks, then append new hubs ----
        w1, w2, w3, w4 = wick(1), wick(2), wick(3), wick(4)
        if D4 is not None:
            D4.densify(K3.sources if K3 is not None else [], D21_ref=D21)
            if l <= 2 and K3 is not None and K3.sources:
                sd3, sd21 = D4.seal_d21()
                print("  d4 пломба l=%d: D21 Sym(E) vs dslices віднос. %.2e, D3 %.2e" % (
                    l, float((sd21 - D4_SEAL["D21_cp"].float()).norm() / D4_SEAL["D21_cp"].float().norm()),
                    float((sd3 - D4_SEAL["D3_cp"].float()).norm() / D4_SEAL["D3_cp"].float().norm())), flush=True)
            D4.birth(dict(var=var, C=C_offdiag,
                          D3=D3 if D3 is not None else torch.zeros(n, dtype=dtype, device=DEV),
                          D21=D21 if D21 is not None else torch.zeros((n, n), dtype=dtype, device=DEV)), wick, w1,
                     pk1=pk[(1,)] if k4_aug else None,
                     T3=get_all_terms_iso(k_max=3), T4=get_all_terms_iso(k_max=4))
        if K3 is not None and K3_INJ is not None:
            K3_INJ(l, K3, D3, D21)
        if K3 is not None:
            K3.wick_(w1)
            # wick-scale the cached dslices of the old content (for get_repeated)
            D3_w = D3 * w1 ** 3
            D21_w = D21 * (w1 ** 2)[:, None] * w1[None, :]
        else:
            K3 = K3Blocks()
            D3_w = torch.zeros(n, dtype=dtype, device=DEV)
            D21_w = torch.zeros((n, n), dtype=dtype, device=DEV)

        WK11 = C_offdiag
        spec = {} if spectra is not None else None
        D21_leg = D21
        if D21 is not None and (spec is not None or (leg_trunc and leg_trunc.get("d21"))):
            U, S, Vh = _svd(D21)
            if spec is not None:
                spec["d21"] = S.clone()
            if leg_trunc and leg_trunc.get("d21"):
                r_ = _rank_at(leg_trunc["d21"], l)
                D21_leg = zero_diag_(U[:, :r_] @ (U[:, :r_].T @ D21))
        if D21_HOOK_FEED is not None:
            D21_leg = D21_HOOK_FEED
        if D21 is None or no_k3_feedback:
            WK21 = torch.zeros((n, n), dtype=dtype, device=DEV)
        else:
            WK21 = 0.5 * D21_leg
        WK12 = WK21.T
        if wk4_mat is not None and not no_wk22_fac:
            WK22 = 0.25 * wk4_mat
        else:
            WK22 = torch.zeros((n, n), dtype=dtype, device=DEV)

        c_w1, r_w1 = w1[:, None], w1[None, :]
        c_w2, r_w2 = w2[:, None], w2[None, :]
        c_w3 = w3[:, None]
        c_w4 = w4[:, None]
        # B1 ("left leg has 1 j idx"); the 3 from fac2 = 3I folded into X1
        if shadow_B1:
            X1, Y1 = None, None
        else:
            X1 = 3.0 * (c_w1 * WK11 + c_w2 * WK21)
            Y1 = (c_w2 * r_w1 * WK11
                  + (0.0 if no_bprime else 1.0) * c_w3 * r_w1 * WK21
                  + c_w2 * r_w2 * WK12 + c_w3 * r_w2 * WK22).T
        # B2 ("left leg has 2 j idxs")
        if shadow_B2:
            X2, Y2 = None, None
        else:
            X2 = 3.0 * (c_w1 * WK12 + c_w2 * WK22)
            Y2 = (c_w3 * r_w1 * WK11 + c_w4 * r_w1 * WK21 + c_w3 * r_w2 * WK12 + c_w4 * r_w2 * WK22).T
        # AUGMENT: K4 (r=1, matrix core) -> pK(1,1,1) hub feeds. Reference factor_k3
        # augment terms 1 & 3 (term 2 vanishes for the diagonal avg metric):
        #   term 1 (2-block on the core diagonal): M-type source with
        #       M_t1 = (metric_c/4) (w2 . diag(G)) x (w1^2)^T          (rank-1)
        #   term 3 (2-block on the metric diagonal): hub pair with
        #       X3 = diag(w1) G diag(w1),  Y3 = (metric_c/4) w2 x 1^T  (rank-1)
        X3, Y3, M_t1 = None, None, None
        if k4_aug and G is not None and not aug_no_feed:
            G_feed = G if g_feed_hook is None else g_feed_hook(G, C_offdiag)
            dG_in = torch.diagonal(G_feed)
            M_t1 = (metric_c / 4.0) * (w2 * dG_in)[:, None] * (w1 * w1)[None, :]
            G_leg = G_feed
            if spec is not None or (leg_trunc and leg_trunc.get("g") is not None):
                G_off = zero_diag_(G.clone())
                ev, evec = torch.linalg.eigh(G_off.detach())
                order = torch.argsort(ev.abs(), descending=True)
                if spec is not None:
                    spec["g_off"] = ev[order].clone()
                if leg_trunc and leg_trunc.get("g") is not None:
                    r_ = _rank_at(leg_trunc["g"], l)
                    keep = order[:r_]
                    G_trunc = evec[:, keep] @ (evec[:, keep].T @ G_off) if r_ > 0                         else torch.zeros((n, n), dtype=dtype, device=DEV)
                    G_leg = torch.diag(dG_in) + zero_diag_(G_trunc)
            X3 = (w1[:, None] * G_leg) * w1[None, :]
            Y3 = (metric_c / 4.0) * w2[:, None] * torch.ones((1, n), dtype=dtype, device=DEV)
        X4, Y4 = None, None
        if PATH4:   # стенд 23.09: діаграма k_max=4 ((0,1,1),(0,1,1),(1,0,1)) = шлях з центром j: 3·w1(a)C_aj · w2(b)C_bj²·w3(j)
            X4 = PATH4 * 3.0 * (c_w1 * WK11)
            Y4 = (w2[:, None] * (WK11 * WK11)) * w3[None, :]
        if l not in SKIP_BIRTH:   # стенд 25.09: пропуск народження κ₃ на ранніх шарах
            if BIRTH_SCALE != 1.0 and l >= BIRTH_SCALE_LMIN:   # стенд 26.09: масштаб хаба народження (усі пари)
                X1 = None if X1 is None else X1 * BIRTH_SCALE
                X2 = None if X2 is None else X2 * BIRTH_SCALE
                X3 = None if X3 is None else X3 * BIRTH_SCALE
                X4 = None if X4 is None else X4 * BIRTH_SCALE
            if HUB_MASK is not None:   # стенд 12.10: маска стовпців хабу (індекс j) усіх X-ніг і M_t1
                _hm = HUB_MASK(l)
                if _hm is not None:
                    _hm = _hm.to(dtype)[None, :]
                    X1 = None if X1 is None else X1 * _hm
                    X2 = None if X2 is None else X2 * _hm
                    X3 = None if X3 is None else X3 * _hm
                    X4 = None if X4 is None else X4 * _hm
                    M_t1 = None if M_t1 is None else M_t1 * _hm
            K3.append_hub(X1, Y1, X2, Y2, X3, Y3, X4, Y4)

        # birth-block dslices of the KEPT hub sub-blocks (P = I): n^2 closed forms
        D21_new = torch.zeros((n, n), dtype=dtype, device=DEV)
        D3_new = torch.zeros(n, dtype=dtype, device=DEV)
        for Xb, Yb in ((X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)):
            if Xb is None:
                continue
            xd, yd = Xb.diagonal(), Yb.diagonal()
            D21_new += (1.0 / 3.0) * (xd[:, None] * Yb.T + (Xb * Yb) + yd[:, None] * Xb.T)
            D3_new += xd * yd
        if M_t1 is not None:
            md = M_t1.diagonal()
            D21_new += (1.0 / 3.0) * (2.0 * torch.diag(md) + M_t1.T)
            D3_new += md
        rep3 = D3_w + D3_new
        rep21 = zero_diag_(D21_w + D21_new)

        # ---- AUGMENT: pK111 -> K211 hub contraction (post-wick sources + M_t1) ----
        contrib_b, birth_part = None, None
        if fast_comb:
            # light remainder of contrib_b: this layer's birth source + n^2 terms
            extra = [(M_t1, None, None)] if M_t1 is not None else []
            birth_srcs = ([K3.sources[-1]]
                          if K3.sources and K3.sources[-1]["P"] is None else [])
            birth_part = k211_from_sources(birth_srcs, u=pk[(1,)], s=None, n=n,
                                           dtype=dtype, extra_triples=extra,
                                           rep=(rep3, rep21))
            if D4 is not None and D4.birth_cap is not None:
                birth_part = birth_part + D4.birth_cap.to(dtype)
        elif k4_aug and not aug_no_b:
            extra = [(M_t1, None, None)] if M_t1 is not None else []
            if aug_recent_k is None:
                srcs, rep_b = K3.sources, (rep3, rep21)
            else:
                srcs = K3.sources[-aug_recent_k:]
                rd3, rd21 = K3.dslices(subset=srcs)
                if M_t1 is not None:
                    md = M_t1.diagonal()
                    rd21 = zero_diag_(rd21 + (1.0 / 3.0)
                                      * (2.0 * torch.diag(md) + M_t1.T))
                    rd3 = rd3 + md
                rep_b = (rd3, rd21)
            contrib_b = k211_from_sources(srcs, u=pk[(1,)], s=None, n=n,
                                          dtype=dtype, extra_triples=extra,
                                          rep=rep_b)

        # ---- pK -> K on slices ----
        def pk_slice(vec):
            nz = tuple(sorted((x for x in vec if x > 0), reverse=True))
            base = pk[nz]
            d_in = len(nz)
            if len(vec) == 1:
                return base
            # len(vec) == 2 placements
            if d_in == 2:
                srt = tuple(sorted(vec, reverse=True))
                return base if vec == srt else base.T
            # d_in == 1: broadcast to the nonzero axis
            return base[:, None] if vec[0] > 0 else base[None, :]

        K = {}
        for out_part, entries in (PK2K_TABLE_AUG if k4_aug else PK2K_TABLE).items():
            d = len(out_part)
            acc = torch.zeros((n,) * d, dtype=dtype, device=DEV)
            for vpart, coef in entries:
                prod = None
                for blk in vpart:
                    f = pk_slice(blk)
                    prod = f if prod is None else prod * f
                acc = acc + coef * prod
            K[out_part] = acc
        for key in [(1, 1), (2, 1), (2, 2)] + ([(3, 1)] if k4_aug else []):
            zero_diag_(K[key])
        K211_cap = (eval_k211_capped(pk, pk211_cap, n, dtype) if k4_aug else None)

        # ---- assemble outputs ----
        mu = K[(1,)]
        C = torch.diag(K[(2,)]) + 0.5 * (K[(1, 1)] + K[(1, 1)].T)
        if K_POST_HOOK is not None:
            K_POST_HOOK(l, K, mu, C)
        S3 = K[(3,)] - rep3
        S21 = zero_diag_(K[(2, 1)] - rep21)
        if spec is not None or (leg_trunc and leg_trunc.get("rres")):
            e_b = wick(1, 2) - 2.0 * pk[(1,)] * w1
            S_sep = zero_diag_(e_b[:, None] * C_offdiag * w1[None, :])
            Rres = S21 - S_sep
            U, S, Vh = _svd(Rres)
            if spec is not None:
                spec["rres"] = S.clone()
                spec["s21_norm"] = float(S21.norm())
                spec["rres_norm"] = float(Rres.norm())
                spec["ssep_norm"] = float(S_sep.norm())
            if leg_trunc and leg_trunc.get("rres"):
                r_ = _rank_at(leg_trunc["rres"], l)
                S21 = S_sep + zero_diag_(U[:, :r_] @ (U[:, :r_].T @ Rres))
        M = torch.diag(S3) + 3.0 * S21.T
        if spec is not None:
            spectra.append(spec)
        M_total = M if not no_m_chain else None
        if M_t1 is not None:
            M_total = M_t1 if M_total is None else M_total + M_t1
        if M_total is not None and HUB_MASK is not None and l not in SKIP_BIRTH:   # стенд 12.10
            _hm = HUB_MASK(l)
            if _hm is not None:
                M_total = M_total * _hm.to(dtype)[None, :]
        if M_total is not None and l not in SKIP_BIRTH:
            K3.append_ds(M_total)
        if K3DAMP != 0.0:   # стенд 15.10 (юзер): експоненційне гасіння джерел κ3, кожне × exp(−λ) за шар віку
            _f = math.exp(-K3DAMP)
            for _s in K3.sources:
                if _s.get("age", 0) >= K3DAMP_AGE:
                    for _k in ("X1", "X2", "X3", "X4", "M"):
                        if _s.get(_k) is not None: _s[_k] = _s[_k] * _f
        if K3_PROBE_POST is not None:
            K3_PROBE_POST(l, K3)
        if k4_aug:
            # harmonic projection of the d=4 K slices onto the r=1 matrix core,
            # plus the two factored contributions (coefs verified numerically)
            K4v, K22, K31 = K[(4,)], K[(2, 2)], K[(3, 1)]
            G_new = (statics["cA"] * (torch.diag(K4v) + torch.diag(K22.sum(dim=1))
                                      + (0.0 if aug_no_k211 else K211_cap))
                     + (0.0 if aug_no_31 else statics["cS"]) * 0.5 * (K31 + K31.T)
                     + statics["cI"] * (K4v.sum() + K22.sum())
                     * torch.eye(n, dtype=dtype, device=DEV))
            if contrib_comb is not None:
                G_new = G_new + statics["cA"] * (contrib_comb + rep_a_term)
            elif contrib_a is not None:
                G_new = G_new + statics["cA"] * contrib_a
            if birth_part is not None:
                G_new = G_new + (-2.0 * statics["cA"]) * birth_part
            elif contrib_b is not None:
                G_new = G_new + (-2.0 * statics["cA"]) * contrib_b
            G = G_new
            if GPARTS is not None:   # стенд 25.09: частини G для атрибуції низькорангового залишку
                GPARTS.clear()
                GPARTS["comb"] = (statics["cA"] * (contrib_comb + rep_a_term)) if contrib_comb is not None else (statics["cA"] * contrib_a if contrib_a is not None else None)
                GPARTS["birth"] = ((-2.0 * statics["cA"]) * birth_part) if birth_part is not None else ((-2.0 * statics["cA"]) * contrib_b if contrib_b is not None else None)
                GPARTS["k31"] = (0.0 if aug_no_31 else statics["cS"]) * 0.5 * (K31 + K31.T)
            if g_hook is not None:
                G = g_hook(l, G, dict(C=C, mu=mu, var=torch.diagonal(C), K21=K[(2, 1)],
                                      K3v=K[(3,)], K31=K31, K22=K22, K4v=K4v,
                                      S3=S3, S21=S21, D21_prev=D21, D3_prev=D3))
            K4_sigma = None
            if debug_state is not None:
                debug_state.append(dict(
                    G=G.clone(), K4v=K4v.clone(), K22=K22.clone(), K31=K31.clone(),
                    K211_cap=K211_cap.clone(), pk211_cap=pk211_cap.clone(),
                    contrib_a=None if contrib_a is None else contrib_a.clone(),
                    contrib_b=None if contrib_b is None else contrib_b.clone(),
                    mu=mu.clone(), C=C.clone(), S3=S3.clone(), S21=S21.clone()))
        else:
            K4_sigma = statics["P2"] * (statics["k4_c4"] * K[(4,)].sum()
                                        + statics["k4_c22"] * K[(2, 2)].sum())
            K4_sigma = float(K4_sigma)
        if k4_vec:
            # per-neuron kappa4 diagonal content; mean equals K4_sigma by construction
            K4_vecstate = (n * statics["P2"]) * (statics["k4_c4"] * K[(4,)]
                                                 + statics["k4_c22"] * K[(2, 2)].sum(dim=0))
        if k4_prop == "mat":
            # (2,2)-slice matrix + (4,) diagonal; mean equals K4_sigma by construction
            K4_matstate = (n * n * statics["P2"] * statics["k4_c22"]) * K[(2, 2)] \
                + torch.diag((n * statics["P2"] * statics["k4_c4"]) * K[(4,)])
        # low-rank confinement of old sources (F52): after this layer's births,
        # truncate/re-truncate per the (min_age, rank) schedule
        if fac_schedule is not None and K3 is not None and K3.sources:
            K3.factorize_schedule_(fac_schedule, basis=fac_basis, eig=fac_eig)
        # F66 option C: age-gated SHARED basis for old sources (after this layer's births)
        if age_rank is not None and K3 is not None and K3.sources:
            tiers = age_rank if isinstance(age_rank, list) else [age_rank]
            for (a_, r_) in tiers:   # nested tiers: apply youngest gate first
                K3.confine_old_(a_, r_, eig=age_eig)
        # symmetric Tucker-rank-R confinement of the accumulated K[3] (floor-play probe):
        # project every tensor leg onto the dominant rank-R subspace; compounds each layer.
        if tucker_rank is not None and tucker_rank < n and K3 is not None and K3.sources:
            G = K3.gram(n, dtype, mu.device)
            evals, evecs = torch.linalg.eigh(G.detach())
            Q = evecs[:, -tucker_rank:].contiguous()  # top-R eigenvectors
            K3.project_(Q)
        # ---- online per-layer mean correction (fitted offline; applied INSIDE the loop
        # so it propagates). delta = feats @ beta[l]; only the mean channel is corrected.
        if mean_corr is not None or collect_feats:
            D3f = D3 if D3 is not None else torch.zeros(n, dtype=dtype, device=DEV)
            D21nf = (D21.abs().sum(dim=1) if D21 is not None
                     else torch.zeros(n, dtype=dtype, device=DEV))
            feats = build_corr_feats(mu_pre, var_pre, mu, D3f, D21nf, k4_in, dtype)
            if collect_feats:
                feats_collect.append(feats)
                predpre_collect.append(mu.clone())
            if mean_corr is not None:
                mu = mu + feats @ mean_corr[l]
        rows.append(mu.clone())
        # teacher-force the mean channel: propagate the TRUE post-ReLU mean to the next
        # layer's linear step (means propagate exactly linearly), isolating fresh error.
        if force_mean is not None:
            mu = torch.as_tensor(force_mean[l], dtype=dtype, device=DEV)
        if MU_HOOK is not None:   # стенд 15.10: корекція середнього після шару l (оракул крон)
            mu = MU_HOOK(l, mu)

    if collect_feats:
        return (torch.stack(rows), torch.stack(feats_collect),
                torch.stack(predpre_collect))
    if return_diag:
        out = [torch.stack(rows), torch.stack(diag_mu_in), torch.stack(diag_var_in),
               torch.stack(diag_d3), torch.stack(diag_d21n),
               torch.tensor(diag_k4, dtype=dtype, device=DEV)]
        if svd_d21:
            out.append(torch.stack(diag_svd))
        return tuple(out)
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def reference_predict(weights: np.ndarray, kind_name="SIMPLE", return_G=False):
    from mlp_kprop.mlp import MLP
    from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
    L, n = weights.shape[0], weights.shape[1]
    mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    mlp = mlp.double()
    for i in range(L):
        mlp.Ws[i].weight.data = torch.from_numpy(weights[i].astype(np.float64).T).contiguous()
    K_in = {1: torch.zeros(n, dtype=torch.float64), 2: torch.eye(n, dtype=torch.float64)}
    K_by_layer = mlp_kprop(mlp, K_in, k_max=3, kind=Kind[kind_name], factor=True,
                           output_all=True, up_to_layer=f"act{L - 1}")
    mus = torch.stack([K_by_layer[f"act{i}"][1].to_tensor() for i in range(L)])
    if not return_G:
        return mus
    Gs = [K_by_layer[f"act{i}"][4].core if 4 in K_by_layer[f"act{i}"] else None
          for i in range(L)]
    return mus, Gs


def main():
    if "--small" in sys.argv:
        # F51 trap: reference proj_coef builds tensors at default dtype (f32) ->
        # ~3e-8 rel noise in ITS coefficients unless default dtype is f64.
        torch.set_default_dtype(torch.float64)
        aug = "--aug" in sys.argv
        fac = None
        for a in sys.argv:
            if a.startswith("--fac="):
                fac = tuple(tuple(int(x) for x in t.split(":"))
                            for t in a.split("=")[1].split(","))
        n, L = 48, 5
        rng = np.random.default_rng(0)
        weights = rng.normal(0.0, math.sqrt(2.0 / n), size=(L, n, n))
        t0 = time.perf_counter()
        lean = lean_k3_predict(weights, k4_aug=aug, fac_schedule=fac)
        t_lean = time.perf_counter() - t0
        t0 = time.perf_counter()
        ref = reference_predict(weights, kind_name="AUGMENT" if aug else "SIMPLE")
        t_ref = time.perf_counter() - t0
        diff = (lean - ref).abs()
        print(f"small parity n={n} L={L} aug={aug}:")
        for i in range(L):
            print(f"  layer {i}: max|lean-ref| = {diff[i].max():.3e}   "
                  f"(scale {ref[i].abs().max():.3e})")
        print(f"  overall max abs diff = {diff.max():.3e}   t_lean={t_lean:.1f}s t_ref={t_ref:.1f}s")
        return

    aug = "--aug" in sys.argv
    abl = dict(aug_no_a="--no_a" in sys.argv, aug_no_b="--no_b" in sys.argv,
               aug_no_feed="--no_feed" in sys.argv, aug_no_31="--no_31" in sys.argv,
               aug_no_211="--no_211" in sys.argv)
    abl_tag = "".join(k.replace("aug_no", ",-")[0:] for k, v in abl.items() if v)
    for a in sys.argv:
        if a.startswith("--recent="):
            abl["aug_recent_k"] = int(a.split("=")[1])
            abl_tag += f",recent{abl['aug_recent_k']}"
        if a.startswith("--fac="):
            # e.g. --fac=2:128 or tiered --fac=2:256,4:128,8:64
            sched = tuple(tuple(int(x) for x in tier.split(":"))
                          for tier in a.split("=")[1].split(","))
            abl["fac_schedule"] = sched
            abl_tag += ",fac" + "|".join(f"{a_}:{r_}" for a_, r_ in sched)
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_mlps = int(pos[0]) if pos else 2
    dump_dir = pos[1] if len(pos) > 1 else DUMP_DIR
    files = sorted(glob.glob(dump_dir + r"\mlp_*.npz"))[:n_mlps]
    statics = None
    for f in files:
        d = np.load(f)
        weights, gt = d["weights"], d["all_layer_means"].astype(np.float64)
        if statics is None:
            statics = build_static_coefs(weights.shape[1])
        t0 = time.perf_counter()
        pred = lean_k3_predict(weights, statics=statics, k4_aug=aug, **abl).numpy()
        dt = time.perf_counter() - t0
        mse_final = float(np.mean((pred[-1] - gt[-1]) ** 2))
        mse_all = float(np.mean((pred - gt) ** 2))
        print(f"{f.split(chr(92))[-1]}: lean(aug={aug}{abl_tag}) "
              f"mse_final={mse_final:.3e}  mse_all={mse_all:.3e}  t={dt:.1f}s")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
