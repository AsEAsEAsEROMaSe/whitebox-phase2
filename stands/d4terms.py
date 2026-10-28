"""Діаграми k_max=4 для зрізу (1,1,1) (стенд 26.09), обчислювані з n²-блоків (var, C_off, D3, D21) і щільного κ₃.
Блоки з κ₄ (3 осі або 4-й порядок) пропускаються. Обчислення по слабу першої осі: Δ[a-slab, :, :].
Кожна діаграма: count · Π блоків · Π_ax w(k_ax, 1), симетризовано (середнє по 6 перестановках), нулі на збігах індексів."""
import itertools, torch
from mlp_kprop.partitions import vec_part_coef

PERMS = list(itertools.permutations(range(3)))


def select_diagrams(T3, T4, with_k4=False):
    """Нові (1,1,1)-діаграми k_max=4 (нема в T3), блоки яких доступні."""
    out = []
    for vp, cnt in T4[(1, 1, 1)].items():
        if vp in T3[(1, 1, 1)]:
            continue
        ok = True
        for v in vp:
            nz = sum(1 for x in v if x > 0)
            if nz == 3 and tuple(v) != (1, 1, 1):
                ok = False
            if nz <= 2 and sum(v) >= 4 and not with_k4:
                ok = False
            if nz <= 2 and sum(v) >= 5:
                ok = False
        if ok:
            out.append((tuple(tuple(v) for v in vp), float(cnt) * float(vec_part_coef(vp, divide_fac=True))))
    return out


def _block_val(v, blocks, a_sl, n, dev, dtype):
    """Значення блоку v=(va,vb,vc) на слабі: тензор, що транслюється до (s, n, n). None = блок κ₃."""
    s = a_sl.stop - a_sl.start
    if tuple(v) == (1, 1, 1):
        return None
    axes = [i for i in range(3) if v[i] > 0]
    if len(axes) == 1:
        ax = axes[0]; k = v[ax]
        vec = {2: blocks["var"], 3: blocks["D3"]}.get(k)
        if vec is None:
            vec = blocks["K4"]
        vec = vec[a_sl] if ax == 0 else vec
        shape = [1, 1, 1]; shape[ax] = -1
        return vec.reshape(shape)
    i, j = axes
    ki, kj = v[i], v[j]
    if (ki, kj) == (1, 1):
        M = blocks["C"]
    elif (ki, kj) == (1, 2):
        M = blocks["D21"].T          # κ_{i j j} = D21[j, i]
    elif (ki, kj) == (2, 1):
        M = blocks["D21"]            # κ_{i i j} = D21[i, j]
    elif (ki, kj) == (2, 2):
        M = blocks["K22"]
    elif (ki, kj) == (1, 3):
        M = blocks["K31"].T
    elif (ki, kj) == (3, 1):
        M = blocks["K31"]
    else:
        raise ValueError(v)
    if i == 0:
        M = M[a_sl]
        return M[:, :, None] if j == 1 else M[:, None, :]
    return M[None, :, :]


def omega_slab(diagrams, blocks, wick, a_sl, n, dev, dtype, with_k3):
    """Сума симетризованих ваг діаграм на слабі: with_k3=True -> ті, що містять блок κ₃ (множник при κ₃);
    False -> діаграми лише з n²-блоків (прямий внесок). Повертає (s, n, n) або None."""
    acc = None
    for vp, cnt in diagrams:
        has3 = any(tuple(v) == (1, 1, 1) for v in vp)
        if has3 != with_k3:
            continue
        for perm in PERMS:
            term = None
            korders = [0, 0, 0]
            for v in vp:
                vs = tuple(v[perm[i]] for i in range(3))   # блок з переставленими осями
                for i in range(3):
                    korders[i] += vs[i]
                bv = _block_val(vs, blocks, a_sl, n, dev, dtype)
                if bv is None:
                    continue
                term = bv if term is None else term * bv
            for i in range(3):
                w = wick(korders[i], 1)
                w = w[a_sl] if i == 0 else w
                shape = [1, 1, 1]; shape[i] = -1
                w = w.reshape(shape)
                term = w if term is None else term * w
            term = term * (cnt / 6.0)
            acc = term if acc is None else acc + term
    if acc is None:
        return None
    return acc.expand(a_sl.stop - a_sl.start, n, n) if acc.dim() == 3 else acc


def distinct_mask(a_sl, n, dev):
    """(s, n, n) маска a≠b, a≠c, b≠c."""
    a = torch.arange(a_sl.start, a_sl.stop, device=dev)
    b = torch.arange(n, device=dev)
    m = (a[:, None, None] != b[None, :, None]) & (a[:, None, None] != b[None, None, :]) & (b[None, :, None] != b[None, None, :])
    return m


def delta_slab(diagrams, blocks, wick, k3_slab, a_sl, n, dev, dtype):
    """Δ_new[a-slab] = Ω_κ3 ∘ κ₃[a-slab] + Ω_0, нулі на збігах індексів. k3_slab: (s, n, n) симетричний κ₃ або None."""
    out = None
    om3 = omega_slab(diagrams, blocks, wick, a_sl, n, dev, dtype, True)
    if om3 is not None and k3_slab is not None:
        out = om3 * k3_slab
    om0 = omega_slab(diagrams, blocks, wick, a_sl, n, dev, dtype, False)
    if om0 is not None:
        out = om0.clone() if out is None else out + om0
    if out is None:
        return None
    return out * distinct_mask(a_sl, n, dev).to(out.dtype)


# ---- швидка форма: групування за блоком (b,c); решта як batched matmul по слабу ----
_PAIRS = ((0, 1), (1, 2), (0, 2))   # (a,b), (b,c), (a,c)


def compile_diagrams(diagrams, blocks, wick, n):
    """Σ_діаграм Σ_перм coef·Π блоків·Π wick -> класи key=(блоки на (b,c), has_k3); у класі терми
    (назви блоків на (a,b), назви на (a,c), α, β, γ). Ω[a,b,c] = Σ_key M_bc ⊙ Σ_t L_t[a,b] R_t[a,c],
    L_t = α_t[a]β_t[b]M_ab, R_t = γ_t[c]M_ac."""
    cls = {}
    ones = torch.ones(n, dtype=blocks["C"].dtype, device=blocks["C"].device)
    for vp, coef in diagrams:
        has3 = any(tuple(v) == (1, 1, 1) for v in vp)
        for perm in PERMS:
            vs_list = [tuple(v[perm[i]] for i in range(3)) for v in vp]
            korders = [sum(vs[i] for vs in vs_list) for i in range(3)]
            vecs = [ones.clone(), ones.clone(), ones.clone()]
            pairs = {p: [] for p in _PAIRS}
            for vs in vs_list:
                if vs == (1, 1, 1):
                    continue
                axes = [i for i in range(3) if vs[i] > 0]
                if len(axes) == 1:
                    ax = axes[0]; k = vs[ax]
                    vecs[ax] = vecs[ax] * {2: blocks["var"], 3: blocks["D3"]}.get(k, blocks.get("K4"))
                else:
                    i, j = axes; ki, kj = vs[i], vs[j]
                    name = {(1, 1): "C", (1, 2): "D21T", (2, 1): "D21", (2, 2): "K22", (1, 3): "K31T", (3, 1): "K31"}[(ki, kj)]
                    pairs[(i, j)].append(name)
            for i in range(3):
                vecs[i] = vecs[i] * wick(korders[i], 1)
            vecs[0] = vecs[0] * (coef / 6.0)
            key = (tuple(sorted(pairs[(1, 2)])), has3)
            cls.setdefault(key, []).append((tuple(sorted(pairs[(0, 1)])), tuple(sorted(pairs[(0, 2)])), vecs[0], vecs[1], vecs[2]))
    return cls


def _mat(names, blocks):
    out = None
    for name in names:
        M = blocks[name[:-1]].T if name.endswith("T") else blocks[name]
        out = M if out is None else out * M
    return out


def omega_fast(compiled, blocks, a_sl, n, with_k3):
    """Ω на слабі (s,n,n) для класів з has_k3 == with_k3, або None."""
    s = a_sl.stop - a_sl.start
    acc = None
    for (bc_names, has3), terms in compiled.items():
        if has3 != with_k3:
            continue
        Ls, Rs = [], []
        for ab_names, ac_names, al, be, ga in terms:
            L = al[a_sl][:, None] * be[None, :]
            Mab = _mat(ab_names, blocks)
            if Mab is not None:
                L = L * Mab[a_sl]
            R = ga[None, :].expand(s, n)
            Mac = _mat(ac_names, blocks)
            if Mac is not None:
                R = R * Mac[a_sl]
            Ls.append(L); Rs.append(R)
        Lt = torch.stack(Ls, 2)            # (s, n, T)
        Rt = torch.stack(Rs, 1)            # (s, T, n)
        V = torch.bmm(Lt, Rt)              # (s, n, n)
        Mbc = _mat(bc_names, blocks)
        if Mbc is not None:
            V = V * Mbc[None]
        acc = V if acc is None else acc + V
    return acc


def delta_slab_fast(compiled, blocks, k3_slab, a_sl, n, dev):
    out = None
    om3 = omega_fast(compiled, blocks, a_sl, n, True)
    if om3 is not None and k3_slab is not None:
        out = om3 * k3_slab
    om0 = omega_fast(compiled, blocks, a_sl, n, False)
    if om0 is not None:
        out = om0 if out is None else out + om0
    if out is None:
        return None
    return out * distinct_mask(a_sl, n, dev).to(out.dtype)
